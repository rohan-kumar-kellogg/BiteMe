import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import math
import pickle
import time
from datetime import datetime
from uuid import uuid4

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

from models.probes import ProbePredictor
from models.hard_negative_reranker import HardNegativePairPredictor
from models.retrieval import predict_dish, predict_dish_with_confidence
from models.tag_head import CLIPTagPredictor
from models.vision import VisionEncoder
from utils.path_utils import normalize_path

DATA_DIR = "data"
PROFILE_DIR = os.path.join(DATA_DIR, "user_profiles")
PROFILE_IMAGE_DIR = os.path.join(PROFILE_DIR, "images")
LABEL_HEADS_PATH = os.path.join(DATA_DIR, "models", "label_heads.pkl")
TAG_HEAD_CKPT_PATH = os.path.join(DATA_DIR, "models", "clip_mlp_tag_head.pt")
PROBES_PATH = os.path.join(DATA_DIR, "models", "probes.pkl")
PAIR_RERANKER_PATH = os.path.join(DATA_DIR, "models", "hard_negative_pair_reranker.pt")


def ensure_dirs():
    os.makedirs(PROFILE_DIR, exist_ok=True)
    os.makedirs(PROFILE_IMAGE_DIR, exist_ok=True)


@st.cache_resource
def load_assets():
    encoder = VisionEncoder()
    cuisines = pd.read_csv(f"{DATA_DIR}/cuisines.csv")["cuisine"].tolist()
    dish_classes_path = f"{DATA_DIR}/dish_classes.csv"
    if os.path.exists(dish_classes_path):
        dish_classes = pd.read_csv(dish_classes_path)["dish_class"].tolist()
    else:
        # Backward compatibility for older generated datasets.
        dishes_df = pd.read_csv(f"{DATA_DIR}/dishes.csv")
        dish_classes = sorted(dishes_df["dish_label"].dropna().astype(str).unique().tolist())
    ingredients_master = pd.read_csv(f"{DATA_DIR}/ingredients_master.csv")["ingredient"].tolist()
    taste_attrs_df = pd.read_csv(f"{DATA_DIR}/taste_attributes.csv")
    affinity_df = pd.read_csv(f"{DATA_DIR}/semantic_affinities.csv")
    archetype_profiles_df = pd.read_csv(f"{DATA_DIR}/archetype_profiles.csv")

    archetypes_df = pd.read_csv(f"{DATA_DIR}/user_archetypes.csv")
    user_embeddings = np.load(f"{DATA_DIR}/user_embeddings.npy")
    dishes_df = pd.read_csv(f"{DATA_DIR}/dishes.csv")
    dish_vectors = np.load(f"{DATA_DIR}/dish_vectors.npy")
    dish_vectors = dish_vectors.astype(np.float32, copy=False)
    dish_vectors = dish_vectors / (np.linalg.norm(dish_vectors, axis=1, keepdims=True) + 1e-12)

    restaurants_df = pd.read_csv(f"{DATA_DIR}/restaurants.csv")
    restaurant_embeddings = np.load(f"{DATA_DIR}/restaurant_embeddings.npy")

    return (
        encoder,
        cuisines,
        dish_classes,
        ingredients_master,
        taste_attrs_df,
        affinity_df,
        archetype_profiles_df,
        archetypes_df,
        user_embeddings,
        dishes_df,
        dish_vectors,
        restaurants_df,
        restaurant_embeddings,
    )


@st.cache_resource
def load_label_heads():
    if not os.path.exists(LABEL_HEADS_PATH):
        return None
    try:
        with open(LABEL_HEADS_PATH, "rb") as f:
            payload = pickle.load(f)
        return payload
    except Exception:
        return None


@st.cache_resource
def load_tag_predictor():
    if not os.path.exists(TAG_HEAD_CKPT_PATH):
        return None
    try:
        return CLIPTagPredictor(TAG_HEAD_CKPT_PATH)
    except Exception:
        return None


@st.cache_resource
def load_probe_predictor():
    if not os.path.exists(PROBES_PATH):
        return None
    try:
        return ProbePredictor.from_path(PROBES_PATH)
    except Exception:
        return None


@st.cache_resource
def load_pair_reranker():
    if not os.path.exists(PAIR_RERANKER_PATH):
        return None
    try:
        return HardNegativePairPredictor(PAIR_RERANKER_PATH)
    except Exception:
        return None


def cuisine_one_hot(cuisine: str, cuisines: list[str]) -> np.ndarray:
    v = np.zeros(len(cuisines), dtype=np.float32)
    if cuisine in cuisines:
        v[cuisines.index(cuisine)] = 1.0
    return v


def dish_class_one_hot(dish_class: str, dish_classes: list[str]) -> np.ndarray:
    v = np.zeros(len(dish_classes), dtype=np.float32)
    if dish_class in dish_classes:
        v[dish_classes.index(dish_class)] = 1.0
    return v


def ingredient_vector(selected: list[str], ingredients_master: list[str]) -> np.ndarray:
    v = np.zeros(len(ingredients_master), dtype=np.float32)
    for ing in selected:
        if ing in ingredients_master:
            v[ingredients_master.index(ing)] = 1.0
    return v


def compute_attribute_vector(
    encoder: VisionEncoder,
    image_emb: np.ndarray,
    taste_attrs_df: pd.DataFrame,
    *,
    temperature: float = 0.07,
) -> np.ndarray:
    out = []
    for row in taste_attrs_df.itertuples(index=False):
        pos_prompts = [p.strip() for p in str(row.positive_prompt).split("|") if p.strip()]
        neg_prompts = [p.strip() for p in str(row.negative_prompt).split("|") if p.strip()]
        pos = encoder.score_image_prompts_from_emb(image_emb, pos_prompts).astype(np.float32)
        neg = encoder.score_image_prompts_from_emb(image_emb, neg_prompts).astype(np.float32)
        gap = float(np.mean(pos) - np.mean(neg))
        t = max(float(temperature), 1e-6)
        out.append(float(1.0 / (1.0 + float(np.exp(-gap / t)))))
    return np.asarray(out, dtype=np.float32)


def compute_affinity_vector(
    encoder: VisionEncoder,
    image_emb: np.ndarray,
    affinity_df: pd.DataFrame,
    *,
    temperature: float = 0.07,
) -> np.ndarray:
    out = []
    for row in affinity_df.itertuples(index=False):
        pos_prompts = [p.strip() for p in str(row.positive_prompt).split("|") if p.strip()]
        neg_prompts = [p.strip() for p in str(row.negative_prompt).split("|") if p.strip()]
        pos = encoder.score_image_prompts_from_emb(image_emb, pos_prompts).astype(np.float32)
        neg = encoder.score_image_prompts_from_emb(image_emb, neg_prompts).astype(np.float32)
        gap = float(np.mean(pos) - np.mean(neg))
        t = max(float(temperature), 1e-6)
        out.append(float(1.0 / (1.0 + float(np.exp(-gap / t)))))
    return np.asarray(out, dtype=np.float32)


def build_dish_vector(
    img_emb: np.ndarray,
) -> np.ndarray:
    vec = img_emb.astype(np.float32, copy=False)
    return vec / (np.linalg.norm(vec) + 1e-12)


def load_user_profile(user_id: str):
    path = os.path.join(PROFILE_DIR, f"{user_id}.npz")
    if not os.path.exists(path):
        return None
    data = np.load(path, allow_pickle=True)
    dish_vecs = data["dish_vectors"]
    return dish_vecs


def save_user_profile(user_id: str, dish_vectors: np.ndarray):
    path = os.path.join(PROFILE_DIR, f"{user_id}.npz")
    np.savez(path, dish_vectors=dish_vectors)


def _safe_user_id(user_id: str) -> str:
    out = "".join(ch for ch in user_id if ch.isalnum() or ch in ("-", "_")).strip()
    return out or "user"


def _user_photo_log_path(user_id: str) -> str:
    return os.path.join(PROFILE_DIR, f"{_safe_user_id(user_id)}_photos.csv")


def load_user_photo_log(user_id: str) -> pd.DataFrame:
    path = _user_photo_log_path(user_id)
    if not os.path.exists(path):
        return pd.DataFrame(
            columns=[
                "timestamp",
                "image_path",
                "dish_class_context",
                "cuisine_context",
                "ingredient_tags",
                "top_flavors",
                "top_affinities",
            ]
        )
    return pd.read_csv(path)


def append_user_photo_log(user_id: str, rows: list[dict]):
    old = load_user_photo_log(user_id)
    new = pd.DataFrame(rows)
    merged = pd.concat([old, new], ignore_index=True)
    merged.to_csv(_user_photo_log_path(user_id), index=False)


def clear_user_history(user_id: str):
    safe_uid = _safe_user_id(user_id)
    profile_path = os.path.join(PROFILE_DIR, f"{safe_uid}.npz")
    photo_log_path = _user_photo_log_path(user_id)
    user_img_dir = os.path.join(PROFILE_IMAGE_DIR, safe_uid)

    if os.path.exists(profile_path):
        os.remove(profile_path)
    if os.path.exists(photo_log_path):
        os.remove(photo_log_path)
    if os.path.isdir(user_img_dir):
        for name in os.listdir(user_img_dir):
            p = os.path.join(user_img_dir, name)
            if os.path.isfile(p):
                os.remove(p)
        try:
            os.rmdir(user_img_dir)
        except OSError:
            pass


def _top_k_labels(names: list[str], values: np.ndarray, k: int = 3) -> str:
    idx = np.argsort(-values)[:k]
    parts = [f"{names[int(i)]} ({float(values[int(i)]):.2f})" for i in idx]
    return ", ".join(parts)


def _confident_labels(names: list[str], values: np.ndarray, threshold: float = 0.60, k: int = 3) -> list[str]:
    idx = np.argsort(-values)
    out = []
    for i in idx:
        v = float(values[int(i)])
        if v >= threshold:
            out.append(f"{names[int(i)]} ({v:.2f})")
        if len(out) >= k:
            break
    return out


ARCHETYPE_DESCRIPTIONS = {
    "The Adventurous Explorer": "You bounce across styles and cuisines with high curiosity.",
    "The Comfort Loyalist": "You favor reliable hits and rich comfort-forward flavors.",
    "The Global Heat-Seeker": "You chase bold, high-intensity dishes with global range.",
    "The Refined Minimalist": "You like cleaner profiles, quality ingredients, and balance.",
    "The Carb Devotee": "You have a clear soft spot for carb-forward mains.",
    "The Spice Enthusiast": "You consistently gravitate toward punchy, spicy flavors.",
    "The Seafood Specialist": "Seafood and brighter, coastal profiles keep showing up.",
    "The Plant-Based Purist": "You trend toward lighter, produce-forward combinations.",
    "The Indulgent Traditionalist": "You lean rich, classic, and satisfying over experimental.",
    "The Fusion Experimenter": "You blend cuisines and styles instead of staying in one lane.",
}

OPEN_VOCAB_DISH_CANDIDATES = [
    "herb-crusted lamb chops",
    "salmon and vegetables",
    "home-cooked pasta",
    "grilled fish plate",
    "seafood tartare",
    "citrus seafood dish",
    "roasted meat and vegetables",
    "creamy pasta",
    "fresh salad bowl",
    "hearty curry plate",
]


def build_dish_prototypes(dishes_df: pd.DataFrame, dish_vectors: np.ndarray):
    """
    Build per-dish-label image centroids from generated dataset.
    dish_vectors shape is [N, D] where D is CLIP embedding dim.
    """
    if dish_vectors.ndim != 2 or dish_vectors.shape[0] == 0:
        return np.zeros((0, 512), dtype=np.float32), [], {}
    img_part = dish_vectors.astype(np.float32, copy=False)
    # ensure normalized
    img_part = img_part / (np.linalg.norm(img_part, axis=1, keepdims=True) + 1e-12)

    centroids = []
    labels = []
    label_to_cuisine = {}
    for label, idx_df in dishes_df.groupby("dish_label"):
        idx = idx_df.index.to_numpy()
        if len(idx) == 0:
            continue
        c = np.mean(img_part[idx], axis=0).astype(np.float32)
        c = c / (np.linalg.norm(c) + 1e-12)
        centroids.append(c)
        labels.append(str(label))
        if "cuisine" in idx_df.columns and len(idx_df["cuisine"]) > 0:
            label_to_cuisine[str(label)] = str(idx_df["cuisine"].mode().iloc[0])
    if not centroids:
        return np.zeros((0, 512), dtype=np.float32), [], {}
    return np.vstack(centroids), labels, label_to_cuisine


def predict_open_vocab_description(encoder: VisionEncoder, image_path: str, dish_classes: list[str]) -> tuple[str, float]:
    class_labels = [c.replace("_", " ") for c in dish_classes[:120]]
    candidates = OPEN_VOCAB_DISH_CANDIDATES + class_labels
    prompts = [f"a close-up food photo of {c}" for c in candidates]
    sims = encoder.score_image_prompts(image_path, prompts).astype(np.float32)
    i = int(np.argmax(sims))
    return candidates[i], float(sims[i])


def predict_from_trained_heads(label_heads_payload, image_emb: np.ndarray) -> dict:
    if not label_heads_payload:
        return {}
    heads = label_heads_payload.get("heads", {})
    out = {}
    for name, clf in heads.items():
        try:
            probs = clf.predict_proba([image_emb])[0]
            i = int(np.argmax(probs))
            out[name] = {"label": str(clf.classes_[i]), "prob": float(probs[i])}
        except Exception:
            continue
    return out


def predict_dish_description(
    encoder: VisionEncoder,
    image_path: str,
    image_emb: np.ndarray,
    dish_centroids: np.ndarray,
    dish_labels: list[str],
    label_to_cuisine: dict[str, str],
    dish_classes: list[str],
) -> tuple[str, str]:
    """
    Hybrid prediction:
    - retrieval against known dish centroids (high precision on known classes)
    - open-vocab fallback when retrieval confidence is weak
    """
    best_label = "unknown dish"
    best_cuisine = "Inferred"
    retrieval_score = -1.0
    if dish_centroids.shape[0] > 0 and image_emb.shape[0] >= 512:
        sims = dish_centroids @ image_emb[:512].astype(np.float32, copy=False)
        j = int(np.argmax(sims))
        retrieval_score = float(sims[j])
        best_label = dish_labels[j].replace("_", " ")
        best_cuisine = label_to_cuisine.get(dish_labels[j], "Inferred")

    ov_label, ov_score = predict_open_vocab_description(encoder, image_path, dish_classes)
    # Prefer open-vocab if retrieval is weak/ambiguous.
    if retrieval_score < 0.24 or ov_score > retrieval_score + 0.04:
        return ov_label, best_cuisine if best_cuisine != "Inferred" else "Inferred"
    return best_label, best_cuisine


def infer_context_label(
    dish_class_context: str,
    cuisine_context: str,
    dish_desc: str,
    affinity_names: list[str],
    affinity_vec: np.ndarray,
) -> tuple[str, str]:
    dish_ctx = dish_desc if dish_class_context == "Auto" else dish_class_context
    if cuisine_context != "Auto":
        return dish_ctx, cuisine_context
    regional = {"peruvian", "japanese", "italian", "mexican"}
    best_idx = int(np.argmax(affinity_vec))
    best_name = affinity_names[best_idx]
    best_val = float(affinity_vec[best_idx])
    if best_name in regional and best_val >= 0.45:
        return dish_ctx, best_name.title()
    return dish_ctx, "Inferred"


def render_user_photo_gallery(user_id: str, max_items: int = 24):
    log_df = load_user_photo_log(user_id)
    if len(log_df) == 0:
        st.caption("No saved photo history yet.")
        return

    st.subheader("Your Uploaded Photos + Tags")
    st.caption("Audit view so you can verify how each photo is being interpreted.")
    view_df = log_df.tail(max_items).iloc[::-1].reset_index(drop=True)
    cols = st.columns(3)
    for i, row in view_df.iterrows():
        col = cols[i % 3]
        with col:
            if os.path.exists(str(row["image_path"])):
                st.image(str(row["image_path"]), use_container_width=True)
            st.markdown(f"**Dish read:** {row.get('dish_description', row.get('dish_class_context', 'Unknown'))}")
            st.caption(f"Context: {row.get('dish_class_context', 'Unknown')} · {row.get('cuisine_context', 'Unknown')}")
            sd = row.get("supervised_dish_name", "")
            sdc = row.get("supervised_dish_conf", np.nan)
            if pd.notna(sd) and str(sd).strip():
                if pd.notna(sdc):
                    st.caption(f"Model dish: {sd} ({float(sdc):.2f})")
                else:
                    st.caption(f"Model dish: {sd}")
            st.caption(f"Flavors: {row.get('top_flavors', '')}")
            st.caption(f"Affinities: {row.get('top_affinities', '')}")
            if pd.notna(row.get("final_score", np.nan)):
                st.caption(
                    "Scores: "
                    f"sim_01={float(row.get('sim_01', np.nan)):.3f} · "
                    f"img_text={float(row.get('image_text_sim_01', np.nan)):.3f} · "
                    f"dish_agreement={float(row.get('dish_agreement', np.nan)):.3f} · "
                    f"protein_agreement={float(row.get('protein_agreement', np.nan)):.3f} · "
                    f"cuisine_agreement={float(row.get('cuisine_agreement', np.nan)):.3f} · "
                    f"final_score={float(row.get('final_score', np.nan)):.3f}"
                )
            tags_val = row.get("ingredient_tags", "")
            if pd.notna(tags_val) and str(tags_val).strip():
                st.caption(f"Tags: {row.get('ingredient_tags', '')}")
            st.caption(str(row.get("timestamp", "")))


def update_user_embedding(user_id: str, new_dish_vectors: np.ndarray) -> np.ndarray:
    existing = load_user_profile(user_id)
    if existing is None:
        all_vecs = new_dish_vectors
    else:
        all_vecs = np.vstack([existing, new_dish_vectors])

    save_user_profile(user_id, all_vecs)
    user_vec = np.mean(all_vecs, axis=0)
    return user_vec


def infer_archetype(user_vec: np.ndarray, archetypes_df: pd.DataFrame, user_embeddings: np.ndarray) -> str:
    sims = cosine_similarity([user_vec], user_embeddings)[0]
    nn = int(np.argmax(sims))
    return archetypes_df.loc[archetypes_df.user_id == nn, "archetype"].values[0]


def to_pct(sim: float) -> float:
    # map [-1, 1] -> [0, 100]
    return round(100.0 * (sim + 1.0) / 2.0, 1)


def find_top_matches(user_vec: np.ndarray, user_embeddings: np.ndarray, archetypes_df: pd.DataFrame, top_k: int = 10):
    sims = cosine_similarity([user_vec], user_embeddings)[0]
    top_idx = np.argsort(-sims)[:top_k]

    rows = []
    for i in top_idx:
        rows.append({
            "match_user_id": int(i),
            "compatibility_pct": to_pct(float(sims[i])),
            "archetype": archetypes_df.loc[archetypes_df.user_id == int(i), "archetype"].values[0],
        })
    return pd.DataFrame(rows)


def rank_restaurants(user_vec: np.ndarray, restaurants_df: pd.DataFrame, restaurant_embeddings: np.ndarray, neighborhood: str | None = None, top_k: int = 10):
    sims = cosine_similarity([user_vec], restaurant_embeddings)[0]

    df = restaurants_df.copy()
    df["fit_pct"] = [to_pct(float(s)) for s in sims]

    if neighborhood and neighborhood != "All":
        df = df[df["neighborhood"] == neighborhood]

    df = df.sort_values("fit_pct", ascending=False).head(top_k)
    return df


def _ensure_restaurant_metadata(restaurants_df: pd.DataFrame) -> pd.DataFrame:
    """
    Backfill zip/rating columns for older generated datasets.
    """
    df = restaurants_df.copy()
    neighborhood_to_zip = {
        "Downtown": "60601",
        "River North": "60654",
        "West Loop": "60607",
        "Wicker Park": "60622",
    }
    if "zip_code" not in df.columns:
        df["zip_code"] = df.get("neighborhood", "Downtown").map(neighborhood_to_zip).fillna("60601")
    if "avg_rating" not in df.columns:
        # Lightweight synthetic quality signal for ranking UX
        rng = np.random.default_rng(42)
        df["avg_rating"] = np.round(rng.uniform(3.7, 4.8, size=len(df)), 1)
    if "review_count" not in df.columns:
        rng = np.random.default_rng(7)
        df["review_count"] = rng.integers(45, 1200, size=len(df))
    return df


def rank_restaurants_by_zip(
    user_vec: np.ndarray,
    restaurants_df: pd.DataFrame,
    restaurant_embeddings: np.ndarray,
    zip_code: str | None,
    top_k: int = 12,
):
    sims = cosine_similarity([user_vec], restaurant_embeddings)[0]
    df = _ensure_restaurant_metadata(restaurants_df)
    df["fit_pct"] = [to_pct(float(s)) for s in sims]

    zip_code = (zip_code or "").strip()
    if zip_code:
        z = df[df["zip_code"] == zip_code].copy()
        # fallback: first 3 digits bucket
        if len(z) == 0 and len(zip_code) >= 3:
            z = df[df["zip_code"].str.startswith(zip_code[:3])].copy()
        if len(z) > 0:
            df = z

    # Blend fit + quality
    df["_rank"] = (
        0.70 * (df["fit_pct"] / 100.0)
        + 0.25 * ((df["avg_rating"] - 3.0) / 2.0).clip(0, 1)
        + 0.05 * np.log1p(df["review_count"]) / np.log1p(max(1, int(df["review_count"].max())))
    )
    return df.sort_values("_rank", ascending=False).head(top_k)


def profile_confidence(dish_vecs: np.ndarray) -> float:
    n = max(1, int(dish_vecs.shape[0]))
    # More photos + tighter internal similarity -> higher confidence.
    if n == 1:
        coherence = 0.55
    else:
        sims = cosine_similarity(dish_vecs)
        tri = sims[np.triu_indices_from(sims, k=1)]
        coherence = float(np.clip(np.mean(tri), 0.0, 1.0))
    count_term = min(1.0, math.log2(n + 1) / math.log2(21))
    conf = 100.0 * (0.65 * count_term + 0.35 * coherence)
    return round(float(np.clip(conf, 0.0, 99.0)), 1)


def build_profile_story(
    user_id: str,
    avg_attr: np.ndarray,
    attr_names: list[str],
    avg_affinity: np.ndarray,
    affinity_names: list[str],
    n_photos: int,
) -> str:
    top_attr_idx = np.argsort(-avg_attr)
    low_attr_idx = np.argsort(avg_attr)
    top_aff_idx = np.argsort(-avg_affinity)

    top_attrs = [attr_names[int(i)] for i in top_attr_idx[:3]]
    low_attr = attr_names[int(low_attr_idx[0])]
    top_affs = [affinity_names[int(i)] for i in top_aff_idx[:3]]

    style_templates = [
        f"You are clearly leaning into {top_attrs[0]} and {top_attrs[1]} right now, with very little {low_attr}.",
        f"So far your uploads point to {top_attrs[0]} + {top_attrs[1]} over {low_attr}, and the pattern is clean.",
        f"Your profile is already signaling {top_attrs[0]} and {top_attrs[1]} much more than {low_attr}.",
    ]
    range_templates = [
        f"From {top_affs[0]} to {top_affs[1]}, you have real range.",
        f"There is a strong {top_affs[0]} / {top_affs[1]} thread in what you upload.",
        f"You keep toggling between {top_affs[0]} and {top_affs[1]} in a way that feels intentional.",
    ]
    roast_templates = [
        "You clearly came here to eat well, not to play it safe.",
        "This is confident food taste with zero apology.",
        "You are picking flavor-forward moves and committing to them.",
    ]
    depth_templates = [
        f"At {n_photos} photos, this is now a pretty personal read on you, {user_id}.",
        f"With {n_photos} images in, this profile is feeling less generic and more specifically you.",
        f"Now that you are at {n_photos} uploads, the signal is getting sharper and more personal.",
    ]
    prediction_templates = [
        f"If this trend holds, you will probably keep picking places that over-index on {top_attrs[0]} and {top_affs[0]}.",
        f"Prediction: you will keep gravitating toward {top_affs[0]} spots with strong {top_attrs[0]} profiles.",
        f"Given this pattern, your best matches will skew toward {top_affs[0]} and {top_attrs[0]} heavy menus.",
    ]

    seed = (sum(ord(c) for c in user_id) + n_photos) % 3
    sentences = [
        style_templates[seed],
        range_templates[(seed + 1) % 3],
        roast_templates[(seed + 2) % 3],
        depth_templates[seed],
        prediction_templates[(seed + 1) % 3],
    ]

    # Length policy: 2-3 by default, expands with more uploads, max 6.
    if n_photos <= 2:
        return (
            f"Okay, I see you — this first pass suggests {top_attrs[0]} and {top_affs[0]} energy. "
            "Upload a few more photos and I can give you a much sharper read."
        )
    if n_photos <= 4:
        keep = 2
    elif n_photos <= 10:
        keep = 3
    elif n_photos <= 18:
        keep = 4
    elif n_photos <= 30:
        keep = 5
    else:
        keep = 6
    out = sentences[: min(keep, len(sentences))]
    return " ".join(out)


def taste_map_plotly(user_embeddings: np.ndarray, archetypes_df: pd.DataFrame, user_vec: np.ndarray, max_points: int = 500):
    n = user_embeddings.shape[0]
    idx = np.arange(n)
    if n > max_points:
        idx = np.random.choice(idx, size=max_points, replace=False)

    sample_emb = user_embeddings[idx]
    sample_meta = archetypes_df.iloc[idx].reset_index(drop=True)

    X = np.vstack([sample_emb, user_vec])
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X)

    df = pd.DataFrame({
        "x": X2[:, 0],
        "y": X2[:, 1],
        "type": (["Community"] * len(sample_emb)) + ["You"],
        "archetype": list(sample_meta["archetype"].values) + [""],
        "size": [7] * len(sample_emb) + [18],
    })

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="archetype",
        symbol="type",
        size="size",
        hover_data=["type", "archetype"],
        title="Taste Map (PCA Projection)"
    )
    fig.update_layout(height=520, margin=dict(l=20, r=20, t=60, b=20))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def main():
    ensure_dirs()
    st.set_page_config(page_title="Bite Me", layout="wide")

    st.title("🍽️ Bite Me")
    st.subheader("Discover your taste profile. Meet people who eat like you.")
    st.write("Upload a few food photos. We’ll learn your taste and show you your best matches + restaurants you’ll love.")

    (
        encoder,
        cuisines,
        dish_classes,
        ingredients_master,
        taste_attrs_df,
        affinity_df,
        archetype_profiles_df,
        archetypes_df,
        community_user_embeddings,
        dishes_df,
        dish_vectors,
        restaurants_df,
        restaurant_embeddings,
    ) = load_assets()
    label_heads_payload = load_label_heads()
    tag_predictor = load_tag_predictor()
    probe_predictor = load_probe_predictor()
    pair_reranker = load_pair_reranker()

    st.sidebar.header("Your profile")
    user_id = st.sidebar.text_input("User ID (to keep evolving)", value="rohan")
    if st.sidebar.button("Clear my photo history"):
        clear_user_history(user_id)
        st.sidebar.success(f"Cleared saved history for '{user_id}'.")
    dish_class_context = st.sidebar.selectbox("Dish class context", ["Auto"] + dish_classes, index=0)
    cuisine_context = st.sidebar.selectbox("Cuisine context", ["Auto"] + cuisines, index=0)
    zip_code = st.sidebar.text_input("ZIP code", value="60601")
    debug_mode = st.sidebar.checkbox("Debug retrieval predictions", value=False)
    use_prompt_rerank = st.sidebar.checkbox("Use prompt-based rerank (experimental)", value=False)
    use_multi_crop = st.sidebar.checkbox("Use multi-crop embedding", value=True)
    use_text_ensemble = st.sidebar.checkbox("Use image-text ensemble retrieval", value=True)
    use_protein_probe = st.sidebar.checkbox("Use protein probe in rerank", value=True)
    scoring_mode_choice = st.sidebar.selectbox("Scoring mode", ["baseline", "blended"], index=0)
    confidence_threshold = st.sidebar.slider("Confidence threshold (abstain below)", min_value=0.50, max_value=0.99, value=0.86, step=0.01)
    rerank_enabled = bool(tag_predictor is not None or (probe_predictor is not None and use_protein_probe))
    if scoring_mode_choice == "blended" and pair_reranker is None:
        st.sidebar.warning("Blended mode requested but pairwise reranker checkpoint is missing. Falling back to baseline mode.")
    active_scoring_mode = "blended" if (scoring_mode_choice == "blended" and pair_reranker is not None) else "baseline"
    rerank_mode = "MLP+Probe" if rerank_enabled else "Retrieval-only"
    st.sidebar.caption(f"Tag head loaded: {bool(tag_predictor is not None)}")
    st.sidebar.caption(f"Probe loaded: {bool(probe_predictor is not None)}")
    st.sidebar.caption(f"Pair reranker loaded: {bool(pair_reranker is not None)}")
    st.sidebar.caption(f"Encoder device: {encoder.device}")
    st.sidebar.caption(f"Dataset size: {len(dishes_df)}")
    st.sidebar.caption(f"Rerank mode: {rerank_mode}")
    st.sidebar.caption(f"Multi-crop: {use_multi_crop}")
    st.sidebar.caption(f"Text ensemble: {use_text_ensemble}")
    st.sidebar.caption(f"Protein probe: {use_protein_probe and (probe_predictor is not None)}")
    st.sidebar.caption(f"Active scoring mode: {active_scoring_mode}")
    st.sidebar.caption(f"Confidence threshold: {confidence_threshold:.2f}")
    if not rerank_enabled:
        st.sidebar.warning("MLP rerank disabled — running retrieval-only.")

    st.sidebar.write("Optional ingredient tags (explainability only)")
    ing = st.sidebar.multiselect("Ingredient tags", ingredients_master, default=[])
    if st.sidebar.button("Run model smoke test (10 random images)"):
        sample_n = min(10, len(dishes_df))
        if sample_n <= 0:
            st.sidebar.error("No dishes available for smoke test.")
        else:
            sample = dishes_df.sample(n=sample_n, random_state=42).reset_index(drop=True)
            rows = []
            top1_good = 0
            for r in sample.itertuples(index=False):
                q = str(r.image_path)
                qn = normalize_path(q)
                top = predict_dish(
                    q,
                    dishes_df,
                    dish_vectors,
                    encoder=encoder,
                    tag_predictor=tag_predictor,
                    top_k=50,
                    top_n=3,
                    use_rerank=rerank_enabled,
                    use_prompt_tags=use_prompt_rerank,
                    multi_crop=use_multi_crop,
                    use_text_ensemble=use_text_ensemble,
                    probe_predictor=probe_predictor,
                    use_protein_probe=use_protein_probe and (probe_predictor is not None),
                    pair_reranker=pair_reranker if active_scoring_mode == "blended" else None,
                    exclude_image_paths={qn},
                    scoring_mode=active_scoring_mode,
                    blended_retrieval_w=0.8,
                    blended_mlp_w=0.1,
                    blended_pair_w=0.1,
                )
                pred1 = str(top[0].get("dish_label", "")) if top else ""
                true1 = str(getattr(r, "dish_label", ""))
                ok = pred1.strip().lower() == true1.strip().lower()
                if ok:
                    top1_good += 1
                rows.append(
                    {
                        "query_image_path": q,
                        "true_dish_label": true1,
                        "pred_top1": pred1,
                        "top1_match": ok,
                        "top3": ", ".join([str(x.get("dish_label", "")) for x in top[:3]]),
                        "top1_final_score": float(top[0].get("final_score", np.nan)) if top else np.nan,
                    }
                )
            top1_acc = float(top1_good / max(1, sample_n))
            st.sidebar.success(f"Smoke test top1 over {sample_n} samples: {top1_acc:.3f}")
            smoke_df = pd.DataFrame(rows)
            st.subheader("Smoke Test Predictions (10 random dataset images)")
            st.dataframe(smoke_df, use_container_width=True)
            fails = smoke_df[~smoke_df["top1_match"]]
            if len(fails) > 0:
                st.caption("Smoke test failures:")
                st.dataframe(fails, use_container_width=True)

    files = st.file_uploader("Upload food photos", type=["jpg", "jpeg", "png", "webp", "heic", "heif"], accept_multiple_files=True)

    if st.button("Update my taste profile"):
        if not files:
            st.error("Upload at least 1 food photo.")
            return

        new_dish_vecs = []
        new_attr_vecs = []
        new_affinity_vecs = []
        upload_rows = []
        timing_limit = 10
        for idx, f in enumerate(files):
            ext = os.path.splitext(f.name)[1].lower() if f.name else ".jpg"
            if ext not in {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}:
                ext = ".jpg"
            safe_uid = _safe_user_id(user_id)
            user_dir = os.path.join(PROFILE_IMAGE_DIR, safe_uid)
            os.makedirs(user_dir, exist_ok=True)
            saved_path = os.path.join(user_dir, f"{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{uuid4().hex[:8]}{ext}")
            with open(saved_path, "wb") as out:
                out.write(f.getbuffer())

            t0 = time.perf_counter()
            img_emb = encoder.encode_image(saved_path)
            t1 = time.perf_counter()
            a_vec = compute_attribute_vector(encoder, img_emb, taste_attrs_df, temperature=0.07)
            t2 = time.perf_counter()
            aff_vec = compute_affinity_vector(encoder, img_emb, affinity_df, temperature=0.07)
            t3 = time.perf_counter()
            new_attr_vecs.append(a_vec)
            new_affinity_vecs.append(aff_vec)
            new_dish_vecs.append(build_dish_vector(img_emb))
            attr_names_local = taste_attrs_df["attribute"].tolist()
            affinity_names_local = affinity_df["affinity"].tolist()
            conf_flavors = _confident_labels(attr_names_local, a_vec, threshold=0.50, k=3)
            conf_affs = _confident_labels(affinity_names_local, aff_vec, threshold=0.50, k=3)
            tr0 = time.perf_counter()
            conf_out = predict_dish_with_confidence(
                saved_path,
                dishes_df,
                dish_vectors,
                encoder=encoder,
                tag_predictor=tag_predictor,
                top_k=50,
                top_n=10 if debug_mode else 3,
                alpha=0.15,
                use_rerank=rerank_enabled,
                use_prompt_tags=use_prompt_rerank,
                debug=debug_mode,
                multi_crop=use_multi_crop,
                use_text_ensemble=use_text_ensemble,
                probe_predictor=probe_predictor,
                use_protein_probe=use_protein_probe and (probe_predictor is not None),
                pair_reranker=pair_reranker if active_scoring_mode == "blended" else None,
                confidence_threshold=confidence_threshold,
                scoring_mode=active_scoring_mode,
                blended_retrieval_w=0.8,
                blended_mlp_w=0.1,
                blended_pair_w=0.1,
            )
            top_dishes = conf_out.get("raw_topn", [])
            tr1 = time.perf_counter()
            if idx < timing_limit:
                print(
                    f"[timing][streamlit][{idx+1}] encode_image={t1-t0:.4f}s "
                    f"attr={t2-t1:.4f}s affinity={t3-t2:.4f}s rerank={tr1-tr0:.4f}s"
                )
            best = top_dishes[0] if top_dishes else {}
            predicted_label = str(conf_out.get("predicted_label", "")).strip()
            dish_desc = (predicted_label if predicted_label else str(best.get("dish_label", "unknown dish"))).replace("_", " ")
            pred_cuisine = str(best.get("cuisine", "Inferred"))
            supervised = predict_from_trained_heads(label_heads_payload, img_emb)
            # Prefer supervised dish_name when confident.
            if "dish_name" in supervised and supervised["dish_name"]["prob"] >= 0.45:
                dish_desc = supervised["dish_name"]["label"]
            if "cuisine" in supervised and supervised["cuisine"]["prob"] >= 0.45:
                pred_cuisine = supervised["cuisine"]["label"]
            inferred_class_ctx, inferred_cuisine_ctx = infer_context_label(
                dish_class_context, cuisine_context, dish_desc, affinity_names_local, aff_vec
            )
            if inferred_cuisine_ctx == "Inferred" and pred_cuisine and pred_cuisine != "Inferred":
                inferred_cuisine_ctx = pred_cuisine
            upload_rows.append(
                {
                    "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
                    "image_path": saved_path,
                    "dish_description": dish_desc,
                    "dish_class_context": inferred_class_ctx,
                    "cuisine_context": inferred_cuisine_ctx,
                    "supervised_dish_name": supervised.get("dish_name", {}).get("label", ""),
                    "supervised_dish_conf": supervised.get("dish_name", {}).get("prob", np.nan),
                    "supervised_cuisine": supervised.get("cuisine", {}).get("label", ""),
                    "supervised_cuisine_conf": supervised.get("cuisine", {}).get("prob", np.nan),
                    "ingredient_tags": ",".join(ing),
                    "top_flavors": ", ".join(conf_flavors) if conf_flavors else "Still learning (add more photos)",
                    "top_affinities": ", ".join(conf_affs) if conf_affs else "Still learning (add more photos)",
                    "sim_01": float(best.get("sim_01", np.nan)),
                    "image_text_sim_01": float(best.get("image_text_sim_01", np.nan)),
                    "dish_agreement": float(best.get("dish_agreement", np.nan)),
                    "protein_agreement": float(best.get("protein_agreement", np.nan)),
                    "cuisine_agreement": float(best.get("cuisine_agreement", np.nan)),
                    "final_score": float(best.get("final_score", np.nan)),
                    "predicted_label": predicted_label,
                    "predicted_score": float(conf_out.get("predicted_score", np.nan)),
                    "abstained": bool(conf_out.get("abstained", False)),
                    "abstain_reason": str(conf_out.get("abstain_reason", "")),
                    "confidence_threshold": float(conf_out.get("confidence_threshold", confidence_threshold)),
                    "scoring_mode": str(conf_out.get("scoring_mode", active_scoring_mode)),
                }
            )
            if bool(conf_out.get("abstained", False)):
                st.warning(str(conf_out.get("fallback_message", "Not confident enough to make a single prediction.")))
                if conf_out.get("abstain_reason"):
                    st.caption(str(conf_out.get("abstain_reason")))
            if top_dishes:
                st.caption(
                    f"{os.path.basename(saved_path)} → "
                    f"mode={str(conf_out.get('scoring_mode', active_scoring_mode))}, "
                    f"sim_01={float(best.get('sim_01', np.nan)):.3f}, "
                    f"img_text={float(best.get('image_text_sim_01', np.nan)):.3f}, "
                    f"dish_agreement={float(best.get('dish_agreement', np.nan)):.3f}, "
                    f"pair_agreement={float(best.get('pair_agreement', np.nan)):.3f}, "
                    f"protein_agreement={float(best.get('protein_agreement', np.nan)):.3f}, "
                    f"cuisine_agreement={float(best.get('cuisine_agreement', np.nan)):.3f}, "
                    f"final_score={float(best.get('final_score', np.nan)):.3f}"
                )
                why_cols = [
                    "scoring_mode",
                    "dish_label",
                    "image_image_sim_01",
                    "image_text_sim_01",
                    "dish_agreement",
                    "pair_agreement",
                    "retrieval_weight",
                    "mlp_weight",
                    "pair_weight",
                    "protein_agreement",
                    "final_score",
                ]
                why_df = pd.DataFrame(top_dishes[:3])
                show_cols = [c for c in why_cols if c in why_df.columns]
                if show_cols:
                    st.markdown(f"**Why this label (`{os.path.basename(saved_path)}`)**")
                    st.dataframe(why_df[show_cols], use_container_width=True)
            if debug_mode:
                st.markdown(f"#### Debug predictions for `{os.path.basename(saved_path)}`")
                st.dataframe(pd.DataFrame(top_dishes[:10]), use_container_width=True)

        new_dish_vecs = np.vstack(new_dish_vecs)
        user_vec = update_user_embedding(user_id, new_dish_vecs)
        append_user_photo_log(user_id, upload_rows)

        archetype = infer_archetype(user_vec, archetypes_df, community_user_embeddings)
        cluster_id = int(archetypes_df.loc[archetypes_df["archetype"] == archetype, "cluster_id"].mode().iloc[0])

        st.success(f"Updated profile for '{user_id}'.")
        st.subheader("Your Profile")
        st.write(f"**Archetype:** {archetype}")
        total_photos = int(load_user_profile(user_id).shape[0])
        st.write(f"**Photos in your history:** {total_photos}")

        attr_names = taste_attrs_df["attribute"].tolist()
        avg_attr = np.mean(np.vstack(new_attr_vecs), axis=0) if new_attr_vecs else np.zeros(len(attr_names), dtype=np.float32)
        affinity_names = affinity_df["affinity"].tolist()
        avg_affinity = (
            np.mean(np.vstack(new_affinity_vecs), axis=0)
            if new_affinity_vecs
            else np.zeros(len(affinity_names), dtype=np.float32)
        )
        conf = profile_confidence(load_user_profile(user_id))
        short_profile = build_profile_story(user_id, avg_attr, attr_names, avg_affinity, affinity_names, total_photos)
        archetype_desc = ARCHETYPE_DESCRIPTIONS.get(archetype, "A recognizable taste cluster in the community.")
        st.markdown("### Your Profile Snapshot")
        st.markdown(f"> {short_profile}")
        st.caption(f"Archetype meaning: {archetype_desc}")
        st.progress(min(int(conf), 100), text=f"Profile confidence: {conf}% (improves as you upload more photos)")
        render_user_photo_gallery(user_id)

        st.subheader("Your Flavor Signals (from your uploads)")
        st.dataframe(
            pd.DataFrame({"attribute": attr_names, "score_0_to_1": avg_attr}).sort_values("score_0_to_1", ascending=False),
            use_container_width=True,
        )

        st.subheader("Regional + Style Affinities")
        st.dataframe(
            pd.DataFrame({"affinity": affinity_names, "score_0_to_1": avg_affinity}).sort_values(
                "score_0_to_1", ascending=False
            ),
            use_container_width=True,
        )

        st.subheader("Archetype Summary (community)")
        prof = archetype_profiles_df[archetype_profiles_df["cluster_id"] == cluster_id]
        if len(prof) > 0:
            st.dataframe(prof, use_container_width=True)

        st.subheader("Your Top Matches")
        matches_df = find_top_matches(user_vec, community_user_embeddings, archetypes_df, top_k=12)
        st.dataframe(matches_df, use_container_width=True)

        st.subheader("Restaurants You’ll Likely Enjoy")
        resto_df = rank_restaurants_by_zip(user_vec, restaurants_df, restaurant_embeddings, zip_code=zip_code, top_k=12)
        display_cols = [c for c in ["name", "cuisine", "zip_code", "neighborhood", "price_tier", "avg_rating", "review_count", "fit_pct"] if c in resto_df.columns]
        st.dataframe(resto_df[display_cols], use_container_width=True)

        st.subheader("Taste Map")
        fig = taste_map_plotly(community_user_embeddings, archetypes_df, user_vec, max_points=600)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Reserve")
        if len(resto_df) > 0:
            chosen = st.selectbox("Pick a restaurant", resto_df["name"].tolist())
            slot = st.selectbox("Pick a time", ["6:00 PM", "6:30 PM", "7:00 PM", "7:30 PM", "8:00 PM"])
            if st.button("Confirm Reservation"):
                st.success(f"✅ Reservation confirmed at {chosen} for {slot}!")
        else:
            st.info("No restaurants match the current neighborhood filter.")


if __name__ == "__main__":
    main()