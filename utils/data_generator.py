import sys
from pathlib import Path
import argparse

# Ensure project root is on PYTHONPATH so "models" imports work
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import os
import random
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from models.vision import VisionEncoder
from semantic_affinities import AFFINITY_SPECS, compute_affinity_scores
from taste_attributes import ATTRIBUTE_SPECS, compute_attribute_scores

try:
    import hdbscan  # type: ignore

    HDBSCAN_AVAILABLE = True
except Exception:
    hdbscan = None
    HDBSCAN_AVAILABLE = False


@dataclass
class Config:
    images_dir: str = "images"
    manifest_csv: str = "images/manifest.csv"
    data_dir: str = "data"

    num_users: int = 60
    meals_per_user_min: int = 6
    meals_per_user_max: int = 18

    rating_min: int = 3
    rating_max: int = 5

    compat_noise_std: float = 0.05
    n_archetypes: int = 10

    restaurants_per_cuisine: int = 6
    restaurant_noise_std: float = 0.03  # small perturbation around cuisine centroid

    seed: int = 42

    attribute_temperature: float = 0.07
    affinity_temperature: float = 0.07
    compute_prompt_features: bool = False
    timing_log_first_n: int = 10


ARCTYPE_NAMES_10 = [
    "The Adventurous Explorer",
    "The Comfort Loyalist",
    "The Global Heat-Seeker",
    "The Refined Minimalist",
    "The Carb Devotee",
    "The Spice Enthusiast",
    "The Seafood Specialist",
    "The Plant-Based Purist",
    "The Indulgent Traditionalist",
    "The Fusion Experimenter",
]

INGREDIENTS_MASTER = [
    "chicken", "beef", "rice", "cheese", "tomato",
    "fish", "beans", "cream", "spice", "pasta",
    "avocado", "tofu", "lamb", "potato", "garlic",
    "shrimp", "mushroom", "spinach", "egg", "chili",
]

CUISINE_INGREDIENT_PRIORS = {
    "Italian": ["pasta", "tomato", "cheese", "garlic", "mushroom"],
    "Japanese": ["rice", "fish", "egg", "mushroom"],
    "Indian": ["chicken", "cream", "spice", "chili"],
    "Mexican": ["beans", "avocado", "chili", "beef"],
    "American": ["beef", "cheese", "potato", "egg"],
    "Thai": ["rice", "chili", "shrimp", "spice"],
    "French": ["cream", "mushroom", "cheese", "garlic"],
    "Mediterranean": ["fish", "tomato", "garlic", "spinach"],
}


def sample_ingredients(cuisine: str, k_min=3, k_max=7) -> list[str]:
    k = random.randint(k_min, k_max)
    priors = CUISINE_INGREDIENT_PRIORS.get(cuisine, [])
    pool = list(set(priors + INGREDIENTS_MASTER))
    return random.sample(pool, k=k)


def discover_images(images_dir: str, manifest_csv: str | None = None):
    root = Path(images_dir)
    exts = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
    rows: list[dict] = []

    manifest_path = Path(manifest_csv) if manifest_csv else (root / "manifest.csv")
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing manifest: {manifest_path}. The manifest is now the only source of truth for labels."
        )
    manifest = pd.read_csv(manifest_path)
    required = {"image_path"}
    if not required.issubset(set(manifest.columns)):
        raise ValueError(f"{manifest_path} must contain `image_path` column.")
    if "dish_class" not in manifest.columns and "dish_family" not in manifest.columns and "dish_label" not in manifest.columns:
        raise ValueError(f"{manifest_path} must include at least one of: dish_label, dish_class, dish_family")

    dish_id = 0
    for row in manifest.itertuples(index=False):
        p = Path(str(row.image_path))
        if p.suffix.lower() not in exts:
            continue
        dish_label = getattr(row, "dish_label", None)
        dish_class = getattr(row, "dish_class", None)
        dish_family = getattr(row, "dish_family", None)
        cuisine = getattr(row, "cuisine", None)
        course = getattr(row, "course", None)
        protein_type = getattr(row, "protein_type", None)
        dish_name = getattr(row, "dish_name", None)
        source = getattr(row, "source", None)
        label_quality = getattr(row, "label_quality", None)
        normalized_label = ""
        if pd.notna(dish_label) and str(dish_label).strip():
            normalized_label = str(dish_label).strip()
        elif pd.notna(dish_class) and str(dish_class).strip():
            normalized_label = str(dish_class).strip()
        elif pd.notna(dish_family) and str(dish_family).strip():
            normalized_label = str(dish_family).strip()
        rows.append(
            {
                "dish_id": dish_id,
                "dish_label": normalized_label,
                "dish_class": str(dish_class) if pd.notna(dish_class) else normalized_label,
                "dish_family": str(dish_family) if pd.notna(dish_family) else "",
                "dish_name": str(dish_name) if pd.notna(dish_name) else "",
                "cuisine": str(cuisine) if pd.notna(cuisine) and str(cuisine).strip() else "Unknown",
                "course": str(course) if pd.notna(course) and str(course).strip() else "Unknown",
                "protein_type": str(protein_type) if pd.notna(protein_type) and str(protein_type).strip() else "Unknown",
                "source": str(source) if pd.notna(source) else "",
                "label_quality": str(label_quality) if pd.notna(label_quality) else "",
                "source_dish_label": str(getattr(row, "source_dish_label", "")) if hasattr(row, "source_dish_label") else "",
                "original_path": str(getattr(row, "original_path", "")) if hasattr(row, "original_path") else "",
                "image_path": str(row.image_path),
            }
        )
        dish_id += 1

    if not rows:
        raise ValueError(f"No images found under {images_dir}/")

    cuisines = sorted({r["cuisine"] for r in rows})
    dish_classes = sorted(
        {
            (
                r["dish_label"]
                if str(r["dish_label"]).strip()
                else (r["dish_class"] if str(r["dish_class"]).strip() else r["dish_family"])
            )
            for r in rows
            if str(r["dish_label"]).strip() or str(r["dish_class"]).strip() or str(r["dish_family"]).strip()
        }
    )
    return rows, cuisines, dish_classes


KEYWORD_INGREDIENT_HINTS = {
    "ramen": ["egg", "mushroom"],
    "sushi": ["rice", "fish"],
    "butter_chicken": ["chicken", "cream", "spice"],
    "chicken_curry": ["chicken", "spice", "chili"],
    "pizza": ["cheese", "tomato"],
    "spaghetti": ["pasta", "tomato"],
    "lasagna": ["pasta", "cheese", "tomato"],
    "risotto": ["rice", "cream"],
    "tacos": ["beans", "beef", "avocado"],
    "burrito": ["beans", "rice", "beef"],
    "guacamole": ["avocado"],
    "fried_rice": ["rice", "egg"],
    "mapo_tofu": ["tofu", "chili"],
    "bibimbap": ["rice", "egg"],
    "kimchi": ["chili"],
    "falafel": ["beans"],
    "hummus": ["beans"],
    "pho": ["rice", "beef"],
    "banh_mi": ["beef", "chili"],
}


def infer_ingredients(cuisine: str, dish_label: str, k_min=3, k_max=7) -> list[str]:
    base = set(CUISINE_INGREDIENT_PRIORS.get(cuisine, []))
    lbl = dish_label.lower()
    for key, vals in KEYWORD_INGREDIENT_HINTS.items():
        if key in lbl:
            base.update(vals)
    # direct token overlap fallback
    for ing in INGREDIENTS_MASTER:
        if ing in lbl:
            base.add(ing)

    pool = list(set(INGREDIENTS_MASTER).union(base))
    k = random.randint(k_min, k_max)
    if len(base) >= k:
        return random.sample(list(base), k=k)
    picks = list(base)
    need = k - len(picks)
    extra_pool = [x for x in pool if x not in picks]
    picks.extend(random.sample(extra_pool, k=min(need, len(extra_pool))))
    return picks


def build_dishes_with_clip(cfg: Config):
    dish_rows, cuisines, dish_classes = discover_images(cfg.images_dir, cfg.manifest_csv)
    encoder = VisionEncoder()

    dish_vectors = []
    dishes_out = []
    attr_names = [a.name for a in ATTRIBUTE_SPECS]
    affinity_names = [a.name for a in AFFINITY_SPECS]

    print(f"Found {len(dish_rows)} images across cuisines: {cuisines}")
    total_t0 = time.perf_counter()

    for idx, row in enumerate(tqdm(dish_rows, desc="Encoding images with CLIP")):
        cuisine = row["cuisine"]
        img_path = row["image_path"]
        dish_class = row.get("dish_class", "")
        dish_family = row.get("dish_family", "")
        dish_name = row.get("dish_name", "")
        dish_label = row.get("dish_label", "") or dish_class or dish_family or dish_name or "unknown"

        t0 = time.perf_counter()
        img_emb = encoder.encode_image(img_path)
        t1 = time.perf_counter()
        ings = infer_ingredients(cuisine, dish_label)
        if cfg.compute_prompt_features:
            a_vec = compute_attribute_scores(encoder, img_path, temperature=cfg.attribute_temperature)
            t2 = time.perf_counter()
            aff_vec = compute_affinity_scores(encoder, img_path, temperature=cfg.affinity_temperature)
            t3 = time.perf_counter()
        else:
            a_vec = np.asarray([], dtype=np.float32)
            aff_vec = np.asarray([], dtype=np.float32)
            t2 = t1
            t3 = t1

        dish_row = {
            "dish_id": row["dish_id"],
            "dish_label": dish_label,
            "dish_class": dish_class,
            "dish_family": dish_family,
            "dish_name": dish_name,
            "cuisine": cuisine,
            "course": row.get("course", "Unknown"),
            "protein_type": row.get("protein_type", "Unknown"),
            "source": row.get("source", ""),
            "label_quality": row.get("label_quality", ""),
            "source_dish_label": row.get("source_dish_label", ""),
            "original_path": row.get("original_path", ""),
            "ingredients": ",".join(ings),
            "image_path": img_path
        }
        if cfg.compute_prompt_features:
            for name, val in zip(attr_names, a_vec.tolist()):
                dish_row[f"attr_{name}"] = float(val)
            for name, val in zip(affinity_names, aff_vec.tolist()):
                dish_row[f"affinity_{name}"] = float(val)

        dishes_out.append(dish_row)
        # Keep embedding space pure CLIP image features.
        dish_vectors.append(img_emb.astype(np.float32))
        if idx < int(cfg.timing_log_first_n):
            print(
                f"[timing][data_generator][{idx+1}] encode_image={t1-t0:.4f}s "
                f"attr={t2-t1:.4f}s affinity={t3-t2:.4f}s"
            )

    dish_vecs = np.vstack(dish_vectors).astype(np.float32)
    dish_vecs = dish_vecs / (np.linalg.norm(dish_vecs, axis=1, keepdims=True) + 1e-12)
    elapsed = time.perf_counter() - total_t0
    eps = float(len(dish_vecs) / max(elapsed, 1e-9))
    norms = np.linalg.norm(dish_vecs, axis=1)
    print(
        f"[embeddings] count={len(dish_vecs)} shape={dish_vecs.shape} dtype={dish_vecs.dtype} "
        f"norm_mean/min/max={float(norms.mean()):.6f}/{float(norms.min()):.6f}/{float(norms.max()):.6f} "
        f"throughput={eps:.2f} img/s"
    )
    return pd.DataFrame(dishes_out), dish_vecs, cuisines, dish_classes


def generate_users(cfg: Config, dish_vectors: np.ndarray):
    num_dishes, dim = dish_vectors.shape
    user_embeddings = np.zeros((cfg.num_users, dim), dtype=np.float32)
    history_rows = []

    for user_id in range(cfg.num_users):
        n_meals = random.randint(cfg.meals_per_user_min, cfg.meals_per_user_max)
        eaten = random.sample(range(num_dishes), k=min(n_meals, num_dishes))
        ratings = np.random.randint(cfg.rating_min, cfg.rating_max + 1, size=len(eaten))

        weighted = []
        for dish_id, rating in zip(eaten, ratings):
            weighted.append(dish_vectors[dish_id] * float(rating))
            history_rows.append({"user_id": user_id, "dish_id": dish_id, "rating": int(rating)})

        user_embeddings[user_id] = np.mean(np.vstack(weighted), axis=0)

    return user_embeddings, pd.DataFrame(history_rows)


def generate_compatibility(cfg: Config, user_embeddings: np.ndarray):
    sim = cosine_similarity(user_embeddings)
    pair_features = []
    pair_rows = []

    for i in range(cfg.num_users):
        for j in range(i + 1, cfg.num_users):
            base = float(sim[i, j])
            satisfaction = base + float(np.random.normal(0, cfg.compat_noise_std))
            pair_features.append(np.abs(user_embeddings[i] - user_embeddings[j]))
            pair_rows.append({
                "userA": i,
                "userB": j,
                "satisfaction_score": satisfaction,
                "cosine_similarity": base
            })

    return np.vstack(pair_features), pd.DataFrame(pair_rows)


def assign_archetypes(cfg: Config, user_embeddings: np.ndarray):
    n_users, dim = user_embeddings.shape
    pca_dim = int(min(50, dim, n_users))
    if pca_dim < 1:
        raise ValueError("user_embeddings must have at least one feature dimension.")
    reduced = PCA(n_components=pca_dim, random_state=cfg.seed).fit_transform(user_embeddings)

    if HDBSCAN_AVAILABLE:
        min_cluster_size = max(3, min(12, n_users // 6))
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=2)
        labels = clusterer.fit_predict(reduced)
    else:
        print(
            "⚠️ HDBSCAN is not installed. Falling back to all users as unclustered (-1). "
            "Install with: pip install hdbscan"
        )
        labels = np.full((n_users,), -1, dtype=np.int32)

    names = ARCTYPE_NAMES_10 if cfg.n_archetypes == 10 else [f"Archetype {i}" for i in range(cfg.n_archetypes)]
    non_noise = sorted(int(x) for x in np.unique(labels) if int(x) >= 0)
    cluster_name_map: dict[int, str] = {-1: "Unclustered / Mixed"}
    for i, c in enumerate(non_noise):
        cluster_name_map[c] = names[i % len(names)]

    archetypes = [cluster_name_map.get(int(l), f"Archetype {int(l)}") for l in labels]
    return labels, archetypes


def build_archetype_profiles(
    dishes_df: pd.DataFrame,
    history_df: pd.DataFrame,
    archetype_labels: np.ndarray,
    archetype_name_map: dict[int, str],
):
    """
    Create human-readable archetype summaries from what users actually ate/rated.
    This is more interpretable than summarizing in raw embedding space.
    """
    attr_cols = [c for c in dishes_df.columns if c.startswith("attr_")]
    affinity_cols = [c for c in dishes_df.columns if c.startswith("affinity_")]
    signal_cols = attr_cols + affinity_cols
    if not signal_cols:
        return pd.DataFrame(columns=["cluster_id", "archetype", "top_cuisines"])

    keep_cols = ["dish_id", "cuisine", "dish_label", "course", "protein_type", *signal_cols]
    merged = history_df.merge(dishes_df[keep_cols], on="dish_id", how="left")
    merged["weight"] = merged["rating"].astype(np.float32)

    # Attach each user's cluster to each history row
    merged["cluster_id"] = merged["user_id"].map(lambda uid: int(archetype_labels[int(uid)]))

    # Global baselines (weighted) for "what's distinctive"
    gw = merged["weight"].to_numpy()
    gw_sum = float(np.sum(gw)) + 1e-12
    global_attr_mean = {}
    global_attr_std = {}
    for col in signal_cols:
        vals = merged[col].to_numpy(dtype=np.float32)
        mean = float(np.sum(vals * gw) / gw_sum)
        var = float(np.sum(((vals - mean) ** 2) * gw) / gw_sum)
        global_attr_mean[col] = mean
        global_attr_std[col] = float(np.sqrt(max(var, 1e-8)))

    rows = []
    for cluster_id in sorted(int(x) for x in np.unique(archetype_labels)):
        dfc = merged[merged["cluster_id"] == cluster_id]
        if len(dfc) == 0:
            continue

        # Cuisine shares
        cuisine_counts = dfc["cuisine"].value_counts()
        top_cuisines = ", ".join([f"{k} ({int(v)})" for k, v in cuisine_counts.head(3).items()])
        class_counts = dfc["dish_label"].value_counts()
        top_dish_classes = ", ".join([f"{k} ({int(v)})" for k, v in class_counts.head(5).items()])
        course_counts = dfc["course"].value_counts()
        top_courses = ", ".join([f"{k} ({int(v)})" for k, v in course_counts.head(2).items()])
        protein_counts = dfc["protein_type"].value_counts()
        top_proteins = ", ".join([f"{k} ({int(v)})" for k, v in protein_counts.head(2).items()])

        # Weighted attribute means
        w = dfc["weight"].to_numpy()
        w_sum = float(np.sum(w)) + 1e-12
        attr_means = {}
        for col in signal_cols:
            vals = dfc[col].to_numpy(dtype=np.float32)
            attr_means[col] = float(np.sum(vals * w) / w_sum)

        # Rank most defining attributes by z-score lift vs global mean
        attr_z = {
            col: (attr_means[col] - global_attr_mean[col]) / (global_attr_std[col] + 1e-12)
            for col in signal_cols
        }
        top_attrs = sorted(attr_z.items(), key=lambda kv: kv[1], reverse=True)[:6]
        top_attributes = ", ".join(
            [f"{k.replace('attr_', '').replace('affinity_', '')} ({v:+.2f}σ)" for k, v in top_attrs]
        )

        row = {
            "cluster_id": int(cluster_id),
            "archetype": archetype_name_map.get(int(cluster_id), f"Archetype {cluster_id}"),
            "top_cuisines": top_cuisines,
            "top_dish_classes": top_dish_classes,
            "top_courses": top_courses,
            "top_proteins": top_proteins,
            "top_attributes": top_attributes,
        }
        for col, val in attr_means.items():
            row[col] = val
        rows.append(row)

    return pd.DataFrame(rows).sort_values("cluster_id")


def generate_restaurants(cfg: Config, dishes_df: pd.DataFrame, dish_vectors: np.ndarray, cuisines: list[str]):
    """
    Make restaurants by cuisine centroid in embedding space.
    Restaurant embedding ~= mean dish vector for that cuisine + noise.
    """
    restaurants = []
    resto_embeddings = []

    for cuisine in cuisines:
        dish_ids = dishes_df.loc[dishes_df["cuisine"] == cuisine, "dish_id"].values
        if len(dish_ids) == 0:
            continue
        centroid = np.mean(dish_vectors[dish_ids], axis=0)

        for k in range(cfg.restaurants_per_cuisine):
            emb = centroid + np.random.normal(0, cfg.restaurant_noise_std, size=centroid.shape).astype(np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-12)

            resto_id = len(restaurants)
            restaurants.append({
                "restaurant_id": resto_id,
                "name": f"{cuisine} Place {k+1}",
                "cuisine": cuisine,
                "price_tier": random.choice([1, 2, 3]),
                "neighborhood": random.choice(["Downtown", "West Loop", "River North", "Wicker Park"]),
            })
            resto_embeddings.append(emb)

    restaurants_df = pd.DataFrame(restaurants)
    return restaurants_df, np.vstack(resto_embeddings)


def check_preprocessing_consistency(dishes_df: pd.DataFrame, dish_vectors: np.ndarray, *, min_cosine: float = 0.999):
    if len(dishes_df) == 0 or dish_vectors.ndim != 2:
        return
    sample_idx = None
    for i, p in enumerate(dishes_df["image_path"].astype(str).tolist()):
        if Path(p).exists():
            sample_idx = i
            break
    if sample_idx is None:
        return
    q_path = str(dishes_df.iloc[int(sample_idx)]["image_path"])
    dataset_emb = dish_vectors[int(sample_idx)].astype(np.float32)
    dataset_emb = dataset_emb / (np.linalg.norm(dataset_emb) + 1e-12)
    query_emb = VisionEncoder().encode_image(q_path).astype(np.float32)
    cos = float(np.dot(dataset_emb, query_emb) / ((np.linalg.norm(dataset_emb) * np.linalg.norm(query_emb)) + 1e-12))
    if cos < float(min_cosine):
        print("WARNING: Preprocessing mismatch: dataset embeddings not comparable to query embeddings.")
        print(f"  sample={q_path}")
        print(f"  cosine={cos:.6f} < threshold={float(min_cosine):.6f}")
    else:
        print(f"Preprocessing consistency check OK (cosine={cos:.6f})")


def save_all(cfg: Config):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    os.makedirs(cfg.data_dir, exist_ok=True)

    dishes_df, dish_vectors, cuisines, dish_classes = build_dishes_with_clip(cfg)
    user_embeddings, history_df = generate_users(cfg, dish_vectors)
    pair_features, pairs_df = generate_compatibility(cfg, user_embeddings)
    cluster_labels, archetypes = assign_archetypes(cfg, user_embeddings)
    archetype_name_map = {
        int(c): str(a)
        for c, a in pd.DataFrame({"cluster_id": cluster_labels, "archetype": archetypes})
        .drop_duplicates("cluster_id")
        .itertuples(index=False)
    }
    archetype_profiles_df = build_archetype_profiles(dishes_df, history_df, cluster_labels, archetype_name_map)

    restaurants_df, restaurant_embeddings = generate_restaurants(cfg, dishes_df, dish_vectors, cuisines)

    # CSVs
    dishes_df.to_csv(os.path.join(cfg.data_dir, "dishes.csv"), index=False)
    history_df.to_csv(os.path.join(cfg.data_dir, "user_history.csv"), index=False)
    pairs_df.to_csv(os.path.join(cfg.data_dir, "compatibility_pairs.csv"), index=False)
    pd.DataFrame({"cuisine": cuisines}).to_csv(os.path.join(cfg.data_dir, "cuisines.csv"), index=False)
    pd.DataFrame({"dish_class": dish_classes}).to_csv(os.path.join(cfg.data_dir, "dish_classes.csv"), index=False)
    pd.DataFrame({"ingredient": INGREDIENTS_MASTER}).to_csv(os.path.join(cfg.data_dir, "ingredients_master.csv"), index=False)
    pd.DataFrame(
        [
            {
                "attribute": a.name,
                "positive_prompt": " | ".join(a.positive_prompts),
                "negative_prompt": " | ".join(a.negative_prompts),
            }
            for a in ATTRIBUTE_SPECS
        ]
    ).to_csv(os.path.join(cfg.data_dir, "taste_attributes.csv"), index=False)
    pd.DataFrame(
        [
            {
                "affinity": a.name,
                "positive_prompt": " | ".join(a.positive_prompts),
                "negative_prompt": " | ".join(a.negative_prompts),
            }
            for a in AFFINITY_SPECS
        ]
    ).to_csv(os.path.join(cfg.data_dir, "semantic_affinities.csv"), index=False)

    pd.DataFrame({
        "user_id": list(range(cfg.num_users)),
        "archetype": archetypes,
        "cluster_id": cluster_labels
    }).to_csv(os.path.join(cfg.data_dir, "user_archetypes.csv"), index=False)
    archetype_profiles_df.to_csv(os.path.join(cfg.data_dir, "archetype_profiles.csv"), index=False)

    restaurants_df.to_csv(os.path.join(cfg.data_dir, "restaurants.csv"), index=False)

    # NPY arrays
    np.save(os.path.join(cfg.data_dir, "dish_vectors.npy"), dish_vectors)
    np.save(os.path.join(cfg.data_dir, "user_embeddings.npy"), user_embeddings)
    np.save(os.path.join(cfg.data_dir, "pair_features.npy"), pair_features)
    np.save(os.path.join(cfg.data_dir, "restaurant_embeddings.npy"), restaurant_embeddings)

    check_preprocessing_consistency(dishes_df, dish_vectors)
    print(f"\n✅ Saved datasets to ./{cfg.data_dir}/ (including restaurants)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate BiteMe data artifacts from manifest.")
    p.add_argument("--images_dir", default="images")
    p.add_argument("--manifest_csv", default="images/manifest.csv")
    p.add_argument("--data_dir", default="data")
    args = p.parse_args()
    cfg = Config(images_dir=args.images_dir, manifest_csv=args.manifest_csv, data_dir=args.data_dir)
    save_all(cfg)