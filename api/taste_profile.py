from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from datetime import datetime, timezone

import numpy as np

from api.dish_family_map import get_family_traits, resolve_dish_family
from api.dish_trait_map import canonical_dish_key, get_canonical_dish_traits
from api.label_normalization import normalize_label
from api.semantic_trait_inference import infer_semantic_traits


@dataclass(frozen=True)
class TasteDimension:
    key: str
    title: str
    description: str


TASTE_DIMENSIONS: list[TasteDimension] = [
    TasteDimension("sweet_leaning", "Sweet leaning", "Preference for sweet-leaning foods and endings."),
    TasteDimension("salty_leaning", "Salty leaning", "Preference for savory and salt-forward dishes."),
    TasteDimension("umami_leaning", "Umami leaning", "Preference for deep savory richness and brothy profiles."),
    TasteDimension("spicy_leaning", "Spicy leaning", "Preference for heat and spice intensity."),
    TasteDimension("richness_preference", "Richness preference", "Preference for creamy, fried, buttery, or indulgent meals."),
    TasteDimension("freshness_preference", "Freshness preference", "Preference for lighter, crisp, bright dishes."),
    TasteDimension("texture_seeking", "Texture seeking", "Preference for textural contrast and crunchy/chewy elements."),
    TasteDimension("comfort_food_tendency", "Comfort-food tendency", "Tendency to favor familiar comfort dishes."),
    TasteDimension("adventurousness", "Adventurousness", "Willingness to explore varied cuisines and dish styles."),
    TasteDimension("variety_seeking", "Variety seeking", "How often choices spread across different dish types."),
    TasteDimension("protein_forward", "Protein forward", "How often selections center protein-focused dishes."),
    TasteDimension("carb_forward", "Carb forward", "How often selections center carb-focused dishes."),
    TasteDimension("dessert_affinity", "Dessert affinity", "How frequently dessert-like choices appear."),
    TasteDimension("global_cuisine_breadth", "Global cuisine breadth", "Breadth of cuisine diversity in uploads."),
]


DIM_KEYS = [d.key for d in TASTE_DIMENSIONS]
TREND_DIM_KEYS = [
    "sweet_leaning",
    "spicy_leaning",
    "richness_preference",
    "freshness_preference",
    "dessert_affinity",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_entropy_norm(vals: list[float]) -> float:
    arr = np.asarray([max(0.0, float(x)) for x in vals], dtype=np.float32)
    if arr.size <= 1 or float(np.sum(arr)) <= 1e-9:
        return 0.0
    p = arr / (np.sum(arr) + 1e-12)
    h = float(-np.sum(p * np.log(p + 1e-12)))
    return float(h / np.log(float(arr.size) + 1e-12))


def _keywords_score(text: str, keys: list[str]) -> float:
    t = str(text).lower()
    if not t:
        return 0.0
    hits = sum(1 for k in keys if k in t)
    return float(min(1.0, hits / max(1, len(keys))))


def _dessert_memory_strength(profile: dict[str, Any]) -> float:
    dishes = profile.get("favorite_dishes", {}) or {}
    if not isinstance(dishes, dict) or not dishes:
        return 0.0
    dessert_terms = (
        "dessert",
        "cake",
        "pie",
        "brownie",
        "cheesecake",
        "tiramisu",
        "gelato",
        "ice cream",
        "pastry",
        "cookie",
        "donut",
    )
    total = 0.0
    dessert_mass = 0.0
    for k, v in dishes.items():
        try:
            fv = max(0.0, float(v))
        except (TypeError, ValueError):
            continue
        if fv <= 0.0:
            continue
        total += fv
        label = str(k).lower().replace("_", " ")
        if any(t in label for t in dessert_terms):
            dessert_mass += fv
    if total <= 1e-9:
        return 0.0
    return float(max(0.0, min(1.0, dessert_mass / total)))


def _candidate_signals(cand: dict) -> dict[str, float]:
    dish_raw = normalize_label(cand.get("dish_label", cand.get("dish_class", "")))
    dish = str(dish_raw).replace("_", " ").lower()
    cuisine = str(cand.get("cuisine", "")).lower()
    protein = str(cand.get("protein_type", "")).lower()
    text = f"{dish} {cuisine}".strip()

    canonical_traits = get_canonical_dish_traits(dish)
    if canonical_traits is not None:
        return canonical_traits

    family_traits = get_family_traits(dish)
    if family_traits is not None:
        return family_traits

    query_emb = None
    for k in ("query_embedding", "image_embedding", "embedding"):
        if k in cand and cand.get(k) is not None:
            try:
                query_emb = np.asarray(cand.get(k), dtype=np.float32).reshape(-1)
                break
            except Exception:
                query_emb = None
    semantic = infer_semantic_traits(label=dish, query_embedding=query_emb, top_k=3, min_similarity=0.20)
    if semantic is not None:
        return semantic.blended_traits

    def _phrase_score(
        source: str,
        terms: list[str],
        *,
        signature_terms: set[str] | None = None,
        exact_weight: float = 0.95,
        signature_hit: float = 0.62,
        phrase_hit: float = 0.38,
        term_hit: float = 0.22,
        combo_bonus: float = 0.08,
    ) -> float:
        src = source.lower()
        if not src:
            return 0.0
        sig = {x.lower() for x in (signature_terms or set())}
        score = 0.0
        hits = 0
        for t in terms:
            tt = t.lower()
            if not tt:
                continue
            if src == tt:
                score += exact_weight
                hits += 1
            elif tt in src:
                if tt in sig:
                    score += signature_hit
                elif " " in tt:
                    score += phrase_hit
                else:
                    score += term_hit
                hits += 1
        if hits > 1:
            score += combo_bonus * float(hits - 1)
        return float(min(1.0, score))

    sweet_terms = [
        "dessert", "sweet", "cake", "cheesecake", "shortcake", "brownie", "cookie", "donut", "doughnut",
        "tiramisu", "parfait", "mousse", "pudding", "custard", "tart", "pie", "apple pie", "churro",
        "ice cream", "gelato", "sundae", "waffle", "pancake", "crepe", "muffin", "scone", "croissant",
        "cream puff", "rare cheese cake", "mango pudding", "almond jelly",
    ]
    sweet_signature_terms = {
        "tiramisu",
        "brownie",
        "cheesecake",
        "apple pie",
        "rare cheese cake",
        "mango pudding",
        "almond jelly",
        "cream puff",
        "parfait",
        "ice cream",
    }
    spicy_terms = [
        "spicy", "hot", "chili", "chilli", "curry", "vindaloo", "green curry", "yellow curry", "mapo tofu",
        "kimchi", "jjigae", "dak galbi", "szechuan", "sichuan", "buffalo wings", "wings", "peri peri",
        "harissa", "jalapeno", "salsa", "papaya salad", "tom yum", "hot and sour", "khao soi", "gochujang",
    ]
    spicy_signature_terms = {
        "mapo tofu",
        "buffalo wings",
        "green curry",
        "yellow curry",
        "vindaloo",
        "khao soi",
        "tom yum",
        "hot and sour",
    }
    rich_terms = [
        "rich", "creamy", "butter", "buttery", "cheese", "cheesy", "carbonara", "alfredo", "bacon", "pork belly",
        "fried", "deep fried", "tempura", "katsu", "cutlet", "gratin", "lasagna", "meat loaf", "beef bowl",
        "roast duck", "lamb kebabs", "cream puff", "brownie", "chocolate", "stew", "buffalo wings", "wings",
    ]
    rich_signature_terms = {
        "buffalo wings",
        "lasagna",
        "gratin",
        "carbonara",
        "alfredo",
        "pork belly",
        "brownie",
        "cream puff",
    }
    fresh_terms = [
        "salad", "green salad", "caesar salad", "papaya salad", "ceviche", "sashimi", "sushi", "poke",
        "gazpacho", "cold tofu", "chilled noodle", "fresh", "light", "spring roll", "steamed rice roll",
        "caprese", "herb", "citrus", "raw", "cold", "lightly roasted fish", "thai papaya salad", "fresh roll",
    ]
    fresh_signature_terms = {
        "ceviche",
        "sashimi",
        "sushi",
        "poke",
        "gazpacho",
        "green salad",
        "caesar salad",
        "papaya salad",
    }
    comfort_terms = [
        "comfort", "mac and cheese", "lasagna", "ramen", "udon", "noodle", "fried rice", "pizza", "burger",
        "fries", "meat loaf", "stew", "omelet", "omelette", "gratin", "cutlet curry", "pork cutlet",
        "chicken cutlet", "beef curry", "hot pot", "oxtail soup", "pot au feu", "jambalaya", "adobo", "pho",
        "buffalo wings", "wings",
    ]
    comfort_signature_terms = {
        "ramen",
        "mac and cheese",
        "lasagna",
        "pizza",
        "burger",
        "fried chicken",
        "buffalo wings",
        "beef curry",
        "jambalaya",
    }
    umami_terms = [
        "umami", "savory", "savoury", "broth", "stock", "ramen", "miso", "soy", "mapo tofu", "mushroom",
        "oyster sauce", "fish sauce", "anchovy", "parmesan", "aged cheese", "grilled", "roast", "stew",
        "jiaozi", "dumpling", "pho", "beef noodle soup", "hot and sour soup", "teriyaki", "yakitori",
    ]
    umami_signature_terms = {
        "mapo tofu",
        "ramen",
        "pho",
        "beef noodle soup",
        "hot and sour soup",
        "yakitori",
        "teriyaki",
    }

    sweet = _phrase_score(dish, sweet_terms, signature_terms=sweet_signature_terms)
    spicy = _phrase_score(text, spicy_terms, signature_terms=spicy_signature_terms)
    rich = _phrase_score(dish, rich_terms, signature_terms=rich_signature_terms)
    fresh = _phrase_score(text, fresh_terms, signature_terms=fresh_signature_terms)
    texture = _phrase_score(dish, ["crispy", "crunchy", "crunch", "tempura", "fried", "taco", "toast", "waffle", "granola"])
    comfort = _phrase_score(dish, comfort_terms, signature_terms=comfort_signature_terms)
    dessert = sweet

    protein_forward = 1.0 if protein in {"beef", "chicken", "pork", "fish", "seafood", "egg", "lamb", "shrimp"} else 0.0
    if _phrase_score(dish, ["steak", "salmon", "chicken", "pork", "tuna", "lamb", "beef", "wings", "tofu"]) > 0:
        protein_forward = max(protein_forward, 0.75)

    carb_forward = _phrase_score(
        dish,
        ["pasta", "rice", "bread", "noodle", "ramen", "pizza", "dumpling", "taco", "burrito", "lasagna", "udon", "soba", "spaghetti"],
    )
    umami = max(protein_forward * 0.7, _phrase_score(text, umami_terms, signature_terms=umami_signature_terms))
    salty = max(comfort * 0.45, _phrase_score(text, ["fries", "chips", "bacon", "anchovy", "soy", "jerky", "savory", "salt", "miso", "oyster sauce"]))

    return {
        "sweet_leaning": sweet,
        "salty_leaning": salty,
        "umami_leaning": umami,
        "spicy_leaning": spicy,
        "richness_preference": rich,
        "freshness_preference": fresh,
        "texture_seeking": texture,
        "comfort_food_tendency": comfort,
        "protein_forward": protein_forward,
        "carb_forward": carb_forward,
        "dessert_affinity": dessert,
    }


def _candidate_confidence(cand: dict, default: float = 0.5) -> float:
    for key in ("final_score", "score", "predicted_score", "retrieval_score", "mlp_score", "pair_score"):
        if key in cand and cand.get(key) is not None:
            try:
                v = float(cand.get(key))
                if np.isfinite(v):
                    return float(np.clip(v, 0.0, 1.0))
            except (TypeError, ValueError):
                continue
    return float(np.clip(default, 0.0, 1.0))


def _base_taste_profile() -> dict[str, Any]:
    dims = {}
    for d in TASTE_DIMENSIONS:
        dims[d.key] = {
            "score": 0.5,
            "explanation": "Still learning from uploads.",
        }
    return {"dimensions": dims, "analysis": {}, "relative_rankings": [], "history": []}


def init_taste_profile(profile: dict[str, Any]) -> None:
    if "taste_profile" not in profile or not isinstance(profile.get("taste_profile"), dict):
        profile["taste_profile"] = _base_taste_profile()
    tp = profile["taste_profile"]
    if "history" not in tp or not isinstance(tp.get("history"), list):
        tp["history"] = []


def _score_text(x: float) -> str:
    if x >= 0.75:
        return "strong"
    if x >= 0.60:
        return "clear"
    if x >= 0.45:
        return "balanced"
    if x >= 0.30:
        return "lighter"
    return "low"


def _dim_explanation(key: str, score: float, profile: dict[str, Any]) -> str:
    top_cuisines = sorted(profile.get("favorite_cuisines", {}).items(), key=lambda x: x[1], reverse=True)[:2]
    top_dishes = sorted(profile.get("favorite_dishes", {}).items(), key=lambda x: x[1], reverse=True)[:2]
    ctxt = ", ".join([c for c, _ in top_cuisines]) if top_cuisines else "recent uploads"
    dtxt = ", ".join([d.replace("_", " ") for d, _ in top_dishes]) if top_dishes else "recent dish picks"
    return f"{_score_text(score).capitalize()} signal from patterns in {ctxt} and dishes like {dtxt}."


def update_taste_profile(profile: dict[str, Any], prediction: dict) -> dict[str, Any]:
    init_taste_profile(profile)
    dims = profile["taste_profile"]["dimensions"]
    top3 = prediction.get("top3_candidates", []) or []
    w = [1.0, 0.6, 0.3]
    agg = {k: 0.0 for k in DIM_KEYS}
    total_w = 0.0
    cuisines = []
    prediction_conf = _candidate_confidence(prediction, default=0.5)
    for i, cand in enumerate(top3[:3]):
        conf = _candidate_confidence(cand, default=prediction_conf)
        wi = float(w[i]) * conf
        total_w += wi
        sig = _candidate_signals(cand)
        for k, v in sig.items():
            agg[k] += wi * float(v)
        c = str(cand.get("cuisine", "")).strip()
        if c:
            cuisines.append(c.lower())
    if total_w > 0:
        for k in list(agg.keys()):
            agg[k] = float(agg[k] / total_w)

    # behavior-derived dimensions from profile memory
    dishes = profile.get("favorite_dishes", {})
    cuisines_map = profile.get("favorite_cuisines", {})
    dish_entropy = _safe_entropy_norm(list(dishes.values()))
    cuisine_entropy = _safe_entropy_norm(list(cuisines_map.values()))
    repeat = float(max(dishes.values()) / (sum(dishes.values()) + 1e-12)) if dishes else 0.0
    agg["variety_seeking"] = float(np.clip(0.5 * dish_entropy + 0.5 * cuisine_entropy, 0.0, 1.0))
    agg["adventurousness"] = float(np.clip(agg["variety_seeking"] * (1.0 - 0.5 * repeat), 0.0, 1.0))
    agg["global_cuisine_breadth"] = float(np.clip(cuisine_entropy, 0.0, 1.0))

    # Calibration: be responsive early (uploads 1-3), then stabilize over time.
    n = int(profile.get("upload_count", 0))
    signal_strength = float(np.clip(total_w / 1.9, 0.0, 1.0))  # top-3 max weighted mass ~= 1.9
    confidence_boost = 0.75 + (0.5 * signal_strength)
    base_alpha = float(np.clip(0.36 / (1.0 + 0.06 * n), 0.10, 0.36))
    alpha = float(np.clip(base_alpha * confidence_boost, 0.10, 0.45))
    dessert_memory = _dessert_memory_strength(profile)
    for k in DIM_KEYS:
        prev = float(dims.get(k, {}).get("score", 0.5))
        target = float(agg.get(k, prev))
        eff_alpha = alpha
        if k in {"dessert_affinity", "sweet_leaning"} and target < prev:
            # Preserve dessert memory: a single savory upload should not erase a clear dessert signal.
            down_scale = 0.60 - (0.30 * min(1.0, dessert_memory / 0.35))
            eff_alpha *= max(0.25, down_scale)
        new_score = float((1.0 - eff_alpha) * prev + eff_alpha * target)
        dims[k] = {"score": float(np.clip(new_score, 0.0, 1.0)), "explanation": _dim_explanation(k, new_score, profile)}
    # Keep lightweight trend snapshots for product UX.
    hist = profile["taste_profile"].setdefault("history", [])
    snapshot = {
        "timestamp": _utc_now_iso(),
        "upload_count": int(profile.get("upload_count", 0)),
        "dimensions": {k: float(dims.get(k, {}).get("score", 0.5)) for k in TREND_DIM_KEYS},
    }
    hist.append(snapshot)
    if len(hist) > 120:
        profile["taste_profile"]["history"] = hist[-120:]
    return profile


def debug_applied_taste_traits(
    raw_label: Any,
    *,
    confidence: float = 0.8,
    cuisine: str = "",
    protein_type: str = "",
) -> dict[str, Any]:
    normalized = normalize_label(raw_label)
    conf = float(np.clip(float(confidence), 0.0, 1.0))
    cand = {
        "dish_label": normalized,
        "cuisine": str(cuisine),
        "protein_type": str(protein_type),
        "final_score": conf,
    }
    base = _candidate_signals(cand)
    weighted = {k: round(float(v) * conf, 4) for k, v in base.items() if float(v) > 0.0}
    return {
        "raw_label": str(raw_label),
        "normalized_label": normalized,
        "confidence": conf,
        "applied_traits": weighted,
    }


def debug_taxonomy_resolution(
    raw_label: Any,
    *,
    confidence: float = 0.8,
    cuisine: str = "",
    protein_type: str = "",
    query_embedding: Any | None = None,
) -> dict[str, Any]:
    normalized = normalize_label(raw_label)
    canon_key = canonical_dish_key(normalized)
    canonical_traits = get_canonical_dish_traits(normalized)
    family_key = resolve_dish_family(normalized)
    query_emb = None
    if query_embedding is not None:
        try:
            query_emb = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        except Exception:
            query_emb = None

    if canonical_traits is not None:
        resolved_by = "canonical"
        base_traits = canonical_traits
        semantic = None
    elif family_key is not None:
        resolved_by = "family"
        base_traits = get_family_traits(normalized) or {}
        semantic = None
    else:
        semantic = infer_semantic_traits(label=normalized, query_embedding=query_emb, top_k=3, min_similarity=0.20)
        if semantic is not None:
            resolved_by = "semantic_embedding"
            base_traits = dict(semantic.blended_traits)
        else:
            resolved_by = "heuristic"
            base_traits = _candidate_signals(
                {
                    "dish_label": normalized,
                    "cuisine": str(cuisine),
                    "protein_type": str(protein_type),
                }
            )

    conf = float(np.clip(float(confidence), 0.0, 1.0))
    weighted = {k: round(float(v) * conf, 4) for k, v in base_traits.items() if float(v) > 0.0}
    return {
        "raw_label": str(raw_label),
        "normalized_label": normalized,
        "resolved_by": resolved_by,
        "canonical_key": canon_key if canonical_traits is not None else None,
        "family_key": family_key if canonical_traits is None and family_key is not None else None,
        "nearest_neighbors": (
            [{"dish_key": n.dish_key, "similarity": round(float(n.similarity), 4)} for n in semantic.neighbors]
            if semantic is not None
            else []
        ),
        "similarity_scores": (
            [round(float(n.similarity), 4) for n in semantic.neighbors]
            if semantic is not None
            else []
        ),
        "blended_traits": ({k: round(float(v), 4) for k, v in base_traits.items() if float(v) > 0.0}),
        "applied_traits": weighted,
    }




def generate_detailed_analysis(profile: dict[str, Any]) -> dict[str, Any]:
    init_taste_profile(profile)
    if int(profile.get("upload_count", 0)) <= 0:
        return {
            "likes": [],
            "less_affinity_for": [],
            "notable_patterns": [
                "Your taste profile is in onboarding mode. Upload a few dishes and BiteMe will start learning your real preferences."
            ],
            "surprising_observations": [],
            "playful_line": "No hot takes yet. Add your first dish to begin your flavor fingerprint.",
        }
    dims = profile["taste_profile"]["dimensions"]
    ranked = sorted([(k, float(v.get("score", 0.5))) for k, v in dims.items()], key=lambda x: x[1], reverse=True)
    likes = [k for k, _ in ranked[:4]]
    avoids = [k for k, _ in ranked[-3:]]
    top_c = sorted(profile.get("favorite_cuisines", {}).items(), key=lambda x: x[1], reverse=True)[:3]
    top_d = sorted(profile.get("favorite_dishes", {}).items(), key=lambda x: x[1], reverse=True)[:3]

    recent = profile.get("last_predictions", [])[:5]
    recent_labels = [str(x.get("predicted_label", "")).replace("_", " ") for x in recent if str(x.get("predicted_label", "")).strip()]
    recent_txt = ", ".join(recent_labels[:3]) if recent_labels else "no strong recent pattern yet"
    top_dishes_txt = ", ".join([x.replace("_", " ") for x, _ in top_d]) or "still building signal"
    notable = [
        f"Your strongest long-term pull is toward {likes[0].replace('_', ' ')} and {likes[1].replace('_', ' ')}." if len(likes) > 1 else "Taste profile is still stabilizing.",
        f"Top cuisines over time: {', '.join([x for x, _ in top_c]) or 'mixed'}; favorite dishes include {top_dishes_txt}.",
        f"Recent uploads lean toward {recent_txt}.",
        f"Repeat-vs-variety currently leans toward {'variety' if float(dims['variety_seeking']['score']) >= 0.5 else 'repeat comfort'}."
    ]
    surprise = []
    if float(dims["dessert_affinity"]["score"]) >= 0.65 and float(dims["protein_forward"]["score"]) >= 0.60:
        surprise.append("You combine dessert curiosity with protein-forward choices more often than typical users.")
    if float(dims["spicy_leaning"]["score"]) >= 0.65 and float(dims["freshness_preference"]["score"]) >= 0.65:
        surprise.append("You seem to like heat and freshness together, not one at the expense of the other.")
    if not surprise:
        surprise.append("Your profile is balanced enough that no single trait dominates every upload.")

    playful = "Your flavor fingerprint reads like someone with a point of view, not random orders."
    return {
        "likes": [x.replace("_", " ") for x in likes],
        "less_affinity_for": [x.replace("_", " ") for x in avoids],
        "notable_patterns": notable,
        "surprising_observations": surprise[:2],
        "playful_line": playful,
    }


def compute_relative_rankings(target_user: dict, others: list[dict]) -> list[dict]:
    tp = target_user.get("profile", {}).get("taste_profile", {}).get("dimensions", {})
    if not tp:
        return []
    out = []
    for d in TASTE_DIMENSIONS:
        key = d.key
        mine = float(tp.get(key, {}).get("score", 0.5))
        ref = []
        for u in others:
            ud = u.get("profile", {}).get("taste_profile", {}).get("dimensions", {})
            ref.append(float(ud.get(key, {}).get("score", 0.5)))
        if ref:
            pct = float(100.0 * sum(1 for x in ref if x <= mine) / len(ref))
        else:
            pct = 50.0
        if pct >= 70:
            phrase = f"More {key.replace('_', ' ')} than most users."
        elif pct <= 30:
            phrase = f"Lower {key.replace('_', ' ')} than most users."
        else:
            phrase = f"Near the middle for {key.replace('_', ' ')} among users."
        out.append(
            {
                "dimension": key,
                "percentile_within_user_base": round(pct, 1),
                "interpretation": phrase,
            }
        )
    return out


def dimension_vector(profile: dict[str, Any]) -> np.ndarray:
    init_taste_profile(profile)
    dims = profile["taste_profile"]["dimensions"]
    return np.asarray([float(dims.get(k, {}).get("score", 0.5)) for k in DIM_KEYS], dtype=np.float32)
