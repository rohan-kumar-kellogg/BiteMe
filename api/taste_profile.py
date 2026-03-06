from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


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


def _candidate_signals(cand: dict) -> dict[str, float]:
    dish = str(cand.get("dish_label", "")).replace("_", " ").lower()
    cuisine = str(cand.get("cuisine", "")).lower()
    protein = str(cand.get("protein_type", "")).lower()

    sweet = _keywords_score(dish, ["cake", "pie", "mousse", "panna cotta", "ice cream", "donut", "dessert", "cookie"])
    spicy = _keywords_score(dish + " " + cuisine, ["spicy", "curry", "hot", "vindaloo", "szechuan", "kimchi"])
    rich = _keywords_score(dish, ["fried", "cheese", "creamy", "carbonara", "butter", "bacon", "ribs"])
    fresh = _keywords_score(dish + " " + cuisine, ["salad", "ceviche", "sashimi", "sushi", "poke", "fresh", "gazpacho"])
    texture = _keywords_score(dish, ["crispy", "crunch", "tempura", "fried", "taco", "granola", "toast"])
    comfort = _keywords_score(dish, ["lasagna", "mac and cheese", "burger", "fries", "pizza", "ramen", "poutine", "meatloaf"])
    dessert = sweet
    protein_forward = 1.0 if protein in {"beef", "chicken", "pork", "fish", "seafood", "egg", "lamb", "shrimp"} else 0.0
    if _keywords_score(dish, ["steak", "salmon", "chicken", "pork", "tuna", "lamb", "beef"]) > 0:
        protein_forward = max(protein_forward, 0.75)
    carb_forward = _keywords_score(dish, ["pasta", "rice", "bread", "noodle", "ramen", "pizza", "dumpling", "taco", "burrito"])
    umami = max(
        protein_forward * 0.7,
        _keywords_score(dish + " " + cuisine, ["broth", "ramen", "mushroom", "soy", "roast", "grilled", "stew"]),
    )
    salty = max(comfort * 0.4, _keywords_score(dish, ["fries", "chips", "bacon", "anchovy", "soy", "jerky"]))

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


def _base_taste_profile() -> dict[str, Any]:
    dims = {}
    for d in TASTE_DIMENSIONS:
        dims[d.key] = {
            "score": 0.5,
            "explanation": "Still learning from uploads.",
        }
    return {"dimensions": dims, "analysis": {}, "relative_rankings": []}


def init_taste_profile(profile: dict[str, Any]) -> None:
    if "taste_profile" not in profile or not isinstance(profile.get("taste_profile"), dict):
        profile["taste_profile"] = _base_taste_profile()


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
    for i, cand in enumerate(top3[:3]):
        wi = w[i]
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

    # stability: decreasing EMA step as profile matures.
    n = int(profile.get("upload_count", 0))
    alpha = float(np.clip(0.22 / (1.0 + 0.08 * n), 0.06, 0.22))
    for k in DIM_KEYS:
        prev = float(dims.get(k, {}).get("score", 0.5))
        target = float(agg.get(k, prev))
        new_score = float((1.0 - alpha) * prev + alpha * target)
        dims[k] = {"score": float(np.clip(new_score, 0.0, 1.0)), "explanation": _dim_explanation(k, new_score, profile)}
    return profile


def generate_detailed_analysis(profile: dict[str, Any]) -> dict[str, Any]:
    init_taste_profile(profile)
    dims = profile["taste_profile"]["dimensions"]
    ranked = sorted([(k, float(v.get("score", 0.5))) for k, v in dims.items()], key=lambda x: x[1], reverse=True)
    likes = [k for k, _ in ranked[:4]]
    avoids = [k for k, _ in ranked[-3:]]
    top_c = sorted(profile.get("favorite_cuisines", {}).items(), key=lambda x: x[1], reverse=True)[:3]
    top_d = sorted(profile.get("favorite_dishes", {}).items(), key=lambda x: x[1], reverse=True)[:3]

    notable = [
        f"Strongest pull: {likes[0].replace('_', ' ')} and {likes[1].replace('_', ' ')}." if len(likes) > 1 else "Taste profile still stabilizing.",
        f"Cuisine pattern: {', '.join([x for x, _ in top_c]) or 'mixed'}." ,
        f"Repeat-vs-variety currently leans toward {'variety' if float(dims['variety_seeking']['score']) >= 0.5 else 'repeat comfort'}."
    ]
    surprise = []
    if float(dims["dessert_affinity"]["score"]) >= 0.65 and float(dims["protein_forward"]["score"]) >= 0.60:
        surprise.append("You combine dessert curiosity with protein-forward choices more often than typical users.")
    if float(dims["spicy_leaning"]["score"]) >= 0.65 and float(dims["freshness_preference"]["score"]) >= 0.65:
        surprise.append("You seem to like heat and freshness together, not one at the expense of the other.")
    if not surprise:
        surprise.append("Your profile is balanced enough that no single trait dominates every upload.")

    playful = "Your uploads read like someone who plans meals with both curiosity and intent."
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
