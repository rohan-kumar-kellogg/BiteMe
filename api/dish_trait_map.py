from __future__ import annotations

from typing import Any


TRAIT_KEYS = [
    "sweet_leaning",
    "salty_leaning",
    "umami_leaning",
    "spicy_leaning",
    "richness_preference",
    "freshness_preference",
    "texture_seeking",
    "comfort_food_tendency",
    "protein_forward",
    "carb_forward",
    "dessert_affinity",
]


def canonical_dish_key(label: Any) -> str:
    x = str(label or "").strip().lower().replace("_", " ").replace("-", " ")
    return " ".join(x.split())


def _dense_trait_vector(partial: dict[str, float]) -> dict[str, float]:
    out = {k: 0.0 for k in TRAIT_KEYS}
    for k, v in partial.items():
        if k in out:
            out[k] = float(max(0.0, min(1.0, float(v))))
    return out


CANONICAL_DISH_TRAIT_MAP: dict[str, dict[str, float]] = {
    "tiramisu": _dense_trait_vector(
        {"sweet_leaning": 0.92, "richness_preference": 0.78, "comfort_food_tendency": 0.55, "dessert_affinity": 0.98}
    ),
    "brownie": _dense_trait_vector(
        {"sweet_leaning": 0.90, "richness_preference": 0.88, "comfort_food_tendency": 0.60, "dessert_affinity": 0.98}
    ),
    "cheesecake": _dense_trait_vector(
        {"sweet_leaning": 0.86, "richness_preference": 0.84, "comfort_food_tendency": 0.62, "dessert_affinity": 0.95}
    ),
    "apple pie": _dense_trait_vector(
        {"sweet_leaning": 0.88, "richness_preference": 0.72, "texture_seeking": 0.35, "comfort_food_tendency": 0.70, "carb_forward": 0.38, "dessert_affinity": 0.96}
    ),
    "ice cream": _dense_trait_vector(
        {"sweet_leaning": 0.94, "richness_preference": 0.62, "freshness_preference": 0.22, "comfort_food_tendency": 0.45, "dessert_affinity": 0.99}
    ),
    "gelato": _dense_trait_vector(
        {"sweet_leaning": 0.90, "richness_preference": 0.54, "freshness_preference": 0.28, "comfort_food_tendency": 0.40, "dessert_affinity": 0.97}
    ),
    "chocolate cake": _dense_trait_vector(
        {"sweet_leaning": 0.89, "richness_preference": 0.84, "comfort_food_tendency": 0.58, "carb_forward": 0.42, "dessert_affinity": 0.98}
    ),
    "buffalo wings": _dense_trait_vector(
        {"salty_leaning": 0.72, "umami_leaning": 0.74, "spicy_leaning": 0.90, "richness_preference": 0.82, "texture_seeking": 0.52, "comfort_food_tendency": 0.80, "protein_forward": 0.80}
    ),
    "mapo tofu": _dense_trait_vector(
        {"salty_leaning": 0.56, "umami_leaning": 0.91, "spicy_leaning": 0.88, "richness_preference": 0.55, "comfort_food_tendency": 0.36, "protein_forward": 0.62}
    ),
    "ramen": _dense_trait_vector(
        {"salty_leaning": 0.62, "umami_leaning": 0.90, "richness_preference": 0.66, "texture_seeking": 0.30, "comfort_food_tendency": 0.86, "protein_forward": 0.58, "carb_forward": 0.84}
    ),
    "pho": _dense_trait_vector(
        {"salty_leaning": 0.48, "umami_leaning": 0.86, "richness_preference": 0.28, "freshness_preference": 0.34, "comfort_food_tendency": 0.70, "protein_forward": 0.62, "carb_forward": 0.68}
    ),
    "fried chicken": _dense_trait_vector(
        {"salty_leaning": 0.60, "umami_leaning": 0.74, "richness_preference": 0.86, "texture_seeking": 0.70, "comfort_food_tendency": 0.82, "protein_forward": 0.86}
    ),
    "caesar salad": _dense_trait_vector(
        {"salty_leaning": 0.38, "umami_leaning": 0.35, "freshness_preference": 0.84, "texture_seeking": 0.20, "comfort_food_tendency": 0.22}
    ),
    "caprese salad": _dense_trait_vector(
        {"salty_leaning": 0.22, "umami_leaning": 0.20, "richness_preference": 0.12, "freshness_preference": 0.94, "texture_seeking": 0.18, "comfort_food_tendency": 0.14}
    ),
    "ceviche": _dense_trait_vector(
        {"salty_leaning": 0.34, "umami_leaning": 0.52, "spicy_leaning": 0.18, "freshness_preference": 0.95, "texture_seeking": 0.24, "protein_forward": 0.70}
    ),
    "mac and cheese": _dense_trait_vector(
        {"salty_leaning": 0.56, "umami_leaning": 0.62, "richness_preference": 0.92, "comfort_food_tendency": 0.94, "carb_forward": 0.64}
    ),
    "burger and fries": _dense_trait_vector(
        {"salty_leaning": 0.76, "umami_leaning": 0.74, "richness_preference": 0.84, "texture_seeking": 0.54, "comfort_food_tendency": 0.88, "protein_forward": 0.72, "carb_forward": 0.68}
    ),
    "sushi": _dense_trait_vector(
        {"salty_leaning": 0.35, "umami_leaning": 0.62, "freshness_preference": 0.86, "texture_seeking": 0.30, "protein_forward": 0.62, "carb_forward": 0.28}
    ),
    "steak": _dense_trait_vector(
        {"salty_leaning": 0.48, "umami_leaning": 0.90, "richness_preference": 0.66, "comfort_food_tendency": 0.64, "protein_forward": 0.96}
    ),
}


ALIASES: dict[str, str] = {
    "burger & fries": "burger and fries",
    "burger and french fries": "burger and fries",
    "burgers and fries": "burger and fries",
    "mac n cheese": "mac and cheese",
    "chocolate gateau": "chocolate cake",
    "caprese": "caprese salad",
}


def get_canonical_dish_traits(label: Any) -> dict[str, float] | None:
    key = canonical_dish_key(label)
    if not key:
        return None
    key = ALIASES.get(key, key)
    traits = CANONICAL_DISH_TRAIT_MAP.get(key)
    if traits is None:
        return None
    return dict(traits)
