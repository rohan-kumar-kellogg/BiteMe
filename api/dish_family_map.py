from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from api.dish_trait_map import TRAIT_KEYS, canonical_dish_key


def _dense_trait_vector(partial: dict[str, float]) -> dict[str, float]:
    out = {k: 0.0 for k in TRAIT_KEYS}
    for k, v in partial.items():
        if k in out:
            out[k] = float(max(0.0, min(1.0, float(v))))
    return out


FAMILY_TRAITS: dict[str, dict[str, float]] = {
    "dessert_family": _dense_trait_vector(
        {
            "sweet_leaning": 0.84,
            "richness_preference": 0.58,
            "comfort_food_tendency": 0.48,
            "carb_forward": 0.24,
            "dessert_affinity": 0.92,
        }
    ),
    "spicy_wing_family": _dense_trait_vector(
        {
            "salty_leaning": 0.64,
            "umami_leaning": 0.62,
            "spicy_leaning": 0.82,
            "richness_preference": 0.74,
            "texture_seeking": 0.46,
            "comfort_food_tendency": 0.70,
            "protein_forward": 0.76,
        }
    ),
    "savory_noodle_soup_family": _dense_trait_vector(
        {
            "salty_leaning": 0.52,
            "umami_leaning": 0.82,
            "richness_preference": 0.48,
            "freshness_preference": 0.18,
            "comfort_food_tendency": 0.72,
            "protein_forward": 0.56,
            "carb_forward": 0.72,
        }
    ),
    "fresh_salad_family": _dense_trait_vector(
        {
            "salty_leaning": 0.24,
            "umami_leaning": 0.28,
            "richness_preference": 0.16,
            "freshness_preference": 0.88,
            "texture_seeking": 0.22,
            "comfort_food_tendency": 0.16,
        }
    ),
}


@dataclass(frozen=True)
class DishFamilyRule:
    family_key: str
    contains_any: tuple[str, ...] = ()
    contains_all_groups: tuple[tuple[str, ...], ...] = ()


FAMILY_RULES: list[DishFamilyRule] = [
    # More specific families first.
    DishFamilyRule(
        family_key="spicy_wing_family",
        contains_any=(
            "buffalo wings",
            "hot wings",
            "peri peri wings",
            "peri-peri wings",
        ),
        contains_all_groups=(
            ("wings", "spicy"),
            ("wings", "hot"),
            ("wings", "chili"),
            ("wings", "peri"),
        ),
    ),
    DishFamilyRule(
        family_key="savory_noodle_soup_family",
        contains_any=(
            "ramen",
            "pho",
            "noodle soup",
            "beef noodle soup",
            "udon soup",
            "soba soup",
            "laksa",
            "khao soi",
            "champon",
            "tanmen",
            "hot and sour soup",
        ),
    ),
    DishFamilyRule(
        family_key="fresh_salad_family",
        contains_any=(
            "caesar salad",
            "green salad",
            "caprese salad",
            "caprese",
            "papaya salad",
            "salad",
            "ceviche",
        ),
    ),
    DishFamilyRule(
        family_key="dessert_family",
        contains_any=(
            "tiramisu",
            "brownie",
            "cheesecake",
            "cake",
            "apple pie",
            "pie",
            "ice cream",
            "gelato",
            "parfait",
            "cookie",
            "donut",
            "doughnut",
            "mousse",
            "pudding",
            "custard",
            "churro",
            "tart",
            "waffle",
            "crepe",
            "muffin",
            "scone",
            "shortcake",
        ),
    ),
]


def resolve_dish_family(label: Any) -> str | None:
    key = canonical_dish_key(label)
    if not key:
        return None
    for rule in FAMILY_RULES:
        if any(term in key for term in rule.contains_any):
            return rule.family_key
        for group in rule.contains_all_groups:
            if all(term in key for term in group):
                return rule.family_key
    return None


def get_family_traits(label: Any) -> dict[str, float] | None:
    fam = resolve_dish_family(label)
    if fam is None:
        return None
    traits = FAMILY_TRAITS.get(fam)
    if traits is None:
        return None
    return dict(traits)
