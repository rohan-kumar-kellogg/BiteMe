from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ArchetypeMeta:
    name: str
    graphic_key: str
    emoji_fallback: str
    joke: str
    long_description_seed: str
    feature_weights: dict[str, float]


ARCHETYPE_CONFIG: dict[str, ArchetypeMeta] = {
    "Global Street-Food Hunter": ArchetypeMeta(
        name="Global Street-Food Hunter",
        graphic_key="global_street_food_hunter",
        emoji_fallback="🌍",
        joke="Your next trip is booked by snacks, not airlines.",
        long_description_seed="You show broad cuisine curiosity and low fear of unfamiliar flavor combinations.",
        feature_weights={"adventurousness": 0.45, "cuisine_variety": 0.35, "repeat_tendency_inv": 0.20},
    ),
    "Comfort-Core Loyalist": ArchetypeMeta(
        name="Comfort-Core Loyalist",
        graphic_key="comfort_core_loyalist",
        emoji_fallback="🍲",
        joke="Your comfort foods could probably beat stress in a straight fight.",
        long_description_seed="You repeatedly return to hearty, familiar plates and predictable satisfaction.",
        feature_weights={"comfort_tendency": 0.55, "repeat_tendency": 0.30, "dessert_tendency": 0.15},
    ),
    "Spice Orbit Captain": ArchetypeMeta(
        name="Spice Orbit Captain",
        graphic_key="spice_orbit_captain",
        emoji_fallback="🌶️",
        joke="You treat 'medium spice' as a warm-up lap.",
        long_description_seed="Your uploads consistently favor heat, aromatic intensity, and punchy profiles.",
        feature_weights={"spice_tendency": 0.65, "adventurousness": 0.20, "cuisine_variety": 0.15},
    ),
    "Seafood Signal Specialist": ArchetypeMeta(
        name="Seafood Signal Specialist",
        graphic_key="seafood_signal_specialist",
        emoji_fallback="🐟",
        joke="If it came from the ocean, you are already interested.",
        long_description_seed="Seafood patterns are strong and persistent across your uploads.",
        feature_weights={"seafood_tendency": 0.70, "protein_tendency": 0.20, "adventurousness": 0.10},
    ),
    "Plant-Forward Explorer": ArchetypeMeta(
        name="Plant-Forward Explorer",
        graphic_key="plant_forward_explorer",
        emoji_fallback="🥗",
        joke="Your produce intake is what nutrition posters aspire to.",
        long_description_seed="You favor fresh, produce-led meals with lighter structure and cleaner finish.",
        feature_weights={"plant_tendency": 0.70, "cuisine_variety": 0.20, "repeat_tendency_inv": 0.10},
    ),
    "Dessert Radar Commander": ArchetypeMeta(
        name="Dessert Radar Commander",
        graphic_key="dessert_radar_commander",
        emoji_fallback="🍰",
        joke="You do not chase dessert; dessert reports to you.",
        long_description_seed="Sweet-leaning choices appear early and repeat often in your profile.",
        feature_weights={"dessert_tendency": 0.70, "comfort_tendency": 0.20, "repeat_tendency": 0.10},
    ),
    "Protein-First Performer": ArchetypeMeta(
        name="Protein-First Performer",
        graphic_key="protein_first_performer",
        emoji_fallback="🥩",
        joke="You can probably estimate protein grams from a photo thumbnail.",
        long_description_seed="You consistently prioritize protein-forward centerplates over side-driven meals.",
        feature_weights={"protein_tendency": 0.65, "comfort_tendency": 0.20, "repeat_tendency": 0.15},
    ),
    "Carb Compass Romantic": ArchetypeMeta(
        name="Carb Compass Romantic",
        graphic_key="carb_compass_romantic",
        emoji_fallback="🍝",
        joke="Your relationship status with carbs is very committed.",
        long_description_seed="Your plate choices repeatedly orbit noodles, breads, rice, and pastry textures.",
        feature_weights={"carb_tendency": 0.65, "comfort_tendency": 0.20, "dessert_tendency": 0.15},
    ),
    "Repeat-Order Strategist": ArchetypeMeta(
        name="Repeat-Order Strategist",
        graphic_key="repeat_order_strategist",
        emoji_fallback="📌",
        joke="When you find a winner, you run it back with zero hesitation.",
        long_description_seed="You optimize for known favorites and show a high repeat-vs-variety ratio.",
        feature_weights={"repeat_tendency": 0.70, "comfort_tendency": 0.20, "adventurousness_inv": 0.10},
    ),
    "Fusion Risk-Taker": ArchetypeMeta(
        name="Fusion Risk-Taker",
        graphic_key="fusion_risk_taker",
        emoji_fallback="🧪",
        joke="Your taste profile reads like a menu collaboration episode.",
        long_description_seed="You mix styles and jump across cuisines with high novelty tolerance.",
        feature_weights={"adventurousness": 0.40, "cuisine_variety": 0.30, "spice_tendency": 0.15, "repeat_tendency_inv": 0.15},
    ),
    "Balanced Bistro Strategist": ArchetypeMeta(
        name="Balanced Bistro Strategist",
        graphic_key="balanced_bistro_strategist",
        emoji_fallback="🍽️",
        joke="You order like someone who already read the room and the menu.",
        long_description_seed="Your profile stays diverse but measured, without over-indexing on one extreme.",
        feature_weights={"balance_index": 0.50, "cuisine_variety": 0.25, "repeat_tendency_inv": 0.25},
    ),
}


def _safe_norm(weights: dict[str, float]) -> dict[str, float]:
    total = float(sum(max(0.0, float(v)) for v in weights.values()))
    if total <= 1e-9:
        return {k: 0.0 for k in weights}
    return {k: float(max(0.0, float(v)) / total) for k, v in weights.items()}


def _entropy_norm(values: list[float]) -> float:
    arr = np.asarray([max(0.0, float(x)) for x in values], dtype=np.float32)
    if arr.size <= 1 or float(np.sum(arr)) <= 1e-9:
        return 0.0
    p = arr / (np.sum(arr) + 1e-12)
    h = float(-np.sum(p * np.log(p + 1e-12)))
    return float(h / np.log(float(arr.size) + 1e-12))


def compute_behavior_features(profile: dict[str, Any]) -> dict[str, float]:
    cuisines = _safe_norm(profile.get("favorite_cuisines", {}))
    dishes = _safe_norm(profile.get("favorite_dishes", {}))
    traits = _safe_norm(profile.get("favorite_traits", {}))

    cuisine_var = _entropy_norm(list(cuisines.values()))
    dish_var = _entropy_norm(list(dishes.values()))
    repeat_tendency = float(max(dishes.values()) if dishes else 0.0)
    adventurousness = float(0.5 * cuisine_var + 0.5 * dish_var) * float(1.0 - 0.5 * repeat_tendency)

    dessert = float(traits.get("dessert-leaning", 0.0))
    spice = float(traits.get("spice-forward", 0.0))
    seafood = float(traits.get("seafood-leaning", 0.0))
    plant = float(traits.get("plant-forward", 0.0))
    comfort = float(traits.get("comfort-food", 0.0))
    protein = float(traits.get("protein-forward", 0.0))
    carb = float(traits.get("carb-forward", 0.0))

    extremes = np.asarray([dessert, spice, seafood, plant, comfort, protein, carb], dtype=np.float32)
    balance_index = float(1.0 - np.std(extremes))
    balance_index = float(np.clip(balance_index, 0.0, 1.0))

    return {
        "cuisine_variety": float(np.clip(cuisine_var, 0.0, 1.0)),
        "dish_variety": float(np.clip(dish_var, 0.0, 1.0)),
        "repeat_tendency": float(np.clip(repeat_tendency, 0.0, 1.0)),
        "repeat_tendency_inv": float(1.0 - np.clip(repeat_tendency, 0.0, 1.0)),
        "adventurousness": float(np.clip(adventurousness, 0.0, 1.0)),
        "adventurousness_inv": float(1.0 - np.clip(adventurousness, 0.0, 1.0)),
        "dessert_tendency": float(np.clip(dessert, 0.0, 1.0)),
        "spice_tendency": float(np.clip(spice, 0.0, 1.0)),
        "seafood_tendency": float(np.clip(seafood, 0.0, 1.0)),
        "plant_tendency": float(np.clip(plant, 0.0, 1.0)),
        "comfort_tendency": float(np.clip(comfort, 0.0, 1.0)),
        "protein_tendency": float(np.clip(protein, 0.0, 1.0)),
        "carb_tendency": float(np.clip(carb, 0.0, 1.0)),
        "balance_index": balance_index,
    }


def _score_archetypes(features: dict[str, float]) -> dict[str, float]:
    scores = {}
    for name, meta in ARCHETYPE_CONFIG.items():
        s = 0.0
        for f, w in meta.feature_weights.items():
            s += float(w) * float(features.get(f, 0.0))
        scores[name] = float(s)
    return scores


def choose_archetype(
    profile: dict[str, Any],
    *,
    previous_archetype: str | None = None,
    min_uploads_for_switch: int = 6,
) -> dict[str, Any]:
    features = compute_behavior_features(profile)
    scores = _score_archetypes(features)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_name, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    upload_count = int(profile.get("upload_count", 0))

    # Stability guard: avoid jitter from tiny updates.
    chosen = best_name
    if previous_archetype and previous_archetype in scores:
        prev_score = float(scores[previous_archetype])
        early_margin = 0.10
        mature_margin = 0.05
        margin = early_margin if upload_count < int(min_uploads_for_switch) else mature_margin
        if prev_score >= (best_score - margin):
            chosen = previous_archetype

    meta = ARCHETYPE_CONFIG[chosen]
    top_cuisines = sorted(profile.get("favorite_cuisines", {}).items(), key=lambda x: x[1], reverse=True)[:3]
    top_dishes = sorted(profile.get("favorite_dishes", {}).items(), key=lambda x: x[1], reverse=True)[:3]
    top_traits = sorted(profile.get("favorite_traits", {}).items(), key=lambda x: x[1], reverse=True)[:4]
    cuisines_txt = ", ".join([c for c, _ in top_cuisines]) or "mixed cuisines"
    dishes_txt = ", ".join([d.replace("_", " ") for d, _ in top_dishes]) or "a broad dish mix"
    traits_txt = ", ".join([t for t, _ in top_traits]) or "balanced preferences"

    long_description = (
        f"{meta.name} — {meta.long_description_seed} Across {upload_count} uploads, your strongest signals show "
        f"{cuisines_txt} with recurring choices like {dishes_txt}. Behavioral markers currently track as {traits_txt}, "
        "which suggests this pattern is intentional rather than random browsing."
    )

    observations = [
        f"Cuisine gravity: {cuisines_txt}.",
        f"Repeat-vs-variety: repeat tendency {features['repeat_tendency']:.2f}, adventurousness {features['adventurousness']:.2f}.",
        f"Flavor lean: spice {features['spice_tendency']:.2f}, comfort {features['comfort_tendency']:.2f}, dessert {features['dessert_tendency']:.2f}.",
        f"Structure lean: protein {features['protein_tendency']:.2f}, carb {features['carb_tendency']:.2f}, plant {features['plant_tendency']:.2f}.",
    ]
    return {
        "archetype": meta.name,
        "archetype_description": long_description,
        "archetype_graphic": f"assets/archetypes/{meta.graphic_key}.png",
        "archetype_graphic_key": meta.graphic_key,
        "archetype_emoji": meta.emoji_fallback,
        "joke": meta.joke,
        "observations": observations,
        "behavior_features": features,
        "archetype_scores": scores,
        "stability": {
            "previous_archetype": previous_archetype or "",
            "selected_best_raw": best_name,
            "selected_best_score": float(best_score),
            "second_best_score": float(second_score),
            "final_archetype": meta.name,
        },
    }
