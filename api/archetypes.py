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
        long_description_seed="You treat menus like maps and rarely pick the safe route.",
        feature_weights={"adventurousness": 0.45, "cuisine_variety": 0.35, "repeat_tendency_inv": 0.20},
    ),
    "Comfort-Core Loyalist": ArchetypeMeta(
        name="Comfort-Core Loyalist",
        graphic_key="comfort_core_loyalist",
        emoji_fallback="🍲",
        joke="Your comfort foods could probably beat stress in a straight fight.",
        long_description_seed="You know your comfort zone and, to be fair, it has excellent food.",
        feature_weights={"comfort_tendency": 0.55, "repeat_tendency": 0.30, "dessert_tendency": 0.15},
    ),
    "Spice Orbit Captain": ArchetypeMeta(
        name="Spice Orbit Captain",
        graphic_key="spice_orbit_captain",
        emoji_fallback="🌶️",
        joke="You treat 'medium spice' as a warm-up lap.",
        long_description_seed="You keep dinner lively and mildly dangerous in the best way.",
        feature_weights={"spice_tendency": 0.65, "adventurousness": 0.20, "cuisine_variety": 0.15},
    ),
    "Seafood Signal Specialist": ArchetypeMeta(
        name="Seafood Signal Specialist",
        graphic_key="seafood_signal_specialist",
        emoji_fallback="🐟",
        joke="If it came from the ocean, you are already interested.",
        long_description_seed="If it came from the water, you are already halfway convinced.",
        feature_weights={"seafood_tendency": 0.70, "protein_tendency": 0.20, "adventurousness": 0.10},
    ),
    "Plant-Forward Explorer": ArchetypeMeta(
        name="Plant-Forward Explorer",
        graphic_key="plant_forward_explorer",
        emoji_fallback="🥗",
        joke="Your produce intake is what nutrition posters aspire to.",
        long_description_seed="You like bright, clean flavors, just not in a self-punishing way.",
        feature_weights={"plant_tendency": 0.70, "cuisine_variety": 0.20, "repeat_tendency_inv": 0.10},
    ),
    "Dessert Radar Commander": ArchetypeMeta(
        name="Dessert Radar Commander",
        graphic_key="dessert_radar_commander",
        emoji_fallback="🍰",
        joke="You do not chase dessert; dessert reports to you.",
        long_description_seed="Dessert keeps showing up near your decisions, very casually.",
        feature_weights={"dessert_tendency": 0.70, "comfort_tendency": 0.20, "repeat_tendency": 0.10},
    ),
    "Protein-First Performer": ArchetypeMeta(
        name="Protein-First Performer",
        graphic_key="protein_first_performer",
        emoji_fallback="🥩",
        joke="You can probably estimate protein grams from a photo thumbnail.",
        long_description_seed="You build meals around the main event and let sides negotiate.",
        feature_weights={"protein_tendency": 0.65, "comfort_tendency": 0.20, "repeat_tendency": 0.15},
    ),
    "Carb Compass Romantic": ArchetypeMeta(
        name="Carb Compass Romantic",
        graphic_key="carb_compass_romantic",
        emoji_fallback="🍝",
        joke="Your relationship status with carbs is very committed.",
        long_description_seed="Carbs are not a phase for you; they are reliable company.",
        feature_weights={"carb_tendency": 0.65, "comfort_tendency": 0.20, "dessert_tendency": 0.15},
    ),
    "Repeat-Order Strategist": ArchetypeMeta(
        name="Repeat-Order Strategist",
        graphic_key="repeat_order_strategist",
        emoji_fallback="📌",
        joke="When you find a winner, you run it back with zero hesitation.",
        long_description_seed="When something works, you see no reason to start improvising.",
        feature_weights={"repeat_tendency": 0.70, "comfort_tendency": 0.20, "adventurousness_inv": 0.10},
    ),
    "Fusion Risk-Taker": ArchetypeMeta(
        name="Fusion Risk-Taker",
        graphic_key="fusion_risk_taker",
        emoji_fallback="🧪",
        joke="Your taste profile reads like a menu collaboration episode.",
        long_description_seed="You like unexpected combinations and usually make them work.",
        feature_weights={"adventurousness": 0.40, "cuisine_variety": 0.30, "spice_tendency": 0.15, "repeat_tendency_inv": 0.15},
    ),
    "Balanced Bistro Strategist": ArchetypeMeta(
        name="Balanced Bistro Strategist",
        graphic_key="balanced_bistro_strategist",
        emoji_fallback="🍽️",
        joke="You order like someone who already read the room and the menu.",
        long_description_seed="You have range, but not the exhausting kind.",
        feature_weights={"balance_index": 0.50, "cuisine_variety": 0.25, "repeat_tendency_inv": 0.25},
    ),
}

DRY_OBSERVATIONS: dict[str, str] = {
    "Global Street-Food Hunter": "You collect cuisines the way other people collect playlists.",
    "Comfort-Core Loyalist": "Your favorites are basically emotional infrastructure.",
    "Spice Orbit Captain": "You treat heat like a feature, not a warning label.",
    "Seafood Signal Specialist": "You trust the ocean more than trend cycles.",
    "Plant-Forward Explorer": "You like freshness, but you still want dinner to feel like dinner.",
    "Dessert Radar Commander": "You are not always ordering dessert. You just keep ending up near it.",
    "Protein-First Performer": "You are not anti-carb; you just like a clear centerpiece.",
    "Carb Compass Romantic": "You and carbs have a long-term understanding.",
    "Repeat-Order Strategist": "You call it repetition; everyone else calls it consistency.",
    "Fusion Risk-Taker": "You flirt with chaos, then somehow make it dinner.",
    "Balanced Bistro Strategist": "You like variety, just with guardrails.",
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


def _dim_score(profile: dict[str, Any], key: str, default: float = 0.5) -> float:
    dims = profile.get("taste_profile", {}).get("dimensions", {}) or {}
    try:
        return float(dims.get(key, {}).get("score", default))
    except (TypeError, ValueError):
        return float(default)


def _pick_provisional_archetype(profile: dict[str, Any], features: dict[str, float]) -> str:
    """
    Early-stage picker (1-2 uploads):
    prioritize strongest directional signals and avoid neutral/balanced fallback.
    """
    dessert = _dim_score(profile, "dessert_affinity")
    sweet = _dim_score(profile, "sweet_leaning")
    spicy = _dim_score(profile, "spicy_leaning")
    rich = _dim_score(profile, "richness_preference")
    fresh = _dim_score(profile, "freshness_preference")
    comfort = _dim_score(profile, "comfort_food_tendency")
    protein = _dim_score(profile, "protein_forward")
    carb = _dim_score(profile, "carb_forward")

    # Strong fresh+light early signal should not be overshadowed by generic seafood/protein tags.
    if fresh >= 0.78 and rich <= 0.48:
        return "Plant-Forward Explorer"

    candidates = [
        ("Dessert Radar Commander", 0.65 * max(0.0, dessert - 0.5) + 0.35 * max(0.0, sweet - 0.5)),
        ("Spice Orbit Captain", 0.65 * max(0.0, spicy - 0.5) + 0.35 * max(0.0, rich - 0.5)),
        ("Plant-Forward Explorer", 0.60 * max(0.0, fresh - 0.5) + 0.20 * max(0.0, features.get("plant_tendency", 0.0)) + 0.20 * max(0.0, 0.55 - rich)),
        ("Protein-First Performer", 0.65 * max(0.0, protein - 0.5) + 0.25 * max(0.0, features.get("protein_tendency", 0.0)) + 0.10 * max(0.0, rich - 0.5)),
        ("Comfort-Core Loyalist", 0.60 * max(0.0, comfort - 0.5) + 0.25 * max(0.0, rich - 0.5) + 0.15 * max(0.0, carb - 0.5)),
        ("Carb Compass Romantic", 0.70 * max(0.0, carb - 0.5) + 0.30 * max(0.0, features.get("carb_tendency", 0.0))),
        ("Seafood Signal Specialist", max(0.0, features.get("seafood_tendency", 0.0))),
    ]
    chosen, score = max(candidates, key=lambda x: x[1])
    if score <= 1e-6:
        # Avoid generic balance in low-evidence mode.
        return "Comfort-Core Loyalist"
    return str(chosen)


def _derive_two_layer_archetype(profile: dict[str, Any], features: dict[str, float]) -> tuple[str, str]:
    """
    Returns:
      - primary archetype (core identity from strongest dimension)
      - secondary trait (next strongest dimension/behavior cue)
    """
    sweet = _dim_score(profile, "sweet_leaning")
    spicy = _dim_score(profile, "spicy_leaning")
    rich = _dim_score(profile, "richness_preference")
    fresh = _dim_score(profile, "freshness_preference")
    umami = _dim_score(profile, "umami_leaning")
    variety = max(_dim_score(profile, "variety_seeking"), float(features.get("cuisine_variety", 0.0)), float(features.get("adventurousness", 0.0)))
    texture = _dim_score(profile, "texture_seeking")
    protein = _dim_score(profile, "protein_forward")
    comfort = _dim_score(profile, "comfort_food_tendency")
    balance = float(features.get("balance_index", 0.0))
    adventurous = float(features.get("adventurousness", 0.0))

    primary_candidates: list[tuple[str, float, str]] = [
        ("sweet_leaning", sweet, "Certified Sweet Tooth"),
        ("spicy_leaning", spicy, "Heat Seeker"),
        ("richness_preference", rich, "Comfort Food Loyalist"),
        ("freshness_preference", fresh, "Bright Bite"),
        ("umami_leaning", umami, "Umami Enthusiast"),
        ("variety_seeking", variety, "Menu Explorer"),
    ]
    primary_key, _primary_score, primary_name = max(primary_candidates, key=lambda x: x[1])

    secondary_candidates: list[tuple[str, float, str]] = [
        ("sweet_leaning", sweet, "Sweet Finish"),
        ("late_night_energy", max(rich, comfort), "Late-Night Energy"),
        ("adventurousness", adventurous, "Adventurous Palate"),
        ("texture_seeking", texture, "Texture Chaser"),
        ("balanced_curiosity", 0.5 * adventurous + 0.5 * balance, "Balanced Curiosity"),
        ("freshness_preference", fresh, "Fresh Lean"),
        ("protein_forward", protein, "Protein Forward"),
    ]
    filtered_secondary = [x for x in secondary_candidates if x[0] != primary_key]
    secondary = max(filtered_secondary, key=lambda x: x[1])[2] if filtered_secondary else "Balanced Curiosity"
    return primary_name, secondary


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

    if upload_count <= 0:
        primary_name, secondary_trait = _derive_two_layer_archetype(profile, features)
        return {
            "archetype": "Profile Warming Up",
            "archetype_description": (
                "Your profile is warming up and still collecting evidence. "
                "One or two uploads can swing things quickly, so we are not locking in a type yet. "
                "You already have hints of a pattern; it is just early. "
                "Add a few more uploads and this read will sharpen."
            ),
            "archetype_graphic": "",
            "archetype_graphic_key": "profile_warming_up",
            "archetype_emoji": "✨",
            "primary_archetype": primary_name,
            "secondary_trait": secondary_trait,
            "joke": "",
            "observations": [
                "Early profile stage: not enough evidence for a stable archetype.",
                "Small sample sizes can overstate one meal.",
                "A few more uploads will make the pattern reliable.",
            ],
            "behavior_features": features,
            "archetype_scores": scores,
            "stability": {
                "previous_archetype": previous_archetype or "",
                "selected_best_raw": best_name,
                "selected_best_score": float(best_score),
                "second_best_score": float(second_score),
                "final_archetype": "Profile Warming Up",
            },
        }

    # Stage-aware selection: strong early signals first, then gradual normalization.
    if upload_count <= 2:
        chosen = _pick_provisional_archetype(profile, features)
    else:
        adjusted_scores = dict(scores)
        if upload_count <= 5:
            # Prevent "balanced" from winning too early unless evidence is broad.
            has_true_balance = (
                features.get("cuisine_variety", 0.0) >= 0.60
                and features.get("dish_variety", 0.0) >= 0.55
                and features.get("repeat_tendency", 0.0) <= 0.52
            )
            if not has_true_balance:
                adjusted_scores["Balanced Bistro Strategist"] = float(adjusted_scores.get("Balanced Bistro Strategist", 0.0) * 0.45)

        adjusted_ranked = sorted(adjusted_scores.items(), key=lambda x: x[1], reverse=True)
        chosen = str(adjusted_ranked[0][0])

        # Stability guard: avoid jitter from tiny updates (kept for mature phase).
        if previous_archetype and previous_archetype in adjusted_scores:
            prev_score = float(adjusted_scores[previous_archetype])
            early_margin = 0.10
            mature_margin = 0.05
            margin = early_margin if upload_count < int(min_uploads_for_switch) else mature_margin
            if prev_score >= (float(adjusted_scores[chosen]) - margin):
                chosen = previous_archetype

    meta = ARCHETYPE_CONFIG[chosen]
    top_cuisines = sorted(profile.get("favorite_cuisines", {}).items(), key=lambda x: x[1], reverse=True)[:3]
    top_dishes = sorted(profile.get("favorite_dishes", {}).items(), key=lambda x: x[1], reverse=True)[:3]
    top_traits = sorted(profile.get("favorite_traits", {}).items(), key=lambda x: x[1], reverse=True)[:4]
    cuisines_txt = ", ".join([c for c, _ in top_cuisines]) or "mixed cuisines"
    dishes_txt = ", ".join([d.replace("_", " ") for d, _ in top_dishes]) or "a broad dish mix"
    traits_txt = ", ".join([t for t, _ in top_traits]) or "balanced tastes"

    if features["adventurousness"] >= 0.62:
        habit_line = "You rotate enough to stay curious, but not enough to lose your center."
    elif features["repeat_tendency"] >= 0.58:
        habit_line = "You trust what works, and your dinner history backs that up."
    elif features["balance_index"] >= 0.62:
        habit_line = "You rarely overcorrect; even your indulgent moments have boundaries."
    else:
        habit_line = "Your habits tilt with your mood, but the pattern is still unmistakably you."

    long_description = " ".join(
        [
            meta.long_description_seed,
            DRY_OBSERVATIONS.get(meta.name, "Your food choices are consistent in a very human way."),
            habit_line,
            "Add a few more uploads and this read will sharpen.",
        ]
    )
    primary_name, secondary_trait = _derive_two_layer_archetype(profile, features)

    observations = [
        DRY_OBSERVATIONS.get(meta.name, "Your food choices are consistent in a very human way."),
        f"Usual rotation: {cuisines_txt}.",
        f"Recurring picks include {dishes_txt}.",
        f"Current mood board: {traits_txt}.",
    ]
    return {
        "archetype": meta.name,
        "primary_archetype": primary_name,
        "secondary_trait": secondary_trait,
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
