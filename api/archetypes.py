from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ArchetypeMeta:
    # Human-facing archetype pack. Keep copy concise and behavior-grounded.
    name: str
    graphic_key: str
    emoji_fallback: str
    joke: str
    description: str
    dry_observation: str
    score_weights: dict[str, float]


@dataclass(frozen=True)
class LowEvidenceMeta:
    # Low-evidence labels are intentionally observational, not identity-heavy.
    name: str
    graphic_key: str
    emoji_fallback: str
    description: str
    dry_observation: str


ARCHETYPE_CONFIG: dict[str, ArchetypeMeta] = {
    "Dessert First": ArchetypeMeta(
        name="Dessert First",
        graphic_key="dessert_first",
        emoji_fallback="🍰",
        joke="You call dessert optional in the same way rain is optional in April.",
        description="You clearly trust sugar to finish the story and usually start reading early.",
        dry_observation="You do not ignore a dessert menu. You review it for completeness.",
        score_weights={"dessert_score": 0.62, "snacky_treat_score": 0.20, "indulgence_score": 0.18},
    ),
    "Fries Non-Negotiable": ArchetypeMeta(
        name="Fries Non-Negotiable",
        graphic_key="fries_non_negotiable",
        emoji_fallback="🍟",
        joke="You don't order fries as a side. You order a second opinion.",
        description="You pick comfort food that actually comforts: crispy, salty, and not pretending.",
        dry_observation="You've never really been against fries. Just briefly unavailable.",
        score_weights={"comfort_score": 0.54, "indulgence_score": 0.26, "routine_score": 0.20},
    ),
    "Spicy or Bust": ArchetypeMeta(
        name="Spicy or Bust",
        graphic_key="spicy_or_bust",
        emoji_fallback="🌶️",
        joke="You call it flavor. Everyone else is looking for water.",
        description="You repeatedly choose dishes that come with heat and a small warning label.",
        dry_observation="Mild is mostly a suggestion to other people.",
        score_weights={"spicy_score": 0.68, "adventurous_score": 0.20, "indulgence_score": 0.12},
    ),
    "Soup Noodle Person": ArchetypeMeta(
        name="Soup Noodle Person",
        graphic_key="soup_noodle_person",
        emoji_fallback="🍜",
        joke="Your ideal weather is anything that justifies broth.",
        description="You keep ending up with bowls, steam, and noodles you have to lean over.",
        dry_observation="You trust dinner more when it arrives in a deep bowl.",
        score_weights={"noodle_score": 0.58, "comfort_score": 0.22, "umami_score": 0.20},
    ),
    "Salad, But Good": ArchetypeMeta(
        name="Salad, But Good",
        graphic_key="salad_but_good",
        emoji_fallback="🥗",
        joke="You eat clean, but never sad.",
        description="You go fresh on purpose, but still expect your food to have a point of view.",
        dry_observation="You'll pick the healthy option if it looks like it wants to live.",
        score_weights={"wellness_score": 0.66, "fresh_score": 0.20, "routine_score": 0.14},
    ),
    "Sushi Too Often": ArchetypeMeta(
        name="Sushi Too Often",
        graphic_key="sushi_too_often",
        emoji_fallback="🍣",
        joke="You can identify good sushi by lighting and chair design.",
        description="You choose sushi with enough frequency that this is no longer random.",
        dry_observation="At this point, chopsticks are muscle memory.",
        score_weights={"sushi_score": 0.60, "trendy_score": 0.24, "adventurous_score": 0.16},
    ),
    "Taco Rotation": ArchetypeMeta(
        name="Taco Rotation",
        graphic_key="taco_rotation",
        emoji_fallback="🌮",
        joke="You don't ask if tacos are the move. You ask which tacos.",
        description="Tacos show up in your upload history like a recurring appointment.",
        dry_observation="You treat taco night like a stable operating system.",
        score_weights={"taco_score": 0.58, "comfort_score": 0.24, "spicy_score": 0.18},
    ),
    "Same Order Energy": ArchetypeMeta(
        name="Same Order Energy",
        graphic_key="same_order_energy",
        emoji_fallback="🧾",
        joke="Your reorder button has seen things.",
        description="You know your lane, and your lane has your order waiting.",
        dry_observation="When you find the move, you keep the move.",
        score_weights={"routine_score": 0.64, "comfort_score": 0.20, "variety_inverse_score": 0.16},
    ),
    "Late Night Menu": ArchetypeMeta(
        name="Late Night Menu",
        graphic_key="late_night_menu",
        emoji_fallback="🍔",
        joke="Your best food decisions happen after regular business hours.",
        description="You lean rich, salty, satisfying, and occasionally very late.",
        dry_observation="You're not chaotic, just easily tempted after dark.",
        score_weights={"indulgence_score": 0.50, "comfort_score": 0.25, "snacky_treat_score": 0.25},
    ),
    "Balanced, Still Hungry": ArchetypeMeta(
        name="Balanced, Still Hungry",
        graphic_key="balanced_still_hungry",
        emoji_fallback="🍽️",
        joke="You contain multitudes, including fries.",
        description="You have range without being random, and your choices stay consistently solid.",
        dry_observation="You don't have one lane; you have standards.",
        score_weights={"balanced_score": 0.58, "variety_score": 0.24, "adventurous_score": 0.18},
    ),
}

LOW_EVIDENCE_ARCHETYPE_CONFIG: dict[str, LowEvidenceMeta] = {
    "Classic Comfort": LowEvidenceMeta(
        name="Classic Comfort",
        graphic_key="classic_comfort",
        emoji_fallback="🍔",
        description="First impression: you are starting in the comfort-food lane.",
        dry_observation="So far this reads savory, familiar, and satisfying.",
    ),
    "Burger Energy": LowEvidenceMeta(
        name="Burger Energy",
        graphic_key="burger_energy",
        emoji_fallback="🍔",
        description="First impression: this is giving burger-and-fries energy.",
        dry_observation="Early signal says classic comfort over experimentation.",
    ),
    "Sweet Start": LowEvidenceMeta(
        name="Sweet Start",
        graphic_key="sweet_start",
        emoji_fallback="🍰",
        description="Not much data yet, but dessert is already making a case.",
        dry_observation="So far this leans sweet in a very convincing way.",
    ),
    "Fresh Lean": LowEvidenceMeta(
        name="Fresh Lean",
        graphic_key="fresh_lean",
        emoji_fallback="🥗",
        description="So far this is reading lighter, brighter, and a little cleaner.",
        dry_observation="Early signal points to fresh over heavy.",
    ),
    "Heat Check": LowEvidenceMeta(
        name="Heat Check",
        graphic_key="heat_check",
        emoji_fallback="🌶️",
        description="Early signs point to someone who is not afraid of spice.",
        dry_observation="Right now this profile looks heat-friendly.",
    ),
    "Savory Start": LowEvidenceMeta(
        name="Savory Start",
        graphic_key="savory_start",
        emoji_fallback="🍜",
        description="First impression: this is leaning savory and umami-forward.",
        dry_observation="So far, rich and savory is winning.",
    ),
    "Noodle Signal": LowEvidenceMeta(
        name="Noodle Signal",
        graphic_key="noodle_signal",
        emoji_fallback="🍜",
        description="Early signal says noodles are already in the lead.",
        dry_observation="First impression: bowls over bites.",
    ),
    "Thai Signal": LowEvidenceMeta(
        name="Thai Signal",
        graphic_key="thai_signal",
        emoji_fallback="🍛",
        description="First impression: there is already a clear Thai direction.",
        dry_observation="So far this is reading bright, spicy, and Thai-leaning.",
    ),
    "Global Start": LowEvidenceMeta(
        name="Global Start",
        graphic_key="global_start",
        emoji_fallback="🌍",
        description="Early signal points to a globally curious plate.",
        dry_observation="Right now this looks like range, not routine.",
    ),
    "Open Plate": LowEvidenceMeta(
        name="Open Plate",
        graphic_key="open_plate",
        emoji_fallback="🍽️",
        description="Still early, so this is a first impression rather than a fixed type.",
        dry_observation="A couple more uploads will sharpen this quickly.",
    ),
}

FALLBACK_ARCHETYPE = "Balanced, Still Hungry"
SYSTEM_ARCHETYPE = "Profile Warming Up"
CANONICAL_ARCHETYPE_NAMES = tuple(ARCHETYPE_CONFIG.keys())
LOW_EVIDENCE_ARCHETYPE_NAMES = tuple(LOW_EVIDENCE_ARCHETYPE_CONFIG.keys())
VALID_ARCHETYPE_NAMES = set(CANONICAL_ARCHETYPE_NAMES) | set(LOW_EVIDENCE_ARCHETYPE_NAMES) | {SYSTEM_ARCHETYPE}

EVIDENCE_THRESHOLDS = {
    "low_max_uploads": 2,
    "medium_max_uploads": 5,
    "high_min_uploads": 6,
}

# Tunable stability behavior for archetype switching.
# Lower values make archetypes adapt faster to new profile evidence.
STABILITY_TUNING = {
    "medium_keep_margin": 0.0,   # medium evidence should evolve readily
    "high_keep_margin": 0.05,    # high evidence can resist tiny jitter
    "high_min_improvement": 0.02,  # require a small lead to replace previous in high mode
}


def is_valid_archetype_name(name: str, *, allow_system: bool = True) -> bool:
    nm = str(name or "").strip()
    if not nm:
        return False
    if allow_system:
        return nm in VALID_ARCHETYPE_NAMES
    return nm in ARCHETYPE_CONFIG


def coerce_archetype_name(name: str, *, allow_system: bool = True) -> str:
    nm = str(name or "").strip()
    if is_valid_archetype_name(nm, allow_system=allow_system):
        return nm
    return FALLBACK_ARCHETYPE


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


def _dim_score(profile: dict[str, Any], key: str, default: float = 0.5) -> float:
    dims = profile.get("taste_profile", {}).get("dimensions", {}) or {}
    try:
        return float(np.clip(float(dims.get(key, {}).get("score", default)), 0.0, 1.0))
    except (TypeError, ValueError):
        return float(default)


def _token_score(source: dict[str, float], terms: set[str]) -> float:
    if not source or not terms:
        return 0.0
    total = float(sum(max(0.0, float(v)) for v in source.values()))
    if total <= 1e-9:
        return 0.0
    hit = 0.0
    for label, weight in source.items():
        txt = str(label).lower().replace("_", " ")
        if any(term in txt for term in terms):
            hit += max(0.0, float(weight))
    return float(np.clip(hit / (total + 1e-12), 0.0, 1.0))


def compute_behavior_features(profile: dict[str, Any]) -> dict[str, float]:
    cuisines = _safe_norm(profile.get("favorite_cuisines", {}))
    dishes = _safe_norm(profile.get("favorite_dishes", {}))
    traits = _safe_norm(profile.get("favorite_traits", {}))

    cuisine_var = _entropy_norm(list(cuisines.values()))
    dish_var = _entropy_norm(list(dishes.values()))
    repeat_tendency = float(max(dishes.values()) if dishes else 0.0)
    adventurousness = float(0.5 * cuisine_var + 0.5 * dish_var) * float(1.0 - 0.45 * repeat_tendency)

    sweet = _dim_score(profile, "sweet_leaning")
    spicy = _dim_score(profile, "spicy_leaning")
    rich = _dim_score(profile, "richness_preference")
    fresh = _dim_score(profile, "freshness_preference")
    dessert_dim = _dim_score(profile, "dessert_affinity")
    protein_dim = _dim_score(profile, "protein_forward")
    comfort_dim = _dim_score(profile, "comfort_food_tendency")
    umami_dim = _dim_score(profile, "umami_leaning")
    variety_dim = _dim_score(profile, "variety_seeking")

    dessert_dishes = {
        "cake", "pie", "tiramisu", "brownie", "cookie", "ice cream", "gelato", "donut", "pastry", "cheesecake",
        "cannoli", "macaron", "eclair", "mousse", "pudding", "tart",
    }
    comfort_dishes = {
        "burger", "fries", "pizza", "wings", "fried chicken", "mac and cheese", "lasagna", "hot dog", "nachos",
        "quesadilla", "sliders", "mozzarella sticks",
    }
    wellness_dishes = {
        "salad", "grain bowl", "bowl", "avocado toast", "smoothie", "poke", "grilled fish", "quinoa", "yogurt",
        "veggie", "vegetable", "fruit",
    }
    trendy_dishes = {
        "sushi", "omakase", "crudo", "ceviche", "matcha", "espresso", "latte", "pastry", "croissant", "brunch",
        "toast", "natural wine", "small plate",
    }
    protein_dishes = {
        "steak", "grilled chicken", "chicken breast", "salmon", "tuna", "egg", "omelette", "protein", "kebab",
        "shawarma", "tofu",
    }
    noodle_dishes = {"ramen", "pho", "udon", "soba", "noodle", "noodle soup", "laksa"}
    sushi_dishes = {"sushi", "sashimi", "nigiri", "maki", "omakase", "chirashi"}
    taco_dishes = {"taco", "tacos", "quesabirria", "burrito", "quesadilla", "enchilada"}
    indulgent_dishes = {
        "fried", "butter", "creamy", "cheese", "bacon", "ribs", "brisket", "burger", "pizza", "wings", "pasta",
        "ramen",
    }
    snacky_dishes = {
        "donut", "cookie", "fries", "nachos", "chips", "boba", "ice cream", "brownie", "pastry", "waffle",
        "croissant",
    }
    fries_explicit_terms = {
        "fries", "french fries", "loaded fries", "curly fries", "truffle fries", "steak fries", "waffle fries",
    }

    trendy_cuisines = {"japanese", "korean", "french", "mediterranean"}
    global_cuisines = {"thai", "indian", "mexican", "japanese", "korean", "middle_eastern", "vietnamese", "chinese"}

    cuisine_global_score = float(np.clip(sum(cuisines.get(c, 0.0) for c in global_cuisines), 0.0, 1.0))
    cuisine_trendy_score = float(np.clip(sum(cuisines.get(c, 0.0) for c in trendy_cuisines), 0.0, 1.0))

    trait_dessert = float(np.clip(traits.get("dessert-leaning", 0.0), 0.0, 1.0))
    trait_spice = float(np.clip(traits.get("spice-forward", 0.0), 0.0, 1.0))
    trait_comfort = float(np.clip(traits.get("comfort-food", 0.0), 0.0, 1.0))
    trait_plant = float(np.clip(traits.get("plant-forward", 0.0), 0.0, 1.0))
    trait_protein = float(np.clip(traits.get("protein-forward", 0.0), 0.0, 1.0))
    trait_carb = float(np.clip(traits.get("carb-forward", 0.0), 0.0, 1.0))

    dessert_score = 0.45 * ((dessert_dim + sweet) / 2.0) + 0.30 * trait_dessert + 0.25 * _token_score(dishes, dessert_dishes)
    comfort_score = 0.38 * comfort_dim + 0.24 * rich + 0.18 * trait_comfort + 0.20 * _token_score(dishes, comfort_dishes)
    indulgence_score = 0.35 * rich + 0.20 * comfort_dim + 0.25 * _token_score(dishes, indulgent_dishes) + 0.20 * trait_carb
    wellness_score = 0.40 * fresh + 0.22 * trait_plant + 0.20 * _token_score(dishes, wellness_dishes) + 0.18 * (1.0 - rich)
    trendy_score = 0.35 * _token_score(dishes, trendy_dishes) + 0.25 * cuisine_trendy_score + 0.20 * variety_dim + 0.20 * cuisine_var
    spicy_score = 0.60 * spicy + 0.25 * trait_spice + 0.15 * cuisine_global_score
    protein_functional_score = 0.45 * protein_dim + 0.25 * trait_protein + 0.20 * _token_score(dishes, protein_dishes) + 0.10 * (1.0 - dessert_dim)
    noodle_score = 0.65 * _token_score(dishes, noodle_dishes) + 0.20 * cuisines.get("japanese", 0.0) + 0.15 * umami_dim
    sushi_score = 0.68 * _token_score(dishes, sushi_dishes) + 0.20 * cuisines.get("japanese", 0.0) + 0.12 * trendy_score
    taco_score = 0.62 * _token_score(dishes, taco_dishes) + 0.26 * cuisines.get("mexican", 0.0) + 0.12 * spicy
    snacky_treat_score = 0.35 * _token_score(dishes, snacky_dishes) + 0.35 * dessert_score + 0.30 * trait_carb
    fries_observed_score = _token_score(dishes, fries_explicit_terms)
    variety_score = 0.45 * cuisine_var + 0.30 * dish_var + 0.25 * variety_dim
    variety_inverse_score = float(np.clip(1.0 - variety_score, 0.0, 1.0))
    routine_score = 0.60 * repeat_tendency + 0.30 * variety_inverse_score + 0.10 * float(max(0.0, comfort_dim - 0.5))
    adventurous_score = 0.50 * adventurousness + 0.35 * variety_score + 0.15 * cuisine_global_score
    umami_score = 0.55 * umami_dim + 0.20 * trait_protein + 0.25 * float(max(0.0, rich))

    directional = np.asarray(
        [dessert_score, comfort_score, spicy_score, wellness_score, protein_functional_score, trendy_score, adventurous_score],
        dtype=np.float32,
    )
    balanced_score = float(np.clip(1.0 - np.std(directional), 0.0, 1.0)) * float(np.clip(variety_score, 0.0, 1.0))

    return {
        "upload_count": float(max(0, int(profile.get("upload_count", 0)))),
        "cuisine_variety": float(np.clip(cuisine_var, 0.0, 1.0)),
        "dish_variety": float(np.clip(dish_var, 0.0, 1.0)),
        "repeat_tendency": float(np.clip(repeat_tendency, 0.0, 1.0)),
        "adventurousness": float(np.clip(adventurousness, 0.0, 1.0)),
        "dessert_score": float(np.clip(dessert_score, 0.0, 1.0)),
        "comfort_score": float(np.clip(comfort_score, 0.0, 1.0)),
        "indulgence_score": float(np.clip(indulgence_score, 0.0, 1.0)),
        "wellness_score": float(np.clip(wellness_score, 0.0, 1.0)),
        "trendy_score": float(np.clip(trendy_score, 0.0, 1.0)),
        "adventurous_score": float(np.clip(adventurous_score, 0.0, 1.0)),
        "routine_score": float(np.clip(routine_score, 0.0, 1.0)),
        "protein_functional_score": float(np.clip(protein_functional_score, 0.0, 1.0)),
        "snacky_treat_score": float(np.clip(snacky_treat_score, 0.0, 1.0)),
        "fries_observed_score": float(np.clip(fries_observed_score, 0.0, 1.0)),
        "spicy_score": float(np.clip(spicy_score, 0.0, 1.0)),
        "noodle_score": float(np.clip(noodle_score, 0.0, 1.0)),
        "sushi_score": float(np.clip(sushi_score, 0.0, 1.0)),
        "taco_score": float(np.clip(taco_score, 0.0, 1.0)),
        "fresh_score": float(np.clip(fresh, 0.0, 1.0)),
        "umami_score": float(np.clip(umami_score, 0.0, 1.0)),
        "variety_score": float(np.clip(variety_score, 0.0, 1.0)),
        "variety_inverse_score": float(np.clip(variety_inverse_score, 0.0, 1.0)),
        "balanced_score": float(np.clip(balanced_score, 0.0, 1.0)),
    }


def _score_archetypes(features: dict[str, float]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for name, meta in ARCHETYPE_CONFIG.items():
        raw = 0.0
        for key, weight in meta.score_weights.items():
            raw += float(weight) * float(features.get(key, 0.0))
        scores[name] = float(np.clip(raw, 0.0, 1.0))
    return scores


def _evidence_mode(upload_count: int) -> str:
    if upload_count <= int(EVIDENCE_THRESHOLDS["low_max_uploads"]):
        return "low"
    if upload_count <= int(EVIDENCE_THRESHOLDS["medium_max_uploads"]):
        return "medium"
    return "high"


def _evidence_count(profile: dict[str, Any]) -> int:
    """
    Count total taste-learning interactions, not just uploads.
    This lets recommendation-click learning influence archetype maturity.
    """
    upload_count = int(profile.get("upload_count", 0) or 0)
    interaction_count = int(profile.get("interaction_count", 0) or 0)
    history = profile.get("taste_profile", {}).get("history", [])
    history_count = len(history) if isinstance(history, list) else 0
    return max(upload_count, interaction_count, history_count)


def _top_label_key(weights: dict[str, Any]) -> str:
    if not isinstance(weights, dict) or not weights:
        return ""
    try:
        top = max(weights.items(), key=lambda x: float(x[1]))
        return str(top[0]).lower().replace("_", " ").strip()
    except Exception:
        return ""


def _pick_low_evidence_archetype(profile: dict[str, Any], features: dict[str, float]) -> str:
    top_dish = _top_label_key(profile.get("favorite_dishes", {}))
    top_cuisine = _top_label_key(profile.get("favorite_cuisines", {}))

    if top_cuisine == "thai" and features.get("spicy_score", 0.0) >= 0.55:
        return "Thai Signal"
    if "burger" in top_dish:
        return "Burger Energy"
    if features.get("dessert_score", 0.0) >= 0.58:
        return "Sweet Start"
    if features.get("spicy_score", 0.0) >= 0.62:
        return "Heat Check"
    if features.get("noodle_score", 0.0) >= 0.60:
        return "Noodle Signal"
    if features.get("wellness_score", 0.0) >= 0.60:
        return "Fresh Lean"
    if features.get("comfort_score", 0.0) >= 0.58:
        return "Classic Comfort"
    if features.get("umami_score", 0.0) >= 0.60:
        return "Savory Start"
    if features.get("adventurous_score", 0.0) >= 0.62 or features.get("variety_score", 0.0) >= 0.62:
        return "Global Start"
    return "Open Plate"


def _derive_two_layer_archetype(profile: dict[str, Any], features: dict[str, float]) -> tuple[str, str]:
    """
    Returns:
      - primary archetype (core identity from strongest dimension)
      - secondary trait (next strongest dimension/behavior cue)
    """
    primary_candidates: list[tuple[str, float]] = [
        ("Dessert Energy", float(features.get("dessert_score", 0.0))),
        ("Comfort Lane", float(features.get("comfort_score", 0.0))),
        ("Heat Lane", float(features.get("spicy_score", 0.0))),
        ("Fresh Lane", float(features.get("wellness_score", 0.0))),
        ("Protein Lane", float(features.get("protein_functional_score", 0.0))),
        ("Curiosity Lane", float(features.get("adventurous_score", 0.0))),
    ]
    primary_name, _primary_score = max(primary_candidates, key=lambda x: x[1])

    secondary_candidates: list[tuple[str, float]] = [
        ("Late-Night Energy", float(features.get("indulgence_score", 0.0))),
        ("Sweet Finish", float(features.get("dessert_score", 0.0))),
        ("Texture Chaser", _dim_score(profile, "texture_seeking")),
        ("Balanced Curiosity", float(features.get("balanced_score", 0.0))),
        ("Protein Forward", float(features.get("protein_functional_score", 0.0))),
        ("Fresh Lean", float(features.get("wellness_score", 0.0))),
    ]
    filtered = [x for x in secondary_candidates if x[0] != primary_name]
    secondary = max(filtered, key=lambda x: x[1])[0] if filtered else "Balanced Curiosity"
    return primary_name, secondary


def _top_signal_rows(features: dict[str, float], n: int = 5) -> list[dict[str, float | str]]:
    skip = {"upload_count", "cuisine_variety", "dish_variety", "adventurousness"}
    rows = [(k, float(v)) for k, v in features.items() if k not in skip]
    rows = sorted(rows, key=lambda x: x[1], reverse=True)[: max(1, int(n))]
    return [{"signal": k, "score": round(v, 4)} for k, v in rows]


def debug_archetype_decision(profile: dict[str, Any]) -> dict[str, Any]:
    """
    Lightweight observability helper for QA:
    shows dominant behavioral signals, archetype scores, and winning label.
    """
    features = compute_behavior_features(profile)
    scores = _score_archetypes(features)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    winner = ranked[0][0] if ranked else FALLBACK_ARCHETYPE
    upload_count = _evidence_count(profile)
    return {
        "evidence_mode": _evidence_mode(upload_count),
        "top_signals": _top_signal_rows(features, n=7),
        "archetype_scores_ranked": [{"archetype": n, "score": round(float(s), 4)} for n, s in ranked],
        "winner": winner,
    }


def choose_archetype(
    profile: dict[str, Any],
    *,
    previous_archetype: str | None = None,
    previous_stability: dict[str, Any] | None = None,
    min_uploads_for_switch: int = 6,
) -> dict[str, Any]:
    features = compute_behavior_features(profile)
    scores = _score_archetypes(features)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_name, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    evidence_count = _evidence_count(profile)
    mode = _evidence_mode(evidence_count)

    if evidence_count <= 0:
        primary_name, secondary_trait = _derive_two_layer_archetype(profile, features)
        return {
            "archetype": SYSTEM_ARCHETYPE,
            "archetype_description": (
                "We need a little more signal before we call your archetype. "
                "Add a few more uploads and this will lock in fast."
            ),
            "archetype_graphic": "",
            "archetype_graphic_key": "profile_warming_up",
            "archetype_emoji": "✨",
            "primary_archetype": primary_name,
            "secondary_trait": secondary_trait,
            "joke": "",
            "observations": [
                "Early stage: one dish can still swing the read.",
                "More uploads = less guesswork.",
            ],
            "behavior_features": features,
            "archetype_scores": scores,
            "debug": debug_archetype_decision(profile),
            "stability": {
                "previous_archetype": previous_archetype or "",
                "selected_best_raw": best_name,
                "selected_best_score": float(best_score),
                "second_best_score": float(second_score),
                "final_archetype": SYSTEM_ARCHETYPE,
                "evidence_mode": _evidence_mode(evidence_count),
                "tuning": STABILITY_TUNING,
            },
        }

    if mode == "low":
        chosen = _pick_low_evidence_archetype(profile, features)
        meta = LOW_EVIDENCE_ARCHETYPE_CONFIG.get(chosen, LOW_EVIDENCE_ARCHETYPE_CONFIG["Open Plate"])
        primary_name, secondary_trait = _derive_two_layer_archetype(profile, features)
        observations = [
            meta.dry_observation,
            "Low evidence mode: first-impression read based on early food signal.",
        ]
        low_scores = {
            k: float(v)
            for k, v in {
                "dessert_score": features.get("dessert_score", 0.0),
                "comfort_score": features.get("comfort_score", 0.0),
                "spicy_score": features.get("spicy_score", 0.0),
                "noodle_score": features.get("noodle_score", 0.0),
                "wellness_score": features.get("wellness_score", 0.0),
                "umami_score": features.get("umami_score", 0.0),
                "adventurous_score": features.get("adventurous_score", 0.0),
            }.items()
        }
        return {
            "archetype": meta.name,
            "primary_archetype": primary_name,
            "secondary_trait": secondary_trait,
            "archetype_description": f"{meta.description} {meta.dry_observation}",
            "archetype_graphic": f"assets/archetypes/{meta.graphic_key}.png",
            "archetype_graphic_key": meta.graphic_key,
            "archetype_emoji": meta.emoji_fallback,
            "joke": "",
            "observations": observations,
            "behavior_features": features,
            "archetype_scores": low_scores,
            "debug": debug_archetype_decision(profile),
            "stability": {
                "previous_archetype": previous_archetype or "",
                "selected_best_raw": chosen,
                "selected_best_score": float(max(low_scores.values()) if low_scores else 0.0),
                "second_best_score": 0.0,
                "final_archetype": meta.name,
                "evidence_mode": mode,
                "tuning": STABILITY_TUNING,
            },
        }

    if mode in {"medium", "high"}:
        # Repetition override: repeated safe favorites should read as routine.
        if (
            float(features.get("repeat_tendency", 0.0)) >= 0.62
            and float(features.get("variety_score", 0.0)) <= 0.42
            and float(features.get("dessert_score", 0.0)) < 0.70
        ):
            chosen = "Same Order Energy"
        else:
            adjusted_scores = dict(scores)
            # Strong specific food lanes should win over broad generic outcomes.
            if float(features.get("dessert_score", 0.0)) >= 0.74:
                adjusted_scores["Dessert First"] = float(adjusted_scores.get("Dessert First", 0.0) + 0.28)
            if float(features.get("sushi_score", 0.0)) >= 0.60:
                adjusted_scores["Sushi Too Often"] = float(adjusted_scores.get("Sushi Too Often", 0.0) + 0.22)
            if float(features.get("noodle_score", 0.0)) >= 0.62:
                adjusted_scores["Soup Noodle Person"] = float(adjusted_scores.get("Soup Noodle Person", 0.0) + 0.22)
            if float(features.get("taco_score", 0.0)) >= 0.60:
                adjusted_scores["Taco Rotation"] = float(adjusted_scores.get("Taco Rotation", 0.0) + 0.20)
            # Guardrail: "Fries Non-Negotiable" must be backed by explicit fries evidence.
            # A generic comfort signal (e.g., one burger) should not produce a fries-specific archetype.
            fries_observed = float(features.get("fries_observed_score", 0.0))
            if fries_observed < 0.16:
                adjusted_scores["Fries Non-Negotiable"] = float(adjusted_scores.get("Fries Non-Negotiable", 0.0) * 0.30)
            elif fries_observed >= 0.30:
                adjusted_scores["Fries Non-Negotiable"] = float(adjusted_scores.get("Fries Non-Negotiable", 0.0) + 0.10)
            # Keep "balanced" as fallback. It should not beat obvious repeated patterns.
            dominant_signal = max(
                float(features.get("dessert_score", 0.0)),
                float(features.get("comfort_score", 0.0)),
                float(features.get("spicy_score", 0.0)),
                float(features.get("wellness_score", 0.0)),
                float(features.get("protein_functional_score", 0.0)),
                float(features.get("trendy_score", 0.0)),
                float(features.get("adventurous_score", 0.0)),
                float(features.get("routine_score", 0.0)),
                float(features.get("indulgence_score", 0.0)),
                float(features.get("noodle_score", 0.0)),
                float(features.get("sushi_score", 0.0)),
                float(features.get("taco_score", 0.0)),
            )
            if dominant_signal >= 0.44:
                adjusted_scores["Balanced, Still Hungry"] = float(adjusted_scores.get("Balanced, Still Hungry", 0.0) * 0.35)
            if mode == "medium":
                adjusted_scores["Balanced, Still Hungry"] = float(adjusted_scores.get("Balanced, Still Hungry", 0.0) * 0.55)
                # In medium evidence, avoid overconfident stable-habit archetype unless very strong.
                if float(features.get("routine_score", 0.0)) < 0.72:
                    adjusted_scores["Same Order Energy"] = float(adjusted_scores.get("Same Order Energy", 0.0) * 0.65)

            adjusted_ranked = sorted(adjusted_scores.items(), key=lambda x: x[1], reverse=True)
            chosen = str(adjusted_ranked[0][0])

            # Stability guard (explicit/tunable):
            # - medium evidence: minimal stickiness so archetype can evolve with profile updates
            # - high evidence: keep previous archetype only when challenger lead is very small
            if previous_archetype and previous_archetype in adjusted_scores:
                prev_score = float(adjusted_scores[previous_archetype])
                chosen_score = float(adjusted_scores[chosen])
                prev_mode = ""
                if isinstance(previous_stability, dict):
                    prev_mode = str(previous_stability.get("evidence_mode", "") or "")
                if mode == "medium":
                    margin = float(STABILITY_TUNING["medium_keep_margin"])
                    if prev_score >= (chosen_score - margin):
                        chosen = previous_archetype
                else:
                    # Do not enforce extra stickiness on the first transition into high evidence.
                    if prev_mode == "high":
                        margin = float(STABILITY_TUNING["high_keep_margin"])
                        improvement_needed = float(STABILITY_TUNING["high_min_improvement"])
                        if (chosen_score - prev_score) < max(margin, improvement_needed):
                            chosen = previous_archetype

    chosen = coerce_archetype_name(chosen, allow_system=False)
    meta = ARCHETYPE_CONFIG[chosen]
    top_cuisines = sorted(profile.get("favorite_cuisines", {}).items(), key=lambda x: x[1], reverse=True)[:3]
    top_dishes = sorted(profile.get("favorite_dishes", {}).items(), key=lambda x: x[1], reverse=True)[:3]
    top_traits = sorted(profile.get("favorite_traits", {}).items(), key=lambda x: x[1], reverse=True)[:4]
    cuisines_txt = ", ".join([c for c, _ in top_cuisines]) or "mixed cuisines"
    dishes_txt = ", ".join([d.replace("_", " ") for d, _ in top_dishes]) or "a broad dish mix"
    traits_txt = ", ".join([t for t, _ in top_traits]) or "balanced tastes"
    primary_name, secondary_trait = _derive_two_layer_archetype(profile, features)

    if mode == "medium":
        long_description = f"So far this is reading as {meta.name.lower()}. {meta.dry_observation}"
    else:
        long_description = f"{meta.description} {meta.dry_observation}"
    observations = [
        meta.dry_observation,
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
        "debug": debug_archetype_decision(profile),
        "stability": {
            "previous_archetype": previous_archetype or "",
            "selected_best_raw": best_name,
            "selected_best_score": float(best_score),
            "second_best_score": float(second_score),
            "final_archetype": meta.name,
            "evidence_mode": mode,
            "tuning": STABILITY_TUNING,
        },
    }
