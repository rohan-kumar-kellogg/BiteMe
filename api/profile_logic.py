from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from api.archetypes import FALLBACK_ARCHETYPE, choose_archetype, is_valid_archetype_name
from api.label_normalization import normalize_label, normalize_prediction_labels, normalize_profile_labels
from api.taste_profile import (
    compute_relative_rankings,
    dimension_vector,
    generate_detailed_analysis,
    init_taste_profile,
    update_taste_profile,
)
from api.storage import utc_now_iso

RECOMMENDATION_CLICK_SIGNAL_WEIGHT_DEFAULT = 0.30


def empty_profile() -> dict[str, Any]:
    base = {
        "upload_count": 0,
        "interaction_count": 0,
        "favorite_cuisines": {},
        "favorite_dishes": {},
        "favorite_traits": {},
        "last_predictions": [],
        "archetype_current": FALLBACK_ARCHETYPE,
    }
    init_taste_profile(base)
    return base


def _top_k(weight_map: dict[str, float], k: int = 5) -> list[dict]:
    items = sorted(weight_map.items(), key=lambda x: x[1], reverse=True)[:k]
    return [{"name": str(n), "score": float(s)} for n, s in items]


def _trait_tokens(cuisine: str, dish_label: str, protein_type: str) -> list[str]:
    t = []
    c = str(cuisine).lower()
    d = str(dish_label).lower().replace("_", " ")
    p = str(protein_type).lower()
    if "spicy" in d or "curry" in d:
        t.append("spice-forward")
    if any(
        x in d
        for x in [
            "cake",
            "pie",
            "dessert",
            "mousse",
            "ice cream",
            "donut",
            "gelato",
            "tiramisu",
            "brownie",
            "pastry",
            "cheesecake",
            "cookie",
        ]
    ):
        t.append("dessert-leaning")
    if p in {"fish", "seafood", "shrimp"}:
        t.append("seafood-leaning")
        t.append("protein-forward")
    if p in {"beef", "chicken", "pork", "lamb", "egg"}:
        t.append("protein-forward")
    if p in {"vegetarian", "vegan"} or any(x in d for x in ["salad", "vegetable"]):
        t.append("plant-forward")
    if any(x in d for x in ["pasta", "rice", "bread", "noodle", "ramen", "pizza", "donut", "dumpling", "pie"]):
        t.append("carb-forward")
    if any(x in d for x in ["burger", "fries", "mac", "cheese", "lasagna", "pizza"]):
        t.append("comfort-food")
    if c and c != "unknown":
        t.append(f"cuisine:{c}")
    return t


def _infer_cuisine_from_dish(dish_label: str, fallback: str = "Unknown") -> str:
    d = str(dish_label).lower().replace("_", " ")
    mapping: list[tuple[list[str], str]] = [
        (["pizza", "pasta", "tiramisu", "gelato", "risotto", "lasagna"], "italian"),
        (
            [
                "ramen",
                "sushi",
                "udon",
                "soba",
                "tempura",
                "yakitori",
                "unagi",
                "donburi",
                "gyudon",
                "katsudon",
                "oyakodon",
            ],
            "japanese",
        ),
        (["taco", "tacos", "burrito", "quesadilla", "enchilada"], "mexican"),
        (["croissant", "pastry", "macaron", "eclair"], "french"),
        (["pho", "banh mi"], "vietnamese"),
    ]
    for terms, cuisine in mapping:
        if any(term in d for term in terms):
            return cuisine
    return str(fallback or "Unknown")


def update_profile_from_prediction(profile: dict[str, Any], prediction: dict) -> dict[str, Any]:
    out, _ = normalize_profile_labels(dict(profile))
    out.setdefault("upload_count", 0)
    out.setdefault("interaction_count", 0)
    out.setdefault("favorite_cuisines", {})
    out.setdefault("favorite_dishes", {})
    out.setdefault("favorite_traits", {})
    out.setdefault("last_predictions", [])

    normalized_prediction = normalize_prediction_labels(prediction)
    predicted_label = str(normalized_prediction.get("predicted_label", "") or "").strip().lower()
    if bool(normalized_prediction.get("abstained", False)) or predicted_label in {"not_food", "unknown"}:
        # Keep upload history, but skip taste-memory updates on abstained/non-food frames.
        out["upload_count"] = int(out["upload_count"]) + 1
        out["interaction_count"] = int(out["interaction_count"]) + 1
        out["last_predictions"] = ([normalized_prediction] + list(out["last_predictions"]))[:30]
        out["taste_profile"]["analysis"] = generate_detailed_analysis(out)
        return out

    top3 = normalized_prediction.get("top3_candidates", []) or []
    weights = [1.0, 0.6, 0.3]
    pred_conf = float(np.clip(float(normalized_prediction.get("predicted_score", 0.5) or 0.5), 0.0, 1.0))
    for i, cand in enumerate(top3[:3]):
        conf = cand.get("final_score", cand.get("score", pred_conf))
        try:
            conf_val = float(np.clip(float(conf), 0.0, 1.0))
        except (TypeError, ValueError):
            conf_val = pred_conf
        # Confidence-weighted memory update: low-confidence candidates have weaker impact.
        w = float(weights[i]) * conf_val
        raw_cuisine = str(cand.get("cuisine", "Unknown"))
        dish = normalize_label(cand.get("dish_label", cand.get("dish_class", "Unknown")))
        cuisine = _infer_cuisine_from_dish(dish, fallback=raw_cuisine)
        protein = str(cand.get("protein_type", ""))
        out["favorite_cuisines"][cuisine] = float(out["favorite_cuisines"].get(cuisine, 0.0) + w)
        out["favorite_dishes"][dish] = float(out["favorite_dishes"].get(dish, 0.0) + w)
        for tok in _trait_tokens(cuisine, dish, protein):
            out["favorite_traits"][tok] = float(out["favorite_traits"].get(tok, 0.0) + w)

    out["upload_count"] = int(out["upload_count"]) + 1
    out["interaction_count"] = int(out["interaction_count"]) + 1
    out["last_predictions"] = ([normalized_prediction] + list(out["last_predictions"]))[:30]
    out = update_taste_profile(out, normalized_prediction)
    out["taste_profile"]["analysis"] = generate_detailed_analysis(out)
    return out


def infer_archetype(profile: dict[str, Any]) -> tuple[str, str, str, str, list[str]]:
    normalized_profile, _ = normalize_profile_labels(profile)
    profile.update(normalized_profile)
    prev_raw = str(profile.get("archetype_current", "")).strip()
    prev = prev_raw if is_valid_archetype_name(prev_raw, allow_system=False) else None
    prev_stability = profile.get("archetype_stability")
    out = choose_archetype(
        profile,
        previous_archetype=prev,
        previous_stability=prev_stability if isinstance(prev_stability, dict) else None,
    )
    profile["archetype_current"] = out["archetype"]
    profile["primary_archetype"] = out.get("primary_archetype", out["archetype"])
    profile["secondary_trait"] = out.get("secondary_trait", "")
    profile["behavior_features"] = out["behavior_features"]
    profile["archetype_scores"] = out["archetype_scores"]
    profile["archetype_stability"] = out["stability"]
    profile["taste_profile"]["analysis"] = generate_detailed_analysis(profile)
    return (
        str(out["archetype"]),
        str(out["archetype_description"]),
        str(out["archetype_graphic"]),
        str(out["joke"]),
        [str(x) for x in out["observations"]],
    )


def update_profile_from_recommendation_click(
    profile: dict[str, Any],
    *,
    dish_label: str,
    cuisine: str = "",
    signal_weight: float = RECOMMENDATION_CLICK_SIGNAL_WEIGHT_DEFAULT,
    record_event: bool = True,
    event_id: str | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """
    Lightweight feedback update for recommendation clicks.
    This is intentionally softer than image-upload updates.
    """
    out, _ = normalize_profile_labels(dict(profile))
    out.setdefault("favorite_cuisines", {})
    out.setdefault("favorite_dishes", {})
    out.setdefault("favorite_traits", {})
    out.setdefault("recommendation_feedback", [])
    out.setdefault("interaction_count", 0)

    dish = normalize_label(dish_label)
    cuisine_clean = str(cuisine or "").strip()
    w = float(np.clip(float(signal_weight), 0.08, 0.45))

    # Softer signal than uploads: only nudge the profile.
    out["favorite_dishes"][dish] = float(out["favorite_dishes"].get(dish, 0.0) + w)
    resolved_cuisine = _infer_cuisine_from_dish(dish, fallback=cuisine_clean or "Unknown")
    if resolved_cuisine and resolved_cuisine.lower() != "unknown":
        out["favorite_cuisines"][resolved_cuisine] = float(out["favorite_cuisines"].get(resolved_cuisine, 0.0) + (0.55 * w))

    for tok in _trait_tokens(resolved_cuisine or "Unknown", dish, ""):
        out["favorite_traits"][tok] = float(out["favorite_traits"].get(tok, 0.0) + (0.35 * w))

    # Keep fingerprint responsive to recommendation clicks (softer than uploads).
    synthetic_conf = float(np.clip(0.55 + (0.70 * w), 0.45, 0.88))
    synthetic_prediction = {
        "predicted_label": dish,
        "predicted_score": synthetic_conf,
        "top3_candidates": [
            {
                "dish_label": dish,
                "dish_class": dish,
                "cuisine": resolved_cuisine,
                "protein_type": "",
                "final_score": synthetic_conf,
            }
        ],
    }
    out["interaction_count"] = int(out["interaction_count"]) + 1
    out = update_taste_profile(out, synthetic_prediction)
    out["taste_profile"]["analysis"] = generate_detailed_analysis(out)

    if record_event:
        out["recommendation_feedback"] = (
            [
                {
                    "event_id": str(event_id or ""),
                    "event_type": "recommendation_click",
                    "dish_label": dish,
                    "cuisine": resolved_cuisine,
                    "signal_weight": w,
                    "timestamp": str(timestamp or utc_now_iso()),
                }
            ]
            + list(out["recommendation_feedback"])
        )[:200]
    return out


def _profile_vector(profile: dict[str, Any]) -> dict[str, float]:
    v: dict[str, float] = defaultdict(float)
    for k, val in profile.get("favorite_cuisines", {}).items():
        v[f"c:{k.lower()}"] += float(val)
    for k, val in profile.get("favorite_dishes", {}).items():
        nk = normalize_label(k).lower()
        v[f"d:{nk}"] += float(val)
    for k, val in profile.get("favorite_traits", {}).items():
        v[f"t:{k.lower()}"] += float(val)
    return dict(v)


def _cosine_sparse(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    keys = set(a.keys()) | set(b.keys())
    num = float(sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys))
    da = float(np.sqrt(sum(v * v for v in a.values())))
    db = float(np.sqrt(sum(v * v for v in b.values())))
    if da <= 1e-12 or db <= 1e-12:
        return 0.0
    return float(num / (da * db))


def _compatibility_explanation(target_profile: dict, other_profile: dict) -> str:
    tc = _top_k(target_profile.get("favorite_cuisines", {}), 4)
    oc = _top_k(other_profile.get("favorite_cuisines", {}), 4)
    td = _top_k(target_profile.get("favorite_dishes", {}), 5)
    od = _top_k(other_profile.get("favorite_dishes", {}), 5)
    tt = _top_k(target_profile.get("favorite_traits", {}), 5)
    ot = _top_k(other_profile.get("favorite_traits", {}), 5)

    overlap_c = sorted(list({x["name"] for x in tc} & {x["name"] for x in oc}))
    overlap_d = sorted(list({x["name"] for x in td} & {x["name"] for x in od}))
    overlap_t = sorted(list({x["name"] for x in tt} & {x["name"] for x in ot}))

    def is_placeholder(x: str) -> bool:
        norm = str(x or "").strip().lower()
        return norm in {"", "unknown", "none", "null", "n/a", "na"}

    def pretty(x: str) -> str:
        return str(x).replace("_", " ").strip()

    def norm_key(x: str) -> str:
        return " ".join(pretty(x).lower().split())

    def unique_phrases(items: list[str], limit: int) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for raw in items:
            if is_placeholder(raw):
                continue
            key = norm_key(raw)
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(pretty(raw))
            if len(out) >= max(0, int(limit)):
                break
        return out

    def join_words(words: list[str]) -> str:
        vals = [w for w in words if w]
        if not vals:
            return ""
        if len(vals) == 1:
            return vals[0]
        if len(vals) == 2:
            return f"{vals[0]} and {vals[1]}"
        return f"{', '.join(vals[:-1])}, and {vals[-1]}"

    trait_map = {
        "spice-forward": "spicy dishes",
        "dessert-leaning": "dessert",
        "comfort-food": "comfort-food meals",
        "plant-forward": "lighter, fresher plates",
        "protein-forward": "protein-heavy meals",
        "carb-forward": "noodle and carb-heavy dishes",
        "seafood-leaning": "seafood",
    }

    dish_words = [d.lower() for d in unique_phrases(overlap_d, limit=2)]
    cuisine_words = [c.title() for c in unique_phrases(overlap_c, limit=1)]
    trait_words = [
        p for p in unique_phrases([trait_map[t] for t in overlap_t if t in trait_map], limit=1)
    ]

    # Remove obvious phrase duplication across clauses (e.g., "noodle" appearing in both dish/trait fragments).
    if dish_words and trait_words:
        dish_blob = " ".join(dish_words)
        if any(tok in dish_blob for tok in ["noodle", "ramen", "pasta", "burger", "taco"]):
            trait_words = [t for t in trait_words if "noodle" not in t.lower()]

    openers = [
        "You both gravitate toward",
        "Your profiles overlap around",
        "You both lean toward",
        "There's strong overlap in",
    ]
    signature = "|".join(dish_words + cuisine_words + trait_words)
    opener = openers[sum(ord(ch) for ch in signature) % len(openers)] if signature else openers[0]

    lines: list[str] = []
    # Priority 1: dishes, then cuisine, then trait.
    if dish_words:
        sentence = f"{opener} {join_words(dish_words)}"
        if cuisine_words:
            sentence += f" and {cuisine_words[0]} cuisine"
        if trait_words:
            sentence += f", especially {trait_words[0]}"
        lines.append(f"{sentence}.")
    elif cuisine_words:
        sentence = f"{opener} {cuisine_words[0]} cuisine"
        if trait_words:
            sentence += f" and {trait_words[0]}"
        lines.append(f"{sentence}.")
    elif trait_words:
        lines.append(f"{opener} {trait_words[0]}.")

    if not lines:
        lines.append("The patterns in what you both like line up pretty naturally.")

    # Keep concise: one sentence preferred; cap at two.
    return " ".join(lines[:2])


def compute_compatible_users(target_user: dict, others: list[dict], limit: int = 5) -> list[dict]:
    tv = _profile_vector(target_user.get("profile", {}))
    tv_dim = dimension_vector(target_user.get("profile", {}))
    rows = []
    for u in others:
        uv = _profile_vector(u.get("profile", {}))
        uv_dim = dimension_vector(u.get("profile", {}))
        sparse_sim = _cosine_sparse(tv, uv)
        dim_sim = float(
            np.dot(tv_dim, uv_dim)
            / ((np.linalg.norm(tv_dim) + 1e-12) * (np.linalg.norm(uv_dim) + 1e-12))
        )
        sim = float(0.65 * sparse_sim + 0.35 * dim_sim)
        rows.append(
            {
                "compatible_username": str(u["username"]),
                "compatible_email": str(u.get("email", "") or ""),
                "compatibility_score": round(float(sim), 4),
                "archetype": str(u.get("archetype", "")),
                "why_you_match": _compatibility_explanation(target_user.get("profile", {}), u.get("profile", {})),
            }
        )
    rows = sorted(rows, key=lambda x: x["compatibility_score"], reverse=True)
    return rows[: max(1, int(limit))]


def build_relative_rankings_for_user(target_user: dict, others: list[dict]) -> list[dict]:
    return compute_relative_rankings(target_user, others)
