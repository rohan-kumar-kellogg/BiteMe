from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from api.archetypes import choose_archetype
from api.taste_profile import (
    compute_relative_rankings,
    dimension_vector,
    generate_detailed_analysis,
    init_taste_profile,
    update_taste_profile,
)


def empty_profile() -> dict[str, Any]:
    base = {
        "upload_count": 0,
        "favorite_cuisines": {},
        "favorite_dishes": {},
        "favorite_traits": {},
        "last_predictions": [],
        "archetype_current": "Balanced Bistro Strategist",
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
    if any(x in d for x in ["cake", "pie", "dessert", "mousse", "ice cream", "donut"]):
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
    if any(x in d for x in ["burger", "fries", "mac", "cheese", "lasagna"]):
        t.append("comfort-food")
    if c and c != "unknown":
        t.append(f"cuisine:{c}")
    return t


def update_profile_from_prediction(profile: dict[str, Any], prediction: dict) -> dict[str, Any]:
    out = dict(profile)
    out.setdefault("upload_count", 0)
    out.setdefault("favorite_cuisines", {})
    out.setdefault("favorite_dishes", {})
    out.setdefault("favorite_traits", {})
    out.setdefault("last_predictions", [])

    top3 = prediction.get("top3_candidates", []) or []
    weights = [1.0, 0.6, 0.3]
    for i, cand in enumerate(top3[:3]):
        w = weights[i]
        cuisine = str(cand.get("cuisine", "Unknown"))
        dish = str(cand.get("dish_label", cand.get("dish_class", "Unknown")))
        protein = str(cand.get("protein_type", ""))
        out["favorite_cuisines"][cuisine] = float(out["favorite_cuisines"].get(cuisine, 0.0) + w)
        out["favorite_dishes"][dish] = float(out["favorite_dishes"].get(dish, 0.0) + w)
        for tok in _trait_tokens(cuisine, dish, protein):
            out["favorite_traits"][tok] = float(out["favorite_traits"].get(tok, 0.0) + w)

    out["upload_count"] = int(out["upload_count"]) + 1
    out["last_predictions"] = ([prediction] + list(out["last_predictions"]))[:30]
    out = update_taste_profile(out, prediction)
    out["taste_profile"]["analysis"] = generate_detailed_analysis(out)
    return out


def infer_archetype(profile: dict[str, Any]) -> tuple[str, str, str, str, list[str]]:
    prev = str(profile.get("archetype_current", "")).strip() or None
    out = choose_archetype(profile, previous_archetype=prev)
    profile["archetype_current"] = out["archetype"]
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


def _profile_vector(profile: dict[str, Any]) -> dict[str, float]:
    v: dict[str, float] = defaultdict(float)
    for k, val in profile.get("favorite_cuisines", {}).items():
        v[f"c:{k.lower()}"] += float(val)
    for k, val in profile.get("favorite_dishes", {}).items():
        v[f"d:{k.lower()}"] += float(val)
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
    tc = _top_k(target_profile.get("favorite_cuisines", {}), 2)
    oc = _top_k(other_profile.get("favorite_cuisines", {}), 2)
    td = _top_k(target_profile.get("favorite_dishes", {}), 2)
    od = _top_k(other_profile.get("favorite_dishes", {}), 2)
    tt = _top_k(target_profile.get("favorite_traits", {}), 3)
    ot = _top_k(other_profile.get("favorite_traits", {}), 3)
    overlap_c = sorted(list({x["name"] for x in tc} & {x["name"] for x in oc}))
    overlap_d = sorted(list({x["name"] for x in td} & {x["name"] for x in od}))
    overlap_t = sorted(list({x["name"] for x in tt} & {x["name"] for x in ot}))

    reasons = []
    if overlap_c:
        reasons.append(f"shared cuisine pull ({', '.join(overlap_c[:2])})")
    if overlap_d:
        reasons.append(f"similar dish choices ({', '.join(overlap_d[:2])})")
    if overlap_t:
        reasons.append(f"matching taste tendencies ({', '.join(overlap_t[:2])})")
    if not reasons:
        reasons.append("complementary variety patterns across cuisines and dish choices")
    return "You match because of " + "; ".join(reasons) + "."


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
                "compatibility_score": round(float(sim), 4),
                "archetype": str(u.get("archetype", "")),
                "why_you_match": _compatibility_explanation(target_user.get("profile", {}), u.get("profile", {})),
            }
        )
    rows = sorted(rows, key=lambda x: x["compatibility_score"], reverse=True)
    return rows[: max(1, int(limit))]


def build_relative_rankings_for_user(target_user: dict, others: list[dict]) -> list[dict]:
    return compute_relative_rankings(target_user, others)
