from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any


def _category_file_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "UECFOOD256" / "category.txt"


def _clean_name(name: str) -> str:
    return " ".join(str(name).strip().split())


def _to_int_if_numeric(x: Any) -> int | None:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    if s.isdigit():
        return int(s)
    try:
        f = float(s)
        if f.is_integer():
            return int(f)
    except ValueError:
        return None
    return None


@lru_cache(maxsize=1)
def load_uec_category_map() -> dict[int, str]:
    path = _category_file_path()
    out: dict[int, str] = {}
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            row = line.strip()
            if not row:
                continue
            if i == 0 and row.lower().startswith("id"):
                continue
            parts = row.split("\t", 1)
            if len(parts) != 2:
                continue
            key_raw, val_raw = parts[0].strip(), parts[1].strip()
            if not key_raw.isdigit():
                continue
            out[int(key_raw)] = _clean_name(val_raw)
    return out


def uec_category_map_report(expected_max_id: int = 256) -> dict[str, Any]:
    m = load_uec_category_map()
    expected_ids = set(range(1, int(expected_max_id) + 1))
    actual_ids = set(int(x) for x in m.keys())
    missing_ids = sorted(list(expected_ids - actual_ids))
    extra_ids = sorted(list(actual_ids - expected_ids))
    return {
        "expected_count": int(expected_max_id),
        "mapped_count": int(len(actual_ids & expected_ids)),
        "total_loaded_rows": int(len(actual_ids)),
        "missing_ids": missing_ids,
        "extra_ids": extra_ids,
        "is_complete_1_to_256": len(missing_ids) == 0,
    }


def normalize_label(raw_label: Any) -> str:
    raw = "" if raw_label is None else str(raw_label).strip()
    if not raw:
        return "unknown dish"
    as_int = _to_int_if_numeric(raw)
    if as_int is not None:
        mapped = load_uec_category_map().get(as_int)
        if mapped:
            return mapped
        return "unknown dish"
    # Food-101 and other already-human labels should pass through unchanged.
    return raw


def normalize_prediction_labels(prediction: dict[str, Any]) -> dict[str, Any]:
    out = dict(prediction or {})

    def _norm_candidate(c: dict[str, Any]) -> dict[str, Any]:
        cc = dict(c or {})
        raw = cc.get("dish_label", cc.get("dish_class", ""))
        norm = normalize_label(raw)
        cc["dish_label"] = norm
        if "dish_class" in cc:
            cc["dish_class"] = norm
        return cc

    out["predicted_label"] = normalize_label(out.get("predicted_label", ""))
    if isinstance(out.get("top3_candidates"), list):
        out["top3_candidates"] = [_norm_candidate(c) for c in out["top3_candidates"]]
    if isinstance(out.get("raw_topn"), list):
        out["raw_topn"] = [_norm_candidate(c) for c in out["raw_topn"]]

    if not out.get("predicted_label") or out.get("predicted_label") == "unknown dish":
        top3 = out.get("top3_candidates") or []
        if top3:
            out["predicted_label"] = normalize_label(top3[0].get("dish_label", ""))
    return out


def normalize_profile_labels(profile: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    out = dict(profile or {})
    changed = False

    dishes_in = out.get("favorite_dishes", {}) or {}
    dishes_out: dict[str, float] = {}
    for k, v in dishes_in.items():
        nk = normalize_label(k)
        dishes_out[nk] = float(dishes_out.get(nk, 0.0) + float(v))
        if nk != str(k):
            changed = True
    out["favorite_dishes"] = dishes_out

    if isinstance(out.get("last_predictions"), list):
        normalized_preds = []
        for p in out["last_predictions"]:
            npred = normalize_prediction_labels(p if isinstance(p, dict) else {})
            if npred != p:
                changed = True
            normalized_preds.append(npred)
        out["last_predictions"] = normalized_preds

    return out, changed
