from __future__ import annotations

from dataclasses import dataclass
import math
import json
import logging
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus


@dataclass(frozen=True)
class RecommendationWeights:
    cuisine_match: float = 0.33
    dish_match: float = 0.24
    trait_match: float = 0.17
    location_match: float = 0.18
    reservation_match: float = 0.05
    popularity_match: float = 0.03


SCORING_WEIGHTS = RecommendationWeights()

CONTEXT_MAX_SWING = 0.12
LOGGER = logging.getLogger(__name__)


CUISINE_ALIASES: dict[str, str] = {
    "middle eastern": "middle_eastern",
    "middle_eastern": "middle_eastern",
    "mediterranean": "mediterranean",
    "american": "american",
    "japanese": "japanese",
    "italian": "italian",
    "thai": "thai",
    "indian": "indian",
    "mexican": "mexican",
    "french": "french",
    "chinese": "chinese",
    "korean": "korean",
}


TRAIT_TO_MENU_TAGS: dict[str, list[str]] = {
    "dessert_affinity": ["dessert", "bakery", "pastry", "ice_cream", "cafe"],
    "sweet_leaning": ["dessert", "brunch", "pastry", "bakery"],
    "spicy_leaning": ["spicy", "chili_forward", "szechuan", "thai", "indian", "mexican", "korean"],
    "adventurousness": ["chef_driven", "niche_regional", "omakase", "tasting_menu", "global"],
    "comfort_food_tendency": ["cozy", "hearty", "casual", "classic"],
    "freshness_preference": ["healthy", "vegetarian_friendly", "salad", "clean_eating", "light"],
    "protein_forward": ["protein_forward", "grill", "steakhouse", "bbq", "chicken"],
}


DISH_KEYWORD_TO_MENU_TAGS: dict[str, list[str]] = {
    "tiramisu": ["dessert", "italian", "pastry"],
    "gelato": ["dessert", "ice_cream", "italian"],
    "cannoli": ["dessert", "pastry", "italian"],
    "pastry": ["pastry", "bakery", "dessert", "cafe"],
    "brownie": ["dessert", "bakery"],
    "cheesecake": ["dessert", "bakery", "pastry"],
    "cake": ["dessert", "bakery", "pastry"],
    "ice cream": ["dessert", "ice_cream", "cafe"],
    "pizza": ["pizza", "italian", "comfort_food", "casual"],
    "margherita": ["pizza", "italian"],
    "pepperoni": ["pizza", "italian", "comfort_food"],
    "ramen": ["ramen", "noodles", "japanese"],
    "udon": ["noodles", "japanese"],
    "pho": ["noodles", "vietnamese", "broth"],
    "sushi": ["sushi", "japanese", "omakase"],
    "taco": ["tacos", "mexican", "casual", "spicy"],
    "tacos": ["tacos", "mexican", "casual", "spicy"],
    "pastry": ["pastry", "dessert", "bakery", "cafe"],
    "mapo tofu": ["spicy", "szechuan", "chinese"],
    "buffalo wings": ["spicy", "casual", "american", "hearty"],
    "fried chicken": ["american", "hearty", "casual", "protein_forward"],
    "ceviche": ["seafood", "light", "mexican", "fresh"],
    "salad": ["salad", "healthy", "clean_eating", "vegetarian_friendly"],
    "bowl": ["healthy", "clean_eating", "protein_forward"],
    "pasta": ["italian", "classic", "hearty"],
}

ZIP_CENTROIDS: dict[str, tuple[float, float]] = {
    "60201": (42.0541, -87.6877),
    "60202": (42.0311, -87.6895),
    "60611": (41.9001, -87.6217),
    "60614": (41.9227, -87.6533),
    "60622": (41.9021, -87.6763),
}

NEARBY_ZIP_EXPANSION: dict[str, list[str]] = {
    "60201": ["60201", "60202", "60626", "60645", "60659", "60640", "60625", "60613", "60618", "60647", "60622", "60614", "60607"],
    "60202": ["60202", "60201", "60626", "60645", "60659", "60640", "60625", "60613", "60618", "60647", "60622", "60614", "60607"],
}

CONTEXT_TO_TAGS: dict[str, list[str]] = {
    "dinner": ["classic", "cozy", "hearty", "sit_down"],
    "dessert": ["dessert", "bakery", "pastry", "ice_cream", "cafe"],
    "drinks": ["bar", "cocktails", "late_night", "casual"],
    "casual_bite": ["quick_bite", "casual", "low_friction", "cafe"],
    "date_night": ["cozy", "premium", "chef_driven", "tasting_menu", "sit_down"],
    "brunch": ["brunch", "cafe", "pastry", "light"],
}


class RestaurantProvider:
    source_name = "unknown"

    def get_restaurants_by_zip(self, zip_code: str, limit: int = 200) -> list[dict[str, Any]]:
        raise NotImplementedError


class LocalJsonRestaurantProvider(RestaurantProvider):
    source_name = "local_json"

    def __init__(self, dataset_path: str | Path | None = None):
        self.dataset_path = Path(dataset_path) if dataset_path else (Path(__file__).resolve().parents[1] / "data" / "restaurants_real.json")
        self._all_rows: list[dict[str, Any]] = []
        self._load_dataset()

    def _load_dataset(self) -> None:
        LOGGER.info("Restaurant dataset path resolved to %s", self.dataset_path)
        if not self.dataset_path.exists():
            LOGGER.warning("Restaurant dataset not found at %s; provider initialized empty.", self.dataset_path)
            self._all_rows = []
            return
        try:
            payload = json.loads(self.dataset_path.read_text(encoding="utf-8"))
        except Exception as exc:
            LOGGER.warning("Failed reading restaurant dataset at %s: %s", self.dataset_path, exc)
            self._all_rows = []
            return
        if not isinstance(payload, list):
            LOGGER.warning("Restaurant dataset at %s is not a list; provider initialized empty.", self.dataset_path)
            self._all_rows = []
            return
        cleaned: list[dict[str, Any]] = []
        for raw in payload:
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("name", "")).strip()
            rest_zip = _clean_zip(str(raw.get("zip_code", "")))
            if not name or len(rest_zip) != 5:
                continue
            row = dict(raw)
            row.setdefault("id", f"{self.source_name}:{name}:{rest_zip}")
            row.setdefault("source", self.source_name)
            row.setdefault("address", str(raw.get("address", "")).strip())
            row.setdefault("cuisine_tags", [])
            row.setdefault("menu_tags", [])
            row.setdefault("trait_tags", [])
            row.setdefault("venue_type", "")
            row.setdefault("service_type", "")
            row.setdefault("rating", 0.0)
            row.setdefault("review_count", 0)
            row.setdefault("reservation_provider", "")
            row.setdefault("reservation_url", "")
            row.setdefault("website_url", "")
            row.setdefault("phone", "")
            for key in ("cuisine_tags", "menu_tags", "trait_tags"):
                if not isinstance(row.get(key), list):
                    row[key] = []
            row["reservation_provider"] = _infer_reservation_provider(row)
            if row["reservation_provider"] == "resy" and not str(row.get("reservation_url", "")).strip():
                row["reservation_url"] = "https://resy.com/"
            if row["reservation_provider"] == "opentable" and not str(row.get("reservation_url", "")).strip():
                row["reservation_url"] = "https://www.opentable.com/"
            cleaned.append(row)
        self._all_rows = cleaned
        LOGGER.info("Loaded %d restaurants from %s", len(self._all_rows), self.dataset_path.name)

    def get_restaurants_by_zip(self, zip_code: str, limit: int = 200) -> list[dict[str, Any]]:
        if not self._all_rows:
            return []
        z = _clean_zip(zip_code)
        candidate_zips = set(NEARBY_ZIP_EXPANSION.get(z, [z] if z else []))
        rows: list[dict[str, Any]] = []
        for row in self._all_rows:
            rest_zip = _clean_zip(str(row.get("zip_code", "")))
            if z and rest_zip not in candidate_zips:
                continue
            rows.append(dict(row))
            if len(rows) >= max(1, int(limit)):
                break
        LOGGER.info("Candidate restaurants near %s: %d", z or "ALL", len(rows))
        return rows


class EmptyRestaurantProvider(RestaurantProvider):
    source_name = "none"

    def get_restaurants_by_zip(self, zip_code: str, limit: int = 200) -> list[dict[str, Any]]:
        return []


_PROVIDER: RestaurantProvider | None = None


def _default_provider() -> RestaurantProvider:
    global _PROVIDER
    if _PROVIDER is None:
        _PROVIDER = LocalJsonRestaurantProvider()
    return _PROVIDER


def current_restaurant_source() -> str:
    p = _default_provider()
    if isinstance(p, LocalJsonRestaurantProvider) and not p.dataset_path.exists():
        return "local_json_missing"
    return p.source_name


def _norm_token(x: str) -> str:
    return str(x).strip().lower().replace("-", "_").replace(" ", "_")


def _clean_zip(zip_code: str) -> str:
    digits = "".join(ch for ch in str(zip_code or "") if ch.isdigit())
    return digits[:5]


def _zip_to_latlon(zip_code: str) -> tuple[float, float] | None:
    z = _clean_zip(zip_code)
    return ZIP_CENTROIDS.get(z)


def _haversine_miles(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    r = 3958.8  # Earth radius in miles
    p1 = math.radians(a_lat)
    p2 = math.radians(b_lat)
    dlat = math.radians(b_lat - a_lat)
    dlon = math.radians(b_lon - a_lon)
    aa = (math.sin(dlat / 2) ** 2) + math.cos(p1) * math.cos(p2) * (math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(aa), math.sqrt(max(1e-12, 1 - aa)))
    return float(r * c)


def _safe_weight_map(raw: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    total = 0.0
    for k, v in raw.items():
        try:
            fv = max(0.0, float(v))
        except (TypeError, ValueError):
            fv = 0.0
        if fv <= 0.0:
            continue
        out[_norm_token(str(k))] = fv
        total += fv
    if total <= 1e-9:
        return {}
    return {k: v / total for k, v in out.items()}


def _alias_cuisine(cuisine: str) -> str:
    c = _norm_token(cuisine)
    return CUISINE_ALIASES.get(c, c)


def _user_cuisine_weights(user_profile: dict[str, Any]) -> dict[str, float]:
    raw = _safe_weight_map(user_profile.get("favorite_cuisines", {}))
    out: dict[str, float] = {}
    for k, v in raw.items():
        out[_alias_cuisine(k)] = out.get(_alias_cuisine(k), 0.0) + v
    return out


def _user_trait_tag_weights(user_profile: dict[str, Any]) -> tuple[dict[str, float], float]:
    dims = user_profile.get("taste_profile", {}).get("dimensions", {}) or {}
    behavior = user_profile.get("behavior_features", {}) or {}
    trait_strength: dict[str, float] = {
        "dessert_affinity": float(dims.get("dessert_affinity", {}).get("score", 0.5)),
        "sweet_leaning": float(dims.get("sweet_leaning", {}).get("score", 0.5)),
        "spicy_leaning": float(dims.get("spicy_leaning", {}).get("score", 0.5)),
        "freshness_preference": float(dims.get("freshness_preference", {}).get("score", 0.5)),
        "comfort_food_tendency": float(dims.get("comfort_food_tendency", {}).get("score", 0.5)),
        "protein_forward": float(dims.get("protein_forward", {}).get("score", 0.5)),
        "adventurousness": float(behavior.get("adventurousness", 0.5)),
    }
    # Keep only meaningful positive preferences above neutral baseline.
    trait_strength = {k: max(0.0, v - 0.5) for k, v in trait_strength.items() if v > 0.5}
    if not trait_strength:
        return {}, 0.0
    total = sum(trait_strength.values()) + 1e-12
    trait_strength = {k: v / total for k, v in trait_strength.items()}
    out: dict[str, float] = {}
    for trait_key, t_weight in trait_strength.items():
        for tag in TRAIT_TO_MENU_TAGS.get(trait_key, []):
            k = _norm_token(tag)
            out[k] = out.get(k, 0.0) + t_weight
    prefs = _safe_weight_map(out)
    upload_count = max(0, int(user_profile.get("upload_count", 0) or 0))
    raw_traits = user_profile.get("favorite_traits", {}) or {}
    trait_mass = sum(max(0.0, float(v)) for v in raw_traits.values() if isinstance(v, (int, float)))
    evidence = max((upload_count / 5.0), (trait_mass / 6.0))
    reliability = max(0.45, min(1.0, evidence))
    if trait_mass < 0.8 and upload_count <= 2:
        reliability *= 0.75
    return prefs, float(reliability)


def _user_dish_tag_weights(user_profile: dict[str, Any]) -> tuple[dict[str, float], float]:
    """
    Returns normalized dish-tag preferences plus a reliability factor.
    Reliability is lower for isolated one-off dish signals and higher for repeated behavior.
    """
    raw_dishes: dict[str, float] = {}
    for k, v in (user_profile.get("favorite_dishes", {}) or {}).items():
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if fv > 0:
            raw_dishes[_norm_token(k)] = fv
    if not raw_dishes:
        return {}, 0.0

    keyword_strength: dict[str, float] = {}
    tag_strength: dict[str, float] = {}
    for dish, d_weight in raw_dishes.items():
        for keyword, tags in DISH_KEYWORD_TO_MENU_TAGS.items():
            kword = _norm_token(keyword)
            if kword in dish:
                keyword_strength[kword] = keyword_strength.get(kword, 0.0) + d_weight
                for tag in tags:
                    tk = _norm_token(tag)
                    # log1p dampens one-offs and rewards repeated stronger behavior.
                    tag_strength[tk] = tag_strength.get(tk, 0.0) + math.log1p(d_weight)

    tag_prefs = _safe_weight_map(tag_strength)
    if not tag_prefs:
        return {}, 0.0

    total_mass = float(sum(raw_dishes.values()))
    strongest_mass = float(max(raw_dishes.values()))
    repeat_like_count = sum(1 for v in raw_dishes.values() if v >= 1.25)
    mass_factor = min(1.0, total_mass / 6.0)
    repeat_factor = min(1.0, repeat_like_count / 3.0)
    concentration = strongest_mass / max(1e-9, total_mass)
    concentration_penalty = 1.0 - max(0.0, concentration - 0.75) * 0.7
    reliability = min(1.0, (0.45 * mass_factor) + (0.35 * repeat_factor) + (0.20 * concentration_penalty))
    if repeat_like_count == 0 and total_mass < 1.2:
        reliability *= 0.7
    upload_count = max(0, int(user_profile.get("upload_count", 0) or 0))
    if upload_count <= 3 and strongest_mass >= 0.8:
        reliability = max(reliability, 0.35)
    reliability = max(0.08, reliability)
    return tag_prefs, float(reliability)


def _weighted_overlap_score(preferences: dict[str, float], candidates: set[str]) -> float:
    if not preferences:
        return 0.0
    return float(sum(w for k, w in preferences.items() if k in candidates))


def _matching_keys(preferences: dict[str, float], candidates: set[str], *, limit: int = 5) -> list[str]:
    keys = [k for k, _ in sorted(preferences.items(), key=lambda x: x[1], reverse=True) if k in candidates]
    return keys[: max(1, int(limit))]


def _signature_dish_overlap_score(user_profile: dict[str, Any], candidates: set[str]) -> float:
    """
    Direct behavior signal:
    if uploaded dish names (pizza/ramen/tacos/tiramisu/etc.) map to restaurant tags, reward strongly.
    """
    raw_dishes: dict[str, float] = {}
    for k, v in (user_profile.get("favorite_dishes", {}) or {}).items():
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if fv > 0.0:
            raw_dishes[_norm_token(k)] = fv
    if not raw_dishes:
        return 0.0

    total = 0.0
    matched = 0.0
    for dish, weight in raw_dishes.items():
        dish_mass = math.log1p(weight)
        total += dish_mass
        hit = False
        for keyword, tags in DISH_KEYWORD_TO_MENU_TAGS.items():
            kword = _norm_token(keyword)
            if kword in dish:
                tag_set = {_norm_token(t) for t in tags}
                if tag_set & candidates:
                    hit = True
                    break
        if hit:
            matched += dish_mass
    if total <= 1e-9:
        return 0.0
    return float(max(0.0, min(1.0, matched / total)))


def _calibrated_percent(raw_score: float) -> float:
    """
    Convert raw compatibility into a more intuitive displayed percent without changing ranking order.
    Keeps weak fits low and lets obvious local+taste fits reach 70-90.
    """
    raw = float(max(0.0, min(1.0, raw_score)))
    x = 1.0 / (1.0 + math.exp(-6.8 * (raw - 0.37)))
    return float(max(1.0, min(99.0, 100.0 * x)))


def _location_match_score(user_zip: str, rest: dict[str, Any]) -> float:
    user_coords = _zip_to_latlon(user_zip)
    rest_coords = None
    if rest.get("latitude") is not None and rest.get("longitude") is not None:
        try:
            rest_coords = (float(rest["latitude"]), float(rest["longitude"]))
        except (TypeError, ValueError):
            rest_coords = None
    if rest_coords is None:
        rest_coords = _zip_to_latlon(str(rest.get("zip_code", "")))
    if user_coords is None or rest_coords is None:
        return 0.5
    d = _haversine_miles(user_coords[0], user_coords[1], rest_coords[0], rest_coords[1])
    # Practical bucket scoring.
    if d <= 1.0:
        return 1.0
    if d <= 3.0:
        return 0.85
    if d <= 7.0:
        return 0.70
    if d <= 15.0:
        return 0.50
    if d <= 25.0:
        return 0.25
    return 0.10


def _reservation_actionability_score(rest: dict[str, Any]) -> float:
    venue = _norm_token(rest.get("venue_type", ""))
    provider = _norm_token(rest.get("reservation_provider", ""))
    has_web = bool(rest.get("website_url"))
    has_phone = bool(rest.get("phone"))
    if provider in {"opentable", "resy"}:
        return 1.0
    if venue in {"dessert", "cafe", "quick_bite"} and (has_web or has_phone):
        return 0.75
    if has_web or has_phone:
        return 0.60
    return 0.25


def _infer_reservation_provider(rest: dict[str, Any]) -> str:
    raw = _norm_token(rest.get("reservation_provider", ""))
    if raw in {"resy", "opentable", "none"}:
        return raw
    tags = {_norm_token(t) for t in (list(rest.get("trait_tags", [])) + list(rest.get("menu_tags", [])))}
    service = _norm_token(rest.get("service_type", ""))
    venue = _norm_token(rest.get("venue_type", ""))
    try:
        rating = float(rest.get("rating", 0.0) or 0.0)
    except (TypeError, ValueError):
        rating = 0.0
    high_end = bool(tags & {"fine_dining", "date_night", "trendy", "tasting_menu"}) or rating >= 4.7
    if high_end and service in {"sit_down", "full_service"}:
        return "resy"
    if venue in {"sit_down", "full_service"} and rating >= 4.4:
        return "opentable"
    return "none"


def _popularity_score(rest: dict[str, Any]) -> float:
    try:
        rating = max(0.0, min(5.0, float(rest.get("rating", 0.0))))
    except (TypeError, ValueError):
        rating = 0.0
    try:
        reviews = max(0.0, float(rest.get("review_count", 0)))
    except (TypeError, ValueError):
        reviews = 0.0
    rating_norm = rating / 5.0
    reviews_norm = min(1.0, reviews / 1200.0)
    return float(0.75 * rating_norm + 0.25 * reviews_norm)


def _booking_action(rest: dict[str, Any]) -> dict[str, str]:
    provider = _norm_token(rest.get("reservation_provider", ""))
    reservation_url = str(rest.get("reservation_url", "") or "")
    website_url = str(rest.get("website_url", "") or "")
    phone = str(rest.get("phone", "") or "")
    if provider == "opentable":
        return {"type": "opentable", "label": "Book on OpenTable", "url": reservation_url or "https://www.opentable.com/"}
    if provider == "resy":
        return {"type": "resy", "label": "Book on Resy", "url": reservation_url or "https://resy.com/"}
    if website_url:
        return {"type": "website", "label": "Visit Website", "url": website_url}
    if phone:
        return {"type": "call", "label": "Call Restaurant", "url": f"tel:{phone}"}
    name = str(rest.get("name", "") or "").strip()
    address = str(rest.get("address", "") or "").strip()
    zip_code = str(rest.get("zip_code", "") or "").strip()
    query = " ".join(part for part in [name, address, zip_code] if part).strip()
    if query:
        return {
            "type": "website",
            "label": "Visit Website",
            "url": f"https://www.google.com/search?q={quote_plus(query + ' restaurant')}",
        }
    return {"type": "none", "label": "No action available", "url": ""}


def _context_match_score(context: str, rest: dict[str, Any]) -> float:
    ctx = _norm_token(context)
    if not ctx or ctx not in CONTEXT_TO_TAGS:
        return 0.5
    target_tags = {_norm_token(t) for t in CONTEXT_TO_TAGS.get(ctx, [])}
    rest_tags = {
        _norm_token(t) for t in (
            list(rest.get("menu_tags", []))
            + list(rest.get("trait_tags", []))
            + list(rest.get("cuisine_tags", []))
        )
    }
    rest_tags.add(_norm_token(rest.get("venue_type", "")))
    rest_tags.add(_norm_token(rest.get("service_type", "")))
    overlap = len(target_tags & rest_tags)
    score = 0.35 + 0.2 * float(overlap)
    if ctx == "date_night" and _norm_token(rest.get("reservation_provider", "")) in {"opentable", "resy"}:
        score += 0.15
    if ctx in {"dessert", "brunch"} and _norm_token(rest.get("venue_type", "")) in {"dessert", "cafe"}:
        score += 0.12
    if ctx == "casual_bite" and _norm_token(rest.get("venue_type", "")) in {"quick_bite", "cafe"}:
        score += 0.12
    return float(max(0.0, min(1.0, score)))


def _context_phrase(context: str) -> str:
    ctx = _norm_token(context)
    return {
        "dinner": "it suits a dinner plan",
        "dessert": "it fits a dessert-focused outing",
        "drinks": "it works for a drinks-first plan",
        "casual_bite": "it fits a quick casual bite",
        "date_night": "it supports a date-night setting",
        "brunch": "it aligns with a brunch vibe",
    }.get(ctx, "")


def _reason_text(top_reasons: list[str]) -> str:
    if not top_reasons:
        return "Ranked for overall fit across taste and practicality."
    if len(top_reasons) == 1:
        return f"Good fit because {top_reasons[0]}."
    if len(top_reasons) == 2:
        return f"Strong match because {top_reasons[0]} and {top_reasons[1]}."
    return f"Ranks highly because {top_reasons[0]}, {top_reasons[1]}, and {top_reasons[2]}."


def _dimension_reasons(
    *,
    cuisine_score: float,
    dish_score: float,
    trait_score: float,
    location_score: float,
    reservation_score: float,
    user_zip: str,
    rest_zip: str,
    context: str,
    context_score: float,
) -> list[str]:
    reasons: list[tuple[str, float]] = []
    if cuisine_score > 0.05:
        reasons.append(("it aligns with your strongest cuisines", cuisine_score))
    if dish_score > 0.05:
        reasons.append(("its menu reflects dishes you repeatedly gravitate to", dish_score))
    if trait_score > 0.05:
        reasons.append(("it matches your taste traits and menu style", trait_score))
    if location_score > 0.05:
        if _clean_zip(user_zip) == _clean_zip(rest_zip):
            reasons.append(("it is in your zip code", location_score + 0.03))
        else:
            reasons.append(("it is practically nearby", location_score))
    if reservation_score > 0.7:
        reasons.append(("it has a clear booking or action path", reservation_score))
    if context and context_score > 0.62:
        phrase = _context_phrase(context)
        if phrase:
            reasons.append((phrase, context_score))
    reasons = sorted(reasons, key=lambda x: x[1], reverse=True)
    return [r[0] for r in reasons[:3]]


def _score_restaurant(user_profile: dict[str, Any], zip_code: str, rest: dict[str, Any], *, context: str = "") -> dict[str, Any]:
    cuisine_pref = _user_cuisine_weights(user_profile)
    trait_pref, trait_reliability = _user_trait_tag_weights(user_profile)
    dish_pref, dish_reliability = _user_dish_tag_weights(user_profile)

    rest_cuisine_tags = {_alias_cuisine(t) for t in rest.get("cuisine_tags", [])}
    rest_menu_tags = {_norm_token(t) for t in rest.get("menu_tags", [])}
    rest_trait_tags = {_norm_token(t) for t in rest.get("trait_tags", [])}
    combined_tags = set(rest_menu_tags) | set(rest_trait_tags) | set(rest_cuisine_tags)
    matched_cuisines = _matching_keys(cuisine_pref, rest_cuisine_tags, limit=4)
    matched_dish_tags = _matching_keys(dish_pref, combined_tags, limit=6)
    matched_trait_tags = _matching_keys(trait_pref, combined_tags, limit=6)

    cuisine_score = _weighted_overlap_score(cuisine_pref, rest_cuisine_tags)
    dish_pref_score = _weighted_overlap_score(dish_pref, combined_tags)
    signature_dish_score = _signature_dish_overlap_score(user_profile, combined_tags)
    dish_blended = max(dish_pref_score, (0.62 * dish_pref_score) + (0.58 * signature_dish_score))
    effective_dish_reliability = max(dish_reliability, 0.30 + (0.45 * signature_dish_score))
    dish_score = min(1.0, dish_blended * effective_dish_reliability)
    trait_score = _weighted_overlap_score(trait_pref, combined_tags) * trait_reliability
    location_score = _location_match_score(zip_code, rest)
    reservation_score = _reservation_actionability_score(rest)
    popularity_score = _popularity_score(rest)
    context_score = _context_match_score(context, rest)

    weighted = {
        "cuisine_match": SCORING_WEIGHTS.cuisine_match * cuisine_score,
        "dish_match": SCORING_WEIGHTS.dish_match * dish_score,
        "trait_match": SCORING_WEIGHTS.trait_match * trait_score,
        "location_match": SCORING_WEIGHTS.location_match * location_score,
        "reservation_match": SCORING_WEIGHTS.reservation_match * reservation_score,
        "popularity_match": SCORING_WEIGHTS.popularity_match * popularity_score,
    }
    base_total = sum(weighted.values())
    # Context should nudge ranking, not replace compatibility.
    context_factor = 1.0 + (CONTEXT_MAX_SWING * (context_score - 0.5))
    total = base_total * context_factor

    # Prevent one-off dessert signals from over-ranking dessert-only venues.
    venue_type = _norm_token(rest.get("venue_type", ""))
    ctx = _norm_token(context)
    dims = user_profile.get("taste_profile", {}).get("dimensions", {}) or {}
    dessert_strength = max(
        float(dims.get("dessert_affinity", {}).get("score", 0.5)),
        float(dims.get("sweet_leaning", {}).get("score", 0.5)),
    )
    if (
        venue_type in {"dessert", "cafe"}
        and ctx not in {"dessert", "brunch"}
        and dessert_strength < 0.78
        and dish_reliability < 0.35
    ):
        total *= 0.86
    if venue_type in {"dessert", "cafe"} and dessert_strength >= 0.88 and dish_reliability >= 0.40:
        total *= 1.08
    reasons = _dimension_reasons(
        cuisine_score=cuisine_score,
        dish_score=dish_score,
        trait_score=trait_score,
        location_score=location_score,
        reservation_score=reservation_score,
        user_zip=zip_code,
        rest_zip=str(rest.get("zip_code", "")),
        context=context,
        context_score=context_score,
    )
    booking_action = _booking_action(rest)
    displayed_percent = _calibrated_percent(total)
    # For obvious practical fits (same ZIP + strong cuisine/dish evidence), nudge display upward.
    if location_score >= 0.95 and cuisine_score >= 0.50 and dish_score >= 0.32:
        displayed_percent = min(95.0, displayed_percent + 8.0)
    elif location_score >= 0.95 and cuisine_score >= 0.45 and dish_score >= 0.24:
        displayed_percent = min(92.0, displayed_percent + 5.0)
    return {
        "restaurant": {
            "id": rest["id"],
            "name": rest["name"],
            "address": rest.get("address", ""),
            "zip_code": rest["zip_code"],
            "latitude": rest.get("latitude"),
            "longitude": rest.get("longitude"),
            "cuisine_tags": rest.get("cuisine_tags", []),
            "menu_tags": rest.get("menu_tags", []),
            "trait_tags": rest.get("trait_tags", []),
            "rating": rest.get("rating", 0.0),
            "review_count": rest.get("review_count", 0),
            "venue_type": rest.get("venue_type", ""),
            "service_type": rest.get("service_type", ""),
            "source": rest.get("source", "unknown"),
        },
        "compatibility_score": round(displayed_percent, 2),
        "score_breakdown": {
            "cuisine_match": {"score": round(cuisine_score, 4), "weight": SCORING_WEIGHTS.cuisine_match, "weighted": round(weighted["cuisine_match"], 4)},
            "trait_match": {"score": round(trait_score, 4), "weight": SCORING_WEIGHTS.trait_match, "weighted": round(weighted["trait_match"], 4)},
            "dish_match": {"score": round(dish_score, 4), "weight": SCORING_WEIGHTS.dish_match, "weighted": round(weighted["dish_match"], 4)},
            "dish_signature_match": {"score": round(signature_dish_score, 4), "weight": 0.0, "weighted": 0.0},
            "location_match": {"score": round(location_score, 4), "weight": SCORING_WEIGHTS.location_match, "weighted": round(weighted["location_match"], 4)},
            "reservation_match": {"score": round(reservation_score, 4), "weight": SCORING_WEIGHTS.reservation_match, "weighted": round(weighted["reservation_match"], 4)},
            "popularity_match": {"score": round(popularity_score, 4), "weight": SCORING_WEIGHTS.popularity_match, "weighted": round(weighted["popularity_match"], 4)},
            "raw_score": round(total, 4),
            "display_percent": round(displayed_percent, 2),
        },
        "explanation": _reason_text(reasons),
        "booking_action": booking_action,
        "action": {
            # Backward-compatible action envelope; frontend can migrate to booking_action.
            "primary": {
                "type": booking_action.get("type", "none"),
                "label": booking_action.get("label", "No action available"),
                "target": booking_action.get("url", ""),
            },
            "reservation_provider": rest.get("reservation_provider", ""),
            "reservation_url": rest.get("reservation_url", ""),
            "website_url": rest.get("website_url", ""),
            "phone": rest.get("phone", ""),
        },
        "debug": {
            "matched_cuisines": matched_cuisines,
            "matched_dishes_menu_tags": matched_dish_tags,
            "matched_traits": matched_trait_tags,
            "location": {
                "user_zip": _clean_zip(zip_code),
                "restaurant_zip": _clean_zip(str(rest.get("zip_code", ""))),
                "score": round(location_score, 4),
                "weighted": round(weighted["location_match"], 4),
            },
            "raw_score": round(total, 4),
            "final_displayed_percent": round(displayed_percent, 2),
        },
    }


def get_compatible_restaurants(
    user_profile: dict[str, Any],
    zip_code: str,
    limit: int = 20,
    context: str = "",
    provider: RestaurantProvider | None = None,
) -> list[dict[str, Any]]:
    src = provider or _default_provider()
    restaurants = src.get_restaurants_by_zip(zip_code=zip_code, limit=max(200, int(limit) * 5))
    if not restaurants:
        LOGGER.info("Candidate restaurants before ranking: 0")
        LOGGER.info("Returning top %d ranked restaurants", 0)
        return []
    LOGGER.info("Candidate restaurants before ranking: %d", len(restaurants))
    rows = [_score_restaurant(user_profile, zip_code, rest, context=context) for rest in restaurants]
    rows = sorted(rows, key=lambda x: x["compatibility_score"], reverse=True)
    top_rows = rows[: max(1, int(limit))]
    LOGGER.info("Returning top %d ranked restaurants", len(top_rows))
    return top_rows


def debug_restaurant_recommendation(
    user_profile: dict[str, Any],
    zip_code: str,
    *,
    restaurant_query: str,
    context: str = "",
    provider: RestaurantProvider | None = None,
) -> dict[str, Any] | None:
    src = provider or _default_provider()
    query = str(restaurant_query or "").strip().lower()
    if not query:
        return None
    rows = src.get_restaurants_by_zip(zip_code=zip_code, limit=500)
    for rest in rows:
        rid = str(rest.get("id", "")).lower()
        name = str(rest.get("name", "")).lower()
        if query == rid or query in rid or query in name:
            scored = _score_restaurant(user_profile, zip_code, rest, context=context)
            return {
                "restaurant_id": rest.get("id", ""),
                "restaurant_name": rest.get("name", ""),
                "debug": scored.get("debug", {}),
                "score_breakdown": scored.get("score_breakdown", {}),
                "explanation": scored.get("explanation", ""),
                "compatibility_score": scored.get("compatibility_score", 0.0),
            }
    return None

