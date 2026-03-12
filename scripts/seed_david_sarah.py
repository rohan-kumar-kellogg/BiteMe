from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from api.profile_logic import compute_compatible_users, empty_profile, infer_archetype, update_profile_from_prediction
from api.restaurant_recommendations import get_compatible_restaurants
from api.storage import ProfileStore, utc_now_iso


@dataclass(frozen=True)
class Dish:
    label: str
    cuisine: str = "thai"
    protein: str = "chicken"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Seed David and Sarah Thai-focused demo users.")
    p.add_argument("--db_path", default="data/app_profiles.db")
    p.add_argument("--zip_code", default="60201")
    p.add_argument("--reset", action="store_true")
    return p.parse_args()


def _prediction(dish: Dish) -> dict:
    # Keep all top-3 entries Thai to make cuisine signal strong and explicit.
    row = {
        "dish_label": dish.label,
        "dish_class": dish.label,
        "cuisine": dish.cuisine,
        "protein_type": dish.protein,
        "final_score": 0.91,
        "score": 0.91,
    }
    return {
        "predicted_label": dish.label,
        "predicted_score": 0.91,
        "abstained": False,
        "top3_candidates": [dict(row), dict(row), dict(row)],
    }


def _enforce_targets(profile: dict, *, gluten_free: bool) -> dict:
    # Explicitly guarantee demo targets for screenshots.
    profile.setdefault("favorite_cuisines", {})
    profile.setdefault("favorite_traits", {})
    profile.setdefault("taste_profile", {}).setdefault("dimensions", {})
    dims = profile["taste_profile"]["dimensions"]

    profile["favorite_cuisines"]["thai"] = max(float(profile["favorite_cuisines"].get("thai", 0.0)), 8.5)
    profile["favorite_cuisines"]["japanese"] = max(float(profile["favorite_cuisines"].get("japanese", 0.0)), 1.2)
    profile["favorite_cuisines"]["american"] = max(float(profile["favorite_cuisines"].get("american", 0.0)), 0.7)

    profile["favorite_traits"]["spice-forward"] = max(float(profile["favorite_traits"].get("spice-forward", 0.0)), 2.2)
    profile["favorite_traits"]["protein-forward"] = max(float(profile["favorite_traits"].get("protein-forward", 0.0)), 1.6)
    profile["favorite_traits"]["dessert-leaning"] = max(float(profile["favorite_traits"].get("dessert-leaning", 0.0)), 1.0)
    profile["favorite_traits"]["cuisine:thai"] = max(float(profile["favorite_traits"].get("cuisine:thai", 0.0)), 2.8)

    targets = {
        "spicy_leaning": 0.67,
        "umami_leaning": 0.62,
        "dessert_affinity": 0.56,
        "variety_seeking": 0.66,
        "global_cuisine_breadth": 0.64,
    }
    for key, target in targets.items():
        cur = float(dims.get(key, {}).get("score", 0.5))
        dims.setdefault(key, {"score": 0.5, "explanation": ""})
        dims[key]["score"] = min(1.0, max(cur, target))

    profile["gluten_free"] = bool(gluten_free)
    profile["is_demo_user"] = True
    return profile


def _seed_user(store: ProfileStore, *, username: str, name: str, bio: str, dishes: list[Dish], gluten_free: bool) -> dict:
    profile = empty_profile()
    for i, dish in enumerate(dishes, start=1):
        pred = _prediction(dish)
        profile = update_profile_from_prediction(profile, pred)
        store.add_upload(
            username=username,
            image_path=f"demo://{username}/upload_{i:03d}.jpg",
            prediction=pred,
        )

    profile = _enforce_targets(profile, gluten_free=gluten_free)
    profile["demo_display_name"] = name
    profile["demo_bio"] = bio
    archetype, desc, graphic, joke, observations = infer_archetype(profile)
    store.upsert_user(
        username=username,
        email=f"{username}@demo.local",
        created_at=utc_now_iso(),
        archetype=archetype,
        archetype_description=desc,
        archetype_graphic=graphic,
        observations="\n".join(observations),
        joke=joke,
        profile=profile,
    )
    return store.get_user(username) or {}


def _cleanup_users(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        for username in ("david_demo", "sarah_demo"):
            conn.execute("DELETE FROM uploads WHERE username = ?", (username,))
            conn.execute("DELETE FROM users WHERE username = ?", (username,))
        conn.commit()
    finally:
        conn.close()


def main() -> None:
    args = parse_args()
    if args.reset:
        _cleanup_users(args.db_path)

    store = ProfileStore(args.db_path)

    david_dishes = [
        Dish("pad thai", protein="shrimp"),
        Dish("green curry", protein="chicken"),
        Dish("mango sticky rice", protein="vegetarian"),
        Dish("papaya salad", protein="vegetarian"),
        Dish("thai basil chicken gluten free", protein="chicken"),
    ]
    sarah_dishes = [
        Dish("pad kee mao", protein="chicken"),
        Dish("tom yum soup", protein="shrimp"),
        Dish("massaman curry", protein="beef"),
        Dish("mango sticky rice", protein="vegetarian"),
        Dish("thai basil stir fry", protein="chicken"),
    ]

    david = _seed_user(
        store,
        username="david_demo",
        name="David",
        bio="Junior software engineer new to Chicago. Gluten-free. Loves Thai food and casual lunch spots.",
        dishes=david_dishes,
        gluten_free=True,
    )
    sarah = _seed_user(
        store,
        username="sarah_demo",
        name="Sarah",
        bio="VP of Marketing at the same company. Big fan of Thai food and spicy dishes.",
        dishes=sarah_dishes,
        gluten_free=False,
    )

    # Re-read after both are seeded for compatibility checks.
    david = store.get_user("david_demo") or david
    sarah = store.get_user("sarah_demo") or sarah
    all_others = store.list_users(exclude_username="david_demo")
    david_matches = compute_compatible_users(david, all_others, limit=5)
    top_match = david_matches[0] if david_matches else {}
    sarah_match = next((m for m in david_matches if m.get("compatible_username") == "sarah_demo"), None)

    # Ensure requested compatibility display band if needed (frontend scales 0-1 to percentage).
    david_profile = david.get("profile", {})
    sarah_profile = sarah.get("profile", {})
    sarah_score = float((sarah_match or {}).get("compatibility_score", 0.0))

    restaurants = get_compatible_restaurants(david_profile, zip_code=args.zip_code, limit=5, context="dinner")
    thai_rest = next(
        (
            r
            for r in restaurants
            if "thai" in {str(x).lower().replace("-", "_") for x in r.get("restaurant", {}).get("cuisine_tags", [])}
        ),
        restaurants[0] if restaurants else None,
    )

    dims = david_profile.get("taste_profile", {}).get("dimensions", {})
    thai_weight = float(david_profile.get("favorite_cuisines", {}).get("thai", 0.0))
    total_cuisine = sum(float(v) for v in david_profile.get("favorite_cuisines", {}).values()) or 1.0
    thai_share = thai_weight / total_cuisine

    print("Seeded users:")
    print("- username: david_demo")
    print("  email: david_demo@demo.local")
    print("  password: (none - username + email login)")
    print("  display_name: David")
    print("  gluten_free: true")
    print("- username: sarah_demo")
    print("  email: sarah_demo@demo.local")
    print("  password: (none - username + email login)")
    print("  display_name: Sarah")
    print("")
    print("David profile checks:")
    print(f"- thai cuisine share: {thai_share:.3f}")
    print(f"- spicy_leaning: {float(dims.get('spicy_leaning', {}).get('score', 0.0)):.3f}")
    print(f"- umami_leaning: {float(dims.get('umami_leaning', {}).get('score', 0.0)):.3f}")
    print(f"- dessert_affinity: {float(dims.get('dessert_affinity', {}).get('score', 0.0)):.3f}")
    print(f"- variety_seeking: {float(dims.get('variety_seeking', {}).get('score', 0.0)):.3f}")
    print("")
    print("David matches:")
    print(f"- top match username: {top_match.get('compatible_username', '<none>')}")
    print(f"- Sarah compatibility raw: {sarah_score:.4f}")
    print(f"- Sarah compatibility display: {round(sarah_score * 100.0, 2):.2f}%")
    if sarah_match:
        print(f"- explanation: {sarah_match.get('why_you_match', '')}")
    print("")
    if thai_rest:
        rest = thai_rest.get("restaurant", {})
        print("Top Thai-leaning restaurant recommendation for David:")
        print(f"- name: {rest.get('name', '')}")
        print(f"- cuisines: {', '.join(rest.get('cuisine_tags', []))}")
        print(f"- compatibility: {thai_rest.get('compatibility_score', 0.0)}")
        print(f"- explanation: {thai_rest.get('explanation', '')}")
    else:
        print("No restaurant recommendations found.")


if __name__ == "__main__":
    main()
