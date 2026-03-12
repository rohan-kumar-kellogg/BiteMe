from __future__ import annotations

import argparse
import random
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from api.archetypes import ARCHETYPE_CONFIG, FALLBACK_ARCHETYPE
from api.profile_logic import empty_profile, infer_archetype, update_profile_from_prediction
from api.storage import ProfileStore, utc_now_iso


@dataclass(frozen=True)
class DishSpec:
    label: str
    cuisine: str
    protein: str


@dataclass(frozen=True)
class DemoUserPlan:
    username: str
    display_name: str
    intended_archetype: str
    uploads: list[DishSpec]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Seed realistic BiteMe demo users aligned to canonical archetypes.")
    p.add_argument("--db_path", default="data/app_profiles.db")
    p.add_argument("--reset_existing_demo", action="store_true", help="Remove existing demo users before seeding.")
    return p.parse_args()


def _prediction(top3: list[DishSpec]) -> dict:
    rows = []
    for i, dish in enumerate(top3[:3]):
        score = max(0.55, 0.90 - (0.08 * i))
        rows.append(
            {
                "dish_label": dish.label,
                "dish_class": dish.label,
                "cuisine": dish.cuisine,
                "protein_type": dish.protein,
                "final_score": float(score),
                "score": float(score),
            }
        )
    top = rows[0] if rows else {"dish_label": "", "final_score": 0.0}
    return {
        "predicted_label": top["dish_label"],
        "predicted_score": top["final_score"],
        "abstained": False,
        "top3_candidates": rows,
    }


def _plans() -> list[DemoUserPlan]:
    return [
        DemoUserPlan(
            username="sugarcommitted",
            display_name="Nina Park",
            intended_archetype="Dessert First",
            uploads=[
                DishSpec("tiramisu", "italian", "vegetarian"),
                DishSpec("apple pie", "american", "vegetarian"),
                DishSpec("chocolate cake", "french", "vegetarian"),
                DishSpec("donuts", "american", "vegetarian"),
                DishSpec("gelato", "italian", "vegetarian"),
                DishSpec("ice cream sundae", "american", "vegetarian"),
            ],
        ),
        DemoUserPlan(
            username="withfriesplease",
            display_name="Jordan Miles",
            intended_archetype="Fries Non-Negotiable",
            uploads=[
                DishSpec("burger and fries", "american", "beef"),
                DishSpec("loaded fries", "american", "beef"),
                DishSpec("buffalo wings", "american", "chicken"),
                DishSpec("fried chicken", "american", "chicken"),
                DishSpec("pepperoni pizza", "italian", "pork"),
                DishSpec("mac and cheese", "american", "vegetarian"),
            ],
        ),
        DemoUserPlan(
            username="chilipriority",
            display_name="Priya Nair",
            intended_archetype="Spicy or Bust",
            uploads=[
                DishSpec("spicy ramen", "japanese", "pork"),
                DishSpec("thai basil chicken", "thai", "chicken"),
                DishSpec("mapo tofu", "chinese", "vegetarian"),
                DishSpec("chili oil dumplings", "chinese", "pork"),
                DishSpec("vindaloo", "indian", "chicken"),
                DishSpec("spicy tacos", "mexican", "beef"),
            ],
        ),
        DemoUserPlan(
            username="ramenweather",
            display_name="Alex Chen",
            intended_archetype="Soup Noodle Person",
            uploads=[
                DishSpec("tonkotsu ramen", "japanese", "pork"),
                DishSpec("pho bo", "vietnamese", "beef"),
                DishSpec("udon soup", "japanese", "beef"),
                DishSpec("miso ramen", "japanese", "pork"),
                DishSpec("laksa", "thai", "seafood"),
                DishSpec("soba noodles", "japanese", "vegetarian"),
            ],
        ),
        DemoUserPlan(
            username="greensbutbetter",
            display_name="Maya Ortiz",
            intended_archetype="Salad, But Good",
            uploads=[
                DishSpec("kale caesar salad", "american", "vegetarian"),
                DishSpec("salmon grain bowl", "mediterranean", "fish"),
                DishSpec("avocado toast", "american", "vegetarian"),
                DishSpec("poke bowl", "japanese", "fish"),
                DishSpec("green smoothie", "american", "vegetarian"),
                DishSpec("quinoa salad", "mediterranean", "vegetarian"),
            ],
        ),
        DemoUserPlan(
            username="rollsagain",
            display_name="Ethan Wu",
            intended_archetype="Sushi Too Often",
            uploads=[
                DishSpec("salmon nigiri", "japanese", "fish"),
                DishSpec("tuna sashimi", "japanese", "fish"),
                DishSpec("spicy tuna roll", "japanese", "fish"),
                DishSpec("chirashi bowl", "japanese", "fish"),
                DishSpec("omakase platter", "japanese", "fish"),
                DishSpec("yellowtail roll", "japanese", "fish"),
            ],
        ),
        DemoUserPlan(
            username="tacorotation",
            display_name="Camila Reyes",
            intended_archetype="Taco Rotation",
            uploads=[
                DishSpec("carne asada tacos", "mexican", "beef"),
                DishSpec("al pastor tacos", "mexican", "pork"),
                DishSpec("birria tacos", "mexican", "beef"),
                DishSpec("chicken burrito", "mexican", "chicken"),
                DishSpec("quesadilla", "mexican", "chicken"),
                DishSpec("enchiladas", "mexican", "chicken"),
            ],
        ),
        DemoUserPlan(
            username="samebowlclub",
            display_name="Ryan Bell",
            intended_archetype="Same Order Energy",
            uploads=[
                DishSpec("chicken burrito bowl", "mexican", "chicken"),
                DishSpec("chicken burrito bowl", "mexican", "chicken"),
                DishSpec("chicken burrito bowl", "mexican", "chicken"),
                DishSpec("chicken burrito bowl", "mexican", "chicken"),
                DishSpec("chicken burrito bowl", "mexican", "chicken"),
                DishSpec("chicken burrito bowl", "mexican", "chicken"),
            ],
        ),
        DemoUserPlan(
            username="midnightcombo",
            display_name="Sam Harper",
            intended_archetype="Late Night Menu",
            uploads=[
                DishSpec("pepperoni pizza", "italian", "pork"),
                DishSpec("loaded fries", "american", "beef"),
                DishSpec("mozzarella sticks", "american", "vegetarian"),
                DishSpec("bacon cheeseburger", "american", "beef"),
                DishSpec("chicken tenders", "american", "chicken"),
                DishSpec("nachos", "mexican", "beef"),
            ],
        ),
        DemoUserPlan(
            username="eatwideawake",
            display_name="Taylor Singh",
            intended_archetype="Balanced, Still Hungry",
            uploads=[
                DishSpec("sushi", "japanese", "fish"),
                DishSpec("pasta primavera", "italian", "vegetarian"),
                DishSpec("salad bowl", "mediterranean", "vegetarian"),
                DishSpec("chicken tacos", "mexican", "chicken"),
                DishSpec("ramen", "japanese", "pork"),
                DishSpec("cheesecake", "american", "vegetarian"),
            ],
        ),
        DemoUserPlan(
            username="brunchcalendar",
            display_name="Ava Morgan",
            intended_archetype="Balanced, Still Hungry",
            uploads=[
                DishSpec("eggs benedict", "american", "egg"),
                DishSpec("croissant", "french", "vegetarian"),
                DishSpec("latte", "american", "vegetarian"),
                DishSpec("brunch toast", "american", "vegetarian"),
                DishSpec("salmon salad", "mediterranean", "fish"),
                DishSpec("fruit tart", "french", "vegetarian"),
            ],
        ),
        DemoUserPlan(
            username="tryeverything",
            display_name="Leo Ahmed",
            intended_archetype="Balanced, Still Hungry",
            uploads=[
                DishSpec("pad thai", "thai", "chicken"),
                DishSpec("bibimbap", "korean", "beef"),
                DishSpec("falafel wrap", "middle_eastern", "vegetarian"),
                DishSpec("dumplings", "chinese", "pork"),
                DishSpec("pizza", "italian", "vegetarian"),
                DishSpec("ceviche", "mexican", "fish"),
            ],
        ),
    ]


BOOSTERS: dict[str, list[DishSpec]] = {
    "Dessert First": [DishSpec("chocolate cake", "french", "vegetarian"), DishSpec("donuts", "american", "vegetarian")],
    "Fries Non-Negotiable": [DishSpec("loaded fries", "american", "beef"), DishSpec("burger and fries", "american", "beef")],
    "Spicy or Bust": [DishSpec("spicy ramen", "japanese", "pork"), DishSpec("mapo tofu", "chinese", "vegetarian")],
    "Soup Noodle Person": [DishSpec("ramen", "japanese", "pork"), DishSpec("pho", "vietnamese", "beef")],
    "Salad, But Good": [DishSpec("salad bowl", "mediterranean", "vegetarian"), DishSpec("quinoa salad", "mediterranean", "vegetarian")],
    "Sushi Too Often": [DishSpec("sushi", "japanese", "fish"), DishSpec("sashimi", "japanese", "fish")],
    "Taco Rotation": [DishSpec("tacos", "mexican", "beef"), DishSpec("al pastor tacos", "mexican", "pork")],
    "Same Order Energy": [DishSpec("chicken burrito bowl", "mexican", "chicken"), DishSpec("chicken burrito bowl", "mexican", "chicken")],
    "Late Night Menu": [DishSpec("loaded fries", "american", "beef"), DishSpec("pepperoni pizza", "italian", "pork")],
    "Balanced, Still Hungry": [DishSpec("sushi", "japanese", "fish"), DishSpec("salad bowl", "mediterranean", "vegetarian")],
}

RECIPE_SEARCH_POOL: dict[str, list[DishSpec]] = {
    "Late Night Menu": [
        DishSpec("loaded fries", "american", "beef"),
        DishSpec("nachos", "mexican", "beef"),
        DishSpec("mozzarella sticks", "american", "vegetarian"),
        DishSpec("pepperoni pizza", "italian", "pork"),
        DishSpec("chicken tenders", "american", "chicken"),
        DishSpec("churros", "mexican", "vegetarian"),
        DishSpec("ice cream", "american", "vegetarian"),
        DishSpec("milkshake", "american", "vegetarian"),
        DishSpec("buffalo wings", "american", "chicken"),
        DishSpec("cheese fries", "american", "vegetarian"),
    ],
    "Balanced, Still Hungry": [
        DishSpec("sushi", "japanese", "fish"),
        DishSpec("pasta primavera", "italian", "vegetarian"),
        DishSpec("salad bowl", "mediterranean", "vegetarian"),
        DishSpec("chicken tacos", "mexican", "chicken"),
        DishSpec("ramen", "japanese", "pork"),
        DishSpec("cheesecake", "american", "vegetarian"),
        DishSpec("pad thai", "thai", "chicken"),
        DishSpec("falafel wrap", "middle_eastern", "vegetarian"),
        DishSpec("dumplings", "chinese", "pork"),
        DishSpec("grilled salmon", "mediterranean", "fish"),
        DishSpec("avocado toast", "american", "vegetarian"),
        DishSpec("tiramisu", "italian", "vegetarian"),
    ],
}


def _remove_existing_demo_users(db_path: str, planned_usernames: set[str]) -> None:
    store = ProfileStore(db_path)
    users = store.list_users()
    to_remove: list[str] = []
    for u in users:
        username = str(u.get("username", ""))
        profile = u.get("profile", {}) or {}
        if username in planned_usernames or bool(profile.get("is_demo_user", False)):
            to_remove.append(username)
    if not to_remove:
        return
    conn = sqlite3.connect(db_path)
    try:
        for username in to_remove:
            conn.execute("DELETE FROM uploads WHERE username = ?", (username,))
            conn.execute("DELETE FROM users WHERE username = ?", (username,))
        conn.commit()
    finally:
        conn.close()


def _seed_one_user(store: ProfileStore, plan: DemoUserPlan) -> tuple[str, int]:
    profile = empty_profile()
    upload_rows = list(plan.uploads)
    seeded_predictions: list[dict] = []

    def apply_upload(dish: DishSpec, idx: int) -> None:
        nonlocal profile
        p2 = DishSpec(dish.label, dish.cuisine, dish.protein)
        pred = _prediction([p2, p2, p2])
        profile = update_profile_from_prediction(profile, pred)
        seeded_predictions.append(pred)

    idx = 1
    for dish in upload_rows:
        apply_upload(dish, idx)
        idx += 1

    archetype, desc, graphic, joke, observations = infer_archetype(profile)
    booster = BOOSTERS.get(plan.intended_archetype, [])
    # Try a few lightweight nudges so intended archetype is demonstrably represented.
    attempt = 0
    while archetype != plan.intended_archetype and booster and attempt < 3:
        for dish in booster:
            apply_upload(dish, idx)
            idx += 1
        archetype, desc, graphic, joke, observations = infer_archetype(profile)
        attempt += 1

    # Recipe search fallback for hard-to-reach archetypes.
    if archetype != plan.intended_archetype and plan.intended_archetype in RECIPE_SEARCH_POOL:
        pool = RECIPE_SEARCH_POOL[plan.intended_archetype]
        rng = random.Random(f"{plan.username}:{plan.intended_archetype}")
        best_recipe = list(upload_rows)
        best_score = -1
        for _ in range(500):
            candidate = [rng.choice(pool) for _ in range(8)]
            temp_profile = empty_profile()
            for dish in candidate:
                p2 = DishSpec(dish.label, dish.cuisine, dish.protein)
                temp_profile = update_profile_from_prediction(temp_profile, _prediction([p2, p2, p2]))
            cand_arch, cand_desc, cand_graphic, cand_joke, cand_obs = infer_archetype(temp_profile)
            score = int(cand_arch == plan.intended_archetype)
            if score > best_score:
                best_score = score
                best_recipe = candidate
            if cand_arch == plan.intended_archetype:
                # Replace with discovered recipe.
                profile = empty_profile()
                seeded_predictions = []
                idx = 1
                for dish in best_recipe:
                    apply_upload(dish, idx)
                    idx += 1
                archetype, desc, graphic, joke, observations = infer_archetype(profile)
                break

    profile["is_demo_user"] = True
    profile["demo_display_name"] = plan.display_name
    profile["demo_intended_archetype"] = plan.intended_archetype
    profile["demo_seed_tag"] = "screenshot_demo_v1"

    store.upsert_user(
        username=plan.username,
        email=f"{plan.username}@demo.local",
        created_at=utc_now_iso(),
        archetype=archetype,
        archetype_description=desc,
        archetype_graphic=graphic,
        observations="\n".join(observations),
        joke=joke,
        profile=profile,
    )
    # Persist upload rows after final recipe selection.
    conn = sqlite3.connect(str(store.db_path))
    try:
        conn.execute("DELETE FROM uploads WHERE username = ?", (plan.username,))
        conn.commit()
    finally:
        conn.close()
    for i, pred in enumerate(seeded_predictions, start=1):
        store.add_upload(
            username=plan.username,
            image_path=f"demo://{plan.username}/upload_{i:03d}.jpg",
            prediction=pred,
        )
    return archetype, idx - 1


def main() -> None:
    args = parse_args()
    plans = _plans()
    canonical = set(ARCHETYPE_CONFIG.keys())
    missing = [p.intended_archetype for p in plans if p.intended_archetype not in canonical]
    if missing:
        raise SystemExit(f"Found intended archetypes not in canonical config: {sorted(set(missing))}")

    if args.reset_existing_demo:
        _remove_existing_demo_users(args.db_path, planned_usernames={p.username for p in plans})

    store = ProfileStore(args.db_path)
    print("Canonical archetypes:", ", ".join(ARCHETYPE_CONFIG.keys()))
    print("Fallback archetype:", FALLBACK_ARCHETYPE)
    print("")

    mismatches = 0
    credential_rows: list[tuple[str, str, str]] = []
    for plan in plans:
        actual, uploads_n = _seed_one_user(store, plan)
        email = f"{plan.username}@demo.local"
        status = "OK" if actual == plan.intended_archetype else "MISMATCH"
        if status != "OK":
            mismatches += 1
        foods_preview = ", ".join(d.label for d in plan.uploads[:4])
        print(f"Created demo user: {plan.username} ({plan.display_name})")
        print(f"  Intended archetype: {plan.intended_archetype}")
        print(f"  Actual archetype:   {actual}   [{status}]")
        print(f"  Uploads seeded:     {uploads_n}")
        print(f"  Food examples:      {foods_preview}")
        print(f"  Login email:        {email}")
        print(f"  Password:           (none - BiteMe demo login is username + email only)")
        print("")
        credential_rows.append((plan.username, email, "(none - username + email login)"))

    print("Generated usernames:")
    print(", ".join(p.username for p in plans))
    print("")
    print(f"Total demo users: {len(plans)}")
    print(f"Archetype mismatches: {mismatches}")
    print("")
    print("LOGIN CREDENTIALS")
    print("-----------------")
    for username, email, password in credential_rows:
        print(f"username: {username}")
        print(f"email: {email}")
        print(f"password: {password}")
        print("")


if __name__ == "__main__":
    main()
