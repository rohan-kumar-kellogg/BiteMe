from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from api.archetypes import ARCHETYPE_CONFIG
from api.profile_logic import compute_compatible_users, empty_profile, infer_archetype, update_profile_from_prediction
from api.storage import ProfileStore, utc_now_iso


@dataclass(frozen=True)
class DishSpec:
    label: str
    cuisine: str
    protein: str


@dataclass(frozen=True)
class DemoPersona:
    username: str
    title: str
    target_archetype: str
    uploads_n: int
    primary: list[DishSpec]
    secondary: list[DishSpec]
    tertiary: list[DishSpec]


def parse_args():
    p = argparse.ArgumentParser(description="Seed demo users with rich taste profiles.")
    p.add_argument("--db_path", default="data/app_profiles.db")
    p.add_argument("--reset_demo_users", action="store_true", help="Delete existing demo_* users before seeding.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--summary_json", default="reports/demo_seed_summary.json")
    return p.parse_args()


def _make_prediction(top3: list[DishSpec], score_base: float = 0.88) -> dict:
    rows = []
    for i, d in enumerate(top3[:3]):
        rows.append(
            {
                "dish_label": d.label,
                "dish_class": d.label,
                "cuisine": d.cuisine,
                "protein_type": d.protein,
                "final_score": float(max(0.5, score_base - 0.05 * i)),
                "retrieval_score": float(max(0.5, score_base - 0.04 * i)),
                "mlp_score": float(max(0.4, score_base - 0.06 * i)),
                "pair_score": float(max(0.4, score_base - 0.07 * i)),
                "image_path": f"demo://{d.label}",
            }
        )
    return {
        "predicted_label": rows[0]["dish_label"] if rows else "",
        "predicted_score": rows[0]["final_score"] if rows else 0.0,
        "abstained": False,
        "abstain_reason": "",
        "top3_candidates": rows,
        "confidence_threshold": 0.86,
        "scoring_mode": "baseline",
        "fallback_message": "",
        "raw_topn": rows,
    }


def _personas() -> list[DemoPersona]:
    return [
        DemoPersona(
            username="demo_global_nomad",
            title="adventurous global eater",
            target_archetype="Global Street-Food Hunter",
            uploads_n=16,
            primary=[DishSpec("pho", "Vietnamese", "beef"), DishSpec("paella", "Spanish", "seafood"), DishSpec("bibimbap", "Korean", "egg")],
            secondary=[DishSpec("tacos", "Mexican", "beef"), DishSpec("ramen", "Japanese", "pork"), DishSpec("chicken_curry", "Indian", "chicken")],
            tertiary=[DishSpec("ceviche", "Peruvian", "seafood"), DishSpec("falafel", "Mediterranean", "vegetarian"), DishSpec("sushi", "Japanese", "fish")],
        ),
        DemoPersona(
            username="demo_comfort_loyalist",
            title="comfort food loyalist",
            target_archetype="Comfort-Core Loyalist",
            uploads_n=14,
            primary=[DishSpec("mac_and_cheese", "American", "vegetarian"), DishSpec("lasagna", "Italian", "beef"), DishSpec("burger_and_fries", "American", "beef")],
            secondary=[DishSpec("pizza", "Italian", "pork"), DishSpec("fried_chicken", "American", "chicken"), DishSpec("poutine", "Canadian", "beef")],
            tertiary=[DishSpec("grilled_cheese", "American", "vegetarian"), DishSpec("chili", "American", "beef"), DishSpec("meatloaf", "American", "beef")],
        ),
        DemoPersona(
            username="demo_dessert_first",
            title="dessert-first person",
            target_archetype="Dessert Radar Commander",
            uploads_n=13,
            primary=[DishSpec("chocolate_mousse", "French", "vegetarian"), DishSpec("cheesecake", "American", "vegetarian"), DishSpec("tiramisu", "Italian", "vegetarian")],
            secondary=[DishSpec("apple_pie", "American", "vegetarian"), DishSpec("ice_cream_sundae", "American", "vegetarian"), DishSpec("panna_cotta", "Italian", "vegetarian")],
            tertiary=[DishSpec("strawberry_shortcake", "American", "vegetarian"), DishSpec("donuts", "American", "vegetarian"), DishSpec("baklava", "Mediterranean", "vegetarian")],
        ),
        DemoPersona(
            username="demo_protein_savory",
            title="protein-heavy savory person",
            target_archetype="Protein-First Performer",
            uploads_n=15,
            primary=[DishSpec("steak", "American", "beef"), DishSpec("grilled_salmon", "American", "fish"), DishSpec("chicken_souvlaki", "Mediterranean", "chicken")],
            secondary=[DishSpec("pork_chop", "American", "pork"), DishSpec("beef_tartare", "French", "beef"), DishSpec("lamb_chops", "Mediterranean", "lamb")],
            tertiary=[DishSpec("shrimp_scampi", "Italian", "seafood"), DishSpec("tuna_steak", "Japanese", "fish"), DishSpec("roast_chicken", "French", "chicken")],
        ),
        DemoPersona(
            username="demo_spice_explorer",
            title="spice-seeking explorer",
            target_archetype="Spice Orbit Captain",
            uploads_n=15,
            primary=[DishSpec("vindaloo", "Indian", "pork"), DishSpec("spicy_tteokbokki", "Korean", "fish"), DishSpec("hot_and_sour_soup", "Chinese", "chicken")],
            secondary=[DishSpec("spicy_ramen", "Japanese", "pork"), DishSpec("green_curry", "Thai", "chicken"), DishSpec("mapo_tofu", "Chinese", "vegetarian")],
            tertiary=[DishSpec("jambalaya", "American", "seafood"), DishSpec("spicy_tacos", "Mexican", "beef"), DishSpec("peri_peri_chicken", "Portuguese", "chicken")],
        ),
        DemoPersona(
            username="demo_balanced_broad",
            title="balanced broad-taste user",
            target_archetype="Balanced Bistro Strategist",
            uploads_n=18,
            primary=[DishSpec("salmon_and_veggies", "American", "fish"), DishSpec("sushi", "Japanese", "fish"), DishSpec("pasta_primavera", "Italian", "vegetarian")],
            secondary=[DishSpec("chicken_tacos", "Mexican", "chicken"), DishSpec("pad_thai", "Thai", "chicken"), DishSpec("bruschetta", "Italian", "vegetarian")],
            tertiary=[DishSpec("grain_bowl", "Mediterranean", "vegetarian"), DishSpec("bibimbap", "Korean", "egg"), DishSpec("dumplings", "Chinese", "pork")],
        ),
        DemoPersona(
            username="demo_fresh_greens",
            title="freshness-forward eater",
            target_archetype="Plant-Forward Explorer",
            uploads_n=12,
            primary=[DishSpec("greek_salad", "Mediterranean", "vegetarian"), DishSpec("ceviche", "Peruvian", "seafood"), DishSpec("caprese_salad", "Italian", "vegetarian")],
            secondary=[DishSpec("sashimi", "Japanese", "fish"), DishSpec("poke_bowl", "Hawaiian", "fish"), DishSpec("gazpacho", "Spanish", "vegetarian")],
            tertiary=[DishSpec("spring_rolls", "Vietnamese", "vegetarian"), DishSpec("quinoa_bowl", "Mediterranean", "vegetarian"), DishSpec("avocado_toast", "American", "vegetarian")],
        ),
        DemoPersona(
            username="demo_carb_crush",
            title="carb-forward enthusiast",
            target_archetype="Carb Compass Romantic",
            uploads_n=14,
            primary=[DishSpec("ramen", "Japanese", "pork"), DishSpec("pasta_carbonara", "Italian", "pork"), DishSpec("margherita_pizza", "Italian", "vegetarian")],
            secondary=[DishSpec("dumplings", "Chinese", "pork"), DishSpec("garlic_bread", "Italian", "vegetarian"), DishSpec("fried_rice", "Chinese", "chicken")],
            tertiary=[DishSpec("risotto", "Italian", "vegetarian"), DishSpec("udon", "Japanese", "beef"), DishSpec("burrito", "Mexican", "beef")],
        ),
    ]


def _reset_demo_users(db_path: str):
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DELETE FROM uploads WHERE username LIKE 'demo_%'")
        conn.execute("DELETE FROM users WHERE username LIKE 'demo_%'")
        conn.commit()
    finally:
        conn.close()


def _archetype_payload(persona: DemoPersona, profile: dict) -> tuple[str, str, str, str, list[str]]:
    _ = infer_archetype(profile)
    meta = ARCHETYPE_CONFIG[persona.target_archetype]
    profile["archetype_scores"] = {k: (1.0 if k == persona.target_archetype else 0.0) for k in ARCHETYPE_CONFIG.keys()}
    profile["archetype_stability"] = {"seeded_demo_profile": True, "target_archetype": persona.target_archetype}
    obs = [
        f"Demo persona: {persona.title}.",
        f"Upload pattern repeatedly reinforces {persona.target_archetype.lower()} tendencies.",
        f"Most frequent dishes include: {', '.join([d.label for d in persona.primary[:2]])}.",
    ]
    desc = f"{meta.long_description_seed} Demo profile seeded to showcase a clear, stable behavior pattern for UI testing."
    return persona.target_archetype, desc, meta.graphic_key, meta.joke, obs


def main():
    args = parse_args()
    random.seed(args.seed)
    Path(args.db_path).parent.mkdir(parents=True, exist_ok=True)
    store = ProfileStore(args.db_path)
    if args.reset_demo_users:
        _reset_demo_users(args.db_path)

    personas = _personas()
    seeded_users = []
    for p in personas:
        profile = empty_profile()
        synthetic_upload_history = []
        for i in range(p.uploads_n):
            bucket = random.random()
            if bucket < 0.65:
                src = p.primary
            elif bucket < 0.90:
                src = p.secondary
            else:
                src = p.tertiary
            top3 = random.sample(src, k=min(3, len(src)))
            while len(top3) < 3:
                top3.append(top3[-1])
            pred = _make_prediction(top3, score_base=0.90 - 0.01 * (i % 3))
            profile = update_profile_from_prediction(profile, pred)
            synthetic_upload_history.append(
                {
                    "image_path": f"demo://{p.username}/upload_{i+1:03d}.jpg",
                    "predicted_label": pred.get("predicted_label", ""),
                    "predicted_score": pred.get("predicted_score", 0.0),
                    "timestamp": utc_now_iso(),
                }
            )
            store.add_upload(
                username=p.username,
                image_path=f"demo://{p.username}/upload_{i+1:03d}.jpg",
                prediction=pred,
            )
        profile["upload_history"] = synthetic_upload_history
        archetype, desc, graphic, joke, observations = _archetype_payload(p, profile)
        created_at = utc_now_iso()
        store.upsert_user(
            username=p.username,
            created_at=created_at,
            archetype=archetype,
            archetype_description=desc,
            archetype_graphic=graphic,
            observations="\n".join(observations),
            joke=joke,
            profile=profile,
        )
        seeded_users.append({"username": p.username, "persona": p.title, "archetype": archetype, "upload_count": p.uploads_n})

    all_users = [store.get_user(x["username"]) for x in seeded_users]
    all_users = [u for u in all_users if u is not None]
    summary = {"seeded_users": seeded_users, "compatible_preview": {}}
    for u in all_users:
        others = [x for x in all_users if x["username"] != u["username"]]
        top = compute_compatible_users(u, others, limit=3)
        summary["compatible_preview"][u["username"]] = top

    out_path = Path(args.summary_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("Seeded demo users:")
    for u in seeded_users:
        print(f"- {u['username']} | {u['persona']} | archetype={u['archetype']} | uploads={u['upload_count']}")
    print(f"Summary saved: {out_path}")


if __name__ == "__main__":
    main()
