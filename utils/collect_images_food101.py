import os
import random
import shutil
import argparse
from pathlib import Path

import pandas as pd
from torchvision.datasets import Food101


# ---- CONFIG ----
OUT_DIR = Path("images")
N_TOTAL = 120  # make a bit more than 100 so you have buffer
SEED = 42

CLASS_TO_CUISINE = {

    # ----------------
    # Italian
    # ----------------
    "pizza": "Italian",
    "spaghetti_bolognese": "Italian",
    "lasagna": "Italian",
    "risotto": "Italian",
    "ravioli": "Italian",
    "gnocchi": "Italian",
    "tiramisu": "Italian",
    "panna_cotta": "Italian",
    "caprese_salad": "Italian",
    "bruschetta": "Italian",
    "focaccia": "Italian",

    # ----------------
    # Japanese
    # ----------------
    "ramen": "Japanese",
    "sushi": "Japanese",
    "takoyaki": "Japanese",
    "miso_soup": "Japanese",
    "tempura": "Japanese",
    "gyoza": "Japanese",
    "tonkatsu": "Japanese",

    # ----------------
    # Indian
    # ----------------
    "butter_chicken": "Indian",
    "naan": "Indian",
    "samosa": "Indian",
    "chicken_curry": "Indian",
    "tandoori_chicken": "Indian",
    "dal": "Indian",

    # ----------------
    # Mexican
    # ----------------
    "tacos": "Mexican",
    "guacamole": "Mexican",
    "nachos": "Mexican",
    "burrito": "Mexican",
    "quesadilla": "Mexican",
    "enchiladas": "Mexican",
    "churros": "Mexican",

    # ----------------
    # American
    # ----------------
    "cheeseburger": "American",
    "hot_dog": "American",
    "french_fries": "American",
    "fried_chicken": "American",
    "macaroni_and_cheese": "American",
    "pulled_pork_sandwich": "American",
    "club_sandwich": "American",
    "grilled_cheese_sandwich": "American",
    "pancakes": "American",
    "waffles": "American",

    # ----------------
    # Thai
    # ----------------
    "pad_thai": "Thai",
    "spring_rolls": "Thai",
    "green_curry": "Thai",
    "tom_yum_soup": "Thai",

    # ----------------
    # French
    # ----------------
    "creme_brulee": "French",
    "croque_madame": "French",
    "beef_tartare": "French",
    "escargots": "French",
    "ratatouille": "French",
    "onion_soup": "French",
    "quiche_lorraine": "French",
    "foie_gras": "French",

    # ----------------
    # Mediterranean
    # ----------------
    "greek_salad": "Mediterranean",
    "falafel": "Mediterranean",
    "hummus": "Mediterranean",
    "shawarma": "Mediterranean",
    "tabbouleh": "Mediterranean",

    # ----------------
    # Chinese
    # ----------------
    "dumplings": "Chinese",
    "kung_pao_chicken": "Chinese",
    "peking_duck": "Chinese",
    "sweet_and_sour_pork": "Chinese",
    "fried_rice": "Chinese",
    "mapo_tofu": "Chinese",

    # ----------------
    # Korean
    # ----------------
    "bibimbap": "Korean",
    "kimchi": "Korean",
    "bulgogi": "Korean",

    # ----------------
    # Vietnamese
    # ----------------
    "pho": "Vietnamese",
    "banh_mi": "Vietnamese",
}


def infer_course(class_name: str) -> str:
    c = class_name.lower()
    if any(k in c for k in ["cake", "pie", "panna_cotta", "tiramisu", "churros", "waffles", "pancakes", "creme_brulee"]):
        return "Dessert"
    if any(k in c for k in ["soup", "salad"]):
        return "Starter"
    if any(k in c for k in ["sandwich", "burger", "hot_dog", "burrito", "tacos", "pizza", "ramen", "pho"]):
        return "Main"
    return "Main"


def infer_protein(class_name: str) -> str:
    c = class_name.lower()
    if any(k in c for k in ["chicken", "duck", "pork", "beef", "lamb"]):
        return "Meat"
    if any(k in c for k in ["fish", "sushi", "shrimp"]):
        return "Seafood"
    if any(k in c for k in ["tofu", "falafel", "hummus", "salad"]):
        return "Plant"
    return "Mixed"


def parse_args():
    p = argparse.ArgumentParser(description="Export balanced Food-101 sample into images/ with metadata manifest.")
    p.add_argument("--n_total", type=int, default=N_TOTAL, help="Total images to export.")
    p.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    p.add_argument("--split", type=str, default="train", choices=["train", "test"], help="Food-101 split.")
    p.add_argument(
        "--per_class_max",
        type=int,
        default=0,
        help="Optional max images per class (0 means no explicit cap beyond balancing).",
    )
    return p.parse_args()


def balanced_sample(class_to_indices: dict[str, list[int]], n_total: int, rng: random.Random, per_class_max: int = 0):
    classes = sorted(class_to_indices.keys())
    n_classes = len(classes)
    if n_classes == 0 or n_total <= 0:
        return []

    base = n_total // n_classes
    rem = n_total % n_classes
    selected: list[tuple[str, int]] = []

    # First pass: balanced allocation
    for i, cls in enumerate(classes):
        idxs = class_to_indices[cls][:]
        rng.shuffle(idxs)
        target = base + (1 if i < rem else 0)
        if per_class_max > 0:
            target = min(target, per_class_max)
        take = idxs[: min(target, len(idxs))]
        selected.extend((cls, j) for j in take)

    # Second pass: top-up from leftovers if we are short
    if len(selected) < n_total:
        used = set(selected)
        leftovers = []
        for cls in classes:
            for j in class_to_indices[cls]:
                t = (cls, j)
                if t not in used:
                    leftovers.append(t)
        rng.shuffle(leftovers)
        need = n_total - len(selected)
        selected.extend(leftovers[:need])

    rng.shuffle(selected)
    return selected[:n_total]


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    # Auto-download Food101 to ./food101_data
    ds = Food101(root="food101_data", split=args.split, download=True)

    # Build index lists for all Food-101 classes (dish-class is primary label).
    class_to_indices = {}
    for idx in range(len(ds)):
        _, label = ds[idx]
        class_name = ds.classes[label]
        class_to_indices.setdefault(class_name, []).append(idx)

    selected = balanced_sample(
        class_to_indices=class_to_indices,
        n_total=args.n_total,
        rng=rng,
        per_class_max=args.per_class_max,
    )

    # Create class folders
    OUT_DIR.mkdir(exist_ok=True)
    for class_name in class_to_indices:
        (OUT_DIR / class_name).mkdir(parents=True, exist_ok=True)

    # Export images
    saved = 0
    manifest_rows = []
    for class_name, idx in selected:
        img, _ = ds[idx]
        cuisine = CLASS_TO_CUISINE.get(class_name, "Unknown")
        class_dir = OUT_DIR / class_name
        out_path = class_dir / f"{class_name}_{idx}.jpg"
        img.save(out_path)
        manifest_rows.append(
            {
                "image_path": str(out_path),
                "dish_class": class_name,
                "dish_family": "",
                "cuisine": cuisine,
                "course": infer_course(class_name),
                "protein_type": infer_protein(class_name),
                "dish_name": class_name.replace("_", " "),
                "source": "food101",
                "label_quality": "dataset_label+heuristic_metadata",
            }
        )
        saved += 1

    pd.DataFrame(manifest_rows).to_csv(OUT_DIR / "manifest.csv", index=False)
    print(
        f"✅ Exported {saved} images into {OUT_DIR}/<DishClass>/ folders + manifest.csv "
        f"(split={args.split}, n_total={args.n_total})"
    )


if __name__ == "__main__":
    main()