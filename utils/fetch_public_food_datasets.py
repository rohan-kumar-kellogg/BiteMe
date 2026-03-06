import argparse
import random
from pathlib import Path

import pandas as pd
from torchvision.datasets import Food101


def _safe_refusal(name: str, reason: str):
    print(f"[{name}] Skipped: {reason}")
    print("No download performed.")


def _export_food101(out_root: Path, n_total: int, split: str, seed: int):
    out_dir = out_root / "food101"
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = Food101(root="food101_data", split=split, download=True)
    rng = random.Random(seed)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    idx = idx[: min(n_total, len(idx))]

    rows = []
    for j in idx:
        img, label = ds[j]
        cls = ds.classes[label]
        class_dir = out_dir / cls
        class_dir.mkdir(parents=True, exist_ok=True)
        p = class_dir / f"{cls}_{j}.jpg"
        img.save(p)
        rows.append(
            {
                "image_path": str(p),
                "dish_class": cls,
                "cuisine": "",
                "source": "food101",
                "label_quality": "dataset_label",
            }
        )
    manifest = out_dir / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    print(f"[food101] Exported {len(rows)} images -> {manifest}")


def parse_args():
    p = argparse.ArgumentParser(description="Fetch public food datasets with license-safe guards.")
    sub = p.add_subparsers(dest="dataset", required=True)

    p_food = sub.add_parser("food101")
    p_food.add_argument("--out_root", default="public_images")
    p_food.add_argument("--n_total", type=int, default=3000)
    p_food.add_argument("--split", default="train", choices=["train", "test"])
    p_food.add_argument("--seed", type=int, default=42)

    p_uec = sub.add_parser("uecfood256")
    p_uec.add_argument("--out_root", default="public_images")

    p_mit = sub.add_parser("mit_places_food")
    p_mit.add_argument("--out_root", default="public_images")
    return p.parse_args()


def main():
    args = parse_args()
    if args.dataset == "food101":
        _export_food101(Path(args.out_root), args.n_total, args.split, args.seed)
        return
    if args.dataset == "uecfood256":
        _safe_refusal(
            "uecfood256",
            "license/redistribution terms are not automatically verified by this script. "
            "Please verify terms manually before use.",
        )
        return
    if args.dataset == "mit_places_food":
        _safe_refusal(
            "mit_places_food",
            "no verified permissive food-specific redistribution terms configured in this script.",
        )
        return


if __name__ == "__main__":
    main()

