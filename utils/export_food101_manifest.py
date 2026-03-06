import argparse
import os
import random
from pathlib import Path

import pandas as pd
from torchvision.datasets import Food101


def _norm_label(x: str) -> str:
    return "_".join(str(x).strip().lower().split())


def infer_course(dish_label: str) -> str:
    s = str(dish_label).lower()
    if any(k in s for k in ["cake", "pie", "tiramisu", "panna_cotta", "churros", "waffles", "pancakes", "ice_cream"]):
        return "Dessert"
    if any(k in s for k in ["soup", "salad"]):
        return "Starter"
    return "Main"


def infer_protein_type(dish_label: str) -> str:
    s = str(dish_label).lower()
    if any(k in s for k in ["beef", "pork", "lamb", "chicken", "duck", "ribs", "steak"]):
        return "Meat"
    if any(k in s for k in ["fish", "salmon", "tuna", "shrimp", "prawn", "seafood", "sushi"]):
        return "Seafood"
    if any(k in s for k in ["tofu", "vegan", "vegetarian", "bean", "falafel", "hummus", "salad"]):
        return "Plant"
    return "Unknown"


def balanced_sample(df: pd.DataFrame, label_col: str, n_total: int, seed: int) -> pd.DataFrame:
    if len(df) < n_total:
        raise ValueError(f"Need {n_total} rows, found only {len(df)}")
    rng = random.Random(seed)
    classes = sorted(df[label_col].astype(str).unique().tolist())
    buckets = {c: df[df[label_col].astype(str) == c].index.tolist() for c in classes}
    for c in classes:
        rng.shuffle(buckets[c])
    base = n_total // len(classes)
    rem = n_total % len(classes)
    picks = []
    for i, c in enumerate(classes):
        target = base + (1 if i < rem else 0)
        picks.extend(buckets[c][:target])
    if len(picks) < n_total:
        used = set(picks)
        leftovers = [i for c in classes for i in buckets[c] if i not in used]
        rng.shuffle(leftovers)
        picks.extend(leftovers[: (n_total - len(picks))])
    return df.loc[picks[:n_total]].reset_index(drop=True)


def parse_args():
    p = argparse.ArgumentParser(description="Export exactly N Food101 images + manifest.")
    p.add_argument("--food101_root", default="food101_data")
    p.add_argument("--split", default="train", choices=["train", "test"])
    p.add_argument("--n_total", type=int, default=3000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="public_images/food101")
    p.add_argument("--out_manifest", default="public_images/food101/manifest_food101.csv")
    p.add_argument("--copy_mode", default="symlink", choices=["symlink", "copy"])
    return p.parse_args()


def _materialize(src: Path, dst: Path, copy_mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy_mode == "symlink":
        os.symlink(src.resolve(), dst)
    else:
        import shutil

        shutil.copy2(src, dst)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_manifest = Path(args.out_manifest)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    ds = Food101(root=args.food101_root, split=args.split, download=True)
    image_files = getattr(ds, "_image_files", None)
    labels = getattr(ds, "_labels", None)
    if image_files is None or labels is None:
        raise RuntimeError("Unexpected torchvision Food101 format")

    rows = []
    for p, y in zip(image_files, labels):
        lbl = _norm_label(ds.classes[int(y)])
        rows.append({"original_path": str(Path(p).resolve()), "dish_label": lbl})
    src_df = pd.DataFrame(rows)
    picked = balanced_sample(src_df, "dish_label", int(args.n_total), int(args.seed))

    out_rows = []
    per_label_counter = {}
    for r in picked.itertuples(index=False):
        src = Path(str(r.original_path))
        lbl = str(r.dish_label)
        per_label_counter[lbl] = per_label_counter.get(lbl, 0) + 1
        dst = out_dir / lbl / f"food101_{lbl}_{per_label_counter[lbl]:06d}{src.suffix.lower() or '.jpg'}"
        _materialize(src, dst, args.copy_mode)
        out_rows.append(
            {
                "image_path": str(dst),
                "dish_label": lbl,
                "cuisine": "Unknown",
                "course": infer_course(lbl),
                "protein_type": infer_protein_type(lbl),
                "source": "food101",
                "label_quality": "dataset_label+heuristic_metadata",
            }
        )
    out_df = pd.DataFrame(out_rows)
    if len(out_df) != int(args.n_total):
        raise RuntimeError(f"Expected {args.n_total} rows, got {len(out_df)}")
    out_df.to_csv(out_manifest, index=False)
    print(f"Exported food101 rows: {len(out_df)}")
    print(f"Saved: {out_manifest}")


if __name__ == "__main__":
    main()

