import argparse
import os
import random
from pathlib import Path

import pandas as pd


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".bmp"}


def _norm_label(x: str) -> str:
    return "_".join(str(x).strip().lower().split())


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
    p = argparse.ArgumentParser(description="Export exactly N UECFOOD256 images + manifest.")
    p.add_argument("--uec_root", default="public_datasets/uecfood256")
    p.add_argument("--n_total", type=int, default=3000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="public_images/uecfood256")
    p.add_argument("--out_manifest", default="public_images/uecfood256/manifest_uecfood256.csv")
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


def _scan_labels_csv(root: Path) -> pd.DataFrame:
    labels_csv = root / "labels.csv"
    if not labels_csv.exists():
        return pd.DataFrame()
    df = pd.read_csv(labels_csv)
    if "image_path" not in df.columns or "dish_label" not in df.columns:
        raise ValueError(f"{labels_csv} must include image_path,dish_label")
    rows = []
    for r in df.itertuples(index=False):
        p = Path(str(r.image_path))
        p = p if p.is_absolute() else (root / p)
        if p.exists():
            rows.append(
                {
                    "original_path": str(p.resolve()),
                    "dish_label": _norm_label(str(r.dish_label)),
                    "cuisine": str(getattr(r, "cuisine", "Unknown")) if str(getattr(r, "cuisine", "")).strip() else "Unknown",
                }
            )
    return pd.DataFrame(rows)


def _scan_class_folders(root: Path) -> pd.DataFrame:
    rows = []
    for class_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        label = _norm_label(class_dir.name)
        for p in class_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                rows.append(
                    {
                        "original_path": str(p.resolve()),
                        "dish_label": label,
                        "cuisine": "Unknown",
                    }
                )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    root = Path(args.uec_root)
    if not root.exists():
        raise FileNotFoundError(f"Missing uec_root: {root}")
    out_dir = Path(args.out_dir)
    out_manifest = Path(args.out_manifest)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    src_df = _scan_labels_csv(root)
    if len(src_df) == 0:
        src_df = _scan_class_folders(root)
    if len(src_df) == 0:
        raise ValueError("No UEC images/labels found. Need labels.csv or class folders.")
    if src_df["dish_label"].astype(str).str.strip().eq("").any():
        raise ValueError("UEC export requires dish_label for all rows.")

    picked = balanced_sample(src_df, "dish_label", int(args.n_total), int(args.seed))

    out_rows = []
    per_label_counter = {}
    for r in picked.itertuples(index=False):
        src = Path(str(r.original_path))
        lbl = _norm_label(str(r.dish_label))
        if not lbl:
            raise ValueError("dish_label missing during export")
        per_label_counter[lbl] = per_label_counter.get(lbl, 0) + 1
        dst = out_dir / lbl / f"uecfood256_{lbl}_{per_label_counter[lbl]:06d}{src.suffix.lower() or '.jpg'}"
        _materialize(src, dst, args.copy_mode)
        out_rows.append(
            {
                "image_path": str(dst),
                "dish_label": lbl,
                "cuisine": str(r.cuisine) if str(r.cuisine).strip() else "Unknown",
                "course": "Unknown",
                "protein_type": "Unknown",
                "source": "uecfood256",
                "label_quality": "dataset_label",
            }
        )
    out_df = pd.DataFrame(out_rows)
    if len(out_df) != int(args.n_total):
        raise RuntimeError(f"Expected {args.n_total} rows, got {len(out_df)}")
    out_df.to_csv(out_manifest, index=False)
    print(f"Exported uecfood256 rows: {len(out_df)}")
    print(f"Saved: {out_manifest}")


if __name__ == "__main__":
    main()

