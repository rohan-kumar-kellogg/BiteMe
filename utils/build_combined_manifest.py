import argparse
import hashlib
import os
import random
from pathlib import Path

import pandas as pd
from torchvision.datasets import Food101


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".bmp"}


def _str2bool(x):
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {x}")


def _normalize_label(x: str) -> str:
    return "_".join(str(x).strip().lower().split())


def _md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _balanced_sample(df: pd.DataFrame, label_col: str, n: int, seed: int) -> pd.DataFrame:
    if n <= 0:
        return df.iloc[0:0].copy()
    if len(df) < n:
        raise ValueError(f"Requested n={n}, but only {len(df)} candidates available.")
    rng = random.Random(seed)
    classes = sorted(df[label_col].astype(str).unique().tolist())
    buckets = {c: df[df[label_col].astype(str) == c].index.tolist() for c in classes}
    for c in classes:
        rng.shuffle(buckets[c])
    base = n // len(classes)
    rem = n % len(classes)
    picks = []
    for i, c in enumerate(classes):
        target = base + (1 if i < rem else 0)
        picks.extend(buckets[c][:target])
    if len(picks) < n:
        used = set(picks)
        leftovers = [idx for c in classes for idx in buckets[c] if idx not in used]
        rng.shuffle(leftovers)
        picks.extend(leftovers[: (n - len(picks))])
    picks = picks[:n]
    out = df.loc[picks].copy().reset_index(drop=True)
    return out


def _materialize_one(src: Path, dst: Path, copy_mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy_mode == "symlink":
        os.symlink(src.resolve(), dst)
    elif copy_mode == "copy":
        import shutil

        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported copy_mode: {copy_mode}")


def _scan_from_labels_csv(root: Path) -> pd.DataFrame:
    labels_csv = root / "labels.csv"
    if not labels_csv.exists():
        return pd.DataFrame()
    df = pd.read_csv(labels_csv)
    if "image_path" not in df.columns or "dish_label" not in df.columns:
        raise ValueError(f"{labels_csv} must include image_path,dish_label columns")
    out = pd.DataFrame(
        {
            "original_path": df["image_path"].map(lambda p: str((root / str(p)).resolve()) if not Path(str(p)).is_absolute() else str(Path(str(p)).resolve())),
            "source_dish_label": df["dish_label"].astype(str),
            "cuisine": df["cuisine"].astype(str) if "cuisine" in df.columns else "Unknown",
        }
    )
    return out


def _scan_from_class_folders(root: Path) -> pd.DataFrame:
    rows = []
    for class_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        label = class_dir.name
        for p in class_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                rows.append(
                    {
                        "original_path": str(p.resolve()),
                        "source_dish_label": str(label),
                        "cuisine": "Unknown",
                    }
                )
    return pd.DataFrame(rows)


def _load_local_dataset(root: Path, source_name: str) -> pd.DataFrame:
    if not root.exists():
        raise FileNotFoundError(f"Missing {source_name} root: {root}")
    from_csv = _scan_from_labels_csv(root)
    if len(from_csv) > 0:
        return from_csv
    from_folders = _scan_from_class_folders(root)
    if len(from_folders) > 0:
        return from_folders
    raise ValueError(f"Could not find labels.csv or class-folder images for {source_name} at {root}")


def _load_food101(food101_root: Path, split: str) -> pd.DataFrame:
    ds = Food101(root=str(food101_root), split=split, download=True)
    image_files = getattr(ds, "_image_files", None)
    labels = getattr(ds, "_labels", None)
    if image_files is None or labels is None:
        raise RuntimeError("Unexpected torchvision Food101 internals; expected _image_files and _labels.")
    rows = []
    for p, lab in zip(image_files, labels):
        rows.append(
            {
                "original_path": str(Path(p).resolve()),
                "source_dish_label": str(ds.classes[int(lab)]),
                "cuisine": "Unknown",
            }
        )
    return pd.DataFrame(rows)


def parse_args():
    p = argparse.ArgumentParser(description="Build merged 8k food manifest from local datasets + Food101.")
    p.add_argument("--out_dir", default="images_combined")
    p.add_argument("--out_manifest", default="images/manifest_merged.csv")
    p.add_argument("--food101_root", default="food101_data")
    p.add_argument("--food101_split", default="train", choices=["train", "test"])
    p.add_argument("--food101_n", type=int, default=3000)
    p.add_argument("--uec_root", default="data/public_datasets/uecfood256")
    p.add_argument("--uec_n", type=int, default=3000)
    p.add_argument("--vireo_root", default="data/public_datasets/vireofood172")
    p.add_argument("--vireo_n", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dedupe", type=_str2bool, default=True)
    p.add_argument("--copy_mode", default="symlink", choices=["symlink", "copy"])
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_manifest = Path(args.out_manifest)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    target_total = int(args.food101_n + args.uec_n + args.vireo_n)
    seed = int(args.seed)

    # Load candidates
    food_all = _load_food101(Path(args.food101_root), args.food101_split)
    food_all["source"] = "food101"
    uec_all = _load_local_dataset(Path(args.uec_root), "uecfood256")
    uec_all["source"] = "uecfood256"
    vireo_all = _load_local_dataset(Path(args.vireo_root), "vireofood172")
    vireo_all["source"] = "vireofood172"

    print("Candidates per source:")
    print(f"- food101: {len(food_all)}")
    print(f"- uecfood256: {len(uec_all)}")
    print(f"- vireofood172: {len(vireo_all)}")

    # Balanced sampling per source
    food_sel = _balanced_sample(food_all, "source_dish_label", int(args.food101_n), seed + 11)
    uec_sel = _balanced_sample(uec_all, "source_dish_label", int(args.uec_n), seed + 23)
    vireo_sel = _balanced_sample(vireo_all, "source_dish_label", int(args.vireo_n), seed + 37)
    merged = pd.concat([food_sel, uec_sel, vireo_sel], ignore_index=True)

    print("Selected per source:")
    print(f"- food101: {len(food_sel)}")
    print(f"- uecfood256: {len(uec_sel)}")
    print(f"- vireofood172: {len(vireo_sel)}")

    merged["source_dish_label"] = merged["source_dish_label"].astype(str)
    merged["dish_label"] = merged["source_dish_label"].map(_normalize_label)
    merged["cuisine"] = merged["cuisine"].astype(str).replace({"": "Unknown"}).fillna("Unknown")
    merged["label_quality"] = "dataset"
    merged["original_path"] = merged["original_path"].astype(str).map(lambda p: str(Path(p).resolve()))

    duplicates_removed = 0
    if bool(args.dedupe):
        keep = []
        seen_hash = set()
        for row in merged.itertuples(index=False):
            p = Path(str(row.original_path))
            if not p.exists():
                continue
            md5 = _md5_file(p)
            if md5 in seen_hash:
                duplicates_removed += 1
                continue
            seen_hash.add(md5)
            keep.append(
                {
                    "original_path": str(row.original_path),
                    "source_dish_label": str(row.source_dish_label),
                    "cuisine": str(row.cuisine),
                    "source": str(row.source),
                    "dish_label": str(row.dish_label),
                    "label_quality": "dataset",
                }
            )
        merged = pd.DataFrame(keep)
    if len(merged) != target_total:
        raise RuntimeError(
            f"Final row count mismatch after selection/dedupe: got {len(merged)} expected {target_total}. "
            "Adjust source roots or dedupe settings."
        )

    # Materialize image links/copies
    out_rows = []
    counters = {}
    for row in merged.itertuples(index=False):
        src = Path(str(row.original_path))
        if not src.exists():
            continue
        lbl = _normalize_label(str(row.dish_label))
        class_dir = out_dir / lbl
        key = (str(row.source), lbl)
        counters[key] = counters.get(key, 0) + 1
        suffix = src.suffix.lower() if src.suffix else ".jpg"
        fname = f"{row.source}_{lbl}_{counters[key]:06d}{suffix}"
        dst = class_dir / fname
        _materialize_one(src, dst, args.copy_mode)
        out_rows.append(
            {
                "image_path": str(dst),
                "dish_label": lbl,
                "source": str(row.source),
                "source_dish_label": str(row.source_dish_label),
                "cuisine": str(row.cuisine) if str(row.cuisine).strip() else "Unknown",
                "label_quality": "dataset",
                "original_path": str(src.resolve()),
            }
        )

    out_df = pd.DataFrame(out_rows)
    if len(out_df) != target_total:
        raise RuntimeError(
            f"Materialized rows mismatch: got {len(out_df)} expected {target_total}. "
            "Check missing files or path permissions."
        )
    out_df.to_csv(out_manifest, index=False)

    print("Dataset merge summary:")
    print(f"- candidates: food101={len(food_all)}, uecfood256={len(uec_all)}, vireofood172={len(vireo_all)}")
    print(f"- selected: food101={len(food_sel)}, uecfood256={len(uec_sel)}, vireofood172={len(vireo_sel)}")
    print(f"- duplicates removed: {duplicates_removed}")
    print(f"- final rows: {len(out_df)}")
    print(f"Saved manifest: {out_manifest}")


if __name__ == "__main__":
    main()

