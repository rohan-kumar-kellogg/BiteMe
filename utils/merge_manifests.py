import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from utils.path_utils import normalize_path


def parse_args():
    p = argparse.ArgumentParser(description="Merge multiple manifest CSV files.")
    p.add_argument("--inputs", nargs="+", required=True, help="Input manifest CSV paths.")
    p.add_argument("--out", required=True, help="Output merged manifest CSV.")
    p.add_argument("--enforce_total", type=int, default=6000)
    p.add_argument("--enforce_source_counts", default="food101=3000,uecfood256=3000")
    return p.parse_args()


def main():
    args = parse_args()
    dfs = []
    per_manifest_counts = {}
    for path in args.inputs:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Missing manifest: {p}")
        df = pd.read_csv(p)
        if "image_path" not in df.columns:
            raise ValueError(f"{p} missing image_path column")
        df["image_path"] = df["image_path"].astype(str).str.strip()
        if "source" not in df.columns:
            stem = p.stem.lower()
            if "food101" in stem:
                df["source"] = "food101"
            elif "uec" in stem:
                df["source"] = "uecfood256"
            else:
                df["source"] = stem
        # Ensure required schema columns
        for col, default in [
            ("dish_label", ""),
            ("cuisine", "Unknown"),
            ("course", "Unknown"),
            ("protein_type", "Unknown"),
            ("label_quality", "dataset"),
        ]:
            if col not in df.columns:
                df[col] = default
            df[col] = df[col].fillna(default).astype(str)
            if col in {"cuisine", "course", "protein_type"}:
                df[col] = df[col].replace({"": "Unknown"})
        if df["dish_label"].astype(str).str.strip().eq("").any():
            raise ValueError(f"{p} has rows missing dish_label")
        df["_norm_image_path"] = df["image_path"].map(normalize_path)
        dfs.append(df)
        per_manifest_counts[p.stem] = int(len(df))
    merged = pd.concat(dfs, ignore_index=True)
    before = len(merged)
    merged = merged.drop_duplicates(subset=["_norm_image_path"]).reset_index(drop=True)
    removed = before - len(merged)
    source_counts = merged["source"].astype(str).value_counts().to_dict()
    expected = {}
    for token in str(args.enforce_source_counts).split(","):
        token = token.strip()
        if not token:
            continue
        k, v = token.split("=")
        expected[k.strip()] = int(v.strip())
    for src, count in expected.items():
        got = int(source_counts.get(src, 0))
        if got != count:
            raise RuntimeError(f"Source count mismatch for {src}: expected {count}, got {got}")
    if int(args.enforce_total) > 0 and len(merged) != int(args.enforce_total):
        raise RuntimeError(f"Merged total mismatch: expected {args.enforce_total}, got {len(merged)}")
    merged = merged.drop(columns=["_norm_image_path"], errors="ignore")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    print("Dataset merge summary:")
    for name, count in per_manifest_counts.items():
        print(f"{name}: {count} images")
    print(f"duplicates removed (normalized image_path): {removed}")
    print("source counts:")
    for src, cnt in source_counts.items():
        print(f"  {src}: {cnt}")
    print(f"merged total: {len(merged)} images")
    print(f"Saved merged manifest: {out}")


if __name__ == "__main__":
    main()

