import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Verify merged dataset growth and embedding alignment.")
    p.add_argument("--manifest_csv", default="images/manifest_merged.csv")
    p.add_argument("--dishes_csv", default="data/dishes.csv")
    p.add_argument("--dish_vectors", default="data/dish_vectors.npy")
    return p.parse_args()


def main():
    args = parse_args()
    manifest = pd.read_csv(args.manifest_csv)
    dishes = pd.read_csv(args.dishes_csv)
    vecs = np.load(args.dish_vectors)

    n_manifest = len(manifest)
    n_dishes = len(dishes)
    n_vec = int(vecs.shape[0]) if vecs.ndim == 2 else -1
    dim = int(vecs.shape[1]) if vecs.ndim == 2 else -1

    print(f"manifest rows: {n_manifest}")
    print(f"dishes rows: {n_dishes}")
    print(f"dish_vectors shape: {vecs.shape}")

    if not (n_manifest == n_dishes == n_vec):
        raise RuntimeError(
            f"Row mismatch detected: manifest={n_manifest}, dishes={n_dishes}, vectors={n_vec}"
        )
    if dim != 512:
        raise RuntimeError(f"Embedding dimension mismatch: expected 512, got {dim}")

    source_col = "source" if "source" in manifest.columns else None
    if source_col:
        print("\nCounts by source:")
        print(manifest[source_col].astype(str).value_counts().to_string())

    label_col = "dish_label" if "dish_label" in manifest.columns else ("dish_class" if "dish_class" in manifest.columns else None)
    if label_col:
        print("\nTop dish labels:")
        print(manifest[label_col].astype(str).value_counts().head(20).to_string())

    if "cuisine" in manifest.columns:
        cuisine = manifest["cuisine"].astype(str).str.strip()
        pct_unknown = float(((cuisine == "") | (cuisine.str.lower() == "unknown")).mean() * 100.0)
        print(f"\nUnknown cuisine: {pct_unknown:.2f}%")

    print("\nOK: dataset growth verification passed.")


if __name__ == "__main__":
    main()

