import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from models.vision import VisionEncoder


def parse_args():
    p = argparse.ArgumentParser(description="Train supervised label heads on CLIP image embeddings.")
    p.add_argument("--labels_csv", default="data/labels.csv", help="CSV with image_path and labels.")
    p.add_argument("--out_path", default="data/models/label_heads.pkl", help="Output pickle path.")
    p.add_argument("--min_count", type=int, default=5, help="Min samples per class to keep.")
    return p.parse_args()


def clean_labels(df: pd.DataFrame, col: str, min_count: int):
    x = df[col].astype(str).str.strip()
    x = x.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    valid = x.dropna()
    if len(valid) == 0:
        return pd.Series([np.nan] * len(df), index=df.index)
    counts = valid.value_counts()
    keep = set(counts[counts >= min_count].index.tolist())
    return x.where(x.isin(keep), np.nan)


def train_head(X: np.ndarray, y: pd.Series):
    mask = y.notna().values
    Xh = X[mask]
    yh = y[mask].astype(str).values
    classes = sorted(pd.Series(yh).unique().tolist())
    if len(classes) < 2 or len(yh) < 20:
        return None
    clf = LogisticRegression(max_iter=2000, multi_class="auto")
    clf.fit(Xh, yh)
    return clf


def main():
    args = parse_args()
    labels_path = Path(args.labels_csv)
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels file: {args.labels_csv}")

    df = pd.read_csv(labels_path)
    required = {"image_path"}
    if not required.issubset(set(df.columns)):
        raise ValueError("labels CSV must include at least `image_path`.")

    df = df.copy()
    df["image_path"] = df["image_path"].astype(str)
    df = df[df["image_path"].map(lambda p: Path(p).exists())].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("No valid image paths found in labels CSV.")

    encoder = VisionEncoder()
    X = []
    kept_rows = []
    for row in df.itertuples(index=False):
        try:
            emb = encoder.encode_image(str(row.image_path))
            X.append(emb.astype(np.float32))
            kept_rows.append(row)
        except Exception:
            continue
    if len(X) < 20:
        raise ValueError("Not enough valid labeled images to train.")

    X = np.vstack(X)
    df = pd.DataFrame(kept_rows)

    # Optional target columns; train what is available.
    target_cols = [
        c
        for c in [
            "dish_class",
            "dish_family",
            "dish_name",
            "cuisine",
            "protein_type",
            "course",
            "protein",
            "prep_style",
        ]
        if c in df.columns
    ]
    if not target_cols:
        raise ValueError(
            "No target columns found. Add one or more: dish_class, dish_family, dish_name, cuisine, protein_type, course."
        )

    heads = {}
    for col in target_cols:
        y = clean_labels(df, col, min_count=args.min_count)
        clf = train_head(X, y)
        if clf is not None:
            heads[col] = clf
            print(f"Trained head: {col} ({len(clf.classes_)} classes)")
        else:
            print(f"Skipped head: {col} (insufficient data/classes)")

    if not heads:
        raise ValueError("No heads trained. Add more labeled examples per class.")

    payload = {"heads": heads, "feature_dim": int(X.shape[1])}
    out = Path(args.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(payload, f)
    print(f"Saved trained heads to: {out}")


if __name__ == "__main__":
    main()

