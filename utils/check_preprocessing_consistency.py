import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from models.vision import VisionEncoder


def parse_args():
    p = argparse.ArgumentParser(description="Check dataset-vs-query CLIP preprocessing consistency.")
    p.add_argument("--manifest_csv", default="images/manifest.csv")
    p.add_argument("--dishes_csv", default="data/dishes.csv")
    p.add_argument("--dish_vectors", default="data/dish_vectors.npy")
    p.add_argument("--min_cosine", type=float, default=0.999)
    return p.parse_args()


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).reshape(-1)
    b = b.astype(np.float32).reshape(-1)
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


def main():
    args = parse_args()
    manifest = pd.read_csv(args.manifest_csv)
    dishes = pd.read_csv(args.dishes_csv)
    vecs = np.load(args.dish_vectors).astype(np.float32)
    vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
    if len(dishes) != vecs.shape[0]:
        print("WARNING: dishes.csv and dish_vectors.npy length mismatch.")
        return

    if len(manifest) == 0:
        print("No rows in manifest.")
        return
    sample_path = None
    for p in manifest["image_path"].astype(str).tolist():
        if Path(p).exists():
            sample_path = str(Path(p).resolve())
            break
    if not sample_path:
        print("No existing image paths found in manifest.")
        return

    dishes["image_path_abs"] = dishes["image_path"].astype(str).map(lambda x: str(Path(x).resolve()))
    hits = dishes.index[dishes["image_path_abs"] == sample_path].tolist()
    if not hits:
        print("WARNING: Sample image not found in dishes.csv image_path entries.")
        return
    idx = int(hits[0])
    dataset_emb = vecs[idx]
    encoder = VisionEncoder()
    query_emb = encoder.encode_image(sample_path)
    cos = _cos(dataset_emb, query_emb)
    print(f"sample_image: {sample_path}")
    print(f"cosine(dataset_emb, query_emb): {cos:.6f}")
    if cos >= float(args.min_cosine):
        print("OK: preprocessing consistency check passed.")
    else:
        print("WARNING: Preprocessing mismatch: dataset embeddings not comparable to query embeddings.")


if __name__ == "__main__":
    main()

