import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Data quality report and suspected label noise.")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--out_csv", default="reports/suspected_mislabeled.csv")
    return p.parse_args()


def _canon(x: str) -> str:
    return str(x).strip().lower().replace("_", " ")


def _norm_rows(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    dishes = pd.read_csv(data_dir / "dishes.csv")
    vectors = np.load(data_dir / "dish_vectors.npy").astype(np.float32)
    vectors = _norm_rows(vectors)

    label_col = "dish_label" if "dish_label" in dishes.columns else "dish_class"
    if label_col not in dishes.columns:
        raise ValueError("dishes.csv must contain dish_label or dish_class")
    if "cuisine" not in dishes.columns:
        raise ValueError("dishes.csv must contain cuisine")

    print("=== Label Counts ===")
    print("\nTop dish labels:")
    print(dishes[label_col].astype(str).value_counts().head(20).to_string())
    print("\nTop cuisines:")
    print(dishes["cuisine"].astype(str).value_counts().head(20).to_string())

    sims = vectors @ vectors.T
    np.fill_diagonal(sims, -1.0)
    top_k = int(max(1, args.top_k))

    suspects = []
    labels = dishes[label_col].astype(str).tolist()
    for i in range(len(dishes)):
        nn_idx = np.argsort(-sims[i])[:top_k]
        nn_labels = [labels[int(j)] for j in nn_idx]
        majority = pd.Series(nn_labels).value_counts().idxmax()
        self_label = labels[i]
        if _canon(majority) != _canon(self_label):
            suspects.append(
                {
                    "dish_id": int(dishes.iloc[i].get("dish_id", i)),
                    "image_path": str(dishes.iloc[i].get("image_path", "")),
                    "self_label": self_label,
                    "majority_neighbor_label": majority,
                    "neighbor_labels_topk": " | ".join(nn_labels),
                    "neighbor_indices_topk": ",".join(str(int(j)) for j in nn_idx),
                    "avg_neighbor_similarity": float(np.mean([sims[i, int(j)] for j in nn_idx])),
                }
            )

    out_df = pd.DataFrame(suspects).sort_values("avg_neighbor_similarity", ascending=False)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\nSuspected mislabeled images: {len(out_df)}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

