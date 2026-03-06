from pathlib import Path

import numpy as np
import pandas as pd


def main():
    manifest_path = Path("images/manifest.csv")
    vectors_path = Path("data/dish_vectors.npy")

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing {manifest_path}")
    if not vectors_path.exists():
        raise FileNotFoundError(f"Missing {vectors_path}")

    manifest = pd.read_csv(manifest_path)
    if "image_path" not in manifest.columns:
        raise ValueError("images/manifest.csv must include image_path")

    image_paths = manifest["image_path"].astype(str)
    unique_count = int(image_paths.nunique())
    missing = [p for p in image_paths.tolist() if not Path(p).exists()]

    print("=== Manifest Integrity ===")
    print(f"total rows: {len(manifest)}")
    print(f"unique image paths: {unique_count}")
    print(f"missing files: {len(missing)}")
    if missing:
        print("sample missing:")
        for p in missing[:10]:
            print(f"  - {p}")

    vectors = np.load(vectors_path)
    norms = np.linalg.norm(vectors.astype(np.float32), axis=1) if vectors.ndim == 2 and len(vectors) else np.array([])
    print("\n=== Embedding Integrity ===")
    print(f"shape: {vectors.shape}")
    print(f"dtype: {vectors.dtype}")
    if len(norms) > 0:
        print(
            f"norms mean/min/max: {float(norms.mean()):.6f} / {float(norms.min()):.6f} / {float(norms.max()):.6f}"
        )
    else:
        print("norms mean/min/max: n/a")

    if vectors.shape[0] != len(manifest):
        print(
            "WARNING: vector/image count mismatch "
            f"(vectors={vectors.shape[0]}, manifest_rows={len(manifest)})."
        )
    else:
        print("vector/image count match: OK")


if __name__ == "__main__":
    main()

