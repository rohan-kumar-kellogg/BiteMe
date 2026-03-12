import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from models.retrieval import predict_dish
from models.tag_head import CLIPTagPredictor
from models.vision import VisionEncoder


EXPECTED_KEYS = {
    "dish_id",
    "dish_label",
    "dish_class",
    "cuisine",
    "course",
    "protein_type",
    "image_path",
    "cosine_similarity",
    "tag_match_score",
    "final_score",
}


def _fmt_mtime(path: Path) -> str:
    return f"{path.stat().st_mtime:.0f}"


def _print_data_info(dishes_path: Path, vectors_path: Path):
    dishes_df = pd.read_csv(dishes_path)
    vectors = np.load(vectors_path)
    norms = np.linalg.norm(vectors.astype(np.float32), axis=1)
    print("=== Data Artifacts ===")
    print(f"dishes.csv mtime: {dishes_path} -> {_fmt_mtime(dishes_path)}")
    print(f"dish_vectors.npy mtime: {vectors_path} -> {_fmt_mtime(vectors_path)}")
    print(f"dishes shape: {dishes_df.shape}")
    print(f"dish_vectors shape: {vectors.shape}, dtype={vectors.dtype}")
    print(
        "L2 norms: "
        f"mean={float(norms.mean()):.6f} min={float(norms.min()):.6f} max={float(norms.max()):.6f}"
    )
    return dishes_df, vectors


def _assert_streamlit_key_usage(streamlit_path: Path):
    text = streamlit_path.read_text()
    assert 'best.get("dish_label"' in text, "archived streamlit app is not reading dish_label from predict_dish results."
    print("Streamlit key check: PASS (uses dish_label)")


def _load_tagger(ckpt_path: Path):
    if not ckpt_path.exists():
        print(f"Tag head not found at {ckpt_path}; running prompt-only rerank.")
        return None
    try:
        return CLIPTagPredictor(str(ckpt_path))
    except Exception as exc:
        print(f"Failed to load tag head ({exc}); continuing prompt-only rerank.")
        return None


def main():
    data_dir = PROJECT_ROOT / "data"
    dishes_path = data_dir / "dishes.csv"
    vectors_path = data_dir / "dish_vectors.npy"
    streamlit_path = PROJECT_ROOT / "archive" / "legacy_streamlit" / "streamlit_app.py"
    tag_ckpt = data_dir / "models" / "clip_mlp_tag_head.pt"

    dishes_df, dish_vectors = _print_data_info(dishes_path, vectors_path)
    _assert_streamlit_key_usage(streamlit_path)

    sample_df = dishes_df.dropna(subset=["image_path"]).sample(n=min(5, len(dishes_df)), random_state=42)
    encoder = VisionEncoder()
    tagger = _load_tagger(tag_ckpt)

    print("\n=== predict_dish sanity checks (N=5) ===")
    for n, row in enumerate(sample_df.itertuples(index=False), start=1):
        img = str(row.image_path)
        print(f"\n[{n}] Query: {img}")
        preds = predict_dish(
            image_path=img,
            dishes_df=dishes_df,
            dish_vectors=dish_vectors,
            encoder=encoder,
            tag_predictor=tagger,
            top_k=50,
            top_n=10,
            alpha=0.15,
            use_rerank=True,
        )
        assert len(preds) > 0, "predict_dish returned empty list."
        for p in preds:
            missing = EXPECTED_KEYS - set(p.keys())
            assert not missing, f"Missing keys in predict_dish output: {missing}"
        for i, p in enumerate(preds, start=1):
            print(
                f"{i:02d}. {p['dish_label']} | {p['cuisine']} | {p['course']} | {p['protein_type']} | "
                f"sim={p['cosine_similarity']:.4f} tag={p['tag_match_score']:.4f} final={p['final_score']:.4f}"
            )

    print("\nVerification suite: PASS")


if __name__ == "__main__":
    main()

