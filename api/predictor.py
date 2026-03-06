from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from models.retrieval import predict_dish_with_confidence
from models.tag_head import CLIPTagPredictor
from models.vision import VisionEncoder


class PredictionService:
    def __init__(self, data_dir: str = "data", tag_head_ckpt: str = "data/models/clip_mlp_tag_head.pt"):
        self.data_dir = Path(data_dir)
        self.dishes_df = pd.read_csv(self.data_dir / "dishes.csv")
        self.dish_vectors = np.load(self.data_dir / "dish_vectors.npy").astype(np.float32, copy=False)
        self.dish_vectors = self.dish_vectors / (np.linalg.norm(self.dish_vectors, axis=1, keepdims=True) + 1e-12)
        self.encoder = VisionEncoder()
        self.tag_predictor = None
        ckpt = Path(tag_head_ckpt)
        if ckpt.exists():
            try:
                self.tag_predictor = CLIPTagPredictor(str(ckpt))
            except Exception:
                self.tag_predictor = None

    def predict(self, image_path: str, *, confidence_threshold: float = 0.86) -> dict:
        raw = predict_dish_with_confidence(
            image_path,
            self.dishes_df,
            self.dish_vectors,
            encoder=self.encoder,
            tag_predictor=self.tag_predictor,
            top_k=50,
            top_n=3,
            alpha=0.15,
            use_rerank=True,
            confidence_threshold=float(confidence_threshold),
            scoring_mode="baseline",
        )
        return _json_safe(raw)


def _json_safe(x):
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_json_safe(v) for v in x]
    if isinstance(x, tuple):
        return [_json_safe(v) for v in x]
    if isinstance(x, (np.floating, float)):
        v = float(x)
        if not np.isfinite(v):
            return None
        return v
    if isinstance(x, (np.integer, int)):
        return int(x)
    return x
