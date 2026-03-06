from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


def normalize_protein_type(x: str) -> str:
    s = str(x).strip().lower()
    if not s or s in {"unknown", "nan", "none"}:
        return "Mixed"
    seafood_terms = {"fish", "salmon", "tuna", "shrimp", "prawn", "crab", "lobster", "seafood"}
    meat_terms = {"beef", "pork", "lamb", "chicken", "turkey", "duck", "ribs", "meat"}
    plant_terms = {"plant", "vegan", "vegetarian", "tofu", "bean", "lentil"}
    if any(t in s for t in seafood_terms):
        return "Seafood"
    if any(t in s for t in meat_terms):
        return "Meat"
    if any(t in s for t in plant_terms):
        return "Plant"
    if "mixed" in s:
        return "Mixed"
    return "Mixed"


class ProbePredictor:
    def __init__(self, payload: dict):
        self.payload = payload
        self.dish_model = payload.get("dish_class_model")
        self.protein_model = payload.get("protein_type_model")

    @classmethod
    def from_path(cls, path: str | Path) -> "ProbePredictor":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Missing probes payload: {p}")
        with open(p, "rb") as f:
            payload = pickle.load(f)
        return cls(payload)

    @staticmethod
    def _predict_proba_dict(model, emb: np.ndarray) -> dict[str, float]:
        if model is None:
            return {}
        probs = model.predict_proba(emb.reshape(1, -1))[0]
        classes = [str(x) for x in model.classes_]
        return {c: float(p) for c, p in zip(classes, probs)}

    def predict(self, emb: np.ndarray) -> dict[str, dict[str, float]]:
        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        return {
            "dish_class_probs": self._predict_proba_dict(self.dish_model, emb),
            "protein_type_probs": self._predict_proba_dict(self.protein_model, emb),
        }

