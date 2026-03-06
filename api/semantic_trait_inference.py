from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import csv

import numpy as np

from api.dish_trait_map import CANONICAL_DISH_TRAIT_MAP, TRAIT_KEYS, canonical_dish_key


@dataclass(frozen=True)
class SemanticNeighbor:
    dish_key: str
    similarity: float


@dataclass(frozen=True)
class SemanticTraitResult:
    blended_traits: dict[str, float]
    neighbors: list[SemanticNeighbor]
    query_source: str


class _SemanticTraitIndex:
    def __init__(self):
        self._ready = False
        self._ref_keys: list[str] = []
        self._ref_vecs: np.ndarray | None = None
        self._ref_traits: np.ndarray | None = None
        self._ref_tokens: list[set[str]] = []
        self._dim = 0
        self._encoder = None
        self._encoder_unavailable = False

    def _build(self) -> None:
        data_root = Path(__file__).resolve().parents[1] / "data"
        dishes_csv = data_root / "dishes.csv"
        vecs_npy = data_root / "dish_vectors.npy"
        if not dishes_csv.exists() or not vecs_npy.exists():
            self._ready = True
            return

        rows: list[dict[str, str]] = []
        with open(dishes_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
        vecs = np.load(vecs_npy).astype(np.float32, copy=False)
        if vecs.ndim != 2 or len(rows) != vecs.shape[0]:
            self._ready = True
            return
        vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)

        # Map each canonical dish key to averaged embedding from existing dataset rows.
        key_to_rows: dict[str, list[int]] = {}
        row_primary_keys: list[str] = []
        for i in range(len(rows)):
            row = rows[int(i)]
            label_candidates = [
                row.get("dish_label", "") if isinstance(row, dict) else "",
                row.get("dish_class", "") if isinstance(row, dict) else "",
                row.get("dish_family", "") if isinstance(row, dict) else "",
                row.get("dish_name", "") if isinstance(row, dict) else "",
            ]
            row_keys = {canonical_dish_key(x) for x in label_candidates if str(x).strip()}
            row_primary = next((k for k in row_keys if k), "")
            row_primary_keys.append(row_primary)
            for k in row_keys:
                if k in CANONICAL_DISH_TRAIT_MAP:
                    key_to_rows.setdefault(k, []).append(int(i))

        # For canonical keys not present exactly in current dataset, approximate from semantically-related rows
        # using token overlap over dish labels. This keeps the index complete without model retraining.
        for ck in CANONICAL_DISH_TRAIT_MAP.keys():
            if key_to_rows.get(ck):
                continue
            c_toks = set(str(ck).split())
            if not c_toks:
                continue
            scored: list[tuple[int, float]] = []
            for i, rk in enumerate(row_primary_keys):
                if not rk:
                    continue
                r_toks = set(rk.split())
                overlap = len(c_toks & r_toks)
                if overlap <= 0:
                    continue
                jacc = float(overlap) / float(max(1, len(c_toks | r_toks)))
                bonus = 0.2 if (ck in rk or rk in ck) else 0.0
                score = float(0.7 * jacc + 0.3 * overlap + bonus)
                scored.append((int(i), score))
            if scored:
                scored = sorted(scored, key=lambda x: x[1], reverse=True)[:24]
                key_to_rows[ck] = [i for i, _ in scored]

        ref_keys: list[str] = []
        ref_vecs: list[np.ndarray] = []
        ref_traits: list[np.ndarray] = []
        for k, rows in key_to_rows.items():
            if not rows:
                continue
            m = vecs[np.asarray(rows, dtype=np.int64)]
            avg = np.mean(m, axis=0).astype(np.float32)
            avg = avg / (np.linalg.norm(avg) + 1e-12)
            ref_keys.append(k)
            ref_vecs.append(avg)
            ref_traits.append(np.asarray([float(CANONICAL_DISH_TRAIT_MAP[k][t]) for t in TRAIT_KEYS], dtype=np.float32))

        if ref_vecs:
            self._ref_keys = ref_keys
            self._ref_vecs = np.vstack(ref_vecs).astype(np.float32, copy=False)
            self._ref_traits = np.vstack(ref_traits).astype(np.float32, copy=False)
            self._ref_tokens = [set(str(k).split()) for k in ref_keys]
            self._dim = int(self._ref_vecs.shape[1])
        self._ready = True

    def _ensure_encoder(self):
        if self._encoder_unavailable:
            return None
        if self._encoder is None:
            try:
                from models.vision import VisionEncoder

                self._encoder = VisionEncoder()
            except Exception:
                self._encoder_unavailable = True
                return None
        return self._encoder

    def _query_from_text(self, label: str) -> np.ndarray | None:
        key = canonical_dish_key(label)
        if not key:
            return None
        enc = self._ensure_encoder()
        if enc is None:
            # Semantic inference should only run from real embeddings.
            return None
        prompts = [
            f"a photo of {key}",
            f"a close-up food photo of {key}",
            f"a plated dish of {key}",
        ]
        txt = enc.encode_texts_cached(prompts).astype(np.float32, copy=False)
        avg = np.mean(txt, axis=0).astype(np.float32)
        return avg / (np.linalg.norm(avg) + 1e-12)

    def infer(
        self,
        *,
        label: str,
        query_embedding: np.ndarray | None = None,
        top_k: int = 3,
        min_similarity: float = 0.20,
    ) -> SemanticTraitResult | None:
        if not self._ready:
            self._build()
        if self._ref_vecs is None or self._ref_traits is None or len(self._ref_keys) == 0:
            return None

        q = None
        source = ""
        if query_embedding is not None:
            arr = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
            if arr.size == self._dim:
                q = arr / (np.linalg.norm(arr) + 1e-12)
                source = "query_embedding"
        if q is None:
            q = self._query_from_text(label)
            source = "label_text_embedding"
        if q is None:
            return None

        sims = (self._ref_vecs @ q).astype(np.float32)
        order = np.argsort(-sims)[: max(1, int(top_k))]
        picked = []
        for j in order:
            sim = float(sims[int(j)])
            if sim >= float(min_similarity):
                picked.append((int(j), sim))
        if not picked:
            return None

        # Similarity-weighted blend; clamp to positive support only.
        weights = np.asarray([max(0.0, s) for _, s in picked], dtype=np.float32)
        sw = float(np.sum(weights))
        if sw <= 1e-12:
            return None
        weights = weights / sw

        trait_vec = np.zeros((len(TRAIT_KEYS),), dtype=np.float32)
        neighbors: list[SemanticNeighbor] = []
        for w, (idx, sim) in zip(weights.tolist(), picked):
            trait_vec += float(w) * self._ref_traits[int(idx)]
            neighbors.append(SemanticNeighbor(dish_key=str(self._ref_keys[int(idx)]), similarity=float(sim)))

        out = {k: float(np.clip(v, 0.0, 1.0)) for k, v in zip(TRAIT_KEYS, trait_vec.tolist())}
        return SemanticTraitResult(blended_traits=out, neighbors=neighbors, query_source=source)


_SEMANTIC_INDEX = _SemanticTraitIndex()


def infer_semantic_traits(
    *,
    label: Any,
    query_embedding: np.ndarray | None = None,
    top_k: int = 3,
    min_similarity: float = 0.20,
) -> SemanticTraitResult | None:
    return _SEMANTIC_INDEX.infer(
        label=canonical_dish_key(label),
        query_embedding=query_embedding,
        top_k=top_k,
        min_similarity=min_similarity,
    )
