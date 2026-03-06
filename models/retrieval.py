from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils.path_utils import normalize_path
from models.probes import ProbePredictor, normalize_protein_type
from models.hard_negative_reranker import HardNegativePairPredictor
from models.tag_head import CLIPTagPredictor
from models.vision import VisionEncoder

RETRIEVAL_IMG_IMG_W = 0.70
RETRIEVAL_IMG_TEXT_W = 0.30
RERANK_RETRIEVAL_W = 0.55
RERANK_DISH_W = 0.30
RERANK_PROTEIN_W = 0.15
RERANK_PAIR_W = 1.00
BLENDED_RETRIEVAL_W = 0.80
BLENDED_MLP_W = 0.10
BLENDED_PAIR_W = 0.10

_LABEL_TEXT_EMB_CACHE: dict[tuple[str, ...], tuple[dict[str, np.ndarray], list[str]]] = {}


def _row_norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def _predict_prompt_tags(encoder: VisionEncoder, image_path: str) -> dict[str, str]:
    cuisine_labels = ["Italian", "Japanese", "Indian", "Mexican", "American", "Thai", "French", "Mediterranean"]
    cuisine_prompts = [f"a close-up food photo of {x.lower()} cuisine" for x in cuisine_labels]
    c = encoder.score_image_prompts(image_path, cuisine_prompts)
    return {
        "cuisine": cuisine_labels[int(np.argmax(c))],
    }


def _canon_label(x: str) -> str:
    return str(x).strip().lower().replace("_", " ")


def _candidate_dish_class(candidate: pd.Series) -> str:
    if "dish_class" in candidate and str(candidate.get("dish_class", "")).strip():
        return str(candidate.get("dish_class", ""))
    return str(candidate.get("dish_label", ""))


def _candidate_dish_label(candidate: pd.Series) -> str:
    if "dish_label" in candidate and str(candidate.get("dish_label", "")).strip():
        return str(candidate.get("dish_label", ""))
    return _candidate_dish_class(candidate)


def _candidate_prob_score(candidate_val: str, probs: dict[str, float] | None) -> float:
    if not probs:
        return 0.0
    key = _canon_label(candidate_val)
    for k, v in probs.items():
        if _canon_label(k) == key:
            return float(v)
    return 0.0


def _label_prompt_set(label: str) -> list[str]:
    lbl = str(label).replace("_", " ").strip()
    return [
        f"a photo of {lbl}",
        f"a close-up food photo of {lbl}",
        f"a plated dish of {lbl}",
        f"a restaurant photo of {lbl}",
        f"a plated serving of {lbl}",
        f"a dish of {lbl}",
    ]


def _build_label_text_embeddings(
    dishes_df: pd.DataFrame,
    encoder: VisionEncoder,
) -> tuple[dict[str, np.ndarray], list[str]]:
    labels = sorted({str(x) for x in dishes_df["dish_label"].dropna().astype(str).tolist() if str(x).strip()})
    key = tuple(labels)
    if key in _LABEL_TEXT_EMB_CACHE:
        return _LABEL_TEXT_EMB_CACHE[key]

    out: dict[str, np.ndarray] = {}
    for lbl in labels:
        prompts = _label_prompt_set(lbl)
        p_emb = encoder.encode_texts_cached(prompts).astype(np.float32, copy=False)
        avg = np.mean(p_emb, axis=0).astype(np.float32)
        avg = avg / (np.linalg.norm(avg) + 1e-12)
        out[lbl] = avg
    _LABEL_TEXT_EMB_CACHE[key] = (out, labels)
    return out, labels


def predict_dish(
    image_path: str,
    dishes_df: pd.DataFrame,
    dish_vectors: np.ndarray,
    *,
    encoder: VisionEncoder | None = None,
    tag_predictor: CLIPTagPredictor | None = None,
    top_k: int = 50,
    top_n: int = 3,
    alpha: float = 0.15,
    use_rerank: bool = False,
    use_prompt_tags: bool = False,
    debug: bool = False,
    multi_crop: bool = False,
    use_text_ensemble: bool = False,
    probe_predictor: ProbePredictor | None = None,
    use_protein_probe: bool = False,
    pair_reranker: HardNegativePairPredictor | None = None,
    exclude_image_paths: set[str] | None = None,
    scoring_mode: str = "baseline",
    blended_retrieval_w: float = BLENDED_RETRIEVAL_W,
    blended_mlp_w: float = BLENDED_MLP_W,
    blended_pair_w: float = BLENDED_PAIR_W,
) -> list[dict]:
    if encoder is None:
        encoder = VisionEncoder()
    if dish_vectors.ndim != 2 or len(dishes_df) != dish_vectors.shape[0]:
        raise ValueError("dishes_df and dish_vectors must align with shape [N, D].")

    emb = encoder.encode_image(image_path, multi_crop=multi_crop).astype(np.float32)
    vecs = dish_vectors.astype(np.float32, copy=False)
    sims = vecs @ emb

    k = int(min(max(1, top_k), len(sims)))
    exclude_norm = {normalize_path(x) for x in (exclude_image_paths or set())}
    idx = []
    for i in np.argsort(-sims):
        cand_path = str(dishes_df.iloc[int(i)].get("image_path", ""))
        if exclude_norm and normalize_path(cand_path) in exclude_norm:
            continue
        idx.append(int(i))
        if len(idx) >= k:
            break

    # If no tag head, fallback is retrieval-only regardless of flags.
    text_label_emb = {}
    if use_text_ensemble:
        text_label_emb, _ = _build_label_text_embeddings(dishes_df, encoder)

    dish_probs: dict[str, float] = {}
    cuisine_probs: dict[str, float] = {}
    protein_probs: dict[str, float] = {}
    use_rerank_effective = bool(
        use_rerank and (tag_predictor is not None or (use_protein_probe and probe_predictor is not None) or pair_reranker is not None)
    )
    if use_rerank_effective:
        if tag_predictor is not None:
            pred = tag_predictor.predict_tags(image_path, top_k=8)
            if "cuisine" in pred:
                cuisine_probs = pred["cuisine"].get("probs", {})
            if "dish_class" in pred:
                dish_probs = pred["dish_class"].get("probs", {})
            elif "dish_family" in pred:
                dish_probs = pred["dish_family"].get("probs", {})

        # Prompt tags are fallback-only (when supervised cuisine is unavailable).
        if use_prompt_tags and not cuisine_probs:
            prompt_tags = _predict_prompt_tags(encoder, image_path)
            p_c = prompt_tags.get("cuisine")
            if p_c:
                cuisine_probs = {p_c: 1.0}
        if use_protein_probe and probe_predictor is not None:
            probe_pred = probe_predictor.predict(emb)
            protein_probs = probe_pred.get("protein_type_probs", {})
            if not dish_probs:
                dish_probs = probe_pred.get("dish_class_probs", {})

    rows = []
    for i in idx:
        cand = dishes_df.iloc[int(i)]
        sim_raw = float(sims[int(i)])
        image_image_sim_01 = float(np.clip((sim_raw + 1.0) / 2.0, 0.0, 1.0))
        cand_lbl = _candidate_dish_label(cand)
        txt_vec = text_label_emb.get(str(cand_lbl))
        if txt_vec is not None:
            image_text_raw = float(np.dot(emb, txt_vec))
            image_text_sim_01 = float(np.clip((image_text_raw + 1.0) / 2.0, 0.0, 1.0))
        else:
            image_text_sim_01 = image_image_sim_01
        if use_text_ensemble:
            combined_retrieval = float(
                np.clip(
                    RETRIEVAL_IMG_IMG_W * image_image_sim_01 + RETRIEVAL_IMG_TEXT_W * image_text_sim_01,
                    0.0,
                    1.0,
                )
            )
        else:
            combined_retrieval = image_image_sim_01

        dish_agreement = _candidate_prob_score(cand_lbl, dish_probs) if dish_probs else float("nan")
        cuisine_agreement = _candidate_prob_score(str(cand.get("cuisine", "")), cuisine_probs) if cuisine_probs else float("nan")
        cand_protein = normalize_protein_type(str(cand.get("protein_type", "")))
        protein_agreement = _candidate_prob_score(cand_protein, protein_probs) if protein_probs else float("nan")
        pair_agreement = (
            float(pair_reranker.score_pair_embeddings(emb, vecs[int(i)]))
            if pair_reranker is not None
            else float("nan")
        )
        if use_rerank_effective:
            alpha_eff = float(np.clip(alpha, 0.0, 1.0))
            mlp_components: list[tuple[float, float]] = []
            if not np.isnan(pair_agreement):
                mlp_components.append((RERANK_PAIR_W, float(pair_agreement)))
            if not np.isnan(dish_agreement):
                mlp_components.append((RERANK_DISH_W, float(dish_agreement)))
            if not np.isnan(protein_agreement):
                mlp_components.append((RERANK_PROTEIN_W, float(protein_agreement)))

            if mlp_components:
                total_w = float(sum(w for w, _ in mlp_components))
                mlp_blend = float(sum((w / total_w) * s for w, s in mlp_components))
                if str(scoring_mode).strip().lower() == "blended":
                    blend_parts: list[tuple[float, float]] = [(float(blended_retrieval_w), float(combined_retrieval))]
                    if not np.isnan(mlp_blend):
                        blend_parts.append((float(blended_mlp_w), float(mlp_blend)))
                    if not np.isnan(pair_agreement):
                        blend_parts.append((float(blended_pair_w), float(pair_agreement)))
                    tw = float(sum(max(0.0, w) for w, _ in blend_parts))
                    if tw > 0:
                        final = float(np.clip(sum((max(0.0, w) / tw) * s for w, s in blend_parts), 0.0, 1.0))
                    else:
                        final = float(combined_retrieval)
                    retrieval_w = float((max(0.0, float(blended_retrieval_w)) / tw) if tw > 0 else 1.0)
                    mlp_w = float((max(0.0, float(blended_mlp_w)) / tw) if tw > 0 else 0.0)
                    pair_w = float((max(0.0, float(blended_pair_w)) / tw) if tw > 0 else 0.0)
                else:
                    final = float(np.clip((1.0 - alpha_eff) * combined_retrieval + alpha_eff * mlp_blend, 0.0, 1.0))
                    retrieval_w = float(1.0 - alpha_eff)
                    mlp_w = float(alpha_eff)
                    pair_w = 0.0
            else:
                mlp_blend = float("nan")
                final = combined_retrieval
                retrieval_w = 1.0
                mlp_w = 0.0
                pair_w = 0.0
        else:
            mlp_blend = float("nan")
            final = combined_retrieval
            retrieval_w = 1.0
            mlp_w = 0.0
            pair_w = 0.0
        dish_class = _candidate_dish_class(cand)
        dish_label = _candidate_dish_label(cand)
        rows.append(
            {
                "dish_id": int(cand.get("dish_id", i)),
                "dish_label": dish_label,
                "dish_class": dish_class,
                "cuisine": str(cand.get("cuisine", "")),
                "course": str(cand.get("course", "")),
                "protein_type": str(cand.get("protein_type", "")),
                "image_path": str(cand.get("image_path", "")),
                "cosine_similarity": sim_raw,
                "image_image_sim_01": image_image_sim_01,
                "image_text_sim_01": image_text_sim_01,
                "combined_retrieval": combined_retrieval,
                "sim_01": combined_retrieval,
                "dish_agreement": float(dish_agreement),
                "cuisine_agreement": float(cuisine_agreement),
                "protein_agreement": float(protein_agreement),
                "pair_agreement": float(pair_agreement),
                "tag_match_score": float(np.nan_to_num(dish_agreement, nan=0.0)),
                "mlp_blend_score": float(mlp_blend),
                "alpha": float(np.clip(alpha, 0.0, 1.0)),
                "retrieval_weight": float(retrieval_w),
                "mlp_weight": float(mlp_w),
                "pair_weight": float(pair_w),
                "scoring_mode": str(scoring_mode),
                "final_score": float(final),
                "rerank_active": bool(use_rerank_effective),
            }
        )
        if debug:
            rows[-1]["candidate_index"] = int(i)
            rows[-1]["similarity_score"] = float(sim_raw)

    rows = sorted(rows, key=lambda r: r["final_score"], reverse=True)
    return rows[: max(1, int(top_n))]


def predict_dish_with_confidence(
    image_path: str,
    dishes_df: pd.DataFrame,
    dish_vectors: np.ndarray,
    *,
    encoder: VisionEncoder | None = None,
    tag_predictor: CLIPTagPredictor | None = None,
    top_k: int = 50,
    top_n: int = 3,
    alpha: float = 0.15,
    use_rerank: bool = False,
    use_prompt_tags: bool = False,
    debug: bool = False,
    multi_crop: bool = False,
    use_text_ensemble: bool = False,
    probe_predictor: ProbePredictor | None = None,
    use_protein_probe: bool = False,
    pair_reranker: HardNegativePairPredictor | None = None,
    exclude_image_paths: set[str] | None = None,
    confidence_threshold: float = 0.86,
    scoring_mode: str = "baseline",
    blended_retrieval_w: float = BLENDED_RETRIEVAL_W,
    blended_mlp_w: float = BLENDED_MLP_W,
    blended_pair_w: float = BLENDED_PAIR_W,
) -> dict:
    top = predict_dish(
        image_path,
        dishes_df,
        dish_vectors,
        encoder=encoder,
        tag_predictor=tag_predictor,
        top_k=top_k,
        top_n=max(3, int(top_n)),
        alpha=alpha,
        use_rerank=use_rerank,
        use_prompt_tags=use_prompt_tags,
        debug=debug,
        multi_crop=multi_crop,
        use_text_ensemble=use_text_ensemble,
        probe_predictor=probe_predictor,
        use_protein_probe=use_protein_probe,
        pair_reranker=pair_reranker,
        exclude_image_paths=exclude_image_paths,
        scoring_mode=scoring_mode,
        blended_retrieval_w=blended_retrieval_w,
        blended_mlp_w=blended_mlp_w,
        blended_pair_w=blended_pair_w,
    )
    best = top[0] if top else {}
    predicted_label = str(best.get("dish_label", best.get("dish_class", ""))) if best else ""
    predicted_score = float(best.get("final_score", np.nan)) if best else float("nan")
    thr = float(np.clip(confidence_threshold, 0.0, 1.0))
    abstained = bool(not best or np.isnan(predicted_score) or predicted_score < thr)
    if abstained:
        top_labels = [str(x.get("dish_label", x.get("dish_class", ""))).replace("_", " ") for x in top[:3]]
        top_labels = [x for x in top_labels if x]
        reason = (
            f"Top prediction confidence {predicted_score:.3f} is below threshold {thr:.2f}."
            if best and not np.isnan(predicted_score)
            else "No valid prediction candidates available."
        )
        fallback_msg = "Not confident enough to make a single prediction."
        if top_labels:
            fallback_msg += f" This looks closest to: {', '.join(top_labels)}"
    else:
        reason = ""
        fallback_msg = ""

    top3_candidates = [
        {
            "dish_label": str(x.get("dish_label", x.get("dish_class", ""))),
            "dish_class": str(x.get("dish_class", x.get("dish_label", ""))),
            "cuisine": str(x.get("cuisine", "")),
            "protein_type": str(x.get("protein_type", "")),
            "final_score": float(x.get("final_score", np.nan)),
            "retrieval_score": float(x.get("combined_retrieval", x.get("sim_01", np.nan))),
            "mlp_score": float(x.get("mlp_blend_score", x.get("dish_agreement", np.nan))),
            "pair_score": float(x.get("pair_agreement", np.nan)),
            "image_path": str(x.get("image_path", "")),
        }
        for x in top[:3]
    ]
    return {
        "predicted_label": ("" if abstained else predicted_label),
        "predicted_score": predicted_score,
        "abstained": abstained,
        "abstain_reason": reason,
        "top3_candidates": top3_candidates,
        "confidence_threshold": thr,
        "scoring_mode": str(scoring_mode),
        "fallback_message": fallback_msg,
        "raw_topn": top[: max(1, int(top_n))],
    }


def predict_text_prototype(
    image_path: str,
    dishes_df: pd.DataFrame,
    *,
    encoder: VisionEncoder | None = None,
    top_n: int = 3,
) -> list[dict]:
    if encoder is None:
        encoder = VisionEncoder()
    emb = encoder.encode_image(image_path).astype(np.float32, copy=False)
    text_label_emb, labels = _build_label_text_embeddings(dishes_df, encoder)
    if not labels:
        return []
    mat = np.vstack([text_label_emb[lbl] for lbl in labels]).astype(np.float32, copy=False)
    sims = mat @ emb
    order = np.argsort(-sims)[: max(1, int(top_n))]
    out = []
    for j in order:
        lbl = str(labels[int(j)])
        s = float(sims[int(j)])
        out.append(
            {
                "dish_label": lbl,
                "dish_class": lbl,
                "final_score": float(np.clip((s + 1.0) / 2.0, 0.0, 1.0)),
                "retrieval_score": float("nan"),
                "mlp_score": float("nan"),
                "text_prototype_score": float(np.clip((s + 1.0) / 2.0, 0.0, 1.0)),
            }
        )
    return out


def _load_default_assets(data_dir: str):
    dishes_df = pd.read_csv(Path(data_dir) / "dishes.csv")
    dish_vectors = np.load(Path(data_dir) / "dish_vectors.npy")
    if dish_vectors.ndim != 2:
        raise ValueError("dish_vectors.npy must be a 2D array [N, D].")
    # Normalize once at load time.
    dish_vectors = _row_norm(dish_vectors.astype(np.float32, copy=False))
    return dishes_df, dish_vectors


def main():
    p = argparse.ArgumentParser(description="Retrieve -> rerank dish prediction")
    p.add_argument("--image_path", required=True, help="Query image path")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_n", type=int, default=3)
    p.add_argument("--alpha", type=float, default=0.15)
    p.add_argument("--use_rerank", action="store_true")
    p.add_argument("--use_prompt_tags", action="store_true")
    p.add_argument("--multi_crop", action="store_true")
    p.add_argument("--use_text_ensemble", action="store_true")
    p.add_argument("--use_protein_probe", action="store_true")
    p.add_argument("--tag_head_ckpt", default="data/models/clip_mlp_tag_head.pt")
    p.add_argument("--probes_path", default="data/models/probes.pkl")
    args = p.parse_args()

    dishes_df, dish_vectors = _load_default_assets(args.data_dir)
    encoder = VisionEncoder()
    tag_predictor = None
    if Path(args.tag_head_ckpt).exists():
        try:
            tag_predictor = CLIPTagPredictor(args.tag_head_ckpt)
        except Exception:
            tag_predictor = None
    probe_predictor = None
    if Path(args.probes_path).exists():
        try:
            probe_predictor = ProbePredictor.from_path(args.probes_path)
        except Exception:
            probe_predictor = None

    top = predict_dish(
        args.image_path,
        dishes_df,
        dish_vectors,
        encoder=encoder,
        tag_predictor=tag_predictor,
        top_k=args.top_k,
        top_n=args.top_n,
        alpha=args.alpha,
        use_rerank=args.use_rerank,
        use_prompt_tags=args.use_prompt_tags,
        multi_crop=args.multi_crop,
        use_text_ensemble=args.use_text_ensemble,
        probe_predictor=probe_predictor,
        use_protein_probe=args.use_protein_probe,
        debug=True,
    )
    print(f"Top {len(top)} results:")
    for i, row in enumerate(top, start=1):
        print(
            f"{i}. {row['dish_label']} | cuisine={row['cuisine']} | "
            f"final={row['final_score']:.4f} | sim={row['cosine_similarity']:.4f} | tag={row['tag_match_score']:.4f}"
        )


if __name__ == "__main__":
    main()

