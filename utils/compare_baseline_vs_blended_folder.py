import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from models.hard_negative_reranker import HardNegativePairPredictor
from models.retrieval import predict_dish_with_confidence
from models.tag_head import CLIPTagPredictor
from models.vision import VisionEncoder
from utils.path_utils import normalize_path


def parse_args():
    p = argparse.ArgumentParser(description="Compare baseline vs blended scoring on a folder of real images.")
    p.add_argument("--images_dir", required=True, help="Folder containing query images (recursive).")
    p.add_argument("--labels_csv", default="", help="Optional labels CSV with image_path and label column.")
    p.add_argument("--label_col", default="dish_label", help="Label column in labels CSV.")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--tag_head_ckpt", default="data/models/clip_mlp_tag_head.pt")
    p.add_argument("--pair_reranker_ckpt", default="data/models/hard_negative_pair_reranker.pt")
    p.add_argument("--confidence_threshold", type=float, default=0.86)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_n", type=int, default=3)
    p.add_argument("--out_json", default="reports/folder_baseline_vs_blended.json")
    return p.parse_args()


def _canon(x: str) -> str:
    return str(x).strip().lower().replace("_", " ")


def _collect_images(root: str) -> list[str]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
    out = []
    for p in Path(root).rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(str(p))
    return sorted(out)


def main():
    args = parse_args()
    images = _collect_images(args.images_dir)
    if not images:
        raise RuntimeError(f"No images found in {args.images_dir}")

    labels_map = {}
    if str(args.labels_csv).strip():
        lab = pd.read_csv(args.labels_csv)
        if "image_path" not in lab.columns or args.label_col not in lab.columns:
            raise ValueError(f"labels_csv must include image_path and {args.label_col}")
        labels_map = {
            normalize_path(str(r.image_path)): _canon(str(getattr(r, args.label_col)))
            for r in lab.itertuples(index=False)
        }

    dishes_df = pd.read_csv(Path(args.data_dir) / "dishes.csv")
    dish_vectors = np.load(Path(args.data_dir) / "dish_vectors.npy").astype(np.float32, copy=False)
    dish_vectors = dish_vectors / (np.linalg.norm(dish_vectors, axis=1, keepdims=True) + 1e-12)

    encoder = VisionEncoder()
    tagger = CLIPTagPredictor(args.tag_head_ckpt) if Path(args.tag_head_ckpt).exists() else None
    pair = HardNegativePairPredictor(args.pair_reranker_ckpt) if Path(args.pair_reranker_ckpt).exists() else None

    rows = []
    for img in images:
        qn = normalize_path(img)
        base = predict_dish_with_confidence(
            img,
            dishes_df,
            dish_vectors,
            encoder=encoder,
            tag_predictor=tagger,
            top_k=int(args.top_k),
            top_n=int(args.top_n),
            alpha=0.15,
            use_rerank=True,
            confidence_threshold=float(args.confidence_threshold),
            scoring_mode="baseline",
        )
        blend = predict_dish_with_confidence(
            img,
            dishes_df,
            dish_vectors,
            encoder=encoder,
            tag_predictor=tagger,
            pair_reranker=pair,
            top_k=int(args.top_k),
            top_n=int(args.top_n),
            alpha=0.15,
            use_rerank=True,
            confidence_threshold=float(args.confidence_threshold),
            scoring_mode="blended",
            blended_retrieval_w=0.8,
            blended_mlp_w=0.1,
            blended_pair_w=0.1,
        )
        true_label = labels_map.get(qn, "")
        row = {
            "image_path": img,
            "true_label": true_label,
            "baseline": {
                "predicted_label": str(base.get("predicted_label", "")),
                "predicted_score": float(base.get("predicted_score", np.nan)),
                "abstained": bool(base.get("abstained", False)),
                "top3": base.get("top3_candidates", []),
            },
            "blended": {
                "predicted_label": str(blend.get("predicted_label", "")),
                "predicted_score": float(blend.get("predicted_score", np.nan)),
                "abstained": bool(blend.get("abstained", False)),
                "top3": blend.get("top3_candidates", []),
            },
            "prediction_changed": str(base.get("predicted_label", "")) != str(blend.get("predicted_label", "")),
        }
        if true_label:
            base_ok = (not row["baseline"]["abstained"]) and (_canon(row["baseline"]["predicted_label"]) == true_label)
            blend_ok = (not row["blended"]["abstained"]) and (_canon(row["blended"]["predicted_label"]) == true_label)
            row["baseline_correct"] = bool(base_ok)
            row["blended_correct"] = bool(blend_ok)
        rows.append(row)

    summary = {
        "n_images": int(len(rows)),
        "n_prediction_changed": int(sum(1 for r in rows if r["prediction_changed"])),
        "baseline_abstain_rate": float(np.mean([1.0 if r["baseline"]["abstained"] else 0.0 for r in rows])),
        "blended_abstain_rate": float(np.mean([1.0 if r["blended"]["abstained"] else 0.0 for r in rows])),
    }
    if labels_map:
        b_cov = [r for r in rows if not r["baseline"]["abstained"]]
        m_cov = [r for r in rows if not r["blended"]["abstained"]]
        summary.update(
            {
                "baseline_top1_accuracy": float(np.mean([1.0 if r.get("baseline_correct", False) else 0.0 for r in rows])),
                "blended_top1_accuracy": float(np.mean([1.0 if r.get("blended_correct", False) else 0.0 for r in rows])),
                "baseline_coverage": float(len(b_cov) / max(1, len(rows))),
                "blended_coverage": float(len(m_cov) / max(1, len(rows))),
                "baseline_selective_accuracy": float(np.mean([1.0 if r.get("baseline_correct", False) else 0.0 for r in b_cov])) if b_cov else 0.0,
                "blended_selective_accuracy": float(np.mean([1.0 if r.get("blended_correct", False) else 0.0 for r in m_cov])) if m_cov else 0.0,
            }
        )

    out = {
        "config": {
            "images_dir": args.images_dir,
            "labels_csv": args.labels_csv,
            "confidence_threshold": float(args.confidence_threshold),
            "weights_blended": {"retrieval": 0.8, "mlp": 0.1, "pair": 0.1},
        },
        "summary": summary,
        "rows": rows,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Saved report: {out_path}")


if __name__ == "__main__":
    main()
