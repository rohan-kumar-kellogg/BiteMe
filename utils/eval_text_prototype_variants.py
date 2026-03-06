import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from models.retrieval import predict_dish, predict_text_prototype
from models.tag_head import CLIPTagPredictor
from models.vision import VisionEncoder
from utils.path_utils import normalize_path


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate retrieval/text-prototype/rerank variants.")
    p.add_argument("--manifest_csv", default="images/manifest.csv")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--tag_head_ckpt", default="data/models/clip_mlp_tag_head.pt")
    p.add_argument("--n_eval", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=0.15)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_n", type=int, default=3)
    p.add_argument("--out_json", default="reports/text_prototype_variant_eval.json")
    return p.parse_args()


def _canon(x: str) -> str:
    return str(x).strip().lower().replace("_", " ")


def _resolve_label_col(df: pd.DataFrame) -> str:
    for col in ("dish_class", "dish_label", "dish_family", "dish_name"):
        if col in df.columns:
            return col
    raise ValueError("manifest CSV needs one of: dish_class, dish_label, dish_family, dish_name")


def _score_variant(
    name: str,
    q_path: str,
    qn: str,
    dishes_df: pd.DataFrame,
    dish_vectors: np.ndarray,
    encoder: VisionEncoder,
    tagger: CLIPTagPredictor | None,
    alpha: float,
    top_k: int,
    top_n: int,
) -> list[dict]:
    if name == "text_prototype_only":
        return predict_text_prototype(q_path, dishes_df, encoder=encoder, top_n=top_n)
    if name == "retrieval_only":
        return predict_dish(
            q_path,
            dishes_df,
            dish_vectors,
            encoder=encoder,
            tag_predictor=tagger,
            top_k=top_k,
            top_n=top_n,
            alpha=alpha,
            use_rerank=False,
            use_text_ensemble=False,
            exclude_image_paths={qn},
        )
    if name == "retrieval_mlp_rerank":
        return predict_dish(
            q_path,
            dishes_df,
            dish_vectors,
            encoder=encoder,
            tag_predictor=tagger,
            top_k=top_k,
            top_n=top_n,
            alpha=alpha,
            use_rerank=True,
            use_text_ensemble=False,
            exclude_image_paths={qn},
        )
    if name == "retrieval_text_prototype":
        return predict_dish(
            q_path,
            dishes_df,
            dish_vectors,
            encoder=encoder,
            tag_predictor=tagger,
            top_k=top_k,
            top_n=top_n,
            alpha=alpha,
            use_rerank=False,
            use_text_ensemble=True,
            exclude_image_paths={qn},
        )
    if name == "retrieval_mlp_rerank_text_prototype":
        return predict_dish(
            q_path,
            dishes_df,
            dish_vectors,
            encoder=encoder,
            tag_predictor=tagger,
            top_k=top_k,
            top_n=top_n,
            alpha=alpha,
            use_rerank=True,
            use_text_ensemble=True,
            exclude_image_paths={qn},
        )
    raise ValueError(f"Unknown variant: {name}")


def run():
    args = parse_args()
    np.random.seed(args.seed)

    manifest = pd.read_csv(args.manifest_csv)
    label_col = _resolve_label_col(manifest)
    manifest = manifest[manifest["image_path"].map(lambda p: Path(str(p)).exists())].copy()
    manifest["eval_label"] = manifest[label_col].astype(str)

    dishes_df = pd.read_csv(Path(args.data_dir) / "dishes.csv")
    dish_vectors = np.load(Path(args.data_dir) / "dish_vectors.npy").astype(np.float32, copy=False)
    dish_vectors = dish_vectors / (np.linalg.norm(dish_vectors, axis=1, keepdims=True) + 1e-12)

    encoder = VisionEncoder()
    tagger = None
    if Path(args.tag_head_ckpt).exists():
        try:
            tagger = CLIPTagPredictor(args.tag_head_ckpt)
        except Exception:
            tagger = None

    n = min(int(args.n_eval), len(manifest))
    sample = manifest.sample(n=n, random_state=args.seed).reset_index(drop=True)

    variants = [
        "retrieval_only",
        "text_prototype_only",
        "retrieval_mlp_rerank",
        "retrieval_text_prototype",
        "retrieval_mlp_rerank_text_prototype",
    ]
    stats = {
        v: {
            "top1_correct": 0,
            "top3_correct": 0,
            "confusions": Counter(),
            "rows": [],
        }
        for v in variants
    }
    per_query = []

    for rec in sample.itertuples(index=False):
        q_path = str(rec.image_path)
        qn = normalize_path(q_path)
        true_label = _canon(str(rec.eval_label))
        query_result = {"query_image_path": q_path, "true_label": true_label, "variants": {}}

        for v in variants:
            pred_rows = _score_variant(
                v,
                q_path,
                qn,
                dishes_df,
                dish_vectors,
                encoder,
                tagger,
                float(args.alpha),
                int(args.top_k),
                int(args.top_n),
            )
            top3_labels = [_canon(str(x.get("dish_class", x.get("dish_label", "")))) for x in pred_rows[: int(args.top_n)]]
            top1 = top3_labels[0] if top3_labels else ""
            top1_ok = bool(top1 == true_label)
            top3_ok = bool(true_label in top3_labels)
            if top1_ok:
                stats[v]["top1_correct"] += 1
            else:
                stats[v]["confusions"][(true_label, top1)] += 1
            if top3_ok:
                stats[v]["top3_correct"] += 1
            score_top1 = float(pred_rows[0].get("final_score", np.nan)) if pred_rows else float("nan")
            stats[v]["rows"].append(
                {
                    "query_image_path": q_path,
                    "true_label": true_label,
                    "pred_top1": top1,
                    "top3": top3_labels,
                    "top1_score": score_top1,
                    "top1_correct": top1_ok,
                    "top3_correct": top3_ok,
                }
            )
            query_result["variants"][v] = {"pred_top1": top1, "top3": top3_labels, "top1_score": score_top1, "top1_correct": top1_ok}
        per_query.append(query_result)

    out_variants = {}
    for v in variants:
        top1 = float(stats[v]["top1_correct"] / max(1, n))
        top3 = float(stats[v]["top3_correct"] / max(1, n))
        confusion_top = [
            {"true_label": t, "pred_label": p, "count": int(c)}
            for (t, p), c in stats[v]["confusions"].most_common(20)
        ]
        out_variants[v] = {
            "top1_accuracy": top1,
            "top3_accuracy": top3,
            "confusion_patterns_top20": confusion_top,
        }

    helps = []
    hurts = []
    for q in per_query:
        base = q["variants"]["retrieval_mlp_rerank"]
        plus = q["variants"]["retrieval_mlp_rerank_text_prototype"]
        if (not base["top1_correct"]) and plus["top1_correct"]:
            helps.append(
                {
                    "query_image_path": q["query_image_path"],
                    "true_label": q["true_label"],
                    "baseline_pred": base["pred_top1"],
                    "with_text_pred": plus["pred_top1"],
                    "baseline_score": base["top1_score"],
                    "with_text_score": plus["top1_score"],
                }
            )
        if base["top1_correct"] and (not plus["top1_correct"]):
            hurts.append(
                {
                    "query_image_path": q["query_image_path"],
                    "true_label": q["true_label"],
                    "baseline_pred": base["pred_top1"],
                    "with_text_pred": plus["pred_top1"],
                    "baseline_score": base["top1_score"],
                    "with_text_score": plus["top1_score"],
                }
            )

    base_top1 = out_variants["retrieval_mlp_rerank"]["top1_accuracy"]
    plus_top1 = out_variants["retrieval_mlp_rerank_text_prototype"]["top1_accuracy"]
    if plus_top1 >= base_top1 + 0.01:
        recommendation = "Add text prototype scoring to default pipeline."
    elif plus_top1 <= base_top1 - 0.01:
        recommendation = "Do not add text prototype to default pipeline yet."
    else:
        recommendation = "Keep as optional toggle; gains are marginal."

    report = {
        "config": {
            "manifest_csv": args.manifest_csv,
            "n_eval": int(n),
            "seed": int(args.seed),
            "alpha": float(args.alpha),
            "top_k": int(args.top_k),
            "top_n": int(args.top_n),
            "tag_head_loaded": bool(tagger is not None),
        },
        "variant_results": out_variants,
        "text_prototype_effect_vs_retrieval_mlp": {
            "help_count": int(len(helps)),
            "hurt_count": int(len(hurts)),
            "help_examples_top20": helps[:20],
            "hurt_examples_top20": hurts[:20],
        },
        "recommendation": recommendation,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v["top1_accuracy"] for k, v in out_variants.items()}, indent=2))
    print(f"Saved report: {args.out_json}")


if __name__ == "__main__":
    run()
