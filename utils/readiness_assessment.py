import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from models.retrieval import predict_dish, predict_dish_with_confidence
from models.tag_head import CLIPTagPredictor
from models.vision import VisionEncoder
from utils.path_utils import normalize_path


def parse_args():
    p = argparse.ArgumentParser(description="Readiness assessment for retrieval+MLP reranker.")
    p.add_argument("--manifest_csv", default="images/manifest.csv")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--tag_head_ckpt", default="data/models/clip_mlp_tag_head.pt")
    p.add_argument("--n_eval", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=0.15)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_n", type=int, default=3)
    p.add_argument("--confidence_threshold", type=float, default=0.86)
    p.add_argument("--out_json", default="reports/readiness_assessment.json")
    p.add_argument("--out_failures_csv", default="reports/readiness_failures_top20.csv")
    return p.parse_args()


def _canon(x: str) -> str:
    return str(x).strip().lower().replace("_", " ")


def _resolve_label_col(df: pd.DataFrame) -> str:
    for col in ("dish_class", "dish_label", "dish_family", "dish_name"):
        if col in df.columns:
            return col
    raise ValueError("manifest CSV needs one of: dish_class, dish_label, dish_family, dish_name")


def _qstats(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    arr = np.asarray(xs, dtype=np.float32)
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


def _to_top3_payload(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows[:3]:
        out.append(
            {
                "label": str(r.get("dish_class", r.get("dish_label", ""))),
                "retrieval_score": float(r.get("combined_retrieval", r.get("sim_01", 0.0))),
                "mlp_score": float(r.get("mlp_blend_score", r.get("dish_agreement", 0.0))),
                "final_score": float(r.get("final_score", 0.0)),
            }
        )
    return out


def _threshold_recommendation(rows: list[dict]) -> dict:
    if not rows:
        return {"recommended_threshold": None, "reason": "no rows"}

    thresholds = np.round(np.arange(0.50, 0.96, 0.01), 2)
    best = None
    total = len(rows)
    for t in thresholds:
        accepted = [r for r in rows if r["top1_final_score"] >= float(t)]
        if not accepted:
            continue
        coverage = len(accepted) / total
        selective_acc = float(np.mean([1.0 if r["top1_correct"] else 0.0 for r in accepted]))
        score = selective_acc * coverage
        row = {
            "threshold": float(t),
            "coverage": float(coverage),
            "accepted_count": int(len(accepted)),
            "selective_top1_accuracy": float(selective_acc),
            "score": float(score),
        }
        if best is None:
            best = row
            continue
        if row["selective_top1_accuracy"] >= 0.90 and best["selective_top1_accuracy"] < 0.90:
            best = row
            continue
        if row["selective_top1_accuracy"] >= 0.90 and best["selective_top1_accuracy"] >= 0.90:
            if row["coverage"] > best["coverage"]:
                best = row
            continue
        if best["selective_top1_accuracy"] < 0.90 and row["score"] > best["score"]:
            best = row

    if best is None:
        return {"recommended_threshold": None, "reason": "no accepted thresholds"}
    return best


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

    rows = []
    confusion = Counter()
    incorrect_examples = []
    self_hits = 0
    same_class_group_total = 0
    same_class_identical_count = 0
    same_class_near_identical_count = 0
    eps_identical = 1e-8
    eps_near = 1e-3

    for rec in sample.itertuples(index=False):
        q_path = str(rec.image_path)
        true_label = _canon(str(rec.eval_label))
        qn = normalize_path(q_path)
        preds = predict_dish(
            q_path,
            dishes_df,
            dish_vectors,
            encoder=encoder,
            tag_predictor=tagger,
            top_k=int(args.top_k),
            top_n=int(args.top_k),
            alpha=float(args.alpha),
            use_rerank=True,
            debug=True,
            exclude_image_paths={qn},
        )
        if any(normalize_path(str(x.get("image_path", ""))) == qn for x in preds):
            self_hits += 1

        top3 = preds[: max(1, int(args.top_n))]
        pred_top1 = _canon(str(top3[0].get("dish_class", top3[0].get("dish_label", "")))) if top3 else ""
        pred_top3_labels = [_canon(str(x.get("dish_class", x.get("dish_label", "")))) for x in top3]
        top1_correct = bool(pred_top1 == true_label)
        top3_correct = bool(true_label in pred_top3_labels)

        top1_retrieval = float(top3[0].get("combined_retrieval", top3[0].get("sim_01", 0.0))) if top3 else 0.0
        top1_mlp = float(top3[0].get("mlp_blend_score", top3[0].get("dish_agreement", 0.0))) if top3 else 0.0
        top1_final = float(top3[0].get("final_score", 0.0)) if top3 else 0.0

        rows.append(
            {
                "query_image_path": q_path,
                "true_label": true_label,
                "pred_top1": pred_top1,
                "pred_top3": pred_top3_labels,
                "top1_correct": top1_correct,
                "top3_correct": top3_correct,
                "top1_retrieval_score": top1_retrieval,
                "top1_mlp_score": top1_mlp,
                "top1_final_score": top1_final,
                "top3_payload": _to_top3_payload(top3),
            }
        )

        if not top1_correct:
            confusion[(true_label, pred_top1)] += 1
            incorrect_examples.append(rows[-1])

        class_scores: dict[str, list[float]] = defaultdict(list)
        for cand in preds:
            cls = _canon(str(cand.get("dish_class", cand.get("dish_label", ""))))
            mlp_score = float(cand.get("mlp_blend_score", cand.get("dish_agreement", 0.0)))
            class_scores[cls].append(mlp_score)
        for scores in class_scores.values():
            if len(scores) < 2:
                continue
            same_class_group_total += 1
            smin = float(np.min(scores))
            smax = float(np.max(scores))
            if abs(smax - smin) <= eps_identical:
                same_class_identical_count += 1
            if abs(smax - smin) <= eps_near:
                same_class_near_identical_count += 1

    total = len(rows)
    top1 = float(np.mean([1.0 if r["top1_correct"] else 0.0 for r in rows])) if rows else 0.0
    top3 = float(np.mean([1.0 if r["top3_correct"] else 0.0 for r in rows])) if rows else 0.0
    self_match_rate = float(self_hits / max(1, total))

    confusion_top = [
        {"true_label": t, "pred_label": p, "count": int(c)}
        for (t, p), c in confusion.most_common(20)
    ]

    incorrect_sorted = sorted(incorrect_examples, key=lambda x: x["top1_final_score"], reverse=True)
    top20_incorrect = incorrect_sorted[:20]
    fail_df = pd.DataFrame(
        [
            {
                "query_image_path": x["query_image_path"],
                "true_label": x["true_label"],
                "pred_top1": x["pred_top1"],
                "pred_top3": json.dumps(x["pred_top3"]),
                "top1_retrieval_score": x["top1_retrieval_score"],
                "top1_mlp_score": x["top1_mlp_score"],
                "top1_final_score": x["top1_final_score"],
                "top3_payload": json.dumps(x["top3_payload"]),
            }
            for x in top20_incorrect
        ]
    )
    Path(args.out_failures_csv).parent.mkdir(parents=True, exist_ok=True)
    fail_df.to_csv(args.out_failures_csv, index=False)

    correct_scores = [r["top1_final_score"] for r in rows if r["top1_correct"]]
    incorrect_scores = [r["top1_final_score"] for r in rows if not r["top1_correct"]]
    threshold = _threshold_recommendation(rows)
    abstain_rows = []
    false_confident_errors = 0
    for rec in sample.itertuples(index=False):
        q_path = str(rec.image_path)
        qn = normalize_path(q_path)
        true_label = _canon(str(rec.eval_label))
        c = predict_dish_with_confidence(
            q_path,
            dishes_df,
            dish_vectors,
            encoder=encoder,
            tag_predictor=tagger,
            top_k=int(args.top_k),
            top_n=int(args.top_n),
            alpha=float(args.alpha),
            use_rerank=True,
            exclude_image_paths={qn},
            confidence_threshold=float(args.confidence_threshold),
        )
        pred = _canon(str(c.get("predicted_label", "")))
        abstained = bool(c.get("abstained", False))
        correct_if_accepted = (pred == true_label) if not abstained else False
        if (not abstained) and (not correct_if_accepted):
            false_confident_errors += 1
        abstain_rows.append({"abstained": abstained, "correct_if_accepted": bool(correct_if_accepted)})
    accepted = [x for x in abstain_rows if not x["abstained"]]
    coverage = float(len(accepted) / max(1, len(abstain_rows)))
    selective_acc = float(np.mean([1.0 if x["correct_if_accepted"] else 0.0 for x in accepted])) if accepted else 0.0
    abstain_rate = float(1.0 - coverage)

    class_score_analysis = {
        "same_class_groups_evaluated": int(same_class_group_total),
        "identical_score_groups": int(same_class_identical_count),
        "near_identical_score_groups_eps_1e-3": int(same_class_near_identical_count),
        "identical_group_rate": float(same_class_identical_count / max(1, same_class_group_total)),
        "near_identical_group_rate": float(same_class_near_identical_count / max(1, same_class_group_total)),
        "interpretation": (
            "High near-identical rates indicate class-level scoring dominates and weak instance discrimination."
        ),
    }

    if top1 >= 0.75 and self_match_rate <= 0.01:
        readiness = "demo-ready"
    else:
        readiness = "not-demo-ready"
    production_ready = bool(top1 >= 0.90 and top3 >= 0.97 and self_match_rate <= 0.001)

    report = {
        "config": {
            "manifest_csv": args.manifest_csv,
            "n_eval": int(total),
            "seed": int(args.seed),
            "alpha": float(args.alpha),
            "top_k": int(args.top_k),
            "top_n": int(args.top_n),
            "tag_head_loaded": bool(tagger is not None),
        },
        "metrics": {
            "top1_accuracy": top1,
            "top3_accuracy": top3,
            "self_match_rate": self_match_rate,
        },
        "confusion_patterns_top20": confusion_top,
        "failure_examples_top20": top20_incorrect,
        "class_level_score_analysis": class_score_analysis,
        "confidence_analysis": {
            "top1_final_score_correct": _qstats(correct_scores),
            "top1_final_score_incorrect": _qstats(incorrect_scores),
            "recommended_abstain_threshold": threshold,
            "abstain_metrics_at_threshold": {
                "confidence_threshold": float(args.confidence_threshold),
                "coverage": coverage,
                "selective_top1_accuracy": selective_acc,
                "abstain_rate": abstain_rate,
                "false_confident_error_count": int(false_confident_errors),
            },
        },
        "readiness_assessment": {
            "demo_status": readiness,
            "production_ready": production_ready,
            "notes": [
                "Demo-ready if top1 is stable and failure modes are acceptable for user expectations.",
                "Production-ready requires stronger calibrated confidence and substantially higher top1/top3.",
            ],
        },
        "artifacts": {
            "failures_csv": args.out_failures_csv,
        },
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report["metrics"], indent=2))
    print(f"Saved report: {args.out_json}")
    print(f"Saved failures: {args.out_failures_csv}")


if __name__ == "__main__":
    run()
