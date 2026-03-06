import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from models.hard_negative_reranker import HardNegativePairPredictor
from models.retrieval import RERANK_DISH_W, RERANK_PROTEIN_W, predict_dish
from models.tag_head import CLIPTagPredictor
from models.vision import VisionEncoder
from utils.path_utils import normalize_path


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate blended retrieval + MLP + pairwise reranker.")
    p.add_argument("--manifest_csv", default="images/manifest.csv")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--tag_head_ckpt", default="data/models/clip_mlp_tag_head.pt")
    p.add_argument("--pair_reranker_ckpt", default="data/models/hard_negative_pair_reranker.pt")
    p.add_argument("--n_eval", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--grid_step", type=float, default=0.1)
    p.add_argument("--out_json", default="reports/rerank_pair_blend_eval.json")
    return p.parse_args()


def _canon(x: str) -> str:
    return str(x).strip().lower().replace("_", " ")


def _resolve_label_col(df: pd.DataFrame) -> str:
    for col in ("dish_label", "dish_class", "dish_family", "dish_name"):
        if col in df.columns:
            return col
    raise ValueError("manifest needs one of: dish_label, dish_class, dish_family, dish_name")


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


def _mlp_from_row(row: dict) -> float:
    comps = []
    dish = float(row.get("dish_agreement", np.nan))
    prot = float(row.get("protein_agreement", np.nan))
    if not np.isnan(dish):
        comps.append((float(RERANK_DISH_W), dish))
    if not np.isnan(prot):
        comps.append((float(RERANK_PROTEIN_W), prot))
    if not comps:
        return 0.0
    tw = float(sum(w for w, _ in comps))
    return float(sum((w / tw) * s for w, s in comps))


def _weight_grid(step: float) -> list[tuple[float, float, float]]:
    vals = np.arange(0.0, 1.0 + 1e-9, float(step))
    out = []
    for wr in vals:
        for wm in vals:
            wp = 1.0 - float(wr) - float(wm)
            if wp < -1e-9:
                continue
            if wp < 0:
                wp = 0.0
            if abs((wr + wm + wp) - 1.0) <= 1e-6:
                out.append((round(float(wr), 4), round(float(wm), 4), round(float(wp), 4)))
    return out


def _variant_match(name: str, wr: float, wm: float, wp: float) -> bool:
    if name == "retrieval_mlp":
        return wr > 0 and wm > 0 and wp == 0
    if name == "retrieval_pair":
        return wr > 0 and wm == 0 and wp > 0
    if name == "mlp_pair":
        return wr == 0 and wm > 0 and wp > 0
    if name == "retrieval_mlp_pair":
        return wr > 0 and wm > 0 and wp > 0
    return False


def _pick_threshold(rows: list[dict]) -> dict:
    thresholds = np.round(np.arange(0.50, 0.96, 0.01), 2)
    best = None
    total = len(rows)
    for t in thresholds:
        accepted = [r for r in rows if r["top1_score"] >= float(t)]
        if not accepted:
            continue
        coverage = len(accepted) / total
        sel_acc = float(np.mean([1.0 if r["top1_correct"] else 0.0 for r in accepted]))
        score = sel_acc * coverage
        cand = {
            "threshold": float(t),
            "coverage": float(coverage),
            "selective_top1_accuracy": float(sel_acc),
            "abstain_rate": float(1.0 - coverage),
            "false_confident_error_count": int(sum(1 for r in accepted if not r["top1_correct"])),
            "score": float(score),
        }
        if best is None:
            best = cand
            continue
        if cand["selective_top1_accuracy"] >= 0.90 and best["selective_top1_accuracy"] < 0.90:
            best = cand
            continue
        if cand["selective_top1_accuracy"] >= 0.90 and best["selective_top1_accuracy"] >= 0.90:
            if cand["coverage"] > best["coverage"]:
                best = cand
            continue
        if best["selective_top1_accuracy"] < 0.90 and cand["score"] > best["score"]:
            best = cand
    return best or {
        "threshold": None,
        "coverage": 0.0,
        "selective_top1_accuracy": 0.0,
        "abstain_rate": 1.0,
        "false_confident_error_count": 0,
    }


def _evaluate_cached_weight(query_candidates: list[dict], weights: tuple[float, float, float]) -> dict:
    wr, wm, wp = weights
    n = len(query_candidates)
    confusions = Counter()
    top1_hits = 0
    top3_hits = 0
    same_class_total = 0
    same_class_ident = 0
    same_class_near = 0
    score_rows = []

    for q in query_candidates:
        scored = []
        for c in q["candidates"]:
            s = float(wr * c["retrieval_score"] + wm * c["mlp_score"] + wp * c["pair_score"])
            scored.append({**c, "blend_score": s})
        scored.sort(key=lambda x: x["blend_score"], reverse=True)
        top3 = scored[:3]
        top3_lbls = [x["dish_class"] for x in top3]
        p1 = top3_lbls[0] if top3_lbls else ""
        ok1 = bool(p1 == q["true_label"])
        ok3 = bool(q["true_label"] in top3_lbls)
        if ok1:
            top1_hits += 1
        else:
            confusions[(q["true_label"], p1)] += 1
        if ok3:
            top3_hits += 1
        top1_score = float(top3[0]["blend_score"]) if top3 else 0.0
        score_rows.append({"top1_score": top1_score, "top1_correct": ok1})

        class_scores: dict[str, list[float]] = defaultdict(list)
        for c in scored[:20]:
            class_scores[c["dish_class"]].append(float(c["blend_score"]))
        for vals in class_scores.values():
            if len(vals) < 2:
                continue
            same_class_total += 1
            if abs(max(vals) - min(vals)) <= 1e-8:
                same_class_ident += 1
            if abs(max(vals) - min(vals)) <= 1e-3:
                same_class_near += 1

    correct_scores = [r["top1_score"] for r in score_rows if r["top1_correct"]]
    incorrect_scores = [r["top1_score"] for r in score_rows if not r["top1_correct"]]
    return {
        "weights": {"retrieval": float(wr), "mlp": float(wm), "pair": float(wp)},
        "top1_accuracy": float(top1_hits / max(1, n)),
        "top3_accuracy": float(top3_hits / max(1, n)),
        "confusion_pairs_top20": [{"true_label": t, "pred_label": p, "count": int(c)} for (t, p), c in confusions.most_common(20)],
        "discriminativeness": {
            "same_class_groups": int(same_class_total),
            "identical_rate": float(same_class_ident / max(1, same_class_total)),
            "near_identical_rate_eps_1e3": float(same_class_near / max(1, same_class_total)),
        },
        "score_distribution": {
            "correct_top1_scores": _qstats(correct_scores),
            "incorrect_top1_scores": _qstats(incorrect_scores),
        },
        "abstain_rows": score_rows,
    }


def main():
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
    tagger = CLIPTagPredictor(args.tag_head_ckpt) if Path(args.tag_head_ckpt).exists() else None
    pair = HardNegativePairPredictor(args.pair_reranker_ckpt) if Path(args.pair_reranker_ckpt).exists() else None
    if tagger is None:
        raise RuntimeError("Tag head checkpoint missing; baseline retrieval+MLP cannot be evaluated.")
    if pair is None:
        raise RuntimeError("Pair reranker checkpoint missing; run hard-negative experiment first.")

    n = min(int(args.n_eval), len(manifest))
    sample = manifest.sample(n=n, random_state=int(args.seed)).reset_index(drop=True)

    query_candidates: list[dict] = []
    for r in sample.itertuples(index=False):
        q_path = str(r.image_path)
        true_label = _canon(str(r.eval_label))
        qn = normalize_path(q_path)
        preds = predict_dish(
            q_path,
            dishes_df,
            dish_vectors,
            encoder=encoder,
            tag_predictor=tagger,
            pair_reranker=pair,
            top_k=int(args.top_k),
            top_n=int(args.top_k),
            alpha=0.15,
            use_rerank=True,
            exclude_image_paths={qn},
            debug=True,
        )
        cands = []
        for row in preds:
            cands.append(
                {
                    "dish_label": str(row.get("dish_label", row.get("dish_class", ""))),
                    "dish_class": _canon(str(row.get("dish_class", row.get("dish_label", "")))),
                    "retrieval_score": float(row.get("combined_retrieval", row.get("sim_01", 0.0))),
                    "mlp_score": float(_mlp_from_row(row)),
                    "pair_score": float(row.get("pair_agreement", 0.0)),
                }
            )
        query_candidates.append({"query_image_path": q_path, "true_label": true_label, "candidates": cands})

    grid = _weight_grid(float(args.grid_step))
    sweep_rows = []
    full_results = {}
    for wr, wm, wp in grid:
        confusions = Counter()
        top1_hits = 0
        top3_hits = 0
        same_class_total = 0
        same_class_ident = 0
        same_class_near = 0
        score_rows = []

        for q in query_candidates:
            cands = q["candidates"]
            scored = []
            for c in cands:
                s = float(wr * c["retrieval_score"] + wm * c["mlp_score"] + wp * c["pair_score"])
                scored.append({**c, "blend_score": s})
            scored.sort(key=lambda x: x["blend_score"], reverse=True)
            top3 = scored[:3]
            top3_lbls = [x["dish_class"] for x in top3]
            p1 = top3_lbls[0] if top3_lbls else ""
            ok1 = bool(p1 == q["true_label"])
            ok3 = bool(q["true_label"] in top3_lbls)
            if ok1:
                top1_hits += 1
            else:
                confusions[(q["true_label"], p1)] += 1
            if ok3:
                top3_hits += 1
            top1_score = float(top3[0]["blend_score"]) if top3 else 0.0
            score_rows.append({"top1_score": top1_score, "top1_correct": ok1})

            class_scores: dict[str, list[float]] = defaultdict(list)
            for c in scored[:20]:
                class_scores[c["dish_class"]].append(float(c["blend_score"]))
            for vals in class_scores.values():
                if len(vals) < 2:
                    continue
                same_class_total += 1
                if abs(max(vals) - min(vals)) <= 1e-8:
                    same_class_ident += 1
                if abs(max(vals) - min(vals)) <= 1e-3:
                    same_class_near += 1

        top1 = float(top1_hits / max(1, n))
        top3 = float(top3_hits / max(1, n))
        correct_scores = [r["top1_score"] for r in score_rows if r["top1_correct"]]
        incorrect_scores = [r["top1_score"] for r in score_rows if not r["top1_correct"]]

        key = f"{wr:.2f}_{wm:.2f}_{wp:.2f}"
        res = {
            "weights": {"retrieval": wr, "mlp": wm, "pair": wp},
            "top1_accuracy": top1,
            "top3_accuracy": top3,
            "confusion_pairs_top20": [{"true_label": t, "pred_label": p, "count": int(c)} for (t, p), c in confusions.most_common(20)],
            "discriminativeness": {
                "same_class_groups": int(same_class_total),
                "identical_rate": float(same_class_ident / max(1, same_class_total)),
                "near_identical_rate_eps_1e3": float(same_class_near / max(1, same_class_total)),
            },
            "score_distribution": {
                "correct_top1_scores": _qstats(correct_scores),
                "incorrect_top1_scores": _qstats(incorrect_scores),
            },
            "abstain_rows": score_rows,
        }
        full_results[key] = res
        sweep_rows.append({"weights": res["weights"], "top1_accuracy": top1, "top3_accuracy": top3})

    variant_names = ["retrieval_mlp", "retrieval_pair", "mlp_pair", "retrieval_mlp_pair"]
    best_by_variant = {}
    for v in variant_names:
        cands = [r for r in full_results.values() if _variant_match(v, r["weights"]["retrieval"], r["weights"]["mlp"], r["weights"]["pair"])]
        if not cands:
            continue
        cands = sorted(cands, key=lambda x: (x["top1_accuracy"], x["top3_accuracy"]), reverse=True)
        best_by_variant[v] = cands[0]

    baseline = None
    for r in full_results.values():
        w = r["weights"]
        if abs(w["retrieval"] - 0.85) < 1e-9 and abs(w["mlp"] - 0.15) < 1e-9 and abs(w["pair"] - 0.0) < 1e-9:
            baseline = r
            break
    if baseline is None:
        baseline = best_by_variant.get("retrieval_mlp")
    baseline_current = _evaluate_cached_weight(query_candidates, (0.85, 0.15, 0.0))

    all_sorted = sorted(full_results.values(), key=lambda x: (x["top1_accuracy"], x["top3_accuracy"]), reverse=True)
    best_overall = all_sorted[0] if all_sorted else None

    abstain_analysis = {}
    if best_overall is not None:
        thr = _pick_threshold(best_overall["abstain_rows"])
        fixed_t = 0.86
        accepted = [r for r in best_overall["abstain_rows"] if r["top1_score"] >= fixed_t]
        cov = float(len(accepted) / max(1, len(best_overall["abstain_rows"])))
        sel = float(np.mean([1.0 if r["top1_correct"] else 0.0 for r in accepted])) if accepted else 0.0
        abstain_analysis = {
            "recommended_threshold": thr,
            "metrics_at_0_86": {
                "coverage": cov,
                "selective_top1_accuracy": sel,
                "abstain_rate": float(1.0 - cov),
                "false_confident_error_count": int(sum(1 for r in accepted if not r["top1_correct"])),
            },
        }

    report = {
        "config": {
            "manifest_csv": args.manifest_csv,
            "n_eval": int(n),
            "grid_step": float(args.grid_step),
            "top_k": int(args.top_k),
        },
        "sweep_summary": sweep_rows,
        "best_by_variant": {
            k: {
                "weights": v["weights"],
                "top1_accuracy": v["top1_accuracy"],
                "top3_accuracy": v["top3_accuracy"],
                "confusion_pairs_top20": v["confusion_pairs_top20"],
                "discriminativeness": v["discriminativeness"],
                "score_distribution": v["score_distribution"],
            }
            for k, v in best_by_variant.items()
        },
        "baseline_retrieval_mlp": {
            "weights": baseline["weights"] if baseline else None,
            "top1_accuracy": baseline["top1_accuracy"] if baseline else None,
            "top3_accuracy": baseline["top3_accuracy"] if baseline else None,
        },
        "baseline_current_anchor_0_85_0_15_0_0": {
            "weights": baseline_current["weights"],
            "top1_accuracy": baseline_current["top1_accuracy"],
            "top3_accuracy": baseline_current["top3_accuracy"],
        },
        "best_overall": {
            "weights": best_overall["weights"] if best_overall else None,
            "top1_accuracy": best_overall["top1_accuracy"] if best_overall else None,
            "top3_accuracy": best_overall["top3_accuracy"] if best_overall else None,
            "confusion_pairs_top20": best_overall["confusion_pairs_top20"] if best_overall else [],
            "discriminativeness": best_overall["discriminativeness"] if best_overall else {},
            "score_distribution": best_overall["score_distribution"] if best_overall else {},
        },
        "abstain_analysis_for_best_overall": abstain_analysis,
        "delta_best_overall_vs_baseline": {
            "top1_accuracy": float((best_overall["top1_accuracy"] - baseline["top1_accuracy"]) if (best_overall and baseline) else 0.0),
            "top3_accuracy": float((best_overall["top3_accuracy"] - baseline["top3_accuracy"]) if (best_overall and baseline) else 0.0),
        },
        "delta_best_overall_vs_current_anchor": {
            "top1_accuracy": float((best_overall["top1_accuracy"] - baseline_current["top1_accuracy"]) if best_overall else 0.0),
            "top3_accuracy": float((best_overall["top3_accuracy"] - baseline_current["top3_accuracy"]) if best_overall else 0.0),
        },
    }

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({"baseline_top1": report["baseline_retrieval_mlp"]["top1_accuracy"], "best_top1": report["best_overall"]["top1_accuracy"]}, indent=2))
    print(f"Saved report: {out}")


if __name__ == "__main__":
    main()
