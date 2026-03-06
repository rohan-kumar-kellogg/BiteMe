import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from models.retrieval import predict_dish
from models.tag_head import CLIPTagPredictor
from models.vision import VisionEncoder


def parse_args():
    p = argparse.ArgumentParser(description="Diagnose dish recognition quality.")
    p.add_argument("--labels_csv", default="data/labels.csv")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--n_eval", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--tag_head_ckpt", default="data/models/clip_mlp_tag_head.pt")
    p.add_argument("--report_dir", default="reports")
    return p.parse_args()


def _norm_rows(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def _canon(x: str) -> str:
    return str(x).strip().lower().replace("_", " ")


def _value_counts_summary(series: pd.Series, top_k: int = 20):
    vc = series.astype(str).value_counts()
    return {
        "n_unique": int(vc.shape[0]),
        "top_counts": [{"label": str(k), "count": int(v)} for k, v in vc.head(top_k).items()],
        "sparsity": {
            "lt_5": int((vc < 5).sum()),
            "lt_10": int((vc < 10).sum()),
            "lt_20": int((vc < 20).sum()),
        },
    }


def _print_distribution(name: str, summary: dict):
    print(f"\n{name} distribution:")
    print(f"  unique classes: {summary['n_unique']}")
    print("  top-20:")
    for row in summary["top_counts"]:
        print(f"    {row['label']}: {row['count']}")
    sp = summary["sparsity"]
    print("  sparsity:")
    print(f"    <5 samples:  {sp['lt_5']}")
    print(f"    <10 samples: {sp['lt_10']}")
    print(f"    <20 samples: {sp['lt_20']}")


def _load_labels(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "dish_class" not in df.columns:
        if "dish_name" in df.columns:
            df["dish_class"] = df["dish_name"]
        else:
            raise ValueError("labels CSV must contain `dish_class` (or `dish_name`).")
    if "cuisine" not in df.columns:
        raise ValueError("labels CSV must contain `cuisine`.")
    if "image_path" not in df.columns:
        raise ValueError("labels CSV must contain `image_path`.")
    df["image_path"] = df["image_path"].astype(str)
    df = df[df["image_path"].map(lambda p: Path(p).exists())].copy()
    if len(df) == 0:
        raise ValueError("No valid image paths found in labels CSV.")
    df["dish_class"] = df["dish_class"].astype(str).str.strip()
    df["cuisine"] = df["cuisine"].astype(str).str.strip()
    return df.reset_index(drop=True)


def _retrieval_topk(
    query_emb: np.ndarray,
    query_image_path: str,
    dishes_df: pd.DataFrame,
    dish_vectors: np.ndarray,
    top_k: int,
):
    sims = _norm_rows(dish_vectors.astype(np.float32, copy=False)) @ query_emb.astype(np.float32, copy=False)
    order = np.argsort(-sims)

    picks = []
    for i in order:
        row = dishes_df.iloc[int(i)]
        cand_path = str(row.get("image_path", ""))
        if cand_path == query_image_path:
            continue
        picks.append(
            {
                "index": int(i),
                "dish_class": str(row.get("dish_class", row.get("dish_label", ""))),
                "cuisine": str(row.get("cuisine", "")),
                "image_path": cand_path,
                "score": float(sims[int(i)]),
            }
        )
        if len(picks) >= top_k:
            break
    return picks


def _topk_metrics(results: list[list[str]], truths: list[str], k: int):
    ok = 0
    for pred, y in zip(results, truths):
        if _canon(y) in [_canon(x) for x in pred[:k]]:
            ok += 1
    return float(ok / max(1, len(truths)))


def _save_failure_bundle(
    failures: list[dict],
    out_dir: Path,
    max_failures: int = 20,
    seed: int = 42,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    if not failures:
        return {"saved_failures": 0}

    rng = random.Random(seed)
    picks = failures if len(failures) <= max_failures else rng.sample(failures, k=max_failures)
    manifest = []
    for idx, f in enumerate(picks):
        case_dir = out_dir / f"case_{idx:02d}"
        case_dir.mkdir(parents=True, exist_ok=True)

        q_src = Path(f["query_image_path"])
        q_dst = case_dir / f"query{q_src.suffix.lower() or '.jpg'}"
        if q_src.exists():
            shutil.copy2(q_src, q_dst)

        neighbors = []
        for j, n in enumerate(f["neighbors_top5"]):
            src = Path(n["image_path"])
            ext = src.suffix.lower() or ".jpg"
            dst = case_dir / f"neighbor_{j+1:02d}{ext}"
            if src.exists():
                shutil.copy2(src, dst)
            n2 = dict(n)
            n2["saved_path"] = str(dst)
            neighbors.append(n2)

        meta = {
            "query_image_path": f["query_image_path"],
            "query_saved_path": str(q_dst),
            "true_dish_class": f["true_dish_class"],
            "true_cuisine": f["true_cuisine"],
            "pred_top1_dish_class": f["pred_top1_dish_class"],
            "neighbors_top5": neighbors,
        }
        with open(case_dir / "meta.json", "w") as fh:
            json.dump(meta, fh, indent=2)
        manifest.append(meta)

    with open(out_dir / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    return {"saved_failures": int(len(manifest)), "path": str(out_dir)}


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    report_dir = Path(args.report_dir)
    failures_dir = report_dir / "failures"
    report_dir.mkdir(parents=True, exist_ok=True)

    labels_df = _load_labels(args.labels_csv)
    dish_summary = _value_counts_summary(labels_df["dish_class"], top_k=20)
    cuisine_summary = _value_counts_summary(labels_df["cuisine"], top_k=20)
    _print_distribution("dish_class", dish_summary)
    _print_distribution("cuisine", cuisine_summary)

    dishes_df = pd.read_csv(Path(args.data_dir) / "dishes.csv")
    dish_vectors = np.load(Path(args.data_dir) / "dish_vectors.npy")
    dish_vectors = _norm_rows(dish_vectors.astype(np.float32, copy=False))
    if len(dishes_df) != dish_vectors.shape[0]:
        raise ValueError("data/dishes.csv and data/dish_vectors.npy row counts do not match.")

    n_eval = min(int(args.n_eval), len(labels_df))
    eval_df = labels_df.sample(n=n_eval, random_state=args.seed).reset_index(drop=True)

    encoder = VisionEncoder()
    tagger = None
    tag_ckpt = Path(args.tag_head_ckpt)
    if tag_ckpt.exists():
        try:
            tagger = CLIPTagPredictor(str(tag_ckpt))
        except Exception:
            tagger = None

    truths = []
    retrieval_preds = []
    rerank_preds = []
    failure_rows = []

    for row in eval_df.itertuples(index=False):
        q_path = str(row.image_path)
        true_dish = _canon(str(row.dish_class).strip())
        true_cuisine = str(row.cuisine).strip()

        emb = encoder.encode_image(q_path)
        ret_top = _retrieval_topk(emb, q_path, dishes_df, dish_vectors, top_k=max(5, args.top_k))
        ret_labels = [str(x["dish_class"]) for x in ret_top]
        retrieval_preds.append(ret_labels)

        rr_top3 = predict_dish(
            q_path,
            dishes_df=dishes_df,
            dish_vectors=dish_vectors,
            encoder=encoder,
            tag_predictor=tagger,
            top_k=args.top_k,
            top_n=3,
            alpha=0.15,
            use_rerank=True,
        )
        rr_labels = [str(x["dish_class"]) for x in rr_top3]
        rerank_preds.append(rr_labels)
        truths.append(true_dish)

        if len(ret_labels) == 0 or _canon(ret_labels[0]) != true_dish:
            failure_rows.append(
                {
                    "query_image_path": q_path,
                    "true_dish_class": true_dish,
                    "true_cuisine": true_cuisine,
                    "pred_top1_dish_class": ret_labels[0] if ret_labels else "",
                    "neighbors_top5": ret_top[:5],
                }
            )

    retrieval_top1 = _topk_metrics(retrieval_preds, truths, k=1)
    retrieval_top3 = _topk_metrics(retrieval_preds, truths, k=3)
    rerank_top1 = _topk_metrics(rerank_preds, truths, k=1)
    rerank_top3 = _topk_metrics(rerank_preds, truths, k=3)

    saved_failures = _save_failure_bundle(failure_rows, failures_dir, max_failures=20, seed=args.seed)

    summary = {
        "n_eval": int(n_eval),
        "label_distribution": {
            "dish_class": dish_summary,
            "cuisine": cuisine_summary,
        },
        "retrieval_only": {
            "top1_dish_class_accuracy": retrieval_top1,
            "top3_dish_class_accuracy": retrieval_top3,
            "n_failures_top1": int((1.0 - retrieval_top1) * n_eval),
        },
        "retrieval_rerank": {
            "top1_dish_class_accuracy": rerank_top1,
            "top3_dish_class_accuracy": rerank_top3,
            "delta_vs_retrieval_only": {
                "top1": float(rerank_top1 - retrieval_top1),
                "top3": float(rerank_top3 - retrieval_top3),
            },
        },
        "failure_report": saved_failures,
    }

    print("\nEvaluation summary:")
    print(json.dumps(summary["retrieval_only"], indent=2))
    print(json.dumps(summary["retrieval_rerank"], indent=2))

    out_path = report_dir / "diagnose_summary.json"
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nSaved summary: {out_path}")


if __name__ == "__main__":
    main()

