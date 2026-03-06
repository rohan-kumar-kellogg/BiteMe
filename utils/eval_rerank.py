import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from models.retrieval import predict_dish
from models.tag_head import CLIPTagPredictor
from models.vision import VisionEncoder
from utils.path_utils import normalize_path


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate retrieval-only vs retrieval+rerank.")
    p.add_argument("--labels_csv", default="data/labels.csv")
    p.add_argument("--manifest_csv", default="")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--n_eval", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag_head_ckpt", default="data/models/clip_mlp_tag_head.pt")
    p.add_argument("--out_json", default="reports/rerank_eval.json")
    p.add_argument("--alpha_values", default="0.15")
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_n", type=int, default=3)
    p.add_argument("--debug_alpha_logging", type=str, default="false")
    p.add_argument("--debug_query_count", type=int, default=5)
    p.add_argument("--debug_candidate_count", type=int, default=5)
    p.add_argument("--debug_alphas", default="0.05,0.30")
    return p.parse_args()


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def _retrieval_only_top3(query_emb: np.ndarray, query_path: str, dish_vectors: np.ndarray, dishes_df: pd.DataFrame):
    sims = dish_vectors.astype(np.float32, copy=False) @ query_emb.astype(np.float32)
    order = np.argsort(-sims)
    qn = normalize_path(query_path)
    out = []
    for i in order:
        cand_path = str(dishes_df.iloc[int(i)].get("image_path", ""))
        if normalize_path(cand_path) == qn:
            continue
        out.append(str(dishes_df.iloc[int(i)].get("dish_class", dishes_df.iloc[int(i)].get("dish_label", ""))))
        if len(out) >= 3:
            break
    # leakage flag is based on returned outputs after exclusion.
    return out, False


def _canon(x: str) -> str:
    return str(x).strip().lower().replace("_", " ")


def _str2bool(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def run_eval(
    *,
    labels_csv: str = "data/labels.csv",
    manifest_csv: str = "",
    data_dir: str = "data",
    n_eval: int = 200,
    seed: int = 42,
    tag_head_ckpt: str = "data/models/clip_mlp_tag_head.pt",
    out_json: str = "reports/rerank_eval.json",
    alpha_values: str = "0.15",
    top_k: int = 50,
    top_n: int = 3,
    debug_alpha_logging: bool = False,
    debug_query_count: int = 5,
    debug_candidate_count: int = 5,
    debug_alphas: str = "0.05,0.30",
) -> dict:
    random.seed(seed)
    np.random.seed(seed)

    labels_path = Path(manifest_csv) if str(manifest_csv).strip() else Path(labels_csv)
    labels = pd.read_csv(labels_path)
    if "dish_label" in labels.columns and "dish_class" not in labels.columns:
        labels["dish_class"] = labels["dish_label"]
    if "dish_family" in labels.columns and "dish_class" not in labels.columns:
        labels["dish_class"] = labels["dish_family"]
    if "dish_class" not in labels.columns and "dish_name" in labels.columns:
        labels["dish_class"] = labels["dish_name"]
    if "dish_class" not in labels.columns:
        raise ValueError("labels/manifest CSV needs one of: dish_label, dish_class, dish_family, dish_name.")
    labels = labels[labels["image_path"].map(lambda p: Path(str(p)).exists())].copy()
    labels["dish_class"] = labels["dish_class"].astype(str)

    dishes_df = pd.read_csv(Path(data_dir) / "dishes.csv")
    dish_vectors = np.load(Path(data_dir) / "dish_vectors.npy")
    dish_vectors = _normalize_rows(dish_vectors.astype(np.float32, copy=False))
    encoder = VisionEncoder()
    tagger = None
    if Path(tag_head_ckpt).exists():
        try:
            tagger = CLIPTagPredictor(tag_head_ckpt)
        except Exception:
            tagger = None

    n = min(int(n_eval), len(labels))
    sample = labels.sample(n=n, random_state=seed).reset_index(drop=True)

    alphas = [float(x.strip()) for x in str(alpha_values).split(",") if x.strip()]
    if not alphas:
        alphas = [0.15]

    r_top1 = 0
    r_top3 = 0
    rerank_hits = {a: {"top1": 0, "top3": 0} for a in alphas}
    retrieval_self_hits = 0
    rerank_self_hits = {a: 0 for a in alphas}

    for row in sample.itertuples(index=False):
        q_path = str(row.image_path)
        true_label = _canon(str(row.dish_class).strip())
        q_emb = encoder.encode_image(q_path)

        top3_ret_raw, self_hit = _retrieval_only_top3(q_emb, q_path, dish_vectors, dishes_df)
        if self_hit:
            retrieval_self_hits += 1
        top3_ret = [_canon(x) for x in top3_ret_raw]
        if len(top3_ret) > 0 and top3_ret[0] == true_label:
            r_top1 += 1
        if true_label in top3_ret:
            r_top3 += 1

        for alpha in alphas:
            exclude = {normalize_path(q_path)}
            top_rerank = predict_dish(
                q_path,
                dishes_df,
                dish_vectors,
                encoder=encoder,
                tag_predictor=tagger,
                top_k=int(top_k),
                top_n=max(int(top_k), int(top_n)),
                alpha=alpha,
                use_rerank=True,
                debug=True,
                exclude_image_paths=exclude,
            )
            qn = normalize_path(q_path)
            self_hit_rr = any(normalize_path(str(x.get("image_path", ""))) == qn for x in top_rerank)
            if self_hit_rr:
                rerank_self_hits[alpha] += 1
            pred_labels = [
                _canon(str(x["dish_class"]))
                for x in top_rerank
                if normalize_path(str(x.get("image_path", ""))) != qn
            ][: int(top_n)]
            if len(pred_labels) > 0 and pred_labels[0] == true_label:
                rerank_hits[alpha]["top1"] += 1
            if true_label in pred_labels:
                rerank_hits[alpha]["top3"] += 1

    alpha_debug_summary: dict[str, object] = {}
    if debug_alpha_logging and len(sample) > 0:
        dbg_alphas = [float(x.strip()) for x in str(debug_alphas).split(",") if x.strip()]
        if not dbg_alphas:
            dbg_alphas = [0.05, 0.30]
        dbg_alphas = [float(np.clip(a, 0.0, 1.0)) for a in dbg_alphas]
        qn = min(int(debug_query_count), len(sample))
        tn = max(1, int(debug_candidate_count))
        debug_rows = sample.head(qn)

        print("\n=== Alpha Debug Samples ===")
        change_count = 0
        per_query_changes: list[dict] = []
        for qi, drow in enumerate(debug_rows.itertuples(index=False), start=1):
            q_path = str(drow.image_path)
            q_label = str(drow.dish_class)
            print(f"\n[query {qi}] {q_path} | true_label={q_label}")

            orders: dict[float, list[tuple[int, str]]] = {}
            for a in dbg_alphas:
                top_dbg = predict_dish(
                    q_path,
                    dishes_df,
                    dish_vectors,
                    encoder=encoder,
                    tag_predictor=tagger,
                    top_k=int(top_k),
                    top_n=tn,
                    alpha=a,
                    use_rerank=True,
                    debug=True,
                    exclude_image_paths={normalize_path(q_path)},
                )
                print(f"  alpha={a:.2f}")
                order_for_alpha: list[tuple[int, str]] = []
                for rank, cand in enumerate(top_dbg, start=1):
                    cid = int(cand.get("dish_id", -1))
                    lbl = str(cand.get("dish_label", cand.get("dish_class", "")))
                    retr = float(cand.get("combined_retrieval", cand.get("sim_01", 0.0)))
                    mlp = float(cand.get("mlp_blend_score", cand.get("dish_agreement", 0.0)))
                    final = float(cand.get("final_score", 0.0))
                    print(
                        f"    rank={rank} | id={cid} | label={lbl} | retrieval={retr:.4f} | "
                        f"mlp={mlp:.4f} | final={final:.4f}"
                    )
                    order_for_alpha.append((cid, lbl))
                orders[a] = order_for_alpha

            if len(dbg_alphas) >= 2:
                ref = orders[dbg_alphas[0]]
                changed = any(orders[a] != ref for a in dbg_alphas[1:])
                if changed:
                    change_count += 1
                per_query_changes.append(
                    {
                        "query_path": q_path,
                        "changed": bool(changed),
                        "orders": {str(a): orders[a] for a in dbg_alphas},
                    }
                )

        alpha_debug_summary = {
            "alphas": dbg_alphas,
            "queries_checked": int(qn),
            "candidate_depth": int(tn),
            "queries_with_rank_change": int(change_count),
            "queries_without_rank_change": int(max(0, qn - change_count)),
            "per_query": per_query_changes,
        }
        print(
            f"\nAlpha ranking changed in {change_count}/{qn} queries "
            f"for alphas {', '.join([f'{a:.2f}' for a in dbg_alphas])}."
        )

    out = {
        "n_eval": int(n),
        "self_match_rate": float(retrieval_self_hits / max(1, n)),
        "retrieval_only": {
            "top1_dish_class_accuracy": float(r_top1 / max(1, n)),
            "top3_dish_class_accuracy": float(r_top3 / max(1, n)),
        },
        "retrieval_mlp_rerank": {},
        "delta_mlp_vs_retrieval": {},
        "alpha_sweep": [],
        "alpha_debug_summary": alpha_debug_summary,
    }
    base_top1 = float(r_top1 / max(1, n))
    base_top3 = float(r_top3 / max(1, n))
    for alpha in alphas:
        t1 = float(rerank_hits[alpha]["top1"] / max(1, n))
        t3 = float(rerank_hits[alpha]["top3"] / max(1, n))
        out["alpha_sweep"].append(
            {
                "alpha": float(alpha),
                "top1_dish_class_accuracy": t1,
                "top3_dish_class_accuracy": t3,
                "self_match_rate": float(rerank_self_hits[alpha] / max(1, n)),
                "delta_top1_vs_retrieval_only": float(t1 - base_top1),
                "delta_top3_vs_retrieval_only": float(t3 - base_top3),
            }
        )
    if out["alpha_sweep"]:
        best = out["alpha_sweep"][0]
        out["retrieval_mlp_rerank"] = {
            "top1_dish_class_accuracy": float(best["top1_dish_class_accuracy"]),
            "top3_dish_class_accuracy": float(best["top3_dish_class_accuracy"]),
        }
        out["delta_mlp_vs_retrieval"] = {
            "top1_dish_class_accuracy": float(best["delta_top1_vs_retrieval_only"]),
            "top3_dish_class_accuracy": float(best["delta_top3_vs_retrieval_only"]),
        }

    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"Saved: {out_path}")
    max_self_rate = max([out["self_match_rate"]] + [x["self_match_rate"] for x in out["alpha_sweep"]])
    if max_self_rate > 0.01:
        raise RuntimeError(
            f"Self-match leakage detected (rate={max_self_rate:.4f} > 0.01). "
            "Check path normalization and self-filter logic."
        )
    return out


def main():
    args = parse_args()
    run_eval(
        labels_csv=args.labels_csv,
        manifest_csv=args.manifest_csv,
        data_dir=args.data_dir,
        n_eval=args.n_eval,
        seed=args.seed,
        tag_head_ckpt=args.tag_head_ckpt,
        out_json=args.out_json,
        alpha_values=args.alpha_values,
        top_k=args.top_k,
        top_n=args.top_n,
        debug_alpha_logging=_str2bool(args.debug_alpha_logging),
        debug_query_count=args.debug_query_count,
        debug_candidate_count=args.debug_candidate_count,
        debug_alphas=args.debug_alphas,
    )


if __name__ == "__main__":
    main()

