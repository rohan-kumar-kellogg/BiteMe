import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from models.hard_negative_reranker import HardNegativePairPredictor, HardNegativePairReranker
from models.retrieval import predict_dish
from models.tag_head import CLIPTagPredictor
from models.vision import VisionEncoder
from utils.path_utils import normalize_path


def parse_args():
    p = argparse.ArgumentParser(description="Hard-negative-only reranker retraining experiment.")
    p.add_argument("--manifest_csv", default="images/manifest.csv")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--tag_head_ckpt", default="data/models/clip_mlp_tag_head.pt")
    p.add_argument("--out_reranker_ckpt", default="data/models/hard_negative_pair_reranker.pt")
    p.add_argument("--report_json", default="reports/rerank_hard_negative_eval.json")
    p.add_argument("--n_eval", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top_k_mine", type=int, default=20)
    p.add_argument("--hard_negs_per_query", type=int, default=3)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--margin", type=float, default=0.10)
    p.add_argument("--alpha", type=float, default=0.15)
    p.add_argument("--confidence_threshold", type=float, default=0.86)
    return p.parse_args()


def _canon(x: str) -> str:
    return str(x).strip().lower().replace("_", " ")


def _resolve_label_col(df: pd.DataFrame) -> str:
    for col in ("dish_label", "dish_class", "dish_family", "dish_name"):
        if col in df.columns:
            return col
    raise ValueError("manifest needs one of: dish_label, dish_class, dish_family, dish_name")


def _mine_triples(
    dishes_df: pd.DataFrame,
    vecs: np.ndarray,
    labels: np.ndarray,
    top_k_mine: int,
    hard_negs_per_query: int,
) -> list[tuple[int, int, int]]:
    triples: list[tuple[int, int, int]] = []
    n = len(dishes_df)
    label_to_indices: dict[str, np.ndarray] = {}
    for lbl in np.unique(labels):
        label_to_indices[str(lbl)] = np.where(labels == lbl)[0]
    sims_all = vecs @ vecs.T
    for qi in range(n):
        q_label = str(labels[qi])
        order = np.argsort(-sims_all[qi])
        order = order[order != qi]
        top = order[: max(1, int(top_k_mine))]
        pos_pool = [int(j) for j in top if str(labels[int(j)]) == q_label]
        if not pos_pool:
            same_cls = label_to_indices.get(q_label, np.array([], dtype=np.int64))
            same_cls = same_cls[same_cls != qi]
            if len(same_cls) == 0:
                continue
            best = int(same_cls[np.argmax(sims_all[qi][same_cls])])
            pos_pool = [best]
        pos_idx = int(pos_pool[0])
        neg_pool = [int(j) for j in top if str(labels[int(j)]) != q_label]
        if not neg_pool:
            continue
        for nj in neg_pool[: max(1, int(hard_negs_per_query))]:
            triples.append((qi, pos_idx, int(nj)))
    return triples


def _train_pair_reranker(
    vecs: np.ndarray,
    triples: list[tuple[int, int, int]],
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    margin: float,
    out_ckpt: str,
) -> dict:
    rng = np.random.default_rng(seed)
    arr = np.asarray(triples, dtype=np.int64)
    if len(arr) == 0:
        raise RuntimeError("No training triples mined.")
    rng.shuffle(arr)
    cut = int(0.9 * len(arr))
    train_arr = arr[:cut]
    val_arr = arr[cut:] if cut < len(arr) else arr[: min(1024, len(arr))]

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = HardNegativePairReranker(emb_dim=vecs.shape[1], hidden_dim=256).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    loss_fn = torch.nn.MarginRankingLoss(margin=float(margin))

    v = torch.from_numpy(vecs.astype(np.float32)).to(device)

    def eval_pair_acc(eval_arr: np.ndarray) -> float:
        if len(eval_arr) == 0:
            return 0.0
        with torch.no_grad():
            q = v[torch.from_numpy(eval_arr[:, 0]).to(device)]
            p = v[torch.from_numpy(eval_arr[:, 1]).to(device)]
            n = v[torch.from_numpy(eval_arr[:, 2]).to(device)]
            sp = model(q, p)
            sn = model(q, n)
            return float((sp > sn).float().mean().item())

    history = []
    for ep in range(int(epochs)):
        rng.shuffle(train_arr)
        losses = []
        for s in range(0, len(train_arr), int(batch_size)):
            b = train_arr[s : s + int(batch_size)]
            q = v[torch.from_numpy(b[:, 0]).to(device)]
            p = v[torch.from_numpy(b[:, 1]).to(device)]
            n = v[torch.from_numpy(b[:, 2]).to(device)]
            sp = model(q, p)
            sn = model(q, n)
            y = torch.ones_like(sp)
            loss = loss_fn(sp, sn, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        val_acc = eval_pair_acc(val_arr)
        history.append({"epoch": ep + 1, "train_loss": float(np.mean(losses) if losses else 0.0), "val_pair_acc": val_acc})

    Path(out_ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "emb_dim": int(vecs.shape[1]),
            "hidden_dim": 256,
            "train_triples": int(len(train_arr)),
            "val_triples": int(len(val_arr)),
        },
        out_ckpt,
    )
    return {"device": device, "history": history, "train_triples": int(len(train_arr)), "val_triples": int(len(val_arr))}


def _evaluate_variant(
    sample: pd.DataFrame,
    dishes_df: pd.DataFrame,
    vecs: np.ndarray,
    encoder: VisionEncoder,
    alpha: float,
    tagger: CLIPTagPredictor | None,
    pair_reranker: HardNegativePairPredictor | None,
    threshold: float,
) -> dict:
    top1 = 0
    top3 = 0
    confusions = Counter()
    same_class_total = 0
    same_class_ident = 0
    same_class_near = 0
    abstain_rows = []

    for r in sample.itertuples(index=False):
        q_path = str(r.image_path)
        qn = normalize_path(q_path)
        true_lbl = _canon(str(r.eval_label))
        pred = predict_dish(
            q_path,
            dishes_df,
            vecs,
            encoder=encoder,
            tag_predictor=tagger,
            pair_reranker=pair_reranker,
            top_k=50,
            top_n=50,
            alpha=alpha,
            use_rerank=True,
            exclude_image_paths={qn},
            debug=True,
        )
        top3_rows = pred[:3]
        top3_lbls = [_canon(str(x.get("dish_class", x.get("dish_label", "")))) for x in top3_rows]
        p1 = top3_lbls[0] if top3_lbls else ""
        p1_ok = bool(p1 == true_lbl)
        p3_ok = bool(true_lbl in top3_lbls)
        if p1_ok:
            top1 += 1
        else:
            confusions[(true_lbl, p1)] += 1
        if p3_ok:
            top3 += 1

        # class-level discriminativeness over top-20 candidates
        class_scores: dict[str, list[float]] = defaultdict(list)
        for c in pred[:20]:
            cls = _canon(str(c.get("dish_class", c.get("dish_label", ""))))
            pair_s = float(c.get("pair_agreement", np.nan))
            if not np.isnan(pair_s):
                score = pair_s
            else:
                score = float(c.get("mlp_blend_score", c.get("dish_agreement", np.nan)))
            class_scores[cls].append(score)
        for scores in class_scores.values():
            valid = [s for s in scores if not np.isnan(s)]
            if len(valid) < 2:
                continue
            same_class_total += 1
            smin = float(np.min(valid))
            smax = float(np.max(valid))
            if abs(smax - smin) <= 1e-8:
                same_class_ident += 1
            if abs(smax - smin) <= 1e-3:
                same_class_near += 1

        top1_score = float(top3_rows[0].get("final_score", np.nan)) if top3_rows else float("nan")
        abstained = bool(np.isnan(top1_score) or top1_score < float(threshold))
        abstain_rows.append({"abstained": abstained, "correct_if_accepted": bool((not abstained) and p1_ok)})

    n = len(sample)
    accepted = [x for x in abstain_rows if not x["abstained"]]
    coverage = float(len(accepted) / max(1, n))
    selective_acc = float(np.mean([1.0 if x["correct_if_accepted"] else 0.0 for x in accepted])) if accepted else 0.0
    false_confident = int(sum(1 for x in abstain_rows if (not x["abstained"]) and (not x["correct_if_accepted"])))

    return {
        "top1_accuracy": float(top1 / max(1, n)),
        "top3_accuracy": float(top3 / max(1, n)),
        "confusion_pairs_top20": [
            {"true_label": t, "pred_label": p, "count": int(c)} for (t, p), c in confusions.most_common(20)
        ],
        "class_score_variation": {
            "same_class_groups": int(same_class_total),
            "identical_rate": float(same_class_ident / max(1, same_class_total)),
            "near_identical_rate_eps_1e3": float(same_class_near / max(1, same_class_total)),
        },
        "abstain_metrics": {
            "confidence_threshold": float(threshold),
            "coverage": coverage,
            "selective_top1_accuracy": selective_acc,
            "abstain_rate": float(1.0 - coverage),
            "false_confident_error_count": false_confident,
        },
    }


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    manifest = pd.read_csv(args.manifest_csv)
    label_col = _resolve_label_col(manifest)
    manifest = manifest[manifest["image_path"].map(lambda p: Path(str(p)).exists())].copy()
    manifest["eval_label"] = manifest[label_col].astype(str)

    dishes_df = pd.read_csv(Path(args.data_dir) / "dishes.csv")
    vecs = np.load(Path(args.data_dir) / "dish_vectors.npy").astype(np.float32, copy=False)
    vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)

    # Build label map over indexed data using image_path.
    idx_label_col = "dish_class" if "dish_class" in dishes_df.columns else "dish_label"
    labels = np.asarray([_canon(x) for x in dishes_df[idx_label_col].astype(str).tolist()])
    triples = _mine_triples(
        dishes_df=dishes_df,
        vecs=vecs,
        labels=labels,
        top_k_mine=int(args.top_k_mine),
        hard_negs_per_query=int(args.hard_negs_per_query),
    )
    train_info = _train_pair_reranker(
        vecs=vecs,
        triples=triples,
        seed=int(args.seed),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        margin=float(args.margin),
        out_ckpt=args.out_reranker_ckpt,
    )

    encoder = VisionEncoder()
    tagger = CLIPTagPredictor(args.tag_head_ckpt) if Path(args.tag_head_ckpt).exists() else None
    pair_predictor = HardNegativePairPredictor(args.out_reranker_ckpt)

    n = min(int(args.n_eval), len(manifest))
    sample = manifest.sample(n=n, random_state=int(args.seed)).reset_index(drop=True)

    baseline = _evaluate_variant(
        sample=sample,
        dishes_df=dishes_df,
        vecs=vecs,
        encoder=encoder,
        alpha=float(args.alpha),
        tagger=tagger,
        pair_reranker=None,
        threshold=float(args.confidence_threshold),
    )
    hardneg = _evaluate_variant(
        sample=sample,
        dishes_df=dishes_df,
        vecs=vecs,
        encoder=encoder,
        alpha=float(args.alpha),
        tagger=None,
        pair_reranker=pair_predictor,
        threshold=float(args.confidence_threshold),
    )

    report = {
        "config": {
            "manifest_csv": args.manifest_csv,
            "n_eval": int(n),
            "alpha": float(args.alpha),
            "confidence_threshold": float(args.confidence_threshold),
            "top_k_mine": int(args.top_k_mine),
            "hard_negs_per_query": int(args.hard_negs_per_query),
        },
        "hard_negative_mining": {
            "triples_total": int(len(triples)),
            "example_format": "query_idx, positive_idx, hard_negative_idx",
        },
        "training": train_info,
        "baseline_retrieval_mlp_rerank": baseline,
        "hard_negative_pair_rerank": hardneg,
        "delta_hardneg_vs_baseline": {
            "top1_accuracy": float(hardneg["top1_accuracy"] - baseline["top1_accuracy"]),
            "top3_accuracy": float(hardneg["top3_accuracy"] - baseline["top3_accuracy"]),
            "identical_rate_delta": float(
                hardneg["class_score_variation"]["identical_rate"] - baseline["class_score_variation"]["identical_rate"]
            ),
            "near_identical_rate_delta": float(
                hardneg["class_score_variation"]["near_identical_rate_eps_1e3"]
                - baseline["class_score_variation"]["near_identical_rate_eps_1e3"]
            ),
        },
    }

    out = Path(args.report_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({"baseline_top1": baseline["top1_accuracy"], "hardneg_top1": hardneg["top1_accuracy"]}, indent=2))
    print(f"Saved report: {out}")


if __name__ == "__main__":
    main()
