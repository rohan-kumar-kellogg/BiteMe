import argparse
import json
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
    p = argparse.ArgumentParser(description="Evaluate retrieval and rerank variants.")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--labels_csv", default="data/labels.csv")
    p.add_argument("--manifest_csv", default="images/manifest.csv")
    p.add_argument("--n_eval", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--alpha", type=float, default=0.15)
    p.add_argument("--tag_head_ckpt", default="data/models/clip_mlp_tag_head.pt")
    p.add_argument("--out_json", default="reports/eval_retrieval.json")
    return p.parse_args()


def _canon(x: str) -> str:
    return str(x).strip().lower().replace("_", " ")


def _load_eval_labels(args) -> pd.DataFrame:
    mpath = Path(args.manifest_csv)
    lpath = Path(args.labels_csv)
    if mpath.exists():
        df = pd.read_csv(mpath)
        source = str(mpath)
    elif lpath.exists():
        df = pd.read_csv(lpath)
        source = str(lpath)
    else:
        raise FileNotFoundError("No eval labels found: expected images/manifest.csv or data/labels.csv")

    if "image_path" not in df.columns:
        raise ValueError("Eval labels must include image_path.")
    if "dish_label" not in df.columns:
        if "dish_class" in df.columns:
            df["dish_label"] = df["dish_class"]
        elif "dish_name" in df.columns:
            df["dish_label"] = df["dish_name"]
        else:
            raise ValueError("Eval labels require dish_label or dish_class/dish_name.")
    df = df[df["image_path"].map(lambda p: Path(str(p)).exists())].copy()
    df["dish_label"] = df["dish_label"].astype(str)
    if "source" not in df.columns:
        df["source"] = "unspecified"
    df["source_file"] = source
    return df.reset_index(drop=True)


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def _retrieval_only_top3(query_emb, query_path, dishes_df, dish_vectors):
    sims = dish_vectors @ query_emb
    idx = np.argsort(-sims)
    out = []
    for i in idx:
        r = dishes_df.iloc[int(i)]
        if str(r.get("image_path", "")) == query_path:
            continue
        out.append(str(r.get("dish_label", r.get("dish_class", ""))))
        if len(out) >= 3:
            break
    return out


def _rerank_top3(query_path, dishes_df, dish_vectors, encoder, tagger, top_k, alpha, use_rerank, use_prompt_tags):
    preds = predict_dish(
        image_path=query_path,
        dishes_df=dishes_df,
        dish_vectors=dish_vectors,
        encoder=encoder,
        tag_predictor=tagger,
        top_k=top_k,
        top_n=top_k,
        alpha=alpha,
        use_rerank=use_rerank,
        use_prompt_tags=use_prompt_tags,
    )
    out = []
    for p in preds:
        if str(p.get("image_path", "")) == query_path:
            continue
        out.append(str(p["dish_label"]))
        if len(out) >= 3:
            break
    return out


def _metrics(true_labels, pred_top3):
    top1 = np.mean([_canon(y) == _canon(p[0]) if len(p) else False for y, p in zip(true_labels, pred_top3)])
    top3 = np.mean([_canon(y) in [_canon(x) for x in p[:3]] for y, p in zip(true_labels, pred_top3)])
    return float(top1), float(top3)


def _save_confusion(true_labels, pred_top1, out_path: Path):
    labels = sorted(set(_canon(x) for x in true_labels) | set(_canon(x) for x in pred_top1 if x))
    idx = {l: i for i, l in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=np.int32)
    for t, p in zip(true_labels, pred_top1):
        tt = _canon(t)
        pp = _canon(p) if p else "unknown"
        if pp not in idx:
            labels.append(pp)
            idx[pp] = len(labels) - 1
            cm2 = np.zeros((len(labels), len(labels)), dtype=np.int32)
            cm2[: cm.shape[0], : cm.shape[1]] = cm
            cm = cm2
        cm[idx[tt], idx[pp]] += 1
    df = pd.DataFrame(cm, index=labels, columns=labels)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path)


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    reports_dir = Path(args.out_json).parent
    reports_dir.mkdir(parents=True, exist_ok=True)

    labels_df = _load_eval_labels(args)
    dishes_df = pd.read_csv(data_dir / "dishes.csv")
    dish_vectors = np.load(data_dir / "dish_vectors.npy").astype(np.float32)
    dish_vectors = _normalize_rows(dish_vectors)

    n = min(int(args.n_eval), len(labels_df))
    eval_df = labels_df.sample(n=n, random_state=args.seed).reset_index(drop=True)
    encoder = VisionEncoder()
    tagger = None
    if Path(args.tag_head_ckpt).exists():
        try:
            tagger = CLIPTagPredictor(args.tag_head_ckpt)
        except Exception:
            tagger = None

    true_labels = []
    r_only = []
    r_prompt = []
    r_mlp = []

    for row in eval_df.itertuples(index=False):
        q_path = str(row.image_path)
        y = str(row.dish_label)
        q_emb = encoder.encode_image(q_path)
        true_labels.append(y)
        r_only.append(_retrieval_only_top3(q_emb, q_path, dishes_df, dish_vectors))
        r_prompt.append(
            _rerank_top3(
                q_path, dishes_df, dish_vectors, encoder, tagger, args.top_k, args.alpha, True, True
            )
        )
        if tagger is not None:
            r_mlp.append(
                _rerank_top3(
                    q_path, dishes_df, dish_vectors, encoder, tagger, args.top_k, args.alpha, True, False
                )
            )

    ro_t1, ro_t3 = _metrics(true_labels, r_only)
    rp_t1, rp_t3 = _metrics(true_labels, r_prompt)
    result = {
        "n_eval": int(n),
        "labels_source": str(eval_df["source_file"].iloc[0]),
        "retrieval_only": {"top1": ro_t1, "top3": ro_t3},
        "retrieval_prompt_rerank": {"top1": rp_t1, "top3": rp_t3},
        "delta_prompt_vs_retrieval": {"top1": rp_t1 - ro_t1, "top3": rp_t3 - ro_t3},
        "alpha": float(args.alpha),
    }

    # Optional source-wise reporting when source column exists.
    source_breakdown = {}
    if "source" in eval_df.columns:
        for source_name, sdf in eval_df.groupby("source"):
            idx = set(sdf.index.tolist())
            t = [true_labels[i] for i in range(len(true_labels)) if i in idx]
            ro = [r_only[i] for i in range(len(r_only)) if i in idx]
            rm = [r_mlp[i] for i in range(len(r_mlp)) if i in idx] if len(r_mlp) > 0 else []
            if len(t) == 0:
                continue
            s_ro_t1, _ = _metrics(t, ro)
            if len(rm) == len(t):
                s_rm_t1, _ = _metrics(t, rm)
            else:
                s_rm_t1 = 0.0
            source_breakdown[str(source_name)] = {
                "n_eval": int(len(t)),
                "retrieval_only_top1": float(s_ro_t1),
                "retrieval_mlp_top1": float(s_rm_t1),
            }
    if source_breakdown:
        result["source_breakdown"] = source_breakdown

    _save_confusion(true_labels, [x[0] if x else "" for x in r_only], reports_dir / "confusion_retrieval_only.csv")
    _save_confusion(
        true_labels, [x[0] if x else "" for x in r_prompt], reports_dir / "confusion_prompt_rerank.csv"
    )

    if tagger is not None:
        rm_t1, rm_t3 = _metrics(true_labels, r_mlp)
        result["retrieval_mlp_rerank"] = {"top1": rm_t1, "top3": rm_t3}
        result["delta_mlp_vs_retrieval"] = {"top1": rm_t1 - ro_t1, "top3": rm_t3 - ro_t3}
        _save_confusion(
            true_labels, [x[0] if x else "" for x in r_mlp], reports_dir / "confusion_mlp_rerank.csv"
        )
    else:
        result["retrieval_mlp_rerank"] = None
        result["delta_mlp_vs_retrieval"] = None

    out_path = Path(args.out_json)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"Saved report: {out_path}")


if __name__ == "__main__":
    main()

