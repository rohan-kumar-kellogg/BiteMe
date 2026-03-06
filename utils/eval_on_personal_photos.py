import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from models.retrieval import predict_dish
from models.probes import ProbePredictor
from models.tag_head import CLIPTagPredictor
from models.vision import VisionEncoder
from utils.path_utils import normalize_path


def _str2bool(x):
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {x}")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate retrieval on personal labeled photos.")
    p.add_argument("--manifest_csv", default="data/personal_manifest.csv")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--tag_head_ckpt", default="data/models/clip_mlp_tag_head.pt")
    p.add_argument("--probes_path", default="data/models/probes.pkl")
    p.add_argument("--n_eval", type=int, default=0, help="0 means evaluate all rows")
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_n", type=int, default=3)
    p.add_argument("--multi_crop", type=_str2bool, default=True)
    p.add_argument("--use_text_ensemble", type=_str2bool, default=True)
    p.add_argument("--use_protein_probe", type=_str2bool, default=True)
    p.add_argument("--out_json", default="reports/personal_eval.json")
    p.add_argument("--out_failures", default="reports/personal_failures.csv")
    return p.parse_args()


def _canon(x: str) -> str:
    return str(x).strip().lower().replace("_", " ")


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def _filter_self(preds: list[dict], query_path: str, top_n: int) -> tuple[list[dict], bool]:
    qn = normalize_path(query_path)
    self_hit = any(normalize_path(str(x.get("image_path", ""))) == qn for x in preds)
    out = [x for x in preds if normalize_path(str(x.get("image_path", ""))) != qn]
    return out[:top_n], self_hit


def _metric(true_labels, pred_lists, k):
    good = 0
    for y, preds in zip(true_labels, pred_lists):
        if _canon(y) in [_canon(p) for p in preds[:k]]:
            good += 1
    return float(good / max(1, len(true_labels)))


def _resolve_label_col(df: pd.DataFrame) -> str:
    for c in ["dish_class", "dish_label", "dish_family"]:
        if c in df.columns:
            return c
    raise ValueError("manifest must include one of: dish_class, dish_label, dish_family")


def _eval_config(
    df: pd.DataFrame,
    dishes_df: pd.DataFrame,
    dish_vectors: np.ndarray,
    encoder: VisionEncoder,
    tagger: CLIPTagPredictor | None,
    probes: ProbePredictor | None,
    *,
    top_k: int,
    top_n: int,
    use_rerank: bool,
    multi_crop: bool,
    use_text_ensemble: bool,
    use_protein_probe: bool,
) -> tuple[dict, list[dict], list[str], list[list[str]], float]:
    truths = []
    preds = []
    failures = []
    self_hits = 0

    for row in df.itertuples(index=False):
        q_path = str(row.image_path)
        true_label = str(row.dish_class)
        truths.append(true_label)
        raw = predict_dish(
            q_path,
            dishes_df,
            dish_vectors,
            encoder=encoder,
            tag_predictor=tagger,
            probe_predictor=probes,
            top_k=top_k,
            top_n=top_k,
            use_rerank=use_rerank,
            use_prompt_tags=False,
            debug=True,
            multi_crop=multi_crop,
            use_text_ensemble=use_text_ensemble,
            use_protein_probe=use_protein_probe,
            exclude_image_paths={normalize_path(q_path)},
        )
        ranked, self_hit = _filter_self(raw, q_path, top_n)
        if self_hit:
            self_hits += 1
        labels = [str(x["dish_label"]) for x in ranked]
        preds.append(labels)
        top1 = ranked[0] if ranked else {}
        if _canon(str(top1.get("dish_label", ""))) != _canon(true_label):
            failures.append(
                {
                    "image_path": q_path,
                    "predicted_label": str(top1.get("dish_label", "")),
                    "true_label": true_label,
                    "cosine_similarity": float(top1.get("cosine_similarity", np.nan)),
                    "final_score": float(top1.get("final_score", np.nan)),
                    "top3_preds": json.dumps(labels),
                    "dish_agreement": float(top1.get("dish_agreement", np.nan)),
                    "protein_agreement": float(top1.get("protein_agreement", np.nan)),
                    "cuisine_agreement": float(top1.get("cuisine_agreement", np.nan)),
                }
            )

    report = {
        "top1": _metric(truths, preds, 1),
        "top3": _metric(truths, preds, 3),
        "n_eval": int(len(df)),
    }
    self_match_rate = float(self_hits / max(1, len(df)))
    return report, failures, truths, preds, self_match_rate


def run_personal_eval(
    *,
    manifest_csv: str = "data/personal_manifest.csv",
    data_dir: str = "data",
    tag_head_ckpt: str = "data/models/clip_mlp_tag_head.pt",
    probes_path: str = "data/models/probes.pkl",
    n_eval: int = 0,
    top_k: int = 50,
    top_n: int = 3,
    multi_crop: bool = True,
    use_text_ensemble: bool = True,
    use_protein_probe: bool = True,
    out_json: str = "reports/personal_eval.json",
    out_failures: str = "reports/personal_failures.csv",
) -> dict:
    man = Path(manifest_csv)
    if not man.exists():
        raise FileNotFoundError(f"Missing personal manifest: {man}")
    df = pd.read_csv(man)
    for c in ["image_path"]:
        if c not in df.columns:
            raise ValueError(f"{manifest_csv} must include `{c}`.")
    label_col = _resolve_label_col(df)
    if "cuisine" not in df.columns:
        df["cuisine"] = ""
    df = df[df["image_path"].map(lambda p: Path(str(p)).exists())].copy().reset_index(drop=True)
    label_series = df.get(label_col, pd.Series([""] * len(df), index=df.index))
    df["dish_class"] = label_series.astype(str).str.strip()
    df = df[df["dish_class"].str.len() > 0].copy().reset_index(drop=True)

    out_json = Path(out_json)
    out_fail = Path(out_failures)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_fail.parent.mkdir(parents=True, exist_ok=True)

    if len(df) == 0:
        print(
            "WARNING: No evaluable images found. Check that personal_manifest.csv contains valid image paths and labels."
        )
        empty = {
            "n_eval": 0,
            "self_match_rate": 0.0,
            "retrieval_only": {"top1": 0.0, "top3": 0.0},
            "retrieval_probe_rerank": {"top1": 0.0, "top3": 0.0},
            "delta_probe_vs_retrieval": {"top1": 0.0, "top3": 0.0},
            "ablation": [],
            "tag_head_loaded": False,
            "probe_loaded": False,
        }
        with open(out_json, "w") as f:
            json.dump(empty, f, indent=2)
        pd.DataFrame(
            columns=[
                "image_path",
                "predicted_label",
                "true_label",
                "cosine_similarity",
                "final_score",
                "top3_preds",
                "dish_agreement",
                "cuisine_agreement",
            ]
        ).to_csv(out_fail, index=False)
        print(json.dumps(empty, indent=2))
        print(f"Saved: {out_json} and {out_fail}")
        return

    if n_eval and n_eval > 0:
        df = df.sample(n=min(int(n_eval), len(df)), random_state=42).reset_index(drop=True)

    dishes_df = pd.read_csv(Path(data_dir) / "dishes.csv")
    dish_vectors = np.load(Path(data_dir) / "dish_vectors.npy").astype(np.float32)
    dish_vectors = _normalize_rows(dish_vectors)

    encoder = VisionEncoder()
    tagger = None
    if Path(tag_head_ckpt).exists():
        try:
            tagger = CLIPTagPredictor(tag_head_ckpt)
        except Exception:
            tagger = None
    probes = None
    if Path(probes_path).exists():
        try:
            probes = ProbePredictor.from_path(probes_path)
        except Exception:
            probes = None

    ablation = []
    failure_rows = []
    max_self_rate = 0.0

    cfgs = [
        ("retrieval-only", dict(use_rerank=False, multi_crop=False, use_text_ensemble=False, use_protein_probe=False)),
        (
            "+multi-crop",
            dict(
                use_rerank=False,
                multi_crop=bool(multi_crop),
                use_text_ensemble=False,
                use_protein_probe=False,
            ),
        ),
        (
            "+text ensemble",
            dict(
                use_rerank=False,
                multi_crop=bool(multi_crop),
                use_text_ensemble=bool(use_text_ensemble),
                use_protein_probe=False,
            ),
        ),
        (
            "+probe rerank",
            dict(
                use_rerank=True,
                multi_crop=bool(multi_crop),
                use_text_ensemble=bool(use_text_ensemble),
                use_protein_probe=bool(use_protein_probe),
            ),
        ),
    ]

    for name, cfg in cfgs:
        use_tagger = tagger if cfg["use_rerank"] else None
        use_probes = probes if cfg["use_protein_probe"] else None
        metrics, failures, _, _, self_rate = _eval_config(
            df,
            dishes_df,
            dish_vectors,
            encoder,
            use_tagger,
            use_probes,
            top_k=top_k,
            top_n=top_n,
            use_rerank=cfg["use_rerank"],
            multi_crop=cfg["multi_crop"],
            use_text_ensemble=cfg["use_text_ensemble"],
            use_protein_probe=cfg["use_protein_probe"],
        )
        max_self_rate = max(max_self_rate, self_rate)
        row = {
            "name": name,
            "top1": float(metrics["top1"]),
            "top3": float(metrics["top3"]),
            "self_match_rate": float(self_rate),
            **cfg,
        }
        ablation.append(row)
        if name == "+probe rerank":
            failure_rows = failures

    base = ablation[0]
    probe = ablation[-1]
    report = {
        "n_eval": int(len(df)),
        "self_match_rate": float(max_self_rate),
        "retrieval_only": {"top1": float(base["top1"]), "top3": float(base["top3"])},
        "retrieval_probe_rerank": {"top1": float(probe["top1"]), "top3": float(probe["top3"])},
        "delta_probe_vs_retrieval": {
            "top1": float(probe["top1"] - base["top1"]),
            "top3": float(probe["top3"] - base["top3"]),
        },
        "ablation": ablation,
        "tag_head_loaded": bool(tagger is not None),
        "probe_loaded": bool(probes is not None),
    }
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)

    fail_df = pd.DataFrame(failure_rows)
    if len(fail_df) > 0:
        fail_df = fail_df.sort_values("final_score", ascending=False)
    fail_df.to_csv(out_fail, index=False)

    print(json.dumps(report, indent=2))
    print(f"Saved report: {out_json}")
    print(f"Saved failures: {out_fail}")
    if max_self_rate > 0.01:
        raise RuntimeError(
            f"Self-match leakage detected (rate={max_self_rate:.4f} > 0.01). "
            "Check path normalization and self-filter logic."
        )
    if len(fail_df) > 0:
        print("\n10 confusing examples (wrong with highest confidence):")
        show = fail_df.head(10)
        for r in show.itertuples(index=False):
            print(
                f"- {r.image_path} | true={r.true_label} pred={r.predicted_label} "
                f"cos={float(r.cosine_similarity):.4f} final={float(r.final_score):.4f}"
            )
    return report


def main():
    args = parse_args()
    run_personal_eval(
        manifest_csv=args.manifest_csv,
        data_dir=args.data_dir,
        tag_head_ckpt=args.tag_head_ckpt,
        probes_path=args.probes_path,
        n_eval=args.n_eval,
        top_k=args.top_k,
        top_n=args.top_n,
        multi_crop=bool(args.multi_crop),
        use_text_ensemble=bool(args.use_text_ensemble),
        use_protein_probe=bool(args.use_protein_probe),
        out_json=args.out_json,
        out_failures=args.out_failures,
    )


if __name__ == "__main__":
    main()

