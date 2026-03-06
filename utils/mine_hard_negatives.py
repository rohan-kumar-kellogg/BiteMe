import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from models.probes import ProbePredictor
from models.retrieval import predict_dish
from models.tag_head import CLIPTagPredictor
from models.vision import VisionEncoder


def parse_args():
    p = argparse.ArgumentParser(description="Mine hard negatives from personal photos.")
    p.add_argument("--manifest_csv", default="data/personal_manifest.csv")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--tag_head_ckpt", default="data/models/clip_mlp_tag_head.pt")
    p.add_argument("--probes_path", default="data/models/probes.pkl")
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--sim_threshold", type=float, default=0.30)
    p.add_argument("--out_csv", default="reports/hard_negatives.csv")
    p.add_argument("--multi_crop", action="store_true")
    p.add_argument("--use_text_ensemble", action="store_true")
    p.add_argument("--use_protein_probe", action="store_true")
    return p.parse_args()


def _canon(x: str) -> str:
    return str(x).strip().lower().replace("_", " ")


def _resolve_label_col(df: pd.DataFrame) -> str:
    for c in ["dish_class", "dish_label", "dish_family"]:
        if c in df.columns:
            return c
    raise ValueError("manifest must include one of: dish_class, dish_label, dish_family")


def main():
    args = parse_args()
    df = pd.read_csv(args.manifest_csv)
    label_col = _resolve_label_col(df)
    df = df[df["image_path"].map(lambda p: Path(str(p)).exists())].copy().reset_index(drop=True)
    label_series = df.get(label_col, pd.Series([""] * len(df), index=df.index))
    df["dish_class"] = label_series.astype(str)
    df = df[df["dish_class"].str.strip().str.len() > 0].copy().reset_index(drop=True)
    if len(df) == 0:
        print("No valid personal images found.")
        return

    dishes_df = pd.read_csv(Path(args.data_dir) / "dishes.csv")
    dish_vectors = np.load(Path(args.data_dir) / "dish_vectors.npy").astype(np.float32)
    dish_vectors = dish_vectors / (np.linalg.norm(dish_vectors, axis=1, keepdims=True) + 1e-12)
    encoder = VisionEncoder()
    tagger = CLIPTagPredictor(args.tag_head_ckpt) if Path(args.tag_head_ckpt).exists() else None
    probes = ProbePredictor.from_path(args.probes_path) if Path(args.probes_path).exists() else None

    rows = []
    for r in df.itertuples(index=False):
        q_path = str(r.image_path)
        true_label = str(r.dish_class)
        top = predict_dish(
            q_path,
            dishes_df,
            dish_vectors,
            encoder=encoder,
            tag_predictor=tagger,
            probe_predictor=probes,
            top_k=max(10, int(args.top_k)),
            top_n=max(10, int(args.top_k)),
            use_rerank=True,
            use_text_ensemble=bool(args.use_text_ensemble),
            use_protein_probe=bool(args.use_protein_probe),
            multi_crop=bool(args.multi_crop),
        )
        if not top:
            continue
        best = top[0]
        pred = str(best.get("dish_label", ""))
        sim = float(best.get("cosine_similarity", 0.0))
        if _canon(pred) != _canon(true_label) and sim > float(args.sim_threshold):
            rows.append(
                {
                    "query_image_path": q_path,
                    "true_label": true_label,
                    "predicted_label": pred,
                    "similarity": sim,
                    "final_score": float(best.get("final_score", np.nan)),
                }
            )

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows).sort_values("similarity", ascending=False) if rows else pd.DataFrame(
        columns=["query_image_path", "true_label", "predicted_label", "similarity", "final_score"]
    )
    out_df.to_csv(out, index=False)
    print(json.dumps({"n_personal": int(len(df)), "n_hard_negatives": int(len(out_df))}, indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

