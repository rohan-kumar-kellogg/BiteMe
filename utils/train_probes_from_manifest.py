import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from models.probes import normalize_protein_type
from models.vision import VisionEncoder


def parse_args():
    p = argparse.ArgumentParser(description="Train lightweight linear probes on CLIP embeddings.")
    p.add_argument("--manifest_csvs", nargs="+", required=True)
    p.add_argument("--out_path", default="data/models/probes.pkl")
    p.add_argument("--cache_path", default="data/cache/probe_embeddings.npz")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--multi_crop", action="store_true")
    p.add_argument("--reports_json", default="reports/probes_eval.json")
    return p.parse_args()


def _load_merged_manifest(paths: list[str]) -> pd.DataFrame:
    dfs = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(f"Missing manifest: {path}")
        df = pd.read_csv(path)
        if "image_path" not in df.columns:
            raise ValueError(f"{path} missing image_path")
        dfs.append(df)
    merged = pd.concat(dfs, ignore_index=True)
    merged["image_path"] = merged["image_path"].astype(str).str.strip()
    merged = merged.drop_duplicates(subset=["image_path"]).reset_index(drop=True)
    return merged


def _resolve_dish_label(row: pd.Series) -> str:
    for col in ["dish_class", "dish_family", "dish_label"]:
        val = str(row.get(col, "")).strip()
        if val:
            return val
    return ""


def _build_or_load_embeddings(df: pd.DataFrame, cache_path: Path, multi_crop: bool) -> np.ndarray:
    if cache_path.exists():
        payload = np.load(cache_path, allow_pickle=True)
        paths = payload["image_paths"].tolist()
        x = payload["embeddings"].astype(np.float32)
        if len(paths) == len(df) and paths == df["image_path"].tolist():
            return x

    encoder = VisionEncoder()
    embs = []
    for p in df["image_path"].tolist():
        embs.append(encoder.encode_image(str(p), multi_crop=multi_crop).astype(np.float32))
    x = np.vstack(embs)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, image_paths=df["image_path"].tolist(), embeddings=x)
    return x


def _train_probe(x: np.ndarray, y: np.ndarray, seed: int):
    clf = LogisticRegression(max_iter=1200, class_weight="balanced", random_state=seed)
    clf.fit(x, y)
    return clf


def main():
    args = parse_args()
    rng_seed = int(args.seed)

    df = _load_merged_manifest(args.manifest_csvs)
    df = df[df["image_path"].map(lambda p: Path(str(p)).exists())].copy().reset_index(drop=True)
    df["dish_class_resolved"] = df.apply(_resolve_dish_label, axis=1)
    df["protein_type_resolved"] = df.get("protein_type", pd.Series([""] * len(df))).map(normalize_protein_type)

    df = df[df["dish_class_resolved"].str.len() > 0].copy().reset_index(drop=True)
    if len(df) < 10:
        raise ValueError("Not enough labeled rows to train probes.")

    x = _build_or_load_embeddings(df, Path(args.cache_path), multi_crop=bool(args.multi_crop))

    # dish_class probe
    dish_mask = df["dish_class_resolved"].str.len() > 0
    x_dish = x[dish_mask.values]
    y_dish = df.loc[dish_mask, "dish_class_resolved"].astype(str).to_numpy()

    # protein probe (coarse categories)
    prot_mask = df["protein_type_resolved"].str.len() > 0
    x_prot = x[prot_mask.values]
    y_prot = df.loc[prot_mask, "protein_type_resolved"].astype(str).to_numpy()

    dish_model = None
    prot_model = None
    report = {"n_rows": int(len(df)), "multi_crop": bool(args.multi_crop)}

    if len(np.unique(y_dish)) >= 2:
        xd_tr, xd_te, yd_tr, yd_te = train_test_split(
            x_dish, y_dish, test_size=float(args.test_size), random_state=rng_seed, stratify=y_dish
        )
        dish_model = _train_probe(xd_tr, yd_tr, rng_seed)
        report["dish_class"] = {
            "n_classes": int(len(np.unique(y_dish))),
            "top1_accuracy": float(dish_model.score(xd_te, yd_te)),
        }
    else:
        report["dish_class"] = {"skipped": "need >=2 classes"}

    if len(np.unique(y_prot)) >= 2:
        xp_tr, xp_te, yp_tr, yp_te = train_test_split(
            x_prot, y_prot, test_size=float(args.test_size), random_state=rng_seed, stratify=y_prot
        )
        prot_model = _train_probe(xp_tr, yp_tr, rng_seed)
        report["protein_type"] = {
            "n_classes": int(len(np.unique(y_prot))),
            "top1_accuracy": float(prot_model.score(xp_te, yp_te)),
        }
    else:
        report["protein_type"] = {"skipped": "need >=2 classes"}

    payload = {
        "dish_class_model": dish_model,
        "protein_type_model": prot_model,
        "meta": {
            "manifest_csvs": args.manifest_csvs,
            "multi_crop": bool(args.multi_crop),
            "n_rows": int(len(df)),
        },
    }
    out = Path(args.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(payload, f)

    reports_json = Path(args.reports_json)
    reports_json.parent.mkdir(parents=True, exist_ok=True)
    with open(reports_json, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"Saved probes: {out}")
    print(f"Saved report: {reports_json}")


if __name__ == "__main__":
    main()

