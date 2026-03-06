import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from models.tag_head import CLIPTagHead
from models.vision import VisionEncoder


TARGET_PREFERENCE = ["cuisine", "dish_family", "dish_class"]


def parse_args():
    p = argparse.ArgumentParser(description="Train frozen-CLIP MLP heads for cuisine + dish_class/dish_family.")
    p.add_argument("--labels_csv", default="data/labels.csv")
    p.add_argument("--cache_path", default="data/cache/clip_embeddings_labels.npz")
    p.add_argument("--out_path", default="data/models/clip_mlp_tag_head.pt")
    p.add_argument("--reports_dir", default="reports")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min_count", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(requested: str) -> str:
    if requested == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "mps":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def normalize_label(x):
    if pd.isna(x):
        return None
    t = str(x).strip()
    if not t or t.lower() in {"none", "nan"}:
        return None
    return t


def _load_labels(labels_csv: str) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    if "image_path" not in df.columns:
        raise ValueError("labels CSV must contain image_path")
    if "dish_label" in df.columns and "dish_class" not in df.columns:
        df["dish_class"] = df["dish_label"]
    if "dish_class" not in df.columns and "dish_name" in df.columns:
        df["dish_class"] = df["dish_name"]
    if "dish_family" not in df.columns:
        df["dish_family"] = None
    for col in TARGET_PREFERENCE:
        if col not in df.columns:
            df[col] = None
    df["image_path"] = df["image_path"].astype(str)
    df = df[df["image_path"].map(lambda p: Path(p).exists())].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("No valid image paths found.")
    return df


def _compute_or_load_embeddings(df: pd.DataFrame, cache_path: str, encoder: VisionEncoder):
    cp = Path(cache_path)
    cp.parent.mkdir(parents=True, exist_ok=True)
    if cp.exists():
        payload = np.load(cp, allow_pickle=True)
        cached_paths = payload["image_paths"].astype(str).tolist()
        cur_paths = df["image_path"].astype(str).tolist()
        if cached_paths == cur_paths:
            return payload["embeddings"].astype(np.float32)

    embs = []
    keep = []
    for row in df.itertuples(index=False):
        try:
            embs.append(encoder.encode_image(str(row.image_path)).astype(np.float32))
            keep.append(True)
        except Exception:
            keep.append(False)
    keep_mask = np.asarray(keep, dtype=bool)
    df.drop(index=np.where(~keep_mask)[0], inplace=True)
    df.reset_index(drop=True, inplace=True)
    X = np.vstack(embs).astype(np.float32)
    np.savez(cp, image_paths=df["image_path"].astype(str).values, embeddings=X)
    return X


def _build_label_maps(df: pd.DataFrame, min_count: int):
    maps = {}
    for col in TARGET_PREFERENCE:
        vals = [normalize_label(x) for x in df[col].tolist()]
        cnt = pd.Series([v for v in vals if v is not None]).value_counts()
        labels = sorted(cnt[cnt >= min_count].index.tolist())
        if len(labels) >= 2:
            maps[col] = {lab: i for i, lab in enumerate(labels)}
    if not maps:
        raise ValueError("Need at least one target with >=2 classes after min_count filtering.")
    return maps


def _encode_targets(df: pd.DataFrame, label_maps: dict[str, dict[str, int]]):
    y = {}
    for col, mp in label_maps.items():
        vals = []
        for v in df[col].tolist():
            n = normalize_label(v)
            vals.append(mp.get(n, -100))
        y[col] = np.asarray(vals, dtype=np.int64)
    return y


def _split_indices(n: int, seed: int, val_frac: float = 0.2):
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_val = max(1, int(n * val_frac))
    return idx[n_val:], idx[:n_val]


def _eval_metrics(model, X_val, y_val, label_maps, device, reports_dir: Path):
    model.eval()
    x = torch.from_numpy(X_val.astype(np.float32)).to(device)
    with torch.no_grad():
        out = model(x)
    report = {}
    for col, logits in out.items():
        yy = y_val[col]
        valid = yy >= 0
        if int(valid.sum()) == 0:
            continue
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        top1 = np.argmax(probs, axis=1)
        top3 = np.argsort(-probs, axis=1)[:, :3]
        y_true = yy[valid]
        y_pred = top1[valid]
        top3_ok = [int(y_true[i] in top3[valid][i]) for i in range(len(y_true))]
        acc1 = float(np.mean(y_true == y_pred))
        acc3 = float(np.mean(top3_ok))
        report[col] = {"top1_accuracy": acc1, "top3_accuracy": acc3, "n_eval": int(len(y_true))}

        labels = sorted(label_maps[col].keys(), key=lambda k: label_maps[col][k])
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        cm_df.to_csv(reports_dir / f"confusion_{col}.csv")
    return report


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    df = _load_labels(args.labels_csv)
    encoder = VisionEncoder(device=device)
    X = _compute_or_load_embeddings(df, args.cache_path, encoder)
    label_maps = _build_label_maps(df, args.min_count)
    y = _encode_targets(df, label_maps)

    tr_idx, va_idx = _split_indices(len(df), args.seed, val_frac=0.2)
    X_tr = X[tr_idx]
    X_va = X[va_idx]
    y_tr = {k: v[tr_idx] for k, v in y.items()}
    y_va = {k: v[va_idx] for k, v in y.items()}

    model = CLIPTagHead(
        input_dim=int(X.shape[1]),
        hidden_dim=args.hidden_dim,
        num_classes={k: len(v) for k, v in label_maps.items()},
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    batch = max(8, int(args.batch_size))
    for ep in range(1, args.epochs + 1):
        model.train()
        order = np.random.permutation(len(X_tr))
        running = 0.0
        steps = 0
        for s in range(0, len(order), batch):
            b = order[s : s + batch]
            xb = torch.from_numpy(X_tr[b].astype(np.float32)).to(device)
            out = model(xb)
            loss = 0.0
            for col, logits in out.items():
                yy = torch.from_numpy(y_tr[col][b]).to(device)
                mask = (yy >= 0).float()
                if float(mask.sum()) == 0:
                    continue
                yy = torch.where(yy < 0, torch.zeros_like(yy), yy)
                losses = loss_fn(logits, yy)
                loss = loss + (losses * mask).sum() / (mask.sum() + 1e-12)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.item())
            steps += 1
        print(f"epoch {ep}/{args.epochs} train_loss={running / max(1, steps):.4f}")

    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    report = _eval_metrics(model, X_va, y_va, label_maps, device, reports_dir)
    report["n_train"] = int(len(X_tr))
    report["n_val"] = int(len(X_va))
    report_path = reports_dir / "clip_mlp_eval.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": int(X.shape[1]),
            "hidden_dim": int(args.hidden_dim),
            "label_maps": label_maps,
        },
        out_path,
    )
    print(f"Saved checkpoint: {out_path}")
    print(f"Saved eval report: {report_path}")


if __name__ == "__main__":
    main()

