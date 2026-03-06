import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except Exception:
    pass


TARGET_COLS = [
    "dish_family",
    "dish_class",
    "dish_name",
    "cuisine",
    "protein_type",
    "course",
    "protein",
    "prep_style",
]


def parse_args():
    p = argparse.ArgumentParser(description="Deep multi-task training for food labels.")
    p.add_argument("--labels_csv", default="data/labels.csv")
    p.add_argument("--out_dir", default="data/models/deep_multitask")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--min_count", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Training device. Use 'auto' to prefer cuda, then mps, then cpu.",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader workers (try 2-8 for faster loading).",
    )
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> str:
    if requested == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        raise ValueError("Requested --device cuda but CUDA is not available.")
    if requested == "mps":
        if torch.backends.mps.is_available():
            return "mps"
        raise ValueError("Requested --device mps but MPS is not available.")
    if requested == "cpu":
        return "cpu"

    # auto mode
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def normalize_label(s):
    if pd.isna(s):
        return None
    t = str(s).strip()
    if not t or t.lower() in {"nan", "none"}:
        return None
    return t


def build_vocab(series: pd.Series, min_count: int):
    vals = [normalize_label(x) for x in series.tolist()]
    vals = [v for v in vals if v is not None]
    cnt = Counter(vals)
    labels = sorted([k for k, v in cnt.items() if v >= min_count])
    idx = {k: i for i, k in enumerate(labels)}
    return labels, idx


class FoodLabelDataset(Dataset):
    def __init__(self, df, target_maps, tfm):
        self.df = df.reset_index(drop=True)
        self.target_maps = target_maps
        self.tfm = tfm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        p = row["image_path"]
        try:
            img = Image.open(p).convert("RGB")
        except UnidentifiedImageError:
            # Return a black image fallback; caller may skip via mask loss.
            img = Image.new("RGB", (224, 224), color="black")
        x = self.tfm(img)
        targets = {}
        masks = {}
        for col, mp in self.target_maps.items():
            val = normalize_label(row.get(col))
            if val is None or val not in mp:
                targets[col] = torch.tensor(0, dtype=torch.long)
                masks[col] = torch.tensor(0.0, dtype=torch.float32)
            else:
                targets[col] = torch.tensor(mp[val], dtype=torch.long)
                masks[col] = torch.tensor(1.0, dtype=torch.float32)
        return x, targets, masks


class MultiTaskResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.heads = nn.ModuleDict({k: nn.Linear(in_dim, v) for k, v in num_classes.items() if v > 1})

    def forward(self, x):
        feat = self.backbone(x)
        return {k: h(feat) for k, h in self.heads.items()}


def split_df(df, seed=42, val_frac=0.15):
    idx = np.arange(len(df))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_val = max(1, int(len(df) * val_frac))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    return df.iloc[tr_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


@torch.no_grad()
def eval_epoch(model, dl, device, loss_fn):
    model.eval()
    total = 0.0
    n = 0
    for x, y, m in dl:
        x = x.to(device)
        out = model(x)
        batch_loss = 0.0
        for col, logits in out.items():
            yy = y[col].to(device)
            mm = m[col].to(device)
            if float(mm.sum()) == 0:
                continue
            losses = loss_fn(logits, yy)
            batch_loss = batch_loss + (losses * mm).sum() / (mm.sum() + 1e-12)
        total += float(batch_loss.item())
        n += 1
    return total / max(1, n)


def main():
    args = parse_args()
    set_seed(args.seed)

    df = pd.read_csv(args.labels_csv)
    if "image_path" not in df.columns:
        raise ValueError("labels CSV must include `image_path` column.")
    df = df[df["image_path"].map(lambda p: Path(str(p)).exists())].reset_index(drop=True)
    if len(df) < 100:
        raise ValueError("Deep training needs more data. Aim for at least ~100 labeled images.")

    target_maps = {}
    target_labels = {}
    for col in TARGET_COLS:
        if col in df.columns:
            labels, mp = build_vocab(df[col], min_count=args.min_count)
            if len(labels) > 1:
                target_maps[col] = mp
                target_labels[col] = labels
    if not target_maps:
        raise ValueError("No trainable target columns with >=2 classes after min_count filtering.")

    tr_df, va_df = split_df(df, seed=args.seed, val_frac=0.15)

    tfm_train = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    tfm_val = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    tr_ds = FoodLabelDataset(tr_df, target_maps, tfm_train)
    va_ds = FoodLabelDataset(va_df, target_maps, tfm_val)
    device = resolve_device(args.device)
    pin_memory = device == "cuda"
    tr_dl = DataLoader(
        tr_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    va_dl = DataLoader(
        va_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    num_classes = {k: len(v) for k, v in target_labels.items()}
    model = MultiTaskResNet(num_classes).to(device)
    print(f"Using device: {device}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    best_val = float("inf")
    best_state = None
    for ep in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        steps = 0
        for x, y, m in tr_dl:
            x = x.to(device)
            out = model(x)
            loss = 0.0
            for col, logits in out.items():
                yy = y[col].to(device)
                mm = m[col].to(device)
                if float(mm.sum()) == 0:
                    continue
                losses = loss_fn(logits, yy)
                loss = loss + (losses * mm).sum() / (mm.sum() + 1e-12)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.item())
            steps += 1

        tr_loss = running / max(1, steps)
        va_loss = eval_epoch(model, va_dl, device, loss_fn)
        print(f"epoch {ep}/{args.epochs} train_loss={tr_loss:.4f} val_loss={va_loss:.4f}")
        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "model.pt"
    meta_path = out_dir / "meta.json"

    torch.save({"state_dict": best_state, "num_classes": num_classes}, ckpt_path)
    with open(meta_path, "w") as f:
        json.dump(
            {
                "target_labels": target_labels,
                "image_size": args.image_size,
                "best_val_loss": best_val,
                "model": "resnet50_multitask",
            },
            f,
            indent=2,
        )
    print(f"Saved model to {ckpt_path}")
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()

