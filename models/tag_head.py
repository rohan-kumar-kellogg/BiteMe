from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from models.vision import VisionEncoder


class CLIPTagHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: dict[str, int]):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.heads = nn.ModuleDict(
            {name: nn.Linear(hidden_dim, int(nc)) for name, nc in num_classes.items() if int(nc) > 1}
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.backbone(x)
        return {k: head(h) for k, head in self.heads.items()}


class CLIPTagPredictor:
    def __init__(self, checkpoint_path: str, device: str | None = None):
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        payload = torch.load(ckpt_path, map_location=self.device)
        self.label_maps: dict[str, dict[str, int]] = payload["label_maps"]
        num_classes = {k: len(v) for k, v in self.label_maps.items()}
        self.model = CLIPTagHead(
            input_dim=int(payload["input_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            num_classes=num_classes,
        ).to(self.device)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.encoder = VisionEncoder(device=self.device)

    def _predict_from_embedding(self, emb: np.ndarray, top_k: int = 3) -> dict[str, dict]:
        x = torch.from_numpy(emb.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(x)
        result: dict[str, dict] = {}
        for head, logits in out.items():
            probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
            idx_to_label = {int(v): k for k, v in self.label_maps[head].items()}
            order = np.argsort(-probs)
            top = []
            for i in order[: max(1, int(top_k))]:
                top.append({"label": idx_to_label[int(i)], "prob": float(probs[int(i)])})
            result[head] = {
                "label": top[0]["label"],
                "prob": top[0]["prob"],
                "top_k": top,
                "probs": {idx_to_label[int(i)]: float(probs[int(i)]) for i in order[: min(10, len(order))]},
            }
        return result

    def predict_tags(self, image_path: str, top_k: int = 3) -> dict[str, dict]:
        emb = self.encoder.encode_image(image_path)
        return self._predict_from_embedding(emb, top_k=top_k)


def predict_tags(image_path: str, checkpoint_path: str = "data/models/clip_mlp_tag_head.pt", top_k: int = 3):
    predictor = CLIPTagPredictor(checkpoint_path)
    return predictor.predict_tags(image_path, top_k=top_k)

