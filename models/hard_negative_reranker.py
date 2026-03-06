from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def _pair_features(query_vec: torch.Tensor, cand_vec: torch.Tensor) -> torch.Tensor:
    return torch.cat([query_vec, cand_vec, torch.abs(query_vec - cand_vec), query_vec * cand_vec], dim=-1)


class HardNegativePairReranker(nn.Module):
    def __init__(self, emb_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        in_dim = int(emb_dim) * 4
        self.net = nn.Sequential(
            nn.Linear(in_dim, int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, query_vec: torch.Tensor, cand_vec: torch.Tensor) -> torch.Tensor:
        feats = _pair_features(query_vec, cand_vec)
        logits = self.net(feats).squeeze(-1)
        return torch.sigmoid(logits)


class HardNegativePairPredictor:
    def __init__(self, checkpoint_path: str, device: str | None = None):
        ckpt = Path(checkpoint_path)
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        payload = torch.load(ckpt, map_location=self.device)
        self.model = HardNegativePairReranker(
            emb_dim=int(payload.get("emb_dim", 512)),
            hidden_dim=int(payload.get("hidden_dim", 256)),
        ).to(self.device)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()

    def score_pair_embeddings(self, query_emb: np.ndarray, cand_emb: np.ndarray) -> float:
        q = torch.from_numpy(query_emb.astype(np.float32)).unsqueeze(0).to(self.device)
        c = torch.from_numpy(cand_emb.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            s = self.model(q, c).squeeze(0).detach().cpu().item()
        return float(s)
