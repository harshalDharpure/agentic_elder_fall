from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..features.biomechanics import extract_biomechanics
from ..agents.evidence import evidence_caption


class SensorEncoder(nn.Module):
    """Lightweight 1D-CNN sensor encoder for Stage-1 alignment."""

    def __init__(self, in_channels: int = 6, embed_dim: int = 384):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(128, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x).squeeze(-1)
        z = self.proj(h)
        return F.normalize(z, dim=-1)


def info_nce(sensor_z: torch.Tensor, text_z: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Symmetric InfoNCE between paired sensor/text embeddings."""
    logits = sensor_z @ text_z.T / temperature
    labels = torch.arange(sensor_z.size(0), device=sensor_z.device)
    loss_s = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_s + loss_t)


def caption_from_window(
    window_tc: torch.Tensor | "np.ndarray",
    sample_rate_hz: float = 200.0,
) -> str:
    import numpy as np

    if isinstance(window_tc, torch.Tensor):
        arr = window_tc.detach().cpu().numpy()
    else:
        arr = window_tc
    if arr.ndim == 2 and arr.shape[0] < arr.shape[1]:
        # (C, T) -> (T, C)
        arr = arr.T
    feats = extract_biomechanics(arr, sample_rate_hz=sample_rate_hz)
    return evidence_caption(feats)
