from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock1d(nn.Module):
    """Optional squeeze-excitation for Tier-1+ ablation."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        w = x.mean(dim=-1)
        w = self.fc(w).unsqueeze(-1)
        return x * w


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        out, _ = self.attn(x, x, x, need_weights=False)
        return self.norm(x + self.drop(out))


class CNNLSTMAttention(nn.Module):
    """Bhatti et al. hybrid CNN–LSTM–Attention (faithful backbone).

    Input: (B, C, T)
    Output: logits (B, num_classes), embedding (B, D)
    """

    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 2,
        conv_channels: int = 32,
        branch_channels: int = 16,
        lstm_hidden: list[int] | tuple[int, ...] = (256, 128),
        attn_heads: int = 4,
        dropout_attn: float = 0.1,
        dropout_fc: float = 0.5,
        use_se: bool = False,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(inplace=True),
        )
        self.branch1 = nn.Sequential(
            nn.Conv1d(conv_channels, branch_channels, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(conv_channels, branch_channels, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
        )
        fused = branch_channels * 2
        self.use_se = use_se
        self.se = SEBlock1d(fused) if use_se else nn.Identity()

        h1, h2 = int(lstm_hidden[0]), int(lstm_hidden[1])
        self.lstm1 = nn.LSTM(fused, h1, batch_first=True)
        self.lstm2 = nn.LSTM(h1, h2, batch_first=True)
        self.attn = MultiHeadSelfAttention(h2, heads=attn_heads, dropout=dropout_attn)
        self.drop = nn.Dropout(dropout_fc)
        self.fc = nn.Linear(h2, num_classes)
        self.embed_dim = h2

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, C, T)
        h = self.stem(x)
        h = torch.cat([self.branch1(h), self.branch2(h)], dim=1)
        h = self.se(h)
        h = h.transpose(1, 2)  # (B, T, F)
        h, _ = self.lstm1(h)
        h, _ = self.lstm2(h)
        h = self.attn(h)
        emb = h.mean(dim=1)
        logits = self.fc(self.drop(emb))
        return logits, emb

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(x)
        return F.softmax(logits, dim=-1)


class CNN1D(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x).squeeze(-1)
        return self.fc(h), h


class LSTMOnly(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, hidden: int = 128):
        super().__init__()
        self.lstm = nn.LSTM(in_channels, hidden, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = x.transpose(1, 2)  # (B, T, C)
        out, _ = self.lstm(h)
        emb = out[:, -1]
        return self.fc(emb), emb


class CNNLSTM(nn.Module):
    """CNN-LSTM without attention (baseline)."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.lstm = nn.LSTM(32, 128, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(x).transpose(1, 2)
        out, _ = self.lstm(h)
        emb = out[:, -1]
        return self.fc(emb), emb


def build_model(name: str, in_channels: int, num_classes: int, **kwargs) -> nn.Module:
    """Deprecated path — use models.factory.build_model (re-exported)."""
    from .factory import build_model as _build

    return _build(name, in_channels, num_classes, **kwargs)
