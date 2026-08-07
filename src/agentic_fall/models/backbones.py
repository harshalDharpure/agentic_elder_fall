from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn_lstm_attn import SEBlock1d


class TemporalBlock(nn.Module):
    """Dilated causal-ish residual block (padding keeps length)."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.drop(h)
        h = self.bn2(self.conv2(h))
        # match length if odd dilation/pad mismatch
        if h.size(-1) != x.size(-1):
            h = F.interpolate(h, size=x.size(-1), mode="linear", align_corners=False)
        return F.relu(h + self.down(x))


class TCN(nn.Module):
    """Temporal Convolutional Network for IMU windows."""

    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 2,
        channels: int = 64,
        levels: int = 4,
        kernel: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        blocks = []
        ch_in = in_channels
        for i in range(levels):
            dil = 2**i
            blocks.append(TemporalBlock(ch_in, channels, kernel, dil, dropout))
            ch_in = channels
        self.net = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels, num_classes)
        self.embed_dim = channels

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        emb = self.pool(h).squeeze(-1)
        return self.fc(emb), emb


class DepthwiseTemporalBlock(nn.Module):
    """Depthwise-separable dilated conv + SE (ModernTCN-style)."""

    def __init__(self, channels: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel - 1) * dilation // 2
        self.dw = nn.Conv1d(
            channels, channels, kernel, padding=pad, dilation=dilation, groups=channels
        )
        self.pw = nn.Conv1d(channels, channels, 1)
        self.bn = nn.BatchNorm1d(channels)
        self.se = SEBlock1d(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dw(x)
        if h.size(-1) != x.size(-1):
            h = F.interpolate(h, size=x.size(-1), mode="linear", align_corners=False)
        h = self.pw(h)
        h = self.bn(h)
        h = self.se(h)
        return F.gelu(self.drop(h) + x)


class ModernTCN(nn.Module):
    """Compact modern TCN: stem + depthwise dilated blocks + SE."""

    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 2,
        channels: int = 64,
        levels: int = 5,
        kernel: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, channels, 3, padding=1),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[
                DepthwiseTemporalBlock(channels, kernel, dilation=2**i, dropout=dropout)
                for i in range(levels)
            ]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels, num_classes)
        self.embed_dim = channels

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.blocks(self.stem(x))
        emb = self.pool(h).squeeze(-1)
        return self.fc(emb), emb


class TSMixerBlock(nn.Module):
    def __init__(self, seq_len: int, channels: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm([seq_len, channels])
        self.time_mlp = nn.Sequential(
            nn.Linear(seq_len, seq_len),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(seq_len, seq_len),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm([seq_len, channels])
        self.feat_mlp = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        h = self.norm1(x)
        h = h.transpose(1, 2)  # (B, C, T)
        h = self.time_mlp(h).transpose(1, 2)
        x = x + h
        h = self.feat_mlp(self.norm2(x))
        return x + h


class TSMixer(nn.Module):
    """TSMixer-style MLP for short IMU windows."""

    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 2,
        seq_len: int = 90,
        d_model: int = 64,
        n_blocks: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.proj = nn.Linear(in_channels, d_model)
        self.blocks = nn.ModuleList(
            [TSMixerBlock(seq_len, d_model, dropout) for _ in range(n_blocks)]
        )
        self.fc = nn.Linear(d_model, num_classes)
        self.embed_dim = d_model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, C, T) -> (B, T, C)
        h = x.transpose(1, 2)
        t = h.size(1)
        if t != self.seq_len:
            h = F.interpolate(
                h.transpose(1, 2), size=self.seq_len, mode="linear", align_corners=False
            ).transpose(1, 2)
        h = self.proj(h)
        for blk in self.blocks:
            h = blk(h)
        emb = h.mean(dim=1)
        return self.fc(emb), emb


class PatchTST(nn.Module):
    """Patch time-series Transformer (channel-independent patches, shared encoder)."""

    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 2,
        seq_len: int = 90,
        patch_len: int = 10,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.seq_len = seq_len
        self.in_channels = in_channels
        self.n_patches = math.ceil(seq_len / patch_len)
        self.patch_embed = nn.Linear(patch_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.fc = nn.Linear(d_model, num_classes)
        self.embed_dim = d_model

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> patches per channel then mean over channels
        b, c, t = x.shape
        if t != self.seq_len:
            x = F.interpolate(x, size=self.seq_len, mode="linear", align_corners=False)
        pad = self.n_patches * self.patch_len - self.seq_len
        if pad > 0:
            x = F.pad(x, (0, pad))
        # (B, C, n_patches, patch_len)
        x = x.view(b, c, self.n_patches, self.patch_len)
        # average channels for compact encoder (IMU-friendly)
        x = x.mean(dim=1)  # (B, n_patches, patch_len)
        return self.patch_embed(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self._patchify(x)
        h = self.encoder(h)
        emb = h.mean(dim=1)
        return self.fc(emb), emb


class TransformerEncoder1D(nn.Module):
    """Vanilla Transformer encoder over time steps (projected channels)."""

    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 2,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        max_len: int = 128,
    ):
        super().__init__()
        self.proj = nn.Linear(in_channels, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.fc = nn.Linear(d_model, num_classes)
        self.embed_dim = d_model
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = x.transpose(1, 2)  # (B, T, C)
        t = h.size(1)
        if t > self.max_len:
            h = h[:, : self.max_len]
            t = self.max_len
        h = self.proj(h) + self.pos[:, :t]
        h = self.encoder(h)
        emb = h.mean(dim=1)
        return self.fc(emb), emb


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
