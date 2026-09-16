"""HARMamba-style bidirectional selective SSM for IMU windows.

Pure-PyTorch drop-in Tier-1 detector (no mamba-ssm CUDA extension).
Inspired by HARMamba (arXiv:2403.20183): bidirectional selective state-space
modeling over short wearable IMU sequences.

Contract matches the rest of the backbone zoo:
  input  (B, C, T)  e.g. (B, 6, 90)
  output (logits, emb) with emb shape (B, d_model)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _selective_scan(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
) -> torch.Tensor:
    """Discretized selective SSM scan (sequential, O(T)).

    Args:
        u:     (B, L, D)     input
        delta: (B, L, D)     step sizes (positive)
        A:     (D, N)        state matrix (log-parameterized externally)
        B:     (B, L, N)     input projection
        C:     (B, L, N)     output projection
        D:     (D,)          skip connection
    Returns:
        y:     (B, L, D)
    """
    bsz, length, d_inner = u.shape
    n = A.size(-1)
    # Discrete A: exp(delta * A)  -> (B, L, D, N)
    deltaA = torch.exp(delta.unsqueeze(-1) * A.view(1, 1, d_inner, n))
    # Discrete B: delta * B * u   -> (B, L, D, N)
    deltaB_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)

    h = u.new_zeros(bsz, d_inner, n)
    ys = []
    for t in range(length):
        h = deltaA[:, t] * h + deltaB_u[:, t]
        y_t = torch.einsum("bdn,bn->bd", h, C[:, t])
        ys.append(y_t)
    y = torch.stack(ys, dim=1)  # (B, L, D)
    return y + u * D.view(1, 1, -1)


class SelectiveSSM(nn.Module):
    """One-directional selective SSM (Mamba-style) over sequence length."""

    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2, dt_rank: int | None = None):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_state = d_state
        self.dt_rank = dt_rank or max(1, math.ceil(d_model / 16))

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=3,
            padding=1,
            groups=self.d_inner,
            bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A parameterized in log space (negative real eigenvalues)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Softplus bias init so delta starts small/positive
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)
        # depthwise conv over time
        x_branch = self.conv1d(x_branch.transpose(1, 2)).transpose(1, 2)
        x_branch = F.silu(x_branch)

        x_dbl = self.x_proj(x_branch)
        dt, B, C = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log.float())
        y = _selective_scan(x_branch, dt, A, B, C, self.D.float())
        y = y * F.silu(z)
        return self.out_proj(y)


class BiMambaBlock(nn.Module):
    """Bidirectional HARMamba block: forward + backward SSM + FFN."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.fwd = SelectiveSSM(d_model, d_state=d_state, expand=expand)
        self.bwd = SelectiveSSM(d_model, d_state=d_state, expand=expand)
        self.mix = nn.Linear(d_model * 2, d_model)
        self.drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        y_f = self.fwd(h)
        y_b = self.bwd(torch.flip(h, dims=[1]))
        y_b = torch.flip(y_b, dims=[1])
        x = x + self.drop(self.mix(torch.cat([y_f, y_b], dim=-1)))
        x = x + self.ffn(self.norm2(x))
        return x


class HARMamba(nn.Module):
    """Bidirectional selective SSM Tier-1 fall detector (HARMamba-style).

    Input:  (B, C, T)
    Output: logits (B, num_classes), embedding (B, d_model)
    """

    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 2,
        d_model: int = 64,
        n_layers: int = 3,
        d_state: int = 16,
        expand: int = 2,
        dropout: float = 0.1,
        seq_len: int = 90,
        patch_len: int = 10,
        use_patch: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.use_patch = use_patch
        self.embed_dim = d_model

        if use_patch:
            self.n_patches = math.ceil(seq_len / patch_len)
            # Channel-mixed patch embedding: each patch is (C * patch_len)
            self.patch_embed = nn.Linear(in_channels * patch_len, d_model)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            self.pos = nn.Parameter(torch.zeros(1, self.n_patches + 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            nn.init.trunc_normal_(self.pos, std=0.02)
            self.stem = None
        else:
            self.stem = nn.Sequential(
                nn.Conv1d(in_channels, d_model, kernel_size=3, padding=1),
                nn.BatchNorm1d(d_model),
                nn.GELU(),
            )
            self.cls_token = None
            self.pos = None
            self.patch_embed = None
            self.n_patches = 0

        self.blocks = nn.ModuleList(
            [
                BiMambaBlock(d_model, d_state=d_state, expand=expand, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def _to_tokens(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> (B, L, D)
        b, c, t = x.shape
        if t != self.seq_len:
            x = F.interpolate(x, size=self.seq_len, mode="linear", align_corners=False)
            t = self.seq_len

        if self.use_patch:
            pad = self.n_patches * self.patch_len - t
            if pad > 0:
                x = F.pad(x, (0, pad))
            # (B, n_patches, C * patch_len)
            x = x.view(b, c, self.n_patches, self.patch_len)
            x = x.permute(0, 2, 1, 3).contiguous().view(b, self.n_patches, c * self.patch_len)
            tokens = self.patch_embed(x)
            cls = self.cls_token.expand(b, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
            tokens = tokens + self.pos[:, : tokens.size(1)]
            return tokens

        # frame-level: (B, T, D)
        h = self.stem(x).transpose(1, 2)
        return h

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self._to_tokens(x)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        if self.use_patch:
            emb = h[:, 0]  # CLS
        else:
            emb = h.mean(dim=1)
        logits = self.fc(self.drop(emb))
        return logits, emb
