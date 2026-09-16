from __future__ import annotations

import torch.nn as nn

from .backbones import ModernTCN, PatchTST, TCN, TSMixer, TransformerEncoder1D
from .cnn_lstm_attn import CNN1D, CNNLSTM, CNNLSTMAttention, LSTMOnly
from .harmamba import HARMamba

MODERN_BACKBONES = ("tcn", "modern_tcn", "tsmixer", "patchtst", "transformer", "harmamba")
CLASSICAL_BACKBONES = ("cnn1d", "lstm", "cnn_lstm", "cnn_lstm_attn")
ALL_BACKBONES = CLASSICAL_BACKBONES + MODERN_BACKBONES


def build_model(name: str, in_channels: int, num_classes: int, **kwargs) -> nn.Module:
    """Factory for classical + modern Tier-1 detectors. Returns (logits, emb) models."""
    name = name.lower().replace("-", "_")
    # Filter kwargs per family to avoid unexpected keyword errors
    if name in ("cnn_lstm_attn", "tier1", "bhatti"):
        allowed = {
            "conv_channels",
            "branch_channels",
            "lstm_hidden",
            "attn_heads",
            "dropout_attn",
            "dropout_fc",
            "use_se",
        }
        kw = {k: v for k, v in kwargs.items() if k in allowed}
        return CNNLSTMAttention(in_channels=in_channels, num_classes=num_classes, **kw)
    if name == "cnn1d":
        return CNN1D(in_channels, num_classes)
    if name == "lstm":
        return LSTMOnly(in_channels, num_classes)
    if name == "cnn_lstm":
        return CNNLSTM(in_channels, num_classes)
    if name == "tcn":
        allowed = {"channels", "levels", "kernel", "dropout"}
        kw = {k: v for k, v in kwargs.items() if k in allowed}
        return TCN(in_channels=in_channels, num_classes=num_classes, **kw)
    if name == "modern_tcn":
        allowed = {"channels", "levels", "kernel", "dropout"}
        kw = {k: v for k, v in kwargs.items() if k in allowed}
        return ModernTCN(in_channels=in_channels, num_classes=num_classes, **kw)
    if name == "tsmixer":
        allowed = {"seq_len", "d_model", "n_blocks", "dropout"}
        kw = {k: v for k, v in kwargs.items() if k in allowed}
        return TSMixer(in_channels=in_channels, num_classes=num_classes, **kw)
    if name == "patchtst":
        allowed = {"seq_len", "patch_len", "d_model", "n_heads", "n_layers", "dropout"}
        kw = {k: v for k, v in kwargs.items() if k in allowed}
        return PatchTST(in_channels=in_channels, num_classes=num_classes, **kw)
    if name == "transformer":
        allowed = {"d_model", "n_heads", "n_layers", "dropout", "max_len"}
        kw = {k: v for k, v in kwargs.items() if k in allowed}
        return TransformerEncoder1D(in_channels=in_channels, num_classes=num_classes, **kw)
    if name in ("harmamba", "mamba", "bi_mamba"):
        allowed = {
            "d_model",
            "n_layers",
            "d_state",
            "expand",
            "dropout",
            "seq_len",
            "patch_len",
            "use_patch",
        }
        kw = {k: v for k, v in kwargs.items() if k in allowed}
        return HARMamba(in_channels=in_channels, num_classes=num_classes, **kw)
    raise ValueError(f"Unknown model: {name}. Choose from {ALL_BACKBONES}")


def kwargs_for_model(name: str, backbones_cfg: dict | None = None) -> dict:
    """Pull model_kwargs for ``name`` from configs/backbones.yaml-style dict."""
    if not backbones_cfg:
        return {}
    name = name.lower().replace("-", "_")
    if name in ("mamba", "bi_mamba"):
        name = "harmamba"
    return dict(backbones_cfg.get("model_kwargs", {}).get(name, {}))
