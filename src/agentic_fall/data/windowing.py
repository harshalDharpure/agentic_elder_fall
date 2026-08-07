from __future__ import annotations

import numpy as np


def sliding_windows(
    signal: np.ndarray,
    window: int,
    hop: int,
) -> np.ndarray:
    """Slice (T, C) into (N, window, C) with given hop."""
    t = signal.shape[0]
    if t < window:
        pad = np.zeros((window - t, signal.shape[1]), dtype=signal.dtype)
        signal = np.concatenate([signal, pad], axis=0)
        return signal[None, ...]
    starts = range(0, t - window + 1, hop)
    return np.stack([signal[s : s + window] for s in starts], axis=0)


def resample_signal(signal: np.ndarray, src_hz: float, dst_hz: float) -> np.ndarray:
    """Linear resample along time for (T, C)."""
    if abs(src_hz - dst_hz) < 1e-6:
        return signal
    t_src = signal.shape[0]
    t_dst = max(1, int(round(t_src * dst_hz / src_hz)))
    x_old = np.linspace(0.0, 1.0, t_src)
    x_new = np.linspace(0.0, 1.0, t_dst)
    out = np.stack(
        [np.interp(x_new, x_old, signal[:, c]) for c in range(signal.shape[1])],
        axis=1,
    )
    return out.astype(np.float32)
