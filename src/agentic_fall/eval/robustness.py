"""IMU window perturbations for robustness evaluation."""
from __future__ import annotations

import numpy as np


def _rotation_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    r, p, y = np.deg2rad([roll_deg, pitch_deg, yaw_deg])
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float32)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float32)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float32)
    return rz @ ry @ rx


def rotate_window(window_tc: np.ndarray, roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Rotate 3-axis channels in (T, C) window; C>=6 assumed accel+gyro triplets."""
    w = np.asarray(window_tc, dtype=np.float32).copy()
    r = _rotation_matrix(roll_deg, pitch_deg, yaw_deg)
    for start in (0, 3):
        if w.shape[1] >= start + 3:
            w[:, start : start + 3] = w[:, start : start + 3] @ r.T
    return w


def resample_window(window_tc: np.ndarray, src_hz: float, tgt_hz: float) -> np.ndarray:
    """Linear resample time axis then restore original length via interpolation."""
    w = np.asarray(window_tc, dtype=np.float32)
    t = np.arange(w.shape[0], dtype=np.float32) / float(src_hz)
    t_new = np.arange(0, t[-1], 1.0 / float(tgt_hz), dtype=np.float32)
    if t_new.size < 2:
        return w
    out = np.zeros((w.shape[0], w.shape[1]), dtype=np.float32)
    src_idx = np.linspace(0, t_new.size - 1, w.shape[0])
    for c in range(w.shape[1]):
        interp = np.interp(t_new, t, w[:, c])
        out[:, c] = np.interp(src_idx, np.arange(interp.size), interp)
    return out
