from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class BiomechanicalFeatures:
    freefall_duration_s: float
    impact_magnitude_g: float
    post_impact_stillness_s: float
    body_tilt_deg: float
    duration_s: float
    mean_acc_g: float
    peak_gyro_dps: float

    def to_dict(self) -> dict:
        return asdict(self)


def _svm(acc: np.ndarray) -> np.ndarray:
    """Signal magnitude vector for (T, 3) accel in g."""
    return np.sqrt(np.sum(acc ** 2, axis=1) + 1e-12)


def extract_biomechanics(
    window: np.ndarray,
    sample_rate_hz: float = 200.0,
    freefall_g_threshold: float = 0.5,
    stillness_var_threshold: float = 0.05,
) -> BiomechanicalFeatures:
    """Extract fall-relevant features from a (T, C) window.

    Expects accel in g on channels 0:3. Gyro (deg/s) optional on 3:6.
    """
    if window.ndim != 2:
        raise ValueError("window must be (T, C)")
    t, c = window.shape
    acc = window[:, :3]
    gyro = window[:, 3:6] if c >= 6 else None
    svm = _svm(acc)
    dt = 1.0 / sample_rate_hz

    # Free-fall: contiguous samples with SVM below threshold (near weightlessness)
    below = svm < freefall_g_threshold
    freefall_s = 0.0
    if below.any():
        # longest run
        runs = np.diff(np.concatenate([[0], below.view(np.int8), [0]]))
        starts = np.where(runs == 1)[0]
        ends = np.where(runs == -1)[0]
        lengths = ends - starts
        freefall_s = float(lengths.max() * dt) if len(lengths) else 0.0

    impact_g = float(svm.max())
    impact_idx = int(np.argmax(svm))

    # Post-impact stillness: low rolling variance after impact
    stillness_s = 0.0
    if impact_idx < t - 2:
        post = svm[impact_idx:]
        # variance in short windows
        w = max(3, int(0.1 * sample_rate_hz))
        still_count = 0
        for i in range(0, len(post) - w + 1):
            if float(np.var(post[i : i + w])) < stillness_var_threshold:
                still_count += 1
            else:
                if still_count:
                    break
        stillness_s = float(still_count * dt)

    # Body tilt from gravity direction change (start vs end mean accel)
    a0 = acc[: max(1, t // 10)].mean(axis=0)
    a1 = acc[-max(1, t // 10) :].mean(axis=0)
    n0 = a0 / (np.linalg.norm(a0) + 1e-8)
    n1 = a1 / (np.linalg.norm(a1) + 1e-8)
    cos = float(np.clip(np.dot(n0, n1), -1.0, 1.0))
    tilt = float(np.degrees(np.arccos(cos)))

    peak_gyro = float(np.max(np.linalg.norm(gyro, axis=1))) if gyro is not None else 0.0

    return BiomechanicalFeatures(
        freefall_duration_s=freefall_s,
        impact_magnitude_g=impact_g,
        post_impact_stillness_s=stillness_s,
        body_tilt_deg=tilt,
        duration_s=float(t * dt),
        mean_acc_g=float(svm.mean()),
        peak_gyro_dps=peak_gyro,
    )
