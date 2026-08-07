from __future__ import annotations

from ..features.biomechanics import BiomechanicalFeatures


def serialize_evidence(
    feats: BiomechanicalFeatures,
    activity: str | None = None,
    p_fall: float | None = None,
) -> str:
    """Convert biomechanical features into structured natural-language evidence."""
    lines = [
        f"Free-fall duration: {feats.freefall_duration_s:.3f}s",
        f"Impact magnitude: {feats.impact_magnitude_g:.2f}g",
        f"Post-impact stillness: {feats.post_impact_stillness_s:.2f}s",
        f"Body tilt change: {feats.body_tilt_deg:.1f} deg",
        f"Event duration: {feats.duration_s:.2f}s",
        f"Mean acceleration: {feats.mean_acc_g:.2f}g",
        f"Peak gyroscope: {feats.peak_gyro_dps:.1f} deg/s",
    ]
    if activity is not None:
        lines.append(f"Activity context code: {activity}")
    if p_fall is not None:
        lines.append(f"Tier-1 fall probability: {p_fall:.3f}")
    return "\n".join(f"- {ln}" for ln in lines)


def evidence_caption(feats: BiomechanicalFeatures) -> str:
    """Short caption for sensor-language alignment / embedding."""
    return (
        f"free-fall {feats.freefall_duration_s:.2f}s; "
        f"impact {feats.impact_magnitude_g:.2f}g; "
        f"stillness {feats.post_impact_stillness_s:.2f}s; "
        f"tilt {feats.body_tilt_deg:.0f} deg"
    )
