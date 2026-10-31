from __future__ import annotations

import re

from ..features.biomechanics import BiomechanicalFeatures

# SisFall / KFall activity tokens that must never reach System-2 prompts.
_ACTIVITY_CODE_RE = re.compile(
    r"(?i)(?:"
    r"activity\s+context\s+code\s*:\s*[DF]\d{2}"
    r"|\b[DF]\d{2}\b"
    r")"
)
_ACTIVITY_LINE_RE = re.compile(
    r"(?im)^\s*[-*]?\s*Activity\s+context\s+code\s*:.*$"
)


def evidence_contains_activity_code(text: str) -> bool:
    """True if free text contains SisFall-style activity / trial codes."""
    if not text:
        return False
    return _ACTIVITY_CODE_RE.search(text) is not None


def strip_activity_from_evidence(text: str) -> str:
    """Remove activity-context lines and lingering Dxx/Fxx tokens from evidence."""
    if not text:
        return text
    cleaned = _ACTIVITY_LINE_RE.sub("", text)
    # Drop bare activity tokens that are not part of numeric units (keep e.g. nothing else).
    cleaned = re.sub(r"(?i)\b[DF]\d{2}\b", "[REDACTED]", cleaned)
    # Collapse blank lines
    lines = [ln for ln in cleaned.splitlines() if ln.strip()]
    return "\n".join(lines)


def serialize_evidence(
    feats: BiomechanicalFeatures,
    activity: str | None = None,
    p_fall: float | None = None,
    *,
    include_activity: bool = False,
) -> str:
    """Convert biomechanical features into structured natural-language evidence.

    ``include_activity`` defaults to False for journal / no-leak runs. Activity
    codes are label-derived in SisFall and must not reach System-2 prompts.
    """
    lines = [
        f"Free-fall duration: {feats.freefall_duration_s:.3f}s",
        f"Impact magnitude: {feats.impact_magnitude_g:.2f}g",
        f"Post-impact stillness: {feats.post_impact_stillness_s:.2f}s",
        f"Body tilt change: {feats.body_tilt_deg:.1f} deg",
        f"Event duration: {feats.duration_s:.2f}s",
        f"Mean acceleration: {feats.mean_acc_g:.2f}g",
        f"Peak gyroscope: {feats.peak_gyro_dps:.1f} deg/s",
    ]
    if include_activity and activity is not None:
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
