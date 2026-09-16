from __future__ import annotations

from dataclasses import asdict, dataclass

from ..features.biomechanics import BiomechanicalFeatures
from .knn_memory import MemoryCase


@dataclass
class ConstraintPack:
    """Biomechanical constraint pack Σ(w) for Actor–Critic adjudication."""

    freefall_duration_s: float
    impact_magnitude_g: float
    post_impact_stillness_s: float
    body_tilt_deg: float
    p_fall: float
    knn_fall_votes: int
    knn_adl_votes: int
    knn_fall_ratio: float
    freefall_band: str  # short | moderate | long
    impact_band: str  # low | moderate | high | severe
    stillness_band: str  # brief | delayed | prolonged
    tilt_band: str  # upright | partial | horizontal
    fall_support_score: float
    near_fall_likely: bool

    def to_dict(self) -> dict:
        return asdict(self)

    def checklist_text(self) -> str:
        return "\n".join(
            [
                f"- Free-fall: {self.freefall_duration_s:.3f}s ({self.freefall_band})",
                f"- Impact: {self.impact_magnitude_g:.2f}g ({self.impact_band})",
                f"- Stillness: {self.post_impact_stillness_s:.2f}s ({self.stillness_band})",
                f"- Tilt: {self.body_tilt_deg:.1f} deg ({self.tilt_band})",
                f"- Tier-1 p(fall): {self.p_fall:.3f}",
                f"- kNN votes: {self.knn_fall_votes} fall / {self.knn_adl_votes} ADL "
                f"(ratio={self.knn_fall_ratio:.2f})",
                f"- Fall-support score: {self.fall_support_score:.3f}",
                f"- Near-fall pattern likely: {self.near_fall_likely}",
            ]
        )


def build_constraint_pack(
    feats: BiomechanicalFeatures,
    p_fall: float,
    cases: list[MemoryCase] | None = None,
) -> ConstraintPack:
    cases = cases or []
    knn_fall = sum(1 for c in cases if c.label == 1)
    knn_adl = sum(1 for c in cases if c.label == 0)
    total = max(1, knn_fall + knn_adl)
    ratio = knn_fall / total if cases else 0.5

    # Bands scaled to a 90-frame window (~0.45s at 200 Hz). Absolute 3s/10s
    # stillness never fires in this window and made the old Critic treat real falls as near-falls.
    if feats.freefall_duration_s >= 0.12:
        ff_band = "long"
    elif feats.freefall_duration_s >= 0.06:
        ff_band = "moderate"
    else:
        ff_band = "short"

    if feats.impact_magnitude_g >= 5.0:
        impact_band = "severe"
    elif feats.impact_magnitude_g >= 3.0:
        impact_band = "high"
    elif feats.impact_magnitude_g >= 2.0:
        impact_band = "moderate"
    else:
        impact_band = "low"

    if feats.post_impact_stillness_s >= 0.20:
        still_band = "prolonged"
    elif feats.post_impact_stillness_s >= 0.10:
        still_band = "delayed"
    else:
        still_band = "brief"

    if feats.body_tilt_deg >= 60:
        tilt_band = "horizontal"
    elif feats.body_tilt_deg >= 35:
        tilt_band = "partial"
    else:
        tilt_band = "upright"

    score = 0.0
    score += 0.25 if ff_band == "long" else (-0.1 if ff_band == "short" else 0.05)
    score += {"severe": 0.3, "high": 0.25, "moderate": 0.1, "low": -0.1}[impact_band]
    score += {"prolonged": 0.3, "delayed": 0.25, "brief": -0.15}[still_band]
    score += {"horizontal": 0.2, "partial": 0.05, "upright": -0.1}[tilt_band]
    score += 0.15 * float(p_fall)
    if cases:
        score += 0.2 * (ratio - 0.5)

    near_fall = (
        ff_band in ("short", "moderate")
        and still_band == "brief"
        and tilt_band != "horizontal"
        and impact_band in ("low", "moderate")
        and score < 0.20
    )

    return ConstraintPack(
        freefall_duration_s=float(feats.freefall_duration_s),
        impact_magnitude_g=float(feats.impact_magnitude_g),
        post_impact_stillness_s=float(feats.post_impact_stillness_s),
        body_tilt_deg=float(feats.body_tilt_deg),
        p_fall=float(p_fall),
        knn_fall_votes=int(knn_fall),
        knn_adl_votes=int(knn_adl),
        knn_fall_ratio=float(ratio),
        freefall_band=ff_band,
        impact_band=impact_band,
        stillness_band=still_band,
        tilt_band=tilt_band,
        fall_support_score=float(score),
        near_fall_likely=bool(near_fall),
    )


_SIGMA_TOKENS = (
    "free-fall",
    "freefall",
    "impact",
    "stillness",
    "tilt",
    "knn",
    "k-nn",
    "p(fall)",
    "p_fall",
    "sigma",
    "σ",
    "biomechan",
)


def rationale_cites_sigma(text: str) -> bool:
    """True if a rationale mentions at least two Σ(w) evidence terms."""
    t = (text or "").lower()
    hits = sum(1 for tok in _SIGMA_TOKENS if tok in t)
    return hits >= 2


def veto_score(pack: ConstraintPack) -> float:
    """Monotone evidence-*against*-fall in [0, 1].

    High score means Σ(w) argues the Actor's fall claim is weakly supported
    (short free-fall, weak impact, brief stillness, upright, ADL neighbours).
    The Critic may veto fall→ADL only when this score exceeds a CRC threshold.
    """
    ff = {"short": 1.0, "moderate": 0.40, "long": 0.0}[pack.freefall_band]
    impact = {"low": 1.0, "moderate": 0.60, "high": 0.20, "severe": 0.0}[pack.impact_band]
    still = {"brief": 1.0, "delayed": 0.35, "prolonged": 0.0}[pack.stillness_band]
    tilt = {"upright": 1.0, "partial": 0.45, "horizontal": 0.0}[pack.tilt_band]
    knn_adl = 1.0 - float(pack.knn_fall_ratio)
    p_adl = 1.0 - float(pack.p_fall)
    near = 1.0 if pack.near_fall_likely else 0.0
    # Invert fall-support (typical range ~[-0.45, 1.1]) into [0, 1].
    support_against = max(0.0, min(1.0, (0.35 - float(pack.fall_support_score)) / 0.80))
    score = (
        0.20 * ff
        + 0.20 * impact
        + 0.16 * still
        + 0.10 * tilt
        + 0.12 * knn_adl
        + 0.08 * p_adl
        + 0.08 * near
        + 0.06 * support_against
    )
    return float(max(0.0, min(1.0, score)))
