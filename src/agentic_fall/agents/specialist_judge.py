"""Near-fall specialist + precision-preserving judge (no activity codes).

The specialist is a tool-grounded tabular model over:
  Tier-1 p_fall, kNN neighbor fall fraction, and biomechanical features.
The judge abstains to Tier-1 unless specialist confidence clears a val-tuned bar.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


FEATURE_NAMES = (
    "p_fall",
    "knn_fall_frac",
    "freefall_duration_s",
    "impact_magnitude_g",
    "post_impact_stillness_s",
    "body_tilt_deg",
    "duration_s",
    "mean_acc_g",
    "peak_gyro_dps",
    "impact_x_stillness",
    "tilt_x_pfall",
)


def featurize_row(
    *,
    p_fall: float,
    knn_fall_frac: float,
    feats: dict[str, float],
) -> np.ndarray:
    ff = float(feats.get("freefall_duration_s", 0.0))
    imp = float(feats.get("impact_magnitude_g", 0.0))
    still = float(feats.get("post_impact_stillness_s", 0.0))
    tilt = float(feats.get("body_tilt_deg", 0.0))
    dur = float(feats.get("duration_s", 0.0))
    mean_a = float(feats.get("mean_acc_g", 0.0))
    gyro = float(feats.get("peak_gyro_dps", 0.0))
    p = float(p_fall)
    kf = float(knn_fall_frac)
    return np.asarray(
        [
            p,
            kf,
            ff,
            imp,
            still,
            tilt,
            dur,
            mean_a,
            gyro,
            imp * still,
            (tilt / 90.0) * p,
        ],
        dtype=np.float64,
    )


@dataclass
class SpecialistJudge:
    """Sklearn classifier + confidence threshold for abstention to Tier-1."""

    model: Any
    conf_threshold: float = 0.55
    prefer_metric: str = "f1"  # f1 | cost

    def predict_proba_fall(self, X: np.ndarray) -> np.ndarray:
        proba = self.model.predict_proba(X)
        # class 1 = fall
        classes = list(self.model.classes_)
        if 1 in classes:
            return proba[:, classes.index(1)]
        return proba[:, -1]

    def decide(
        self,
        X: np.ndarray,
        tier1_pred: np.ndarray,
        escalated: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (y_pred, used_specialist, p_spec)."""
        p_spec = self.predict_proba_fall(X)
        y_spec = (p_spec >= 0.5).astype(np.int64)
        conf = np.maximum(p_spec, 1.0 - p_spec)
        use = escalated.astype(bool) & (conf >= float(self.conf_threshold))
        out = tier1_pred.astype(np.int64).copy()
        out[use] = y_spec[use]
        return out, use.astype(np.int64), p_spec


def train_specialist(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = 42,
) -> Any:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    y = np.asarray(y, dtype=np.int64)
    X = np.asarray(X, dtype=np.float64)
    if len(np.unique(y)) < 2:
        # degenerate — constant predictor
        clf = LogisticRegression()
        clf.fit(X, y if len(y) else np.array([0]))
        return clf

    # Sample weights for class balance (works across sklearn versions)
    n_pos = max(1, int((y == 1).sum()))
    n_neg = max(1, int((y == 0).sum()))
    w = np.where(y == 1, n_neg / n_pos, 1.0).astype(np.float64)
    clf = HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.08,
        max_iter=200,
        l2_regularization=0.1,
        random_state=seed,
    )
    clf.fit(X, y, sample_weight=w)
    return clf


def calibrate_judge_threshold(
    specialist_model: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    tier1_val: np.ndarray,
    esc_val: np.ndarray,
    *,
    cost_fn: float = 10.0,
    cost_fp: float = 1.0,
    metric: str = "f1",
) -> tuple[float, dict[str, float]]:
    from sklearn.metrics import f1_score

    y_val = np.asarray(y_val, dtype=np.int64)
    tier1_val = np.asarray(tier1_val, dtype=np.int64)
    esc_val = np.asarray(esc_val, dtype=bool)
    tmp = SpecialistJudge(model=specialist_model, conf_threshold=0.5)
    p_spec = tmp.predict_proba_fall(X_val)

    best_t, best_score, best_stats = 0.55, -1e18, {}
    for t in np.linspace(0.50, 0.95, 46):
        conf = np.maximum(p_spec, 1.0 - p_spec)
        use = esc_val & (conf >= float(t))
        pred = tier1_val.copy()
        pred[use] = (p_spec[use] >= 0.5).astype(np.int64)
        f1 = float(f1_score(y_val, pred, zero_division=0))
        fp = int(((pred == 1) & (y_val == 0)).sum())
        fn = int(((pred == 0) & (y_val == 1)).sum())
        c = cost_fn * fn + cost_fp * fp
        # primary: beat/match F1; secondary: lower cost
        if metric == "cost":
            score = -c + 0.01 * f1
        else:
            score = f1 * 1000.0 - 0.001 * c
        if score > best_score:
            best_score = score
            best_t = float(t)
            best_stats = {
                "f1": f1,
                "cost": float(c),
                "override_rate": float(use.mean()),
                "conf_threshold": best_t,
            }
    return best_t, best_stats
