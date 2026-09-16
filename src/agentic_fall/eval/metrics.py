from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def binary_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> dict[str, float]:
    yt = np.asarray(list(y_true), dtype=int)
    yp = np.asarray(list(y_pred), dtype=int)
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    spec = tn / (tn + fp + 1e-12)
    sens = tp / (tp + fn + 1e-12)
    return {
        "accuracy": float(accuracy_score(yt, yp)),
        "precision": float(precision_score(yt, yp, zero_division=0)),
        "recall": float(recall_score(yt, yp, zero_division=0)),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "f1": float(f1_score(yt, yp, zero_division=0)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def expected_calibration_error(
    y_true: Iterable[int],
    p_fall: Iterable[float],
    n_bins: int = 10,
) -> float:
    yt = np.asarray(list(y_true), dtype=float)
    pp = np.asarray(list(p_fall), dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (pp >= bins[i]) & (pp < bins[i + 1] if i < n_bins - 1 else pp <= bins[i + 1])
        if not np.any(m):
            continue
        acc = yt[m].mean()
        conf = pp[m].mean()
        ece += m.mean() * abs(acc - conf)
    return float(ece)


GRADED_ACTION_COST = {
    "emergency": 10.0,
    "notify_caregiver": 4.0,
    "monitor": 1.0,
    "log": 0.0,
}


def agentic_metrics(
    y_true: list[int],
    y_pred: list[int],
    escalated: list[bool],
    activities: list[str] | None = None,
    ambiguous_codes: list[str] | None = None,
    latencies_ms: list[float] | None = None,
    p_falls: list[float] | None = None,
    cost_fn: float = 10.0,
    cost_fp: float = 1.0,
    actions: list[str] | None = None,
    rationales: list[str] | None = None,
) -> dict[str, float]:
    """Safety / agentic metrics with explicit near-fall and escalated counts.

    - ``false_alarm_rate_near_fall``: FAR on activity codes in ``ambiguous_codes`` (D18/D19).
    - ``escalated_f1``: F1 on gate-escalated windows.
    - Legacy aliases ``false_alarm_rate_ambiguous`` / ``ambiguous_f1`` are kept for CSV compat.
    """
    base = binary_metrics(y_true, y_pred)
    esc = np.asarray(escalated, dtype=bool)
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    out = dict(base)
    out["escalation_rate"] = float(esc.mean()) if len(esc) else 0.0
    out["n_escalated"] = int(esc.sum()) if len(esc) else 0

    codes = list(ambiguous_codes or [])
    acts = [str(a) for a in activities] if activities is not None else None
    out["n_d18"] = int(sum(1 for a in acts if a == "D18")) if acts is not None else 0
    out["n_d19"] = int(sum(1 for a in acts if a == "D19")) if acts is not None else 0
    out["n_near_fall"] = 0
    out["false_alarm_rate_near_fall"] = 0.0
    out["escalated_f1"] = 0.0

    if acts is not None and codes:
        amb_set = set(codes)
        amb_idx = [i for i, a in enumerate(acts) if a in amb_set]
        out["n_near_fall"] = len(amb_idx)
        if not amb_idx:
            warnings.warn(
                f"No near-fall activities {codes} in eval slice "
                f"(n_d18={out['n_d18']}, n_d19={out['n_d19']}). "
                "FAR reported as 0.0 — re-prepare data if this is unexpected.",
                stacklevel=2,
            )
            out["false_alarm_rate_near_fall"] = 0.0
        else:
            n_neg = sum(1 for i in amb_idx if yt[i] == 0)
            far = sum(1 for i in amb_idx if yt[i] == 0 and yp[i] == 1) / max(1, n_neg)
            out["false_alarm_rate_near_fall"] = float(far)
    elif acts is None or not codes:
        warnings.warn(
            "activities/ambiguous_codes missing — near-fall FAR set to 0.0",
            stacklevel=2,
        )

    if esc.any():
        out["escalated_f1"] = float(f1_score(yt[esc], yp[esc], zero_division=0))
    else:
        out["escalated_f1"] = 0.0

    # Legacy aliases used by earlier tables / export scripts
    out["false_alarm_rate_ambiguous"] = out["false_alarm_rate_near_fall"]
    out["ambiguous_f1"] = out["escalated_f1"]

    out["missed_fall_rate"] = float(base["fn"] / max(1, base["fn"] + base["tp"]))
    out["expected_response_cost"] = float(cost_fn * base["fn"] + cost_fp * base["fp"])
    out["false_alarms_per_1000"] = float(1000.0 * base["fp"] / max(1, len(yt)))

    if latencies_ms is not None and escalated is not None:
        lat = np.asarray(latencies_ms, dtype=float)
        out["latency_tier1_ms"] = float(lat[~esc].mean()) if (~esc).any() else 0.0
        out["latency_escalated_ms"] = float(lat[esc].mean()) if esc.any() else 0.0
        out["latency_mean_ms"] = float(lat.mean()) if len(lat) else 0.0

    if p_falls is not None:
        out["ece"] = expected_calibration_error(y_true, p_falls)

    n = max(1, len(yt))
    acts_list = [str(a) for a in (actions or [])]
    out["emergency_rate"] = 0.0
    out["emergency_rate_adl"] = 0.0
    out["emergency_rate_near_fall"] = 0.0
    out["graded_action_cost"] = 0.0
    out["rationale_grounding_rate"] = 0.0
    if acts_list and len(acts_list) == len(yt):
        emerg = [a == "emergency" for a in acts_list]
        out["emergency_rate"] = float(sum(emerg) / n)
        n_adl = int((yt == 0).sum())
        out["emergency_rate_adl"] = float(
            sum(1 for i, e in enumerate(emerg) if e and yt[i] == 0) / max(1, n_adl)
        )
        if acts is not None and codes:
            amb_set = set(codes)
            nf_idx = [i for i, a in enumerate(acts) if a in amb_set]
            out["emergency_rate_near_fall"] = float(
                sum(1 for i in nf_idx if acts_list[i] == "emergency") / max(1, len(nf_idx))
            )
        graded = 0.0
        for i, act in enumerate(acts_list):
            if yt[i] == 1 and yp[i] == 0:
                graded += cost_fn
            elif yt[i] == 0:
                graded += GRADED_ACTION_COST.get(act, 0.0)
        out["graded_action_cost"] = float(graded)
    if rationales:
        from ..agents.constraints import rationale_cites_sigma

        out["rationale_grounding_rate"] = float(
            sum(1 for r in rationales if rationale_cites_sigma(str(r))) / max(1, len(rationales))
        )
    return out


def crc_veto_metrics(
    y_true: list[int],
    pred_before: list[int],
    pred_after: list[int],
    *,
    vetoed: list[bool] | None = None,
    certified_alpha: float | None = None,
    lambda_star: float | None = None,
    feasible: bool | None = None,
    aurc: float | None = None,
) -> dict[str, float]:
    """Metrics for one-directional fall→ADL vetoes."""
    yt = np.asarray(y_true, dtype=int)
    before = np.asarray(pred_before, dtype=int)
    after = np.asarray(pred_after, dtype=int)
    if vetoed is None:
        mask = (before == 1) & (after == 0)
    else:
        mask = np.asarray(vetoed, dtype=bool)
    n = max(1, len(yt))
    n_veto = int(mask.sum())
    induced_fn = int((mask & (yt == 1)).sum())
    veto_tp_adl = int((mask & (yt == 0)).sum())
    return {
        "veto_rate": float(n_veto / n),
        "veto_n": float(n_veto),
        "induced_fn": float(induced_fn),
        "veto_precision": float(veto_tp_adl / max(1, n_veto)),
        "certified_alpha": float(certified_alpha) if certified_alpha is not None else float("nan"),
        "lambda_star": float(lambda_star) if lambda_star is not None else float("nan"),
        "crc_feasible": 1.0 if feasible else 0.0,
        "aurc": float(aurc) if aurc is not None else float("nan"),
    }


def metrics_to_latex(rows: list[dict], caption: str = "Results") -> str:
    if not rows:
        return ""
    prefer = [
        "name",
        "f1",
        "recall",
        "specificity",
        "precision",
        "accuracy",
        "missed_fall_rate",
        "false_alarm_rate_near_fall",
        "escalated_f1",
        "escalation_rate",
        "expected_response_cost",
        "graded_action_cost",
        "emergency_rate_adl",
        "emergency_rate_near_fall",
        "rationale_grounding_rate",
        "ece",
        "params",
        "latency_ms",
        "n_near_fall",
        "n_escalated",
        "false_alarms_per_1000",
        "veto_rate",
        "veto_precision",
        "induced_fn",
        "certified_alpha",
        "lambda_star",
        "aurc",
    ]
    keys = [k for k in prefer if any(k in r for r in rows)]
    extras = sorted({k for r in rows for k in r if k not in keys and k != "name"})
    # Keep tables readable for paper exports
    keys = keys + [k for k in extras if k in ("delta_f1", "detector_f1", "backbone", "mode", "n")]
    header = " & ".join(["Method"] + [k for k in keys if k != "name"]) + " \\\\"
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\begin{tabular}{" + "l" + "c" * max(1, len(keys) - (1 if "name" in keys else 0)) + "}",
        "\\toprule",
        header,
        "\\midrule",
    ]
    col_keys = [k for k in keys if k != "name"]
    for r in rows:
        vals = [str(r.get("name", ""))]
        for k in col_keys:
            v = r.get(k, "")
            if isinstance(v, float):
                vals.append("nan" if v != v else f"{v:.3f}")
            else:
                vals.append(str(v))
        lines.append(" & ".join(vals) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)
