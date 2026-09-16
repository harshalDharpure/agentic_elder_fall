#!/usr/bin/env python3
"""Offline α-sweep from cached val calibration + optional test window dump."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_fall.eval.conformal import apply_veto, area_under_risk_coverage, risk_coverage_curve
from agentic_fall.eval.metrics import binary_metrics
from agentic_fall.utils.io import save_json

ALPHAS = (0.01, 0.05, 0.10, 0.20)


def _metrics_from_windows(windows: list[dict], lam: float | None) -> dict:
    actor = [str(w["actor_pred"]) for w in windows]
    scores = [float(w["veto_score"] if w.get("veto_score") is not None else 0.0) for w in windows]
    final = apply_veto(actor, scores, lam)
    yt = [int(w["y"]) for w in windows]
    yp = [1 if p == "fall" else 0 for p in final]
    m = binary_metrics(yt, yp)
    n = max(1, len(yt))
    veto = [(a == "fall" and f == "adl") for a, f in zip(actor, final)]
    induced_fn = sum(1 for v, y in zip(veto, yt) if v and y == 1)
    veto_adl = sum(1 for v, y in zip(veto, yt) if v and y == 0)
    n_veto = sum(veto)
    m["veto_rate"] = n_veto / n
    m["induced_fn"] = induced_fn
    m["veto_precision"] = veto_adl / max(1, n_veto)
    m["false_alarms_per_1000"] = 1000.0 * m["fp"] / n
    m["expected_response_cost"] = 10.0 * m["fn"] + 1.0 * m["fp"]
    m["lambda"] = lam
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()
    fold = args.fold
    cal_path = ROOT / args.results_dir / f"fold{fold}" / f"veto_calibration_fold{fold}.json"
    if not cal_path.exists():
        raise SystemExit(f"Missing {cal_path}; run scripts/calibrate_crc_veto.py first")
    cal = json.loads(cal_path.read_text())
    windows_path = ROOT / args.results_dir / f"fold{fold}" / f"crc_veto_windows_fold{fold}.json"
    windows = None
    if windows_path.exists():
        windows = json.loads(windows_path.read_text())
        if isinstance(windows, dict):
            windows = windows.get("windows") or windows.get("records")

    rows = []
    for a in ALPHAS:
        sel = (cal.get("by_alpha") or {}).get(f"{float(a):.2f}", {})
        lam = sel.get("lambda_star")
        feasible = bool(sel.get("feasible"))
        row = {
            "alpha": a,
            "feasible": feasible,
            "lambda_star": lam,
            "U": sel.get("U"),
            "r_hat": sel.get("r_hat"),
            "n_eligible": sel.get("n_eligible"),
        }
        if windows:
            row["test"] = _metrics_from_windows(windows, lam if feasible else None)
        rows.append(row)

    curve = cal.get("curve") or []
    payload = {
        "fold": fold,
        "alpha_break_even": cal.get("alpha_break_even"),
        "aurc_val": cal.get("aurc") or area_under_risk_coverage(curve),
        "route_fp_diagnostic": cal.get("route_fp_diagnostic"),
        "rows": rows,
        "curve": curve,
    }
    if windows:
        scores, labels = [], []
        for w in windows:
            if w.get("actor_pred") == "fall" and w.get("veto_score") is not None:
                scores.append(float(w["veto_score"]))
                labels.append(int(w["y"]))
        payload["test_curve"] = risk_coverage_curve(scores, labels)
        payload["test_aurc"] = area_under_risk_coverage(payload["test_curve"])

    out = ROOT / args.results_dir / f"fold{fold}" / f"crc_alpha_sweep_fold{fold}.json"
    save_json(payload, out)

    # Tiny PNG if matplotlib is available
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        xs = [c["coverage"] for c in curve]
        ys = [c["r_hat"] for c in curve]
        order = np.argsort(xs)
        ax.plot(np.asarray(xs)[order], np.asarray(ys)[order], label="val R-C")
        ax.axhline(cal.get("alpha_break_even") or 0.09, ls="--", color="gray", label="break-even α")
        ax.set_xlabel("Coverage (Actor-fall kept)")
        ax.set_ylabel("Empirical miss rate among vetoes")
        ax.set_title(f"CRC risk-coverage fold {fold}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out.with_suffix(".png"), dpi=140)
        plt.close(fig)
    except Exception as exc:
        print(f"skip figure: {exc}")
    print(json.dumps({"wrote": str(out), "rows": rows}, indent=2, default=str))


if __name__ == "__main__":
    main()
