#!/usr/bin/env python3
"""JBHI robustness suite: A1 leakage + B1 event metrics + B2 cheap baselines + B3/B4.

Expected CSV columns (one row per window):
  fold, subject, trial_id, activity_code, t_start, y_true, p, escalated,
  pred_primary, pred_stack, split
Optional: pred_stack_r1, pred_stack_r2, and sig_* evidence features.
Include train/calibration rows as well as test; for B1 include every window
of each test trial (not only the 1000-window sample).

Usage:
  python scripts/robustness_suite.py predictions.csv --prompts results/prompt_audit
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_fall.agents.evidence import evidence_contains_activity_code  # noqa: E402


def _binary_metrics(y: np.ndarray, pred: np.ndarray, cost_fn: float = 10.0) -> dict:
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-12, prec + rec)
    return {
        "n": int(len(y)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "f1": float(f1),
        "recall": float(rec),
        "precision": float(prec),
        "cost": float(cost_fn * fn + fp),
        "cost_per_1000": float(1000.0 * (cost_fn * fn + fp) / max(1, len(y))),
    }


def section_a1(prompts_dir: Path | None, df: pd.DataFrame) -> dict:
    hits = []
    scanned = 0
    if prompts_dir and prompts_dir.exists():
        for p in prompts_dir.rglob("*"):
            if not p.is_file():
                continue
            try:
                text = p.read_text(errors="ignore")
            except OSError:
                continue
            scanned += 1
            if evidence_contains_activity_code(text):
                hits.append(str(p))
    # Ambiguous D18/D19 System-2 accuracy (should not be near 1.0 if no leakage)
    amb = df[(df["split"] == "test") & (df["activity_code"].isin(["D18", "D19"]))]
    amb_esc = amb[amb["escalated"].astype(bool)] if len(amb) else amb
    s2_acc = None
    if len(amb_esc) and "pred_stack" in amb_esc:
        s2_acc = float((amb_esc["pred_stack"].astype(int) == amb_esc["y_true"].astype(int)).mean())
    return {
        "prompts_scanned": scanned,
        "activity_code_hits": len(hits),
        "hit_paths_sample": hits[:20],
        "d18_d19_escalated_n": int(len(amb_esc)),
        "d18_d19_system2_accuracy": s2_acc,
        "pass": len(hits) == 0 and (s2_acc is None or s2_acc < 0.98),
    }


def section_b1(df: pd.DataFrame, hop_s: float = 0.05) -> dict:
    """Trial-level recall, FA/hour on ADL trials, median time-to-detection."""
    te = df[df["split"] == "test"].copy()
    if te.empty or "trial_id" not in te.columns:
        return {"error": "need test rows with trial_id"}

    fall_trials = te[te["y_true"].astype(int) == 1].groupby(["fold", "trial_id"])
    caught = 0
    total = 0
    ttds = []
    for (_, _), g in fall_trials:
        total += 1
        g = g.sort_values("t_start")
        hits = g[g["pred_stack"].astype(int) == 1]
        if len(hits):
            caught += 1
            t0 = float(g["t_start"].iloc[0])
            ttds.append(float(hits["t_start"].iloc[0]) - t0)

    adl = te[te["y_true"].astype(int) == 0]
    # Approximate hours: n_windows * hop
    adl_hours = max(1e-9, len(adl) * hop_s / 3600.0)
    fa = int((adl["pred_stack"].astype(int) == 1).sum())
    fa_primary = int((adl["pred_primary"].astype(int) == 1).sum()) if "pred_primary" in adl else None

    return {
        "trial_recall_stack": float(caught / max(1, total)),
        "fall_trials": total,
        "trials_caught": caught,
        "false_alarms_per_hour_adl_stack": float(fa / adl_hours),
        "false_alarms_per_hour_adl_primary": (
            float(fa_primary / adl_hours) if fa_primary is not None else None
        ),
        "median_time_to_detection_s": float(np.median(ttds)) if ttds else None,
        "mean_time_to_detection_s": float(np.mean(ttds)) if ttds else None,
    }


def section_b2(df: pd.DataFrame) -> dict:
    """Non-LLM baselines on escalated test windows using sig_* features + p."""
    te = df[(df["split"] == "test") & (df["escalated"].astype(bool))].copy()
    if te.empty:
        return {"error": "no escalated test rows"}
    feat_cols = [c for c in te.columns if c.startswith("sig_")] + (
        ["p"] if "p" in te.columns else []
    )
    if len(feat_cols) < 2:
        return {"error": "need sig_* feature columns (and preferably p)"}

    # Train on escalated train/cal rows when available and both classes present.
    tr = df[(df["split"].isin(["train", "calibration", "cal"])) & (df["escalated"].astype(bool))]
    in_sample = False
    if tr.empty or tr["y_true"].nunique() < 2:
        tr = te  # fallback: report in-sample (flagged)
        in_sample = True

    Xtr = tr[feat_cols].astype(float).fillna(0.0).to_numpy()
    ytr = tr["y_true"].astype(int).to_numpy()
    Xte = te[feat_cols].astype(float).fillna(0.0).to_numpy()
    yte = te["y_true"].astype(int).to_numpy()

    out = {
        "n_escalated_test": int(len(te)),
        "n_train": int(len(tr)),
        "features": feat_cols,
        "in_sample_fallback": in_sample,
        "stack_llm": _binary_metrics(yte, te["pred_stack"].astype(int).to_numpy()),
    }
    if "pred_primary" in te:
        out["primary_on_escalated"] = _binary_metrics(
            yte, te["pred_primary"].astype(int).to_numpy()
        )

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.preprocessing import StandardScaler
    except ImportError as e:
        out["error"] = f"sklearn required: {e}"
        return out

    if len(set(ytr.tolist())) < 2:
        out["error"] = "need both classes among escalated windows for non-LLM baselines"
        return out

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    models = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "knn_vote": KNeighborsClassifier(n_neighbors=5),
    }
    try:
        models["hist_gradient_boosting"] = HistGradientBoostingClassifier(max_depth=4)
    except Exception:
        models["gradient_boosting"] = GradientBoostingClassifier(max_depth=3)

    for name, clf in models.items():
        clf.fit(Xtr_s, ytr)
        pred = clf.predict(Xte_s)
        out[name] = _binary_metrics(yte, pred)

    # Temperature scaling + Bayes threshold (β_FN=10 => τ* = c_FP/(c_FN+c_FP)=1/11)
    # Fit temperature on train logits from p
    if "p" in te.columns and "p" in tr.columns:
        p_tr = np.clip(tr["p"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
        # grid-search T
        best_t, best_cost = 1.0, 1e18
        for T in np.linspace(0.5, 5.0, 19):
            logit = np.log(p_tr / (1 - p_tr)) / T
            p_cal = 1 / (1 + np.exp(-logit))
            pred = (p_cal >= 1.0 / 11.0).astype(int)
            cost = 10 * ((pred == 0) & (ytr == 1)).sum() + ((pred == 1) & (ytr == 0)).sum()
            if cost < best_cost:
                best_cost, best_t = cost, float(T)
        p_te = np.clip(te["p"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
        logit = np.log(p_te / (1 - p_te)) / best_t
        p_cal = 1 / (1 + np.exp(-logit))
        pred = (p_cal >= 1.0 / 11.0).astype(int)
        out["temperature_bayes"] = {
            **_binary_metrics(yte, pred),
            "temperature": best_t,
            "tau_star": 1.0 / 11.0,
        }
    return out


def section_b3(df: pd.DataFrame) -> dict:
    cols = [c for c in ("pred_stack", "pred_stack_r1", "pred_stack_r2") if c in df.columns]
    te = df[df["split"] == "test"]
    if len(cols) < 2:
        return {"error": "need pred_stack_r1/r2 for run variance"}
    f1s = []
    for c in cols:
        f1s.append(_binary_metrics(te["y_true"], te[c])["f1"])
    # pairwise agreement
    agree = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            agree.append(float((te[cols[i]].astype(int) == te[cols[j]].astype(int)).mean()))
    return {
        "f1_runs": f1s,
        "f1_mean": float(np.mean(f1s)),
        "f1_sd": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
        "label_agreement_mean": float(np.mean(agree)) if agree else None,
    }


def section_b4(df: pd.DataFrame) -> dict:
    """Per-subject paired Wilcoxon on F1 improvement (stack - primary)."""
    te = df[df["split"] == "test"]
    if "subject" not in te.columns:
        return {"error": "need subject column"}
    deltas = []
    for subj, g in te.groupby("subject"):
        m_p = _binary_metrics(g["y_true"], g["pred_primary"])
        m_s = _binary_metrics(g["y_true"], g["pred_stack"])
        deltas.append(m_s["f1"] - m_p["f1"])
    deltas = np.asarray(deltas, dtype=float)
    pval = None
    try:
        from scipy.stats import wilcoxon

        if len(deltas) >= 5 and np.any(deltas != 0):
            pval = float(wilcoxon(deltas).pvalue)
    except Exception as e:
        pval = f"unavailable: {e}"
    # subject bootstrap CI on mean delta
    rng = np.random.default_rng(42)
    boots = []
    for _ in range(10000):
        samp = rng.choice(deltas, size=len(deltas), replace=True)
        boots.append(float(samp.mean()))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "n_subjects": int(len(deltas)),
        "mean_f1_delta": float(deltas.mean()),
        "wilcoxon_p": pval,
        "bootstrap_mean_delta_ci95": [float(lo), float(hi)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions_csv")
    ap.add_argument("--prompts", default="")
    ap.add_argument("--hop-s", type=float, default=0.05)
    ap.add_argument("--out", default="results/robustness_suite_report.json")
    args = ap.parse_args()

    df = pd.read_csv(args.predictions_csv)
    report = {
        "A1_leakage": section_a1(Path(args.prompts) if args.prompts else None, df),
        "B1_event_metrics": section_b1(df, hop_s=args.hop_s),
        "B2_non_llm_baselines": section_b2(df),
        "B3_llm_variance": section_b3(df),
        "B4_subject_stats": section_b4(df),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
