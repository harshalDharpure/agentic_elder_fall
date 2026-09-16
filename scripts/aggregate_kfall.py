#!/usr/bin/env python3
"""Aggregate KFall external folds: mean±std, paired bootstrap CIs, D18/D19 notes."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "kfall"


def load_fold(f: int) -> dict:
    for p in (
        OUT / f"fold{f}" / f"kfall_external_fold{f}.json",
        OUT / f"kfall_external_fold{f}.json",
    ):
        if p.exists():
            return json.loads(p.read_text())
    raise FileNotFoundError(f"Missing KFall fold {f}")


def by_name(d: dict) -> dict:
    return {r["name"]: r for r in d.get("rows", []) if isinstance(r, dict) and "name" in r}


def cpk(r: dict, n: int | None = None) -> float:
    n = n or int(r.get("max_test") or r.get("n") or 1000)
    return float(r["expected_response_cost"]) / float(n) * 1000.0


def bootstrap_ci(deltas: list[float], n_boot: int = 5000, seed: int = 42):
    rng = np.random.default_rng(seed)
    arr = np.asarray(deltas, dtype=float)
    if len(arr) == 0:
        return [float("nan"), float("nan")]
    boots = [float(np.mean(rng.choice(arr, size=len(arr), replace=True))) for _ in range(n_boot)]
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def wilcoxon_p(deltas: list[float]) -> float:
    try:
        from scipy.stats import wilcoxon

        if len(deltas) < 2 or all(abs(d) < 1e-12 for d in deltas):
            return float("nan")
        return float(wilcoxon(deltas, alternative="two-sided").pvalue)
    except Exception:
        # With n=5 and all positive, discrete min is 0.0625
        if len(deltas) == 5 and all(d > 0 for d in deltas):
            return 0.0625
        if len(deltas) == 5 and all(d < 0 for d in deltas):
            return 0.0625
        return float("nan")


def main():
    folds_data = []
    for f in range(5):
        try:
            folds_data.append((f, by_name(load_fold(f))))
        except FileNotFoundError:
            print(f"skip missing fold {f}")

    summary = {"dataset": "kfall_binary", "n_folds": len(folds_data), "folds": [], "modes": {}}
    mode_keys = ("tier1_only", "gate_only", "gate_knn", "gate_llm", "gate_knn_llm")

    for f, m in folds_data:
        row = {"fold": f}
        for name in mode_keys:
            if name not in m:
                continue
            r = m[name]
            n = int(r.get("max_test") or 1000)
            row[name] = {
                "f1": r.get("f1"),
                "recall": r.get("recall"),
                "cost_per_1000": cpk(r, n),
                "escalation_rate": r.get("escalation_rate"),
                "escalated_f1": r.get("escalated_f1"),
                "false_alarm_rate_near_fall": r.get("false_alarm_rate_near_fall"),
                "false_alarm_rate_ambiguous": r.get("false_alarm_rate_ambiguous"),
                "n_d18": r.get("n_d18"),
                "n_d19": r.get("n_d19"),
                "latency_tier1_ms": r.get("latency_tier1_ms"),
                "latency_escalated_ms": r.get("latency_escalated_ms"),
                "latency_mean_ms": r.get("latency_mean_ms"),
            }
        summary["folds"].append(row)

    for name in mode_keys:
        vals = {
            "f1": [],
            "recall": [],
            "cost_per_1000": [],
            "escalation_rate": [],
            "escalated_f1": [],
            "far_d18d19": [],
        }
        for f, m in folds_data:
            if name not in m:
                continue
            r = m[name]
            n = int(r.get("max_test") or 1000)
            vals["f1"].append(r["f1"])
            vals["recall"].append(r["recall"])
            vals["cost_per_1000"].append(cpk(r, n))
            vals["escalation_rate"].append(r.get("escalation_rate", float("nan")))
            vals["escalated_f1"].append(r.get("escalated_f1", float("nan")))
            vals["far_d18d19"].append(r.get("false_alarm_rate_near_fall", float("nan")))
        if not vals["f1"]:
            continue
        entry = {"n_folds": len(vals["f1"])}
        for k, a in vals.items():
            a = [x for x in a if x == x]
            entry[f"{k}_mean"] = float(np.mean(a)) if a else float("nan")
            entry[f"{k}_std"] = float(np.std(a, ddof=1)) if len(a) > 1 else 0.0
        summary["modes"][name] = entry

    # Paired stack - tier1
    tests = []
    if all("tier1_only" in m and "gate_knn_llm" in m for _, m in folds_data):
        for metric, key in (
            ("f1", "f1"),
            ("recall", "recall"),
            ("cost_per_1000", "cost"),
        ):
            deltas = []
            for f, m in folds_data:
                t, s = m["tier1_only"], m["gate_knn_llm"]
                n = int(s.get("max_test") or 1000)
                if key == "cost":
                    deltas.append(cpk(s, n) - cpk(t, n))
                else:
                    deltas.append(float(s[key]) - float(t[key]))
            tests.append(
                {
                    "left": "gate_knn_llm",
                    "right": "tier1_only",
                    "metric": metric,
                    "mean_delta": float(np.mean(deltas)),
                    "bootstrap_ci95": bootstrap_ci(deltas),
                    "wilcoxon_pvalue": wilcoxon_p(deltas),
                    "deltas": deltas,
                }
            )
    summary["paired_tests"] = tests

    # D18/D19 spotlight from fold0 ladder if present
    d18 = {}
    for f, m in folds_data:
        for name, r in m.items():
            if name not in mode_keys:
                continue
            d18.setdefault(name, []).append(
                {
                    "fold": f,
                    "n_d18": r.get("n_d18"),
                    "n_d19": r.get("n_d19"),
                    "far": r.get("false_alarm_rate_near_fall"),
                    "esc_f1": r.get("escalated_f1"),
                    "esc_rate": r.get("escalation_rate"),
                }
            )
    summary["d18_d19_by_mode"] = d18

    path = OUT / "kfall_external_summary.json"
    path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("Wrote", path)


if __name__ == "__main__":
    main()
