#!/usr/bin/env python3
"""Aggregate per-fold results into mean±std tables and paired significance tests.

Only reads results/fold*/ (never root mirrors) to avoid double-counting.
Adds cost_per_1000 so hybrid budgets (n=4000 vs n=1000) are comparable.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load_rows(path: Path):
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "rows" in data:
        return data["rows"]
    if isinstance(data, dict) and "metrics" in data:
        r = dict(data["metrics"])
        r.setdefault("name", path.stem)
        return [r]
    if isinstance(data, list):
        return data
    return []


def _fold_id(path: Path) -> int | None:
    m = re.search(r"fold(\d+)", path.as_posix())
    return int(m.group(1)) if m else None


def _infer_n(row: dict, fold: int | None, protocol_n: dict[int, int]) -> int:
    for key in ("n", "max_test", "n_samples"):
        v = row.get(key)
        if isinstance(v, (int, float)) and int(v) > 0:
            return int(v)
    # tp+tn+fp+fn if present
    parts = [row.get(k) for k in ("tp", "tn", "fp", "fn")]
    if all(isinstance(x, (int, float)) for x in parts):
        return int(sum(parts))
    if fold is not None and fold in protocol_n:
        return protocol_n[fold]
    # hybrid default: folds 0–1 full, 2–4 fast
    if fold is not None:
        return 4000 if fold <= 1 else 1000
    return 1000


def _enrich_rows(rows: list[dict], fold: int | None, protocol_n: dict[int, int]) -> list[dict]:
    out = []
    for row in rows:
        r = dict(row)
        n = _infer_n(r, fold, protocol_n)
        r["n"] = n
        r["fold"] = fold
        cost = r.get("expected_response_cost")
        if isinstance(cost, (int, float)) and n > 0:
            r["cost_per_1000"] = float(cost) / float(n) * 1000.0
        out.append(r)
    return out


def _collect(results_dir: Path, pattern: str, protocol_n: dict[int, int]) -> dict[str, list[dict]]:
    """Collect only from results/fold*/ — never root mirrors."""
    by_name: dict[str, list[dict]] = {}
    for path in sorted(results_dir.glob(f"fold*/{pattern}")):
        # skip archived full4000 copies
        if "full4000" in path.name:
            continue
        fold = _fold_id(path)
        for row in _enrich_rows(_load_rows(path), fold, protocol_n):
            name = str(row.get("name") or row.get("model_name") or path.stem)
            by_name.setdefault(name, []).append(row)
    # sort each list by fold for paired tests
    for name in by_name:
        by_name[name] = sorted(by_name[name], key=lambda r: (r.get("fold") is None, r.get("fold", -1)))
    return by_name


def _numeric_keys(rows: list[dict]) -> list[str]:
    keys = set()
    for r in rows:
        for k, v in r.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    continue
                keys.add(k)
    return sorted(keys)


def mean_std_table(by_name: dict[str, list[dict]]) -> list[dict]:
    out = []
    for name, rows in sorted(by_name.items()):
        keys = _numeric_keys(rows)
        row = {"name": name, "n_folds": len(rows)}
        for k in keys:
            if k == "fold":
                continue
            vals = [float(r[k]) for r in rows if k in r and isinstance(r[k], (int, float))]
            vals = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
            if not vals:
                continue
            row[f"{k}_mean"] = float(np.mean(vals))
            row[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        out.append(row)
    return out


def paired_wilcoxon(a: list[float], b: list[float]) -> dict:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    out = {
        "n": int(len(d)),
        "mean_delta": float(d.mean()) if len(d) else 0.0,
        "std_delta": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
    }
    try:
        from scipy.stats import wilcoxon

        if len(d) >= 3 and np.any(d != 0):
            stat = wilcoxon(d)
            out["wilcoxon_stat"] = float(stat.statistic)
            out["wilcoxon_pvalue"] = float(stat.pvalue)
    except Exception:
        pass
    if len(d):
        rng = np.random.default_rng(42)
        boots = []
        for _ in range(2000):
            idx = rng.integers(0, len(d), size=len(d))
            boots.append(float(d[idx].mean()))
        out["bootstrap_ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    return out


def paired_metric(
    by_name: dict[str, list[dict]], left: str, right: str, metric: str
) -> dict | None:
    if left not in by_name or right not in by_name:
        return None
    left_rows = {r.get("fold"): r for r in by_name[left] if r.get("fold") is not None}
    right_rows = {r.get("fold"): r for r in by_name[right] if r.get("fold") is not None}
    folds = sorted(set(left_rows) & set(right_rows))
    if not folds:
        # fallback positional
        a = [float(r[metric]) for r in by_name[left] if metric in r]
        b = [float(r[metric]) for r in by_name[right] if metric in r]
        n = min(len(a), len(b))
        if n == 0:
            return None
        return {"left": left, "right": right, "metric": metric, **paired_wilcoxon(a[:n], b[:n])}
    a, b = [], []
    for f in folds:
        if metric in left_rows[f] and metric in right_rows[f]:
            a.append(float(left_rows[f][metric]))
            b.append(float(right_rows[f][metric]))
    if not a:
        return None
    return {"left": left, "right": right, "metric": metric, "folds": folds, **paired_wilcoxon(a, b)}


def write_csv(rows: list[dict], path: Path):
    if not rows:
        return
    keys = sorted({k for r in rows for k in r})
    keys = ["name"] + [k for k in keys if k != "name"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _protocol_n_map(results_dir: Path) -> dict[int, int]:
    out: dict[int, int] = {}
    for path in sorted(results_dir.glob("fold*/protocol_fold*.json")):
        try:
            data = json.loads(path.read_text())
            fold = int(data.get("fold", _fold_id(path) or -1))
            mt = data.get("max_test")
            if isinstance(mt, (int, float)):
                out[fold] = int(mt)
            elif str(data.get("protocol_tag", "")).startswith("full"):
                out[fold] = 4000
            elif "1000" in str(data.get("protocol_tag", "")):
                out[fold] = 1000
        except Exception:
            continue
    # defaults for missing
    for f in range(5):
        out.setdefault(f, 4000 if f <= 1 else 1000)
    return out


def _collect_protocol_notes(results_dir: Path) -> list[dict]:
    notes = []
    for path in sorted(results_dir.glob("fold*/protocol_fold*.json")):
        try:
            notes.append(json.loads(path.read_text()))
        except Exception:
            continue
    return notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    out_dir = args.out_dir or args.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    protocol_n = _protocol_n_map(args.results_dir)
    protocol_notes = _collect_protocol_notes(args.results_dir)
    if protocol_notes:
        (out_dir / "aggregate_protocol_notes.json").write_text(json.dumps(protocol_notes, indent=2))

    summaries = {}
    for label, pattern in [
        ("backbone", "backbone_compare_fold*.json"),
        ("ablation", "ablation_fold*.json"),
        ("llm", "llm_compare_fold*.json"),
        ("agentic", "agentic_fold*.json"),
    ]:
        by_name = _collect(args.results_dir, pattern, protocol_n)
        table = mean_std_table(by_name)
        summaries[label] = {"by_name": {k: len(v) for k, v in by_name.items()}, "table": table}
        write_csv(table, out_dir / f"aggregate_{label}_mean_std.csv")
        (out_dir / f"aggregate_{label}_mean_std.json").write_text(json.dumps(table, indent=2))
        print(f"{label}: {summaries[label]['by_name']}")

        tests = []
        if label == "ablation":
            for metric in (
                "expected_response_cost",
                "cost_per_1000",
                "false_alarm_rate_near_fall",
                "f1",
                "recall",
            ):
                t = paired_metric(by_name, "gate_knn_llm", "tier1_only", metric)
                if t:
                    tests.append(t)
        if label == "llm":
            for metric in ("f1", "expected_response_cost", "cost_per_1000"):
                left = next((n for n in by_name if "mistral" in n), None)
                if left:
                    t = paired_metric(by_name, left, "heuristic", metric)
                    if t:
                        tests.append(t)
        if label == "backbone":
            for name in list(by_name):
                if name.endswith("+agentic"):
                    base = name.replace("+agentic", "")
                    t = paired_metric(by_name, name, base, "cost_per_1000")
                    if t:
                        tests.append(t)
        if tests:
            (out_dir / f"aggregate_{label}_tests.json").write_text(json.dumps(tests, indent=2))
            print(f"{label} tests:", json.dumps(tests, indent=2))

    footnote = {
        "guide_meeting_protocol": {
            "primary_claim_row": "gate_knn_llm vs tier1_only",
            "cost_metric": "cost_per_1000 (normalized; folds 0-1 n=4000, folds 2-4 n=1000 until homogenized)",
            "folds_0_1_ablation": "full_4000 (gate_llm available for appendix)",
            "folds_2_4_ablation": "fast_1000_no_gate_llm",
            "agentic_eval": "use ollama for full stack; heuristic is NOT gate_knn_llm",
            "do_not_claim": [
                "detector F1 >= 0.90",
                "near-fall FAR as primary win",
                "monotonic gain from gate/kNN alone",
                "KFall external validity",
            ],
        },
        "protocol_n": protocol_n,
        "protocol_notes": protocol_notes,
    }
    (out_dir / "aggregate_paper_footnotes.json").write_text(json.dumps(footnote, indent=2))
    print(f"Wrote aggregate tables under {out_dir}")


if __name__ == "__main__":
    main()
