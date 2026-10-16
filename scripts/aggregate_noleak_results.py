#!/usr/bin/env python3
"""Aggregate results/noleak fold JSONs into a comparison markdown vs original paper numbers."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOLEAK = ROOT / "results" / "noleak"
OUT = NOLEAK / "NOLEAK_COMPARISON.md"


def load_rows(path: Path) -> dict[str, dict]:
    d = json.loads(path.read_text())
    rows = d.get("rows", d) if isinstance(d, dict) else d
    out = {}
    for r in rows:
        if isinstance(r, dict) and "name" in r:
            out[r["name"]] = r
    return out


def mean_sd(vals):
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.stdev(vals)


def main():
    modes = ["tier1_only", "gate_only", "gate_knn", "gate_llm", "gate_knn_llm"]
    by_mode: dict[str, list] = {m: [] for m in modes}
    for fold in range(5):
        p = NOLEAK / f"fold{fold}" / f"ablation_fold{fold}.json"
        if not p.exists():
            # try alternate naming
            cands = list((NOLEAK / f"fold{fold}").glob("ablation*.json")) if (NOLEAK / f"fold{fold}").exists() else []
            if not cands:
                continue
            p = cands[0]
        rows = load_rows(p)
        for m in modes:
            if m in rows:
                by_mode[m].append(rows[m])

    lines = [
        "# No-leak System-2 re-run comparison",
        "",
        "Activity codes stripped from evidence + memory; `include_activity: false`.",
        "",
        "| Mode | n_folds | F1 mean±sd | Recall mean±sd | Cost mean±sd |",
        "|------|--------:|------------|----------------|--------------|",
    ]
    for m, rows in by_mode.items():
        f1s = [float(r.get("f1", float("nan"))) for r in rows]
        recs = [float(r.get("recall", float("nan"))) for r in rows]
        costs = [float(r.get("expected_response_cost", float("nan"))) for r in rows]
        f1m, f1s_ = mean_sd(f1s)
        rm, rs = mean_sd(recs)
        cm, cs = mean_sd(costs)
        lines.append(
            f"| `{m}` | {len(rows)} | {f1m:.3f}±{f1s_:.3f} | {rm:.3f}±{rs:.3f} | {cm:.1f}±{cs:.1f} |"
        )
    lines += [
        "",
        "Paper (leaky) SisFall main `gate_knn_llm` reference: F1 0.887, recall 0.977, cost 200.",
        "If no-leak F1/cost collapse toward `tier1_only`, revise System-2 claims before submission.",
        "",
    ]
    OUT.write_text("\n".join(lines))
    print(OUT.read_text())


if __name__ == "__main__":
    main()
