#!/usr/bin/env python3
"""Summarize Critic impact: action_critique vs gate_knn_llm (labels frozen)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
KEYS = [
    "f1",
    "recall",
    "expected_response_cost",
    "graded_action_cost",
    "emergency_rate_adl",
    "rationale_grounding_rate",
    "false_alarms_per_1000",
]


def main():
    rows_g, rows_c = [], []
    lines = [
        "# Critic impact (Option A): action_critique vs gate_knn_llm",
        "",
        "Labels are frozen. Impact should appear in graded action cost and rationale grounding.",
        "",
        "| Fold | Mode | F1 | Recall | Cost | GradedAction | EmADL | Ground |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for f in range(5):
        p = ROOT / f"results/fold{f}/ablation_fold{f}_action_critique.json"
        if not p.exists():
            continue
        by = {r["name"]: r for r in json.loads(p.read_text())["rows"]}
        for name, bucket in (
            ("gate_knn_llm", rows_g),
            ("gate_knn_llm_action_critique", rows_c),
        ):
            r = by.get(name)
            if not r:
                continue
            bucket.append(r)
            lines.append(
                f"| {f} | {name} | {r['f1']:.3f} | {r['recall']:.3f} | "
                f"{r['expected_response_cost']:.0f} | {r['graded_action_cost']:.0f} | "
                f"{r.get('emergency_rate_adl', 0):.3f} | {r.get('rationale_grounding_rate', 0):.3f} |"
            )
    lines += ["", "## Means", ""]
    summary = {}
    for label, rs in (("gate_knn_llm", rows_g), ("action_critique", rows_c)):
        if not rs:
            continue
        m = {k: float(np.mean([r[k] for r in rs if k in r])) for k in KEYS if all(k in r for r in rs)}
        summary[label] = m
        lines.append(
            f"- **{label}**: F1={m.get('f1', float('nan')):.3f}, "
            f"graded_action={m.get('graded_action_cost', float('nan')):.1f}, "
            f"grounding={m.get('rationale_grounding_rate', float('nan')):.3f}"
        )
    if rows_g and rows_c:
        lines += [
            "",
            "## Readout",
            "",
            "Old action_critique barely changes detection (by design). "
            "Grounding rises a little; graded cost barely moves because Critic "
            "did not screen confident-path falls. The new "
            "`gate_knn_llm_contrastive_action_critique` mode screens those "
            "windows for action downgrades and feeds hard-negative ADLs to the Critic.",
        ]
    out = ROOT / "results/CRITIC_ACTION_IMPACT.md"
    out.write_text("\n".join(lines) + "\n")
    (ROOT / "results/critic_action_impact.json").write_text(json.dumps(summary, indent=2))
    print(out.read_text())


if __name__ == "__main__":
    main()
