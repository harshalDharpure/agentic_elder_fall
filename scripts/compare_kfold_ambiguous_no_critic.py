#!/usr/bin/env python3
"""Compare tier1_only vs gate_knn_llm on K-fold ambiguous bench (no Critic)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_fold(out_dir: Path, fold: int) -> dict:
    p = out_dir / f"fold{fold}_eval.json"
    return json.loads(p.read_text())


def aggregate_mode(out_dir: Path, mode: str) -> list[dict]:
    rows = []
    for f in range(5):
        d = load_fold(out_dir, f)
        m = d["metrics"]
        amb = (m.get("per_bucket") or {}).get("ambiguous", {})
        rows.append(
            {
                "fold": f,
                "mode": mode,
                "n": m.get("n", 300),
                "acc": m["accuracy"],
                "f1": m["f1"],
                "rec": m["recall"],
                "fn": m["fn"],
                "fp": m["fp"],
                "far": m.get("far", m["fp"] / max(1, m["fp"] + m["tn"])),
                "cost": m["expected_response_cost"],
                "cost_win": m.get("cost_per_window", m["expected_response_cost"] / 300),
                "gr_cost": m.get("graded_action_cost", 0),
                "amb_acc": amb.get("accuracy"),
                "esc": m.get("escalate_rate", 0),
            }
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier1-dir", default="results/kfold_ambiguous_tier1")
    ap.add_argument("--stack-dir", default="results/kfold_ambiguous")
    ap.add_argument("--out", default="results/kfold_ambiguous/COMPARISON_NO_CRITIC.txt")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    tier1 = aggregate_mode(root / args.tier1_dir, "tier1_only")
    stack = aggregate_mode(root / args.stack_dir, "gate_knn_llm")

    lines = [
        "K-fold ambiguous bench — NO CRITIC comparison",
        "Dataset: 300/fold (100 D18/D19 ambiguous + 100 fall + 100 clear ADL)",
        "Cost = 10×FN + 1×FP (clinical safety cost, NOT compute cost)",
        "",
        f"{'fold':>4} {'mode':<16} {'Acc':>6} {'F1':>6} {'Rec':>6} {'FN':>4} {'FP':>4} "
        f"{'FAR':>6} {'Cost':>6} {'$/win':>6} {'AmbAcc':>7}",
        "-" * 88,
    ]
    for f in range(5):
        t = tier1[f]
        s = stack[f]
        lines.append(
            f"{f:>4} {'tier1_only':<16} {t['acc']:6.3f} {t['f1']:6.3f} {t['rec']:6.3f} "
            f"{t['fn']:4.0f} {t['fp']:4.0f} {t['far']:6.3f} {t['cost']:6.0f} {t['cost_win']:6.2f} {t['amb_acc']:7.3f}"
        )
        lines.append(
            f"{f:>4} {'gate_knn_llm':<16} {s['acc']:6.3f} {s['f1']:6.3f} {s['rec']:6.3f} "
            f"{s['fn']:4.0f} {s['fp']:4.0f} {s['far']:6.3f} {s['cost']:6.0f} {s['cost_win']:6.2f} {s['amb_acc']:7.3f}"
        )
        dc = s["cost"] - t["cost"]
        df1 = s["f1"] - t["f1"]
        lines.append(f"     {'Δ stack−tier1':<16} {'':>6} {df1:+6.3f} {'':>6} "
                     f"{s['fn']-t['fn']:+4.0f} {s['fp']-t['fp']:+4.0f} {'':>6} {dc:+6.0f}")
        lines.append("")

    def mean(rows, k):
        return float(np.mean([r[k] for r in rows]))

    lines.append("-" * 88)
    lines.append("MEAN across folds:")
    for mode, rows in [("tier1_only", tier1), ("gate_knn_llm", stack)]:
        lines.append(
            f"  {mode:<16} F1={mean(rows,'f1'):.3f} Rec={mean(rows,'rec'):.3f} "
            f"FN={mean(rows,'fn'):.1f} FP={mean(rows,'fp'):.1f} "
            f"Cost={mean(rows,'cost'):.0f} Cost/win={mean(rows,'cost_win'):.2f} "
            f"AmbAcc={mean(rows,'amb_acc'):.3f}"
        )
    cost_red = (1 - mean(stack, "cost") / max(1, mean(tier1, "cost"))) * 100
    lines.append(f"  Cost reduction (stack vs tier1): {cost_red:.1f}%")
    lines.append(f"  ΔF1 (stack vs tier1): {mean(stack,'f1')-mean(tier1,'f1'):+.3f}")
    lines.append(f"  ΔRec (stack vs tier1): {mean(stack,'rec')-mean(tier1,'rec'):+.3f}")

    report = "\n".join(lines) + "\n"
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    summary = {"tier1_only": tier1, "gate_knn_llm": stack}
    (out.parent / "comparison_no_critic.json").write_text(json.dumps(summary, indent=2))
    print(report)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
