#!/usr/bin/env python3
"""Aggregate per-fold K-fold ambiguous eval JSONs into ALL_FOLDS_REPORT.txt."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _metric(m: dict, key: str, fallback=None):
    v = m.get(key, fallback)
    return float(v) if v is not None else None


def _far(m: dict) -> float:
    far = m.get("far")
    if far is not None:
        return float(far)
    fp, tn = int(m.get("fp", 0)), int(m.get("tn", 0))
    return fp / max(1, fp + tn)


def aggregate(out_dir: Path, backend: str | None = None) -> dict:
    rows = []
    for f in range(5):
        p = out_dir / f"fold{f}_eval.json"
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}")
        d = json.loads(p.read_text())
        m = d["metrics"]
        amb = (m.get("per_bucket") or {}).get("ambiguous", {})
        rows.append(
            {
                "fold": f,
                "backend": d.get("backend", backend or "?"),
                "n": m.get("n", len(d.get("rows", []))),
                "acc": float(m["accuracy"]),
                "f1": float(m["f1"]),
                "prec": float(m["precision"]),
                "rec": float(m["recall"]),
                "far": _far(m),
                "esc": _metric(m, "escalate_rate", 0.0),
                "cost": _metric(m, "expected_response_cost", 0.0),
                "amb_acc": amb.get("accuracy"),
                "fall_acc": (m.get("per_bucket") or {}).get("strict_fall", {}).get("accuracy"),
                "adl_acc": (m.get("per_bucket") or {}).get("clear_adl", {}).get("accuracy"),
            }
        )

    def fmt(v, width, prec=3):
        if v is None:
            return f"{'n/a':>{width}}"
        if isinstance(v, float) and prec == 0:
            return f"{v:>{width}.0f}"
        return f"{v:>{width}.{prec}f}"

    backend_label = rows[0]["backend"] if rows else (backend or "?")
    lines = [
        f"Complete architecture: gate_knn_llm | backend={backend_label}",
        "Dataset: data/kfold_ambiguous (100 amb + 100 fall + 100 ADL per fold)",
        "",
        f"{'fold':>4} {'n':>4} {'Acc':>7} {'F1':>7} {'Prec':>7} {'Rec':>7} "
        f"{'FAR':>7} {'Esc':>7} {'AmbAcc':>7} {'Cost':>8}",
        "-" * 80,
    ]
    for r in rows:
        lines.append(
            f"{r['fold']:>4} {r['n']:>4} {fmt(r['acc'],7)} {fmt(r['f1'],7)} "
            f"{fmt(r['prec'],7)} {fmt(r['rec'],7)} {fmt(r['far'],7)} "
            f"{fmt(r['esc'],7)} {fmt(r['amb_acc'],7)} {fmt(r['cost'],8,0)}"
        )
    lines.append("-" * 80)
    for key, label in [
        ("acc", "Acc"),
        ("f1", "F1"),
        ("prec", "Prec"),
        ("rec", "Rec"),
        ("far", "FAR"),
        ("esc", "Esc"),
        ("amb_acc", "AmbAcc"),
        ("cost", "Cost"),
    ]:
        vals = [r[key] for r in rows if r[key] is not None]
        if vals:
            lines.append(f"mean±std {label}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")

    report = "\n".join(lines) + "\n"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ALL_FOLDS_REPORT.txt").write_text(report)
    summary = {"folds": rows}
    (out_dir / "all_folds_summary.json").write_text(json.dumps(summary, indent=2))
    print(report)
    print(f"Wrote {out_dir / 'ALL_FOLDS_REPORT.txt'}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="e.g. results/kfold_ambiguous")
    ap.add_argument("--backend", default=None)
    args = ap.parse_args()
    aggregate(Path(args.out_dir), backend=args.backend)


if __name__ == "__main__":
    main()
