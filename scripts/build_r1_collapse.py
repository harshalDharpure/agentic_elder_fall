#!/usr/bin/env python3
"""Build R1 collapse table from standard_protocol + ambiguous + main backbone results."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "r1_collapse"
OUT.mkdir(parents=True, exist_ok=True)

PAPER = [
    "cnn1d",
    "lstm",
    "cnn_lstm",
    "cnn_lstm_attn",
    "tcn",
    "modern_tcn",
    "tsmixer",
    "patchtst",
    "transformer",
]


def mean_std(xs):
    xs = [float(x) for x in xs if x is not None]
    if not xs:
        return None, None
    if len(xs) == 1:
        return xs[0], 0.0
    return float(np.mean(xs)), float(np.std(xs, ddof=1))


def load_standard():
    p = ROOT / "results" / "standard_protocol" / "standard_protocol_summary.json"
    if not p.exists():
        return {}
    data = json.loads(p.read_text())
    return {s["name"]: s for s in data.get("summary", [])}


def load_ambiguous_tier1():
    """Detector-only on ambiguous bench if present; else lead mean 0.713."""
    rows = {}
    base = ROOT / "results" / "kfold_ambiguous"
    # per-fold eval often has tier1 f1
    f1s = []
    for fold in range(5):
        fp = base / f"fold{fold}_eval.json"
        if not fp.exists():
            continue
        d = json.loads(fp.read_text())
        # try common keys
        for key in ("tier1_f1", "tier1", "baseline_f1"):
            if key in d and isinstance(d[key], (int, float)):
                f1s.append(float(d[key]))
                break
        else:
            if "modes" in d:
                for m in d["modes"]:
                    if m.get("name") in ("tier1_only", "tier1", "detector"):
                        f1s.append(float(m["f1"]))
                        break
    if f1s:
        mu, sd = mean_std(f1s)
        rows["cnn_lstm_attn"] = {"f1_mean": mu, "f1_std": sd, "source": "kfold_ambiguous"}
    # locked lead from Q1 pack
    cmp = base / "comparison_no_critic.json"
    if cmp.exists():
        d = json.loads(cmp.read_text())
        if "tier1_f1_mean" in d:
            rows.setdefault(
                "cnn_lstm_attn",
                {
                    "f1_mean": float(d["tier1_f1_mean"]),
                    "f1_std": float(d.get("tier1_f1_std", 0)),
                    "source": "comparison_no_critic",
                },
            )
    return rows


def load_main_detector():
    """From backbone_compare aggregates or per-fold."""
    out = {}
    agg = ROOT / "results" / "aggregate_backbone_mean_std.json"
    if agg.exists():
        d = json.loads(agg.read_text())
        # structure varies
        if isinstance(d, dict):
            for name, stats in d.items():
                if not isinstance(stats, dict):
                    continue
                if "f1_mean" in stats:
                    out[name] = {
                        "f1_mean": float(stats["f1_mean"]),
                        "f1_std": float(stats.get("f1_std", 0)),
                        "source": "aggregate_backbone",
                    }
                elif "f1" in stats and isinstance(stats["f1"], dict):
                    out[name] = {
                        "f1_mean": float(stats["f1"].get("mean", 0)),
                        "f1_std": float(stats["f1"].get("std", 0)),
                        "source": "aggregate_backbone",
                    }
    # fallback fold0 detector rows
    for fold in range(5):
        fp = ROOT / "results" / f"fold{fold}" / f"backbone_compare_fold{fold}.json"
        if not fp.exists():
            fp = ROOT / "results" / f"backbone_compare_fold{fold}.json"
        if not fp.exists():
            continue
        d = json.loads(fp.read_text())
        items = d if isinstance(d, list) else d.get("detector", d.get("rows", []))
        if isinstance(d, dict) and "models" in d:
            items = d["models"]
        if not isinstance(items, list):
            continue
        for row in items:
            if not isinstance(row, dict):
                continue
            name = row.get("name") or row.get("model")
            if not name or "+agentic" in str(name):
                continue
            if row.get("mode") and row["mode"] != "detector":
                continue
            out.setdefault(name, {"_f1s": []})
            if "_f1s" not in out[name]:
                out[name]["_f1s"] = []
            if "f1" in row:
                out[name]["_f1s"].append(float(row["f1"]))
    for name, v in list(out.items()):
        if "_f1s" in v and v["_f1s"]:
            mu, sd = mean_std(v["_f1s"])
            out[name] = {"f1_mean": mu, "f1_std": sd, "source": "backbone_compare_folds"}
    return out


def main() -> None:
    std = load_standard()
    main = load_main_detector()
    amb = load_ambiguous_tier1()

    rows = []
    for name in PAPER:
        s = std.get(name, {})
        m = main.get(name, {})
        a = amb.get(name, {})
        rows.append(
            {
                "backbone": name,
                "standard_f1": s.get("f1_mean"),
                "standard_f1_std": s.get("f1_std"),
                "main_enriched_f1": m.get("f1_mean"),
                "main_enriched_f1_std": m.get("f1_std"),
                "nearfall_f1": a.get("f1_mean") if name in amb else None,
                "nearfall_f1_std": a.get("f1_std") if name in amb else None,
                "g1_pass": s.get("g1_pass"),
            }
        )

    (OUT / "r1_collapse.json").write_text(json.dumps({"rows": rows}, indent=2))
    with (OUT / "r1_collapse.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def fmt(mu, sd):
        if mu is None:
            return "---"
        if sd is None:
            return f"{mu:.3f}"
        return f"{mu:.3f}$\\pm${sd:.3f}"

    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{R1 collapse: nine backbones under standard vs enriched vs near-fall stress (detector-only).}",
        r"\label{tab:collapse}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Backbone & Standard F1 & Enriched F1 & Near-fall F1 \\",
        r"\midrule",
    ]
    for r in rows:
        nm = r["backbone"].replace("_", r"\_")
        lines.append(
            f"{nm} & {fmt(r['standard_f1'], r['standard_f1_std'])} & "
            f"{fmt(r['main_enriched_f1'], r['main_enriched_f1_std'])} & "
            f"{fmt(r['nearfall_f1'], r['nearfall_f1_std'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (OUT / "tab_collapse.tex").write_text("\n".join(lines))
    # also copy into paper table dir
    paper_tab = ROOT / "paper" / "table" / "tab_collapse.tex"
    if paper_tab.parent.exists():
        paper_tab.write_text("\n".join(lines))
    print(f"Wrote {OUT}/r1_collapse.json and tab_collapse.tex")
    for r in rows:
        print(
            f"{r['backbone']:16s} std={r['standard_f1']}  "
            f"main={r['main_enriched_f1']}  nf={r['nearfall_f1']}"
        )


if __name__ == "__main__":
    main()
