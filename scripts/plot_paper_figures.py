#!/usr/bin/env python3
"""Generate paper result figures from locked results JSON (no re-training)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper" / "fig"
FIG.mkdir(parents=True, exist_ok=True)


def load_abl(fold: int):
    p = ROOT / f"results/fold{fold}/ablation_fold{fold}.json"
    d = json.loads(p.read_text())
    rows = d.get("rows", d)
    return {r["name"]: r for r in rows if isinstance(r, dict) and "name" in r}, d


def cpk(row, n=None):
    n = n or row.get("n") or row.get("max_test") or 1000
    return float(row["expected_response_cost"]) / float(n) * 1000.0


def fig_cost_f1_per_fold():
    folds = list(range(5))
    t_f1, s_f1, t_c, s_c = [], [], [], []
    for f in folds:
        m, meta = load_abl(f)
        n = meta.get("max_test", 1000)
        t_f1.append(m["tier1_only"]["f1"])
        s_f1.append(m["gate_knn_llm"]["f1"])
        t_c.append(cpk(m["tier1_only"], n))
        s_c.append(cpk(m["gate_knn_llm"], n))

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))
    x = np.arange(5)
    w = 0.35
    axes[0].bar(x - w / 2, t_f1, w, label="tier1_only", color="#4C78A8")
    axes[0].bar(x + w / 2, s_f1, w, label="gate_knn_llm", color="#F58518")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"F{f}" for f in folds])
    axes[0].set_ylabel("F1")
    axes[0].set_ylim(0.5, 1.0)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("F1 by fold")

    axes[1].bar(x - w / 2, t_c, w, label="tier1_only", color="#4C78A8")
    axes[1].bar(x + w / 2, s_c, w, label="gate_knn_llm", color="#F58518")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"F{f}" for f in folds])
    axes[1].set_ylabel("Cost / 1000 windows")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].set_title("Normalized response cost")
    fig.tight_layout()
    out = FIG / "cost_f1_per_fold.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(FIG / "cost_f1_per_fold.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def fig_ablation_ladder():
    # Prefer Fold0 full4000 archive for the complete ladder including gate_llm
    p = ROOT / "results/fold0/ablation_fold0_full4000.json"
    if p.exists():
        rows = json.loads(p.read_text())
        if isinstance(rows, dict):
            rows = rows.get("rows", rows)
        m = {r["name"]: r for r in rows if isinstance(r, dict) and "name" in r}
        title_n = 4000
    else:
        m, _ = load_abl(0)
        title_n = 1000
    modes = [x for x in ("tier1_only", "gate_only", "gate_knn", "gate_llm", "gate_knn_llm") if x in m]
    f1s = [m[x]["f1"] for x in modes]
    costs = [m[x]["expected_response_cost"] for x in modes]

    fig, ax1 = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(len(modes))
    ax1.plot(x, f1s, "o-", color="#4C78A8", label="F1")
    ax1.set_xticks(x)
    ax1.set_xticklabels([n.replace("_", "\n") for n in modes], fontsize=8)
    ax1.set_ylabel("F1", color="#4C78A8")
    ax1.set_ylim(0.7, 1.0)
    ax2 = ax1.twinx()
    ax2.bar(x, costs, alpha=0.35, color="#F58518", label="Cost")
    ax2.set_ylabel("Expected response cost", color="#F58518")
    ax1.set_title(f"Ablation ladder (Fold 0, n={title_n})")
    fig.tight_layout()
    out = FIG / "ablation_ladder.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(FIG / "ablation_ladder.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def fig_latency():
    # Prefer end-to-end ollama agentic metrics for escalated latency
    tier, esc, mean = [], [], []
    for f in range(5):
        p = ROOT / f"results/fold{f}/agentic_fold{f}.json"
        if p.exists():
            d = json.loads(p.read_text())
            m = d.get("metrics", d)
            tier.append(m.get("latency_tier1_ms", 0))
            esc.append(m.get("latency_escalated_ms", 0) / 1000.0)  # seconds
            mean.append(m.get("latency_mean_ms", 0) / 1000.0)
        else:
            m, _ = load_abl(f)
            r = m.get("gate_knn_llm", {})
            tier.append(r.get("latency_tier1_ms", 0))
            esc.append(r.get("latency_escalated_ms", 0) / 1000.0)
            mean.append(r.get("latency_mean_ms", 0) / 1000.0)

    fig, ax = plt.subplots(figsize=(6.5, 3.3))
    x = np.arange(5)
    ax.bar(x - 0.25, tier, 0.25, label="Tier-1 (ms)", color="#4C78A8")
    ax2 = ax.twinx()
    ax2.bar(x, esc, 0.25, label="Escalated path (s)", color="#E45756")
    ax2.bar(x + 0.25, mean, 0.25, label="Mean stack (s)", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels([f"F{f}" for f in range(5)])
    ax.set_ylabel("Tier-1 latency (ms)")
    ax2.set_ylabel("LLM-path latency (s)")
    ax.set_title("Latency: Tier-1 vs escalated LLM path")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    out = FIG / "latency_bars.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(FIG / "latency_bars.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def fig_llm_backends():
    stores = {"heuristic": [], "mistral": [], "qwen": []}
    for f in range(5):
        rows = json.loads((ROOT / f"results/fold{f}/llm_compare_fold{f}.json").read_text())
        if isinstance(rows, dict):
            rows = rows.get("rows", rows)
        for r in rows:
            n = str(r.get("name", ""))
            if n == "heuristic":
                stores["heuristic"].append(r["f1"])
            elif "mistral" in n:
                stores["mistral"].append(r["f1"])
            elif "qwen" in n:
                stores["qwen"].append(r["f1"])
    labels = ["Heuristic", "Mistral", "Qwen2.5"]
    means = [np.mean(stores[k]) for k in ("heuristic", "mistral", "qwen")]
    stds = [np.std(stores[k], ddof=1) for k in ("heuristic", "mistral", "qwen")]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.bar(labels, means, yerr=stds, capsize=4, color=["#9e9e9e", "#4C78A8", "#72B7B2"])
    ax.set_ylabel("Ambiguous-case F1")
    ax.set_ylim(0, 1.05)
    ax.set_title("LLM backends on paired ambiguous IDs")
    fig.tight_layout()
    out = FIG / "llm_backends.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(FIG / "llm_backends.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def fig_larger_n_compare():
    """Compare n=1000 vs archived n=4000 on folds that have both."""
    rows_out = []
    for f in (0, 1):
        m1000, meta = load_abl(f)
        p4 = ROOT / f"results/fold{f}/ablation_fold{f}_full4000.json"
        if not p4.exists():
            continue
        raw = json.loads(p4.read_text())
        rows = raw.get("rows", raw) if isinstance(raw, dict) else raw
        m4000 = {r["name"]: r for r in rows if isinstance(r, dict) and "name" in r}
        for mode in ("tier1_only", "gate_knn_llm"):
            if mode in m1000 and mode in m4000:
                rows_out.append(
                    {
                        "fold": f,
                        "mode": mode,
                        "f1_1000": m1000[mode]["f1"],
                        "f1_4000": m4000[mode]["f1"],
                        "cost_1000": cpk(m1000[mode], meta.get("max_test", 1000)),
                        "cost_4000": cpk(m4000[mode], 4000),
                    }
                )
    if not rows_out:
        print("skip larger_n_compare (no full4000 archives)")
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.2))
    # F1 scatter 1000 vs 4000
    for r in rows_out:
        c = "#F58518" if r["mode"] == "gate_knn_llm" else "#4C78A8"
        axes[0].scatter(r["f1_1000"], r["f1_4000"], color=c, s=60, label=r["mode"])
    axes[0].plot([0.7, 1.0], [0.7, 1.0], "--", color="gray", linewidth=1)
    axes[0].set_xlabel("F1 @ n=1000")
    axes[0].set_ylabel("F1 @ n=4000")
    axes[0].set_title("Budget sensitivity (F1)")
    handles, labels = axes[0].get_legend_handles_labels()
    by = dict(zip(labels, handles))
    axes[0].legend(by.values(), by.keys(), frameon=False, fontsize=8)

    for r in rows_out:
        c = "#F58518" if r["mode"] == "gate_knn_llm" else "#4C78A8"
        axes[1].scatter(r["cost_1000"], r["cost_4000"], color=c, s=60)
    mx = max(max(r["cost_1000"] for r in rows_out), max(r["cost_4000"] for r in rows_out))
    axes[1].plot([0, mx], [0, mx], "--", color="gray", linewidth=1)
    axes[1].set_xlabel("Cost/1k @ n=1000")
    axes[1].set_ylabel("Cost/1k @ n=4000")
    axes[1].set_title("Budget sensitivity (cost)")
    fig.tight_layout()
    out = FIG / "budget_sensitivity.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(FIG / "budget_sensitivity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main():
    fig_cost_f1_per_fold()
    fig_ablation_ladder()
    fig_latency()
    fig_llm_backends()
    fig_larger_n_compare()


if __name__ == "__main__":
    main()
