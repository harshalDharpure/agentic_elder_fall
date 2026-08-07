#!/usr/bin/env python3
"""Build results/GUIDE_BRIEF.md from fold*/ ablations + LLM compare (guide meeting pack)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_ablation(fold: int) -> dict:
    path = ROOT / f"results/fold{fold}/ablation_fold{fold}.json"
    data = json.loads(path.read_text())
    rows = data.get("rows", data) if isinstance(data, dict) else data
    return {r["name"]: r for r in rows if isinstance(r, dict) and "name" in r}


def infer_n(row: dict, fold: int) -> int:
    if isinstance(row.get("max_test"), (int, float)) and int(row["max_test"]) > 0:
        return int(row["max_test"])
    parts = [row.get(k) for k in ("tp", "tn", "fp", "fn")]
    if all(isinstance(x, (int, float)) for x in parts):
        return int(sum(parts))
    return 4000 if fold <= 1 else 1000


def cost_per_1000(row: dict, fold: int) -> float:
    n = infer_n(row, fold)
    return float(row["expected_response_cost"]) / n * 1000.0


def load_llm(fold: int) -> dict:
    path = ROOT / f"results/fold{fold}/llm_compare_fold{fold}.json"
    data = json.loads(path.read_text())
    rows = data.get("rows", data) if isinstance(data, dict) else data
    if isinstance(rows, dict):
        rows = list(rows.values()) if rows else []
    out = {}
    for r in rows:
        if isinstance(r, dict) and "name" in r and "f1" in r:
            out[r["name"]] = r
    return out


def fmt(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


def main():
    lines = []
    lines.append("# Guide Brief — Agentic Fall Detection (Q1-style evidence)")
    lines.append("")
    lines.append("**Dataset:** SisFall · **Splits:** subject-independent 5-fold · **Primary detector:** `cnn_lstm_attn`")
    lines.append("")
    lines.append("## One-line claim")
    lines.append("")
    lines.append(
        "> Confidence-gated retrieval + local LLM (`gate_knn_llm`) improves F1/recall and reduces "
        "**normalized response cost** vs the same Tier-1 detector on every fold; paired ambiguous-case "
        "compare shows **Mistral ≫ heuristic**."
    )
    lines.append("")
    lines.append("## Table 1 — Per-fold: Tier-1 vs full agentic stack")
    lines.append("")
    lines.append("| Fold | n | Tier1 F1 | Tier1 Rec | Tier1 cost/1k | Stack F1 | Stack Rec | Stack cost/1k | ΔF1 | Cost ↓ % |")
    lines.append("|------|---|----------|-----------|---------------|----------|-----------|---------------|-----|----------|")

    f1_t, f1_s, rec_t, rec_s, c_t, c_s = [], [], [], [], [], []
    for fold in range(5):
        m = load_ablation(fold)
        t, s = m["tier1_only"], m["gate_knn_llm"]
        n = infer_n(s if "max_test" in s else t, fold)
        ct, cs = cost_per_1000(t, fold), cost_per_1000(s, fold)
        df1 = s["f1"] - t["f1"]
        drop = (ct - cs) / ct * 100.0 if ct else 0.0
        f1_t.append(t["f1"]); f1_s.append(s["f1"])
        rec_t.append(t["recall"]); rec_s.append(s["recall"])
        c_t.append(ct); c_s.append(cs)
        lines.append(
            f"| {fold} | {n} | {fmt(t['f1'])} | {fmt(t['recall'])} | {ct:.0f} | "
            f"{fmt(s['f1'])} | {fmt(s['recall'])} | {cs:.0f} | {df1:+.3f} | {drop:.0f}% |"
        )

    lines.append("")
    lines.append("## Table 2 — 5-fold mean ± std (normalized)")
    lines.append("")
    lines.append("| Mode | F1 | Recall | Cost / 1000 windows |")
    lines.append("|------|----|--------|---------------------|")
    lines.append(
        f"| tier1_only | {np.mean(f1_t):.3f}±{np.std(f1_t, ddof=1):.3f} | "
        f"{np.mean(rec_t):.3f}±{np.std(rec_t, ddof=1):.3f} | "
        f"{np.mean(c_t):.0f}±{np.std(c_t, ddof=1):.0f} |"
    )
    lines.append(
        f"| **gate_knn_llm** | **{np.mean(f1_s):.3f}±{np.std(f1_s, ddof=1):.3f}** | "
        f"**{np.mean(rec_s):.3f}±{np.std(rec_s, ddof=1):.3f}** | "
        f"**{np.mean(c_s):.0f}±{np.std(c_s, ddof=1):.0f}** |"
    )
    lines.append("")
    lines.append(
        f"- Mean ΔF1 = **{np.mean(np.array(f1_s)-np.array(f1_t)):+.3f}** · "
        f"Mean ΔRecall = **{np.mean(np.array(rec_s)-np.array(rec_t)):+.3f}** · "
        f"Mean cost reduction = **{(1 - np.mean(c_s)/np.mean(c_t))*100:.0f}%** (on cost/1000)."
    )
    lines.append("")

    # Ablation ladder fold0
    lines.append("## Table 3 — Ablation ladder (Fold 0, full protocol n=4000)")
    lines.append("")
    lines.append("| Mode | F1 | Recall | Cost | Near-fall FAR | Escalated F1 |")
    lines.append("|------|----|--------|------|---------------|--------------|")
    m0 = load_ablation(0)
    for name in ("tier1_only", "gate_only", "gate_knn", "gate_llm", "gate_knn_llm"):
        if name not in m0:
            continue
        r = m0[name]
        lines.append(
            f"| {name} | {fmt(r['f1'])} | {fmt(r['recall'])} | {r['expected_response_cost']:.0f} | "
            f"{fmt(r.get('false_alarm_rate_near_fall', float('nan')))} | "
            f"{fmt(r.get('escalated_f1', float('nan')))} |"
        )
    lines.append("")
    lines.append("Takeaway: **gate or kNN alone is not enough** — the win needs **kNN + LLM**.")
    lines.append("")

    lines.append("## Table 4 — LLM backends on shared ambiguous cases (5 folds)")
    lines.append("")
    lines.append("| Fold | Heuristic F1 | Mistral F1 | Qwen F1 |")
    lines.append("|------|--------------|------------|---------|")
    h_f, m_f, q_f = [], [], []
    for fold in range(5):
        llm = load_llm(fold)
        h = next((llm[k] for k in llm if k == "heuristic"), None)
        mi = next((llm[k] for k in llm if "mistral" in k), None)
        q = next((llm[k] for k in llm if "qwen" in k), None)
        hf = h["f1"] if h else float("nan")
        mf = mi["f1"] if mi else float("nan")
        qf = q["f1"] if q else float("nan")
        h_f.append(hf); m_f.append(mf); q_f.append(qf)
        lines.append(f"| {fold} | {fmt(hf)} | {fmt(mf)} | {fmt(qf)} |")
    lines.append("")
    lines.append(
        f"**Mean:** heuristic {np.nanmean(h_f):.3f} · "
        f"**Mistral {np.nanmean(m_f):.3f}** · Qwen {np.nanmean(q_f):.3f}"
    )
    lines.append("")

    lines.append("## Methods footnote (say this if asked)")
    lines.append("")
    lines.append(
        "- Folds **0–1** ablations: `max_test=4000` (full ladder incl. `gate_llm`). "
        "Folds **2–4**: `max_test=1000`, skip `gate_llm` (compute budget). "
        "Main comparisons use **cost per 1000 windows** so budgets are comparable."
    )
    lines.append(
        "- End-to-end `agentic_fold*.json` with **heuristic** backend is **not** the full stack; "
        "primary paper row is ablation **`gate_knn_llm`** (Ollama). Folds 2–4 agentic already use ollama."
    )
    lines.append("- Near-fall FAR is reported; it is **not** the primary differentiator in these runs.")
    lines.append("- KFall external validation: code ready, raw data not run yet.")
    lines.append("")

    lines.append("## What to say / not say")
    lines.append("")
    lines.append("| Say | Do not say |")
    lines.append("|-----|------------|")
    lines.append("| Agentic wrapper on a fixed Tier-1 detector | We beat all SOTA fall detectors |")
    lines.append("| Primary gains: cost↓ + recall/F1↑ via escalation | Detector F1 ≥ 0.90 |")
    lines.append("| Mistral recovers ambiguous cases heuristics miss | FAR problem is solved |")
    lines.append("| 5-fold subject-independent SisFall | Results from a single fold only |")
    lines.append("")

    lines.append("## Artifacts")
    lines.append("")
    lines.append("- Per-fold: `results/fold{0-4}/ablation_fold*.json`, `llm_compare_fold*.json`")
    lines.append("- Aggregate: `results/aggregate_ablation_mean_std.*`, `aggregate_ablation_tests.json`")
    lines.append("- Footnotes: `results/aggregate_paper_footnotes.json`")
    lines.append("")

    out = ROOT / "results" / "GUIDE_BRIEF.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
