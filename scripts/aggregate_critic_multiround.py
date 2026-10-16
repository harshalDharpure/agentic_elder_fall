#!/usr/bin/env python3
"""Aggregate SisFall/KFall critic ladder into CRITIC_MULTIROUND_COMPARISON.md."""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _mean_std(vals: list[float]) -> str:
    if not vals:
        return "—"
    if len(vals) == 1:
        return f"{vals[0]:.3f}"
    return f"{st.mean(vals):.3f} ± {st.stdev(vals):.3f}"


def _row_metrics(r: dict) -> dict:
    return {
        "f1": r.get("f1"),
        "recall": r.get("recall"),
        "cost": r.get("cost_per_1000") or r.get("expected_response_cost"),
        "graded": r.get("graded_action_cost"),
        "ground": r.get("rationale_grounding_rate"),
        "esc": r.get("escalation_rate"),
        "llm": r.get("mean_llm_calls") or r.get("mean_llm_calls_escalated"),
    }


def load_sisfall_mode(mode: str) -> list[dict]:
    rows = []
    for fold in range(5):
        cands = [
            ROOT / f"results/fold{fold}/ablation_fold{fold}_multi_round_critique.json",
            ROOT / f"results/fold{fold}/ablation_fold{fold}_modes_gate_knn_llm_gate_knn_llm_action_critique.json",
            ROOT / f"results/fold{fold}/ablation_fold{fold}_modes_gate_knn_llm_gate_knn_llm_contrastive_action_critique.json",
            ROOT / f"results/fold{fold}/ablation_fold{fold}_action_critique.json",
            ROOT / f"results/fold{fold}/ablation_fold{fold}_contrastive_action.json",
            ROOT / f"results/fold{fold}/ablation_fold{fold}.json",
            ROOT / f"results/ablation_fold{fold}_action_critique.json",
            ROOT / f"results/ablation_fold{fold}_contrastive_action.json",
        ]
        found = None
        for p in cands:
            if not p.exists():
                continue
            data = json.loads(p.read_text())
            by = {r["name"]: r for r in data.get("rows", []) if "name" in r}
            if mode in by:
                found = by[mode]
                break
        if found:
            rows.append({"fold": fold, **_row_metrics(found)})
    return rows


def load_kfall_mode(mode: str) -> list[dict]:
    rows = []
    for fold in range(5):
        p = ROOT / f"results/kfall/cnn_lstm_attn_critic_ladder/fold{fold}/kfall_external_fold{fold}.json"
        # also try generic names
        cands = list((ROOT / f"results/kfall/cnn_lstm_attn_critic_ladder/fold{fold}").glob("*.json")) if (ROOT / f"results/kfall/cnn_lstm_attn_critic_ladder/fold{fold}").exists() else []
        if p.exists():
            cands = [p] + cands
        found = None
        for path in cands:
            data = json.loads(path.read_text())
            by = {r["name"]: r for r in data.get("rows", []) if isinstance(r, dict) and "name" in r}
            if mode in by:
                found = by[mode]
                break
        if found:
            rows.append({"fold": fold, **_row_metrics(found)})
    return rows


def table_block(title: str, modes: list[str], loader) -> str:
    lines = [f"### {title}", "", "| Mode | F1 | Recall | Cost | Graded cost | Grounding | Esc rate | LLM calls | n_folds |",
             "|------|---:|-------:|-----:|------------:|----------:|---------:|----------:|--------:|"]
    for mode in modes:
        rows = loader(mode)
        if not rows:
            lines.append(f"| `{mode}` | — | — | — | — | — | — | — | 0 |")
            continue
        def col(k):
            vals = [r[k] for r in rows if r.get(k) is not None]
            return _mean_std([float(v) for v in vals])
        lines.append(
            f"| `{mode}` | {col('f1')} | {col('recall')} | {col('cost')} | {col('graded')} | {col('ground')} | {col('esc')} | {col('llm')} | {len(rows)} |"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    sis_modes = [
        "gate_knn_llm",
        "gate_knn_llm_action_critique",
        "gate_knn_llm_contrastive_action_critique",
        "gate_knn_llm_multi_round_critique",
    ]
    kfall_modes = list(sis_modes)

    # Prefer existing analysis numbers for SisFall non-multiround if folds incomplete
    analysis = ROOT / "results/CRITIQUE_VS_NO_CRITIQUE_ANALYSIS.md"

    md = []
    md.append("# Critic / Multi-round Comparison (SisFall + KFall)")
    md.append("")
    md.append("Protocol: subject-independent 5-fold, n=1000 test/fold, train-only RAG, frozen Tier-1, Ollama/Mistral.")
    md.append("")
    md.append("## Recommendation")
    md.append("")
    md.append("- **Main paper system:** `gate_knn_llm` (**no critic**) — best detection F1 / FN-weighted cost.")
    md.append("- **Optional appendix critic:** `gate_knn_llm_contrastive_action_critique` (frozen labels) if grounding / graded action cost wins.")
    md.append("- **Multi-round:** take forward only if it clearly beats one-shot contrastive on grounding/action without hurting recall or exploding latency.")
    md.append("- **Do not forward:** label-flipping `actor_critic`.")
    md.append("")
    md.append(table_block("SisFall (cnn_lstm_attn)", sis_modes, load_sisfall_mode))
    md.append(table_block("KFall binary (cnn_lstm_attn)", kfall_modes, load_kfall_mode))

    # Decision from available numbers
    sis_base = load_sisfall_mode("gate_knn_llm")
    sis_cont = load_sisfall_mode("gate_knn_llm_contrastive_action_critique")
    sis_multi = load_sisfall_mode("gate_knn_llm_multi_round_critique")
    md.append("## Decision notes")
    md.append("")
    if sis_base:
        md.append(f"- SisFall no-critic folds available: {len(sis_base)}")
    if sis_cont:
        g = [r["ground"] for r in sis_cont if r.get("ground") is not None]
        md.append(f"- SisFall contrastive grounding mean: {st.mean(g):.3f}" if g else "- SisFall contrastive: present")
    if sis_multi:
        f1s = [r["f1"] for r in sis_multi if r.get("f1") is not None]
        gs = [r["ground"] for r in sis_multi if r.get("ground") is not None]
        md.append(f"- SisFall multi-round F1 mean: {st.mean(f1s):.3f} (n={len(f1s)})" if f1s else "")
        md.append(f"- SisFall multi-round grounding mean: {st.mean(gs):.3f}" if gs else "")
    else:
        md.append("- SisFall multi-round: **not finished / missing**")
    k_multi = load_kfall_mode("gate_knn_llm_multi_round_critique")
    md.append(f"- KFall multi-round folds: {len(k_multi)}")
    if analysis.exists():
        md.append(f"- Prior SisFall critic analysis: `{analysis.relative_to(ROOT)}`")
    md.append("")
    md.append("## Take-forward rule")
    md.append("")
    md.append("1. Keep `gate_knn_llm` as primary claim.")
    md.append("2. If contrastive grounding ≫ base and graded cost ↓ without FN rise → appendix.")
    md.append("3. If multi-round ≤ contrastive on grounding/action at higher LLM cost → **drop multi-round**.")
    md.append("")

    out = ROOT / "results/CRITIC_MULTIROUND_COMPARISON.md"
    out.write_text("\n".join(md) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
