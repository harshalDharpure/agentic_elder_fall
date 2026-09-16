#!/usr/bin/env python3
"""Patch paper/main.tex and Q1 briefs with KFall / Bhatti-limitation sections if missing."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "paper" / "main.tex"
PACK = ROOT / "results" / "Q1_SUBMISSION_PACK.md"
BRIEF = ROOT / "results" / "PROFESSOR_MEETING_BRIEF.md"
SUM = ROOT / "results" / "kfall" / "kfall_external_summary.json"


KFALL_SECTION = r"""
\subsection{External Validation on KFall}
\label{sec:kfall}

Bhatti \textit{et al.}~\cite{bhatti2025beyond} evaluate multi-class multi-phase detection on KFall and report approximately 98\% F1 with $\sim$20\,ms inference. Their manuscript also notes persistent confusion on sit/stand ADLs (D18/D19) and leaves open uncertainty routing and response policy. We therefore evaluate our \emph{agentic safety layer} on KFall under a locked binary Fall/ADL protocol that mirrors SisFall: same CNN--LSTM--Attention Tier-1, subject-independent five-fold splits, and cost $10\cdot\mathrm{FN}+1\cdot\mathrm{FP}$.

Table~\ref{tab:bhattigaps} maps their gaps to our agents. Tables~\ref{tab:kfallperfold}--\ref{tab:kfallmean} report Tier-1 vs.\ \texttt{gate\_knn\_llm}; Table~\ref{tab:kfallablation} shows the Fold~0 ablation ladder; Table~\ref{tab:kfallllm} compares LLM backends on paired ambiguous IDs. Table~\ref{tab:kfallmc} reports a context-only 36-class Tier-1 run and is \emph{not} a claim against the published multi-class leaderboard.

\input{table/tab_bhatti_gaps}
\input{table/tab_kfall_perfold}
\input{table/tab_kfall_mean}
\input{table/tab_kfall_ablation}
\input{table/tab_kfall_llm}
\input{table/tab_kfall_multiclass}

"""


def patch_tex():
    text = TEX.read_text()
    if "sec:kfall" in text:
        print("main.tex already has KFall section")
        return
    # Insert before Discussion
    needle = r"\section{Discussion}"
    if needle not in text:
        raise SystemExit("Could not find Discussion section in main.tex")
    # Also ensure Results ends with KFall after backbone
    insert_at = text.find(needle)
    text = text[:insert_at] + KFALL_SECTION + "\n" + text[insert_at:]
    # Strengthen discussion limitations sentence
    old = "External validation on KFall remains future work pending raw data."
    new = (
        "KFall external validation under our binary agentic protocol is reported in "
        "Section~\\ref{sec:kfall}; we still do not claim to supersede Bhatti's published "
        "36-class multi-phase tables."
    )
    text = text.replace(old, new)
    # Update abstract slightly if needed
    if "KFall external" not in text.split(r"\begin{abstract}")[1].split(r"\end{abstract}")[0]:
        text = text.replace(
            "External validation tooling for binary KFall is released with the codebase;",
            "External validation on binary KFall is included alongside SisFall;",
        )
    TEX.write_text(text)
    print("patched main.tex")


def patch_pack():
    summary = json.loads(SUM.read_text()) if SUM.exists() else {}
    modes = summary.get("modes", {})
    lines = [
        "",
        "## KFall external validation (vs Bhatti gaps)",
        "",
        "- Claim: agentic wrapper improves recall/cost vs same Tier-1 on KFall binary (not beating published 98% multi-class).",
        f"- Updated: {datetime.now().isoformat(timespec='seconds')}",
    ]
    if "tier1_only" in modes and "gate_knn_llm" in modes:
        t, s = modes["tier1_only"], modes["gate_knn_llm"]
        lines += [
            f"- Tier-1 F1 {t['f1_mean']:.3f}±{t['f1_std']:.3f} → stack {s['f1_mean']:.3f}±{s['f1_std']:.3f}",
            f"- Tier-1 recall {t['recall_mean']:.3f} → stack {s['recall_mean']:.3f}",
            f"- Cost/1000 {t['cost_per_1000_mean']:.0f} → {s['cost_per_1000_mean']:.0f}",
        ]
    lines += [
        "- [x] KFall downloaded + prepared",
        "- [x] Binary agentic 5-fold pipeline",
        "- [x] Bhatti-limitation mapping in paper",
        "- [ ] Rotate exposed Kaggle API token",
        "",
    ]
    text = PACK.read_text() if PACK.exists() else ""
    if "## KFall external validation" in text:
        # replace block
        pre = text.split("## KFall external validation")[0]
        PACK.write_text(pre.rstrip() + "\n" + "\n".join(lines))
    else:
        PACK.write_text(text.rstrip() + "\n" + "\n".join(lines))
    print("updated Q1_SUBMISSION_PACK.md")


def patch_brief():
    summary = json.loads(SUM.read_text()) if SUM.exists() else {}
    modes = summary.get("modes", {})
    block = [
        "",
        "## KFall vs Bhatti (say this)",
        "",
        "> Sir, Bhatti’s paper is multi-class detection (~98% F1). Ours is an agentic safety layer on the same CNN–LSTM–Attention idea. On KFall binary 5-fold we compare against the *same* Tier-1 checkpoint—not their published 36-class table.",
        "",
    ]
    if "tier1_only" in modes and "gate_knn_llm" in modes:
        t, s = modes["tier1_only"], modes["gate_knn_llm"]
        block.append(
            f"- KFall mean: F1 {t['f1_mean']:.3f}→{s['f1_mean']:.3f}; "
            f"recall {t['recall_mean']:.3f}→{s['recall_mean']:.3f}; "
            f"cost/1k {t['cost_per_1000_mean']:.0f}→{s['cost_per_1000_mean']:.0f}"
        )
    block += [
        "- Their D18/D19 confusion → our Confidence Gate + LLM escalation.",
        "- Their no response policy → our Action Agent with 10×FN cost.",
        "",
    ]
    text = BRIEF.read_text() if BRIEF.exists() else ""
    if "## KFall vs Bhatti" in text:
        pre = text.split("## KFall vs Bhatti")[0]
        BRIEF.write_text(pre.rstrip() + "\n" + "\n".join(block))
    else:
        BRIEF.write_text(text.rstrip() + "\n" + "\n".join(block))
    print("updated PROFESSOR_MEETING_BRIEF.md")


def main():
    # Always generate table stubs / current numbers
    import subprocess

    subprocess.check_call(["python", "scripts/generate_kfall_paper_tables.py"], cwd=ROOT)
    patch_tex()
    patch_pack()
    patch_brief()


if __name__ == "__main__":
    main()
