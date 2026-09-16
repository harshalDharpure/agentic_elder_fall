#!/usr/bin/env python3
"""Write paper/table/tab_kfall_*.tex from results/kfall summaries."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "table"
OUT.mkdir(parents=True, exist_ok=True)
KF = ROOT / "results" / "kfall"


def write(name: str, text: str):
    (OUT / name).write_text(text)
    print("wrote", name)


def load_summary():
    p = KF / "kfall_external_summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def table_positioning_bhatti():
    write(
        "tab_bhatti_gaps.tex",
        r"""\begin{table*}[!t]
\centering
\caption{Bhatti \textit{et al.}~2025 limitations versus our multi-agent answers (complementary Q1 claim).}
\label{tab:bhattigaps}
\begin{tabular}{p{0.42\linewidth}p{0.48\linewidth}}
\toprule
Bhatti \textit{et al.} gap & Our multi-agent response \\
\midrule
Softmax force-labels every window; D18/D19 sit/stand confusion & Dual-threshold \textbf{Confidence Gate} escalates ambiguous band \\
Explainability limited to attention weights & Biomechanical \textbf{evidence} + LLM chain-of-thought JSON \\
No retrieval / case memory & MiniLM $k$-NN \textbf{Memory} agent \\
Detection only; no clinical response policy & Cost-sensitive \textbf{Action Agent} ($10\cdot\mathrm{FN}+1\cdot\mathrm{FP}$) \\
Published focus: 36-class multi-phase F1~$\approx$98\%, $\sim$20\,ms & Fair claim: same Tier-1 + agents improve recall/cost under uncertainty (SisFall+KFall) \\
\bottomrule
\end{tabular}
\end{table*}
""",
    )


def table_kfall_perfold(summary: dict | None):
    if not summary or not summary.get("folds"):
        write(
            "tab_kfall_perfold.tex",
            "% KFall per-fold pending — run scripts/run_kfall_q1_pipeline.sh\n"
            r"\begin{table}[!t]\centering\caption{KFall binary external validation (pending).}\label{tab:kfallperfold}"
            r"\begin{tabular}{c}Pending\end{tabular}\end{table}"
            "\n",
        )
        return
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{KFall binary external validation: Tier-1 vs.\ full agentic stack ($n{=}1000$/fold, subject-independent). Cost normalized per 1000 windows.}",
        r"\label{tab:kfallperfold}",
        r"\begin{tabular}{c ccc ccc c}",
        r"\toprule",
        r"Fold & \multicolumn{3}{c}{Tier-1} & \multicolumn{3}{c}{gate\_knn\_llm} & $\Delta$F1 \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r" & F1 & Rec. & Cost/1k & F1 & Rec. & Cost/1k & \\",
        r"\midrule",
    ]
    for fr in summary["folds"]:
        t = fr.get("tier1_only")
        s = fr.get("gate_knn_llm")
        if not t or not s:
            continue
        lines.append(
            f"{fr['fold']} & {t['f1']:.3f} & {t['recall']:.3f} & {t['cost_per_1000']:.0f} & "
            f"{s['f1']:.3f} & {s['recall']:.3f} & {s['cost_per_1000']:.0f} & "
            f"{s['f1']-t['f1']:+.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    write("tab_kfall_perfold.tex", "\n".join(lines))


def table_kfall_mean(summary: dict | None):
    if not summary or "modes" not in summary or "tier1_only" not in summary.get("modes", {}):
        write(
            "tab_kfall_mean.tex",
            "% pending\n"
            r"\begin{table}[!t]\centering\caption{KFall 5-fold mean$\pm$std (pending).}\label{tab:kfallmean}"
            r"\begin{tabular}{c}Pending\end{tabular}\end{table}"
            "\n",
        )
        return
    modes = summary["modes"]
    tests = {t["metric"]: t for t in summary.get("paired_tests", [])}

    def ms(mode, key):
        m = modes[mode]
        return f"{m[key+'_mean']:.3f}$\\pm${m[key+'_std']:.3f}"

    def msi(mode, key):
        m = modes[mode]
        return f"{m[key+'_mean']:.0f}$\\pm${m[key+'_std']:.0f}"

    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{KFall five-fold mean$\pm$std (binary Fall/ADL). Paired deltas are stack$-$Tier-1.}",
        r"\label{tab:kfallmean}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Mode & F1 & Recall & Cost/1000 \\",
        r"\midrule",
        f"tier1\\_only & {ms('tier1_only','f1')} & {ms('tier1_only','recall')} & {msi('tier1_only','cost_per_1000')} \\\\",
    ]
    if "gate_knn_llm" in modes:
        lines.append(
            f"\\textbf{{gate\\_knn\\_llm}} & \\textbf{{{ms('gate_knn_llm','f1')}}} & "
            f"\\textbf{{{ms('gate_knn_llm','recall')}}} & \\textbf{{{msi('gate_knn_llm','cost_per_1000')}}} \\\\"
        )
    if tests:
        lines += [
            r"\midrule",
            r"\multicolumn{4}{l}{\textit{Paired $\Delta$ (bootstrap 95\% CI)}} \\",
        ]
        for metric in ("f1", "recall", "cost_per_1000"):
            if metric not in tests:
                continue
            t = tests[metric]
            ci = t["bootstrap_ci95"]
            label = metric.replace("_", r"\_")
            lines.append(
                f"{label} & {t['mean_delta']:+.3f} & "
                f"[{ci[0]:.3f}, {ci[1]:.3f}] & $p$={t.get('wilcoxon_pvalue', float('nan')):.4f} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write("tab_kfall_mean.tex", "\n".join(lines))


def table_kfall_ablation(summary: dict | None):
    # Prefer fold0 ladder from raw JSON
    p = KF / "fold0" / "kfall_external_fold0.json"
    if not p.exists():
        write("tab_kfall_ablation.tex", "% pending ablation\n")
        return
    rows = {r["name"]: r for r in json.loads(p.read_text()).get("rows", [])}
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{KFall Fold~0 ablation ladder ($n{=}1000$). Full stack requires $k$-NN+LLM.}",
        r"\label{tab:kfallablation}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Mode & F1 & Recall & Cost & Esc.\ F1 \\",
        r"\midrule",
    ]
    for name in ("tier1_only", "gate_only", "gate_knn", "gate_llm", "gate_knn_llm"):
        if name not in rows:
            continue
        r = rows[name]
        n = int(r.get("max_test") or 1000)
        cost = float(r["expected_response_cost"])
        nm = name.replace("_", r"\_")
        if name == "gate_knn_llm":
            nm = r"\textbf{gate\_knn\_llm}"
        lines.append(
            f"{nm} & {r['f1']:.3f} & {r['recall']:.3f} & {cost:.0f} & "
            f"{r.get('escalated_f1', float('nan')):.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write("tab_kfall_ablation.tex", "\n".join(lines))


def table_kfall_llm():
    stores = {"heuristic": [], "mistral": [], "qwen": []}
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{KFall paired LLM backends on shared ambiguous IDs.}",
        r"\label{tab:kfallllm}",
        r"\begin{tabular}{cccc}",
        r"\toprule",
        r"Fold & Heuristic F1 & Mistral F1 & Qwen F1 \\",
        r"\midrule",
    ]
    any_row = False
    for f in range(5):
        p = KF / f"fold{f}" / f"kfall_llm_compare_fold{f}.json"
        if not p.exists():
            p = KF / f"kfall_llm_compare_fold{f}.json"
        if not p.exists():
            continue
        any_row = True
        rows = json.loads(p.read_text()).get("rows", [])
        h = m = q = float("nan")
        for r in rows:
            n = str(r.get("name", ""))
            if n == "heuristic":
                h = r["f1"]
                stores["heuristic"].append(h)
            elif "mistral" in n:
                m = r["f1"]
                stores["mistral"].append(m)
            elif "qwen" in n:
                q = r["f1"]
                stores["qwen"].append(q)
        lines.append(f"{f} & {h:.3f} & {m:.3f} & {q:.3f} \\\\")
    if any_row and stores["mistral"]:
        lines += [
            r"\midrule",
            f"Mean & {np.mean(stores['heuristic']):.3f} & \\textbf{{{np.mean(stores['mistral']):.3f}}} & "
            f"{np.mean(stores['qwen']) if stores['qwen'] else float('nan'):.3f} \\\\",
        ]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    if not any_row:
        write("tab_kfall_llm.tex", "% LLM compare pending\n")
    else:
        write("tab_kfall_llm.tex", "\n".join(lines))


def table_kfall_multiclass():
    p = KF / "kfall_multiclass_fold0.json"
    if not p.exists():
        write("tab_kfall_multiclass.tex", "% multiclass pending\n")
        return
    d = json.loads(p.read_text())
    write(
        "tab_kfall_multiclass.tex",
        "\n".join(
            [
                r"\begin{table}[!t]",
                r"\centering",
                r"\caption{Context-only 36-class Tier-1 on KFall transition (Fold~0). Not a claim against Bhatti's published multi-class table.}",
                r"\label{tab:kfallmc}",
                r"\begin{tabular}{lccc}",
                r"\toprule",
                r"Setting & Acc. & Macro-F1 & Weighted-F1 \\",
                r"\midrule",
                f"Our Tier-1 (transition, fold0) & {d['accuracy']:.3f} & {d['macro_f1']:.3f} & {d['weighted_f1']:.3f} \\\\",
                r"Bhatti \textit{et al.} (published, multi-phase) & --- & $\approx$0.98 & --- \\",
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
                "",
            ]
        ),
    )


def main():
    summary = load_summary()
    table_positioning_bhatti()
    table_kfall_perfold(summary)
    table_kfall_mean(summary)
    table_kfall_ablation(summary)
    table_kfall_llm()
    table_kfall_multiclass()


if __name__ == "__main__":
    main()
