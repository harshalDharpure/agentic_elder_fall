#!/usr/bin/env python3
"""Generate paper/tables/*.tex from results JSON."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "table"
OUT.mkdir(parents=True, exist_ok=True)


def load_abl(fold: int, archive: bool = False):
    name = f"ablation_fold{fold}_full4000.json" if archive else f"ablation_fold{fold}.json"
    p = ROOT / f"results/fold{fold}" / name
    if archive and not p.exists():
        p = ROOT / f"results/fold{fold}/ablation_fold{fold}.json"
    d = json.loads(p.read_text())
    rows = d.get("rows", d) if isinstance(d, dict) else d
    return {r["name"]: r for r in rows if isinstance(r, dict) and "name" in r}, (d if isinstance(d, dict) else {})


def n_of(row, meta):
    for k in ("n", "max_test"):
        if isinstance(row.get(k), (int, float)) and int(row[k]) > 0:
            return int(row[k])
    if isinstance(meta.get("max_test"), (int, float)):
        return int(meta["max_test"])
    parts = [row.get(x) for x in ("tp", "tn", "fp", "fn")]
    if all(isinstance(x, (int, float)) for x in parts):
        return int(sum(parts))
    return 1000


def cpk(row, meta):
    return float(row["expected_response_cost"]) / n_of(row, meta) * 1000.0


def write(name: str, text: str):
    (OUT / name).write_text(text)
    print("wrote", name)


def table_positioning():
    write(
        "tab_positioning.tex",
        r"""\begin{table*}[!t]
\centering
\caption{Positioning relative to Bhatti \textit{et al.} (base detector paper).}
\label{tab:positioning}
\begin{tabular}{lll}
\toprule
Aspect & Bhatti \textit{et al.}~2025~\cite{bhatti2025beyond} & This work \\
\midrule
Primary task & Multi-phase fall/ADL classification & Detection + uncertainty + explanation + response \\
Backbone & CNN--LSTM--Attention & Same primary backbone + modern zoo \\
Uncertainty & Softmax confidence only & Dual-threshold confidence gate ($\tau_{\mathrm{low}},\tau_{\mathrm{high}}$) \\
Explainability & Attention weights & Structured biomechanical evidence + CoT JSON \\
Case memory & None & Retrieval-augmented $k$-NN memory \\
Response policy & None & Cost-sensitive Action Agent \\
Primary dataset in original paper & KFall (multi-class) & SisFall binary + near-fall ADLs (D18/D19) \\
Fair baseline in our tables & --- & \texttt{tier1\_only} (same checkpoint/folds) \\
\bottomrule
\end{tabular}
\end{table*}
""",
    )


def table_per_fold():
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Per-fold comparison of Tier-1 (\texttt{tier1\_only}) vs.\ full agentic stack (\texttt{gate\_knn\_llm}) under the homogeneous protocol ($n{=}1000$ stratified windows/fold with forced D18/D19). Cost is normalized per 1000 windows.}",
        r"\label{tab:perfold}",
        r"\begin{tabular}{c r ccc ccc c}",
        r"\toprule",
        r"Fold & $n$ & \multicolumn{3}{c}{Tier-1} & \multicolumn{3}{c}{gate\_knn\_llm} & $\Delta$F1 \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r" & & F1 & Rec. & Cost/1k & F1 & Rec. & Cost/1k & \\",
        r"\midrule",
    ]
    for f in range(5):
        m, meta = load_abl(f)
        t, s = m["tier1_only"], m["gate_knn_llm"]
        n = n_of(s, meta)
        lines.append(
            f"{f} & {n} & {t['f1']:.3f} & {t['recall']:.3f} & {cpk(t, meta):.0f} & "
            f"{s['f1']:.3f} & {s['recall']:.3f} & {cpk(s, meta):.0f} & {s['f1']-t['f1']:+.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    write("tab_perfold.tex", "\n".join(lines))


def table_mean_std():
    f1t, f1s, rt, rs, ct, cs = [], [], [], [], [], []
    for f in range(5):
        m, meta = load_abl(f)
        t, s = m["tier1_only"], m["gate_knn_llm"]
        f1t.append(t["f1"])
        f1s.append(s["f1"])
        rt.append(t["recall"])
        rs.append(s["recall"])
        ct.append(cpk(t, meta))
        cs.append(cpk(s, meta))
    tests = json.loads((ROOT / "results/aggregate_ablation_tests.json").read_text())
    by_m = {t["metric"]: t for t in tests}

    def ms(a):
        return f"{np.mean(a):.3f}$\\pm${np.std(a, ddof=1):.3f}"

    def msi(a):
        return f"{np.mean(a):.0f}$\\pm${np.std(a, ddof=1):.0f}"

    def row_delta(metric, label):
        t = by_m[metric]
        ci = t["bootstrap_ci95"]
        return (
            f"{label} & {t['mean_delta']:+.3f} & "
            f"[{ci[0]:.3f}, {ci[1]:.3f}] & {t.get('wilcoxon_pvalue', float('nan')):.4f} \\\\"
        )

    write(
        "tab_meanstd.tex",
        "\n".join(
            [
                r"\begin{table}[!t]",
                r"\centering",
                r"\caption{Five-fold mean$\pm$std on SisFall (homogeneous $n{=}1000$). Paired deltas are \texttt{gate\_knn\_llm}$-$\texttt{tier1\_only}. With five folds, Wilcoxon $p{=}0.0625$ is at the discrete minimum for a complete sign agreement; we therefore emphasize bootstrap 95\% CIs.}",
                r"\label{tab:meanstd}",
                r"\begin{tabular}{lccc}",
                r"\toprule",
                r"Mode & F1 & Recall & Cost / 1000 \\",
                r"\midrule",
                f"tier1\\_only & {ms(f1t)} & {ms(rt)} & {msi(ct)} \\\\",
                f"\\textbf{{gate\\_knn\\_llm}} & \\textbf{{{ms(f1s)}}} & \\textbf{{{ms(rs)}}} & \\textbf{{{msi(cs)}}} \\\\",
                r"\midrule",
                r"\multicolumn{4}{l}{\textit{Paired $\Delta$ (stack $-$ Tier-1)}} \\",
                r"Metric & Mean $\Delta$ & Bootstrap 95\% CI & Wilcoxon $p$ \\",
                r"\midrule",
                row_delta("f1", "F1"),
                row_delta("recall", "Recall"),
                row_delta("cost_per_1000", "Cost/1000"),
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
                "",
            ]
        ),
    )


def table_ablation():
    m0, _ = load_abl(0, archive=True)
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Ablation ladder on Fold~0 under the full evaluation budget ($n{=}4000$). Gate or $k$-NN alone does not yield the headline gain; the full stack (\texttt{gate\_knn\_llm}) is required.}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Mode & F1 & Recall & Cost & Near-fall FAR & Esc.\ F1 \\",
        r"\midrule",
    ]
    for name in ("tier1_only", "gate_only", "gate_knn", "gate_llm", "gate_knn_llm"):
        r = m0[name]
        bold = name == "gate_knn_llm"
        cells = [
            f"{r['f1']:.3f}",
            f"{r['recall']:.3f}",
            f"{r['expected_response_cost']:.0f}",
            f"{r.get('false_alarm_rate_near_fall', float('nan')):.3f}",
            f"{r.get('escalated_f1', float('nan')):.3f}",
        ]
        if bold:
            cells = [f"\\textbf{{{c}}}" for c in cells]
            name_s = r"\textbf{gate\_knn\_llm}"
        else:
            name_s = name.replace("_", r"\_")
        lines.append(f"{name_s} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write("tab_ablation.tex", "\n".join(lines))


def table_llm():
    stores = {"heuristic": [], "mistral": [], "qwen": []}
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Paired LLM backend comparison on shared ambiguous case IDs (up to 150 cases/fold).}",
        r"\label{tab:llm}",
        r"\begin{tabular}{cccc}",
        r"\toprule",
        r"Fold & Heuristic F1 & Mistral F1 & Qwen2.5 F1 \\",
        r"\midrule",
    ]
    for f in range(5):
        rows = json.loads((ROOT / f"results/fold{f}/llm_compare_fold{f}.json").read_text())
        if isinstance(rows, dict):
            rows = rows.get("rows", rows)
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
    lines += [
        r"\midrule",
        f"Mean & {np.mean(stores['heuristic']):.3f} & \\textbf{{{np.mean(stores['mistral']):.3f}}} & {np.mean(stores['qwen']):.3f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    write("tab_llm.tex", "\n".join(lines))


def table_backbone():
    bb = json.loads((ROOT / "results/fold0/backbone_compare_fold0.json").read_text())
    rows = bb if isinstance(bb, list) else bb.get("rows", bb)
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Tier-1 backbone zoo on Fold~0 (detector-only metrics). Primary paper backbone is \texttt{cnn\_lstm\_attn} (Bhatti-style).}",
        r"\label{tab:backbone}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Model & F1 & Recall & Spec. & Params \\",
        r"\midrule",
    ]
    for r in rows:
        name = str(r.get("name", ""))
        if "+agentic" in name or "f1" not in r:
            continue
        params = r.get("params", r.get("n_params", ""))
        if isinstance(params, (int, float)):
            params = f"{int(params):,}"
        spec = r.get("specificity", r.get("spec", float("nan")))
        nm = name.replace("_", r"\_")
        if name == "cnn_lstm_attn":
            nm = r"\textbf{cnn\_lstm\_attn}"
        lines.append(f"{nm} & {r['f1']:.3f} & {r['recall']:.3f} & {spec:.3f} & {params} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write("tab_backbone.tex", "\n".join(lines))


def main():
    table_positioning()
    table_per_fold()
    table_mean_std()
    table_ablation()
    table_llm()
    table_backbone()


if __name__ == "__main__":
    main()
