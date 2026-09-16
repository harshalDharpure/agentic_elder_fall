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
        if name not in m0:
            continue
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

    # Homogeneous 5-fold mean ablation including gate_llm when available
    modes = ("tier1_only", "gate_only", "gate_knn", "gate_llm", "gate_knn_llm")
    store = {m: {"f1": [], "rec": [], "cost": []} for m in modes}
    have_gate_llm = True
    for f in range(5):
        m, meta = load_abl(f)
        if "gate_llm" not in m:
            have_gate_llm = False
        for name in modes:
            if name not in m:
                continue
            store[name]["f1"].append(m[name]["f1"])
            store[name]["rec"].append(m[name]["recall"])
            store[name]["cost"].append(cpk(m[name], meta))

    def ms(a):
        if len(a) < 2:
            return f"{a[0]:.3f}" if a else "---"
        return f"{np.mean(a):.3f}$\\pm${np.std(a, ddof=1):.3f}"

    def msi(a):
        if len(a) < 2:
            return f"{a[0]:.0f}" if a else "---"
        return f"{np.mean(a):.0f}$\\pm${np.std(a, ddof=1):.0f}"

    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Five-fold mean$\pm$std ablation under the homogeneous $n{=}1000$ protocol"
        + ("" if have_gate_llm else r" (\texttt{gate\_llm} pending on some folds)")
        + r".}",
        r"\label{tab:ablationmean}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Mode & F1 & Recall & Cost/1000 \\",
        r"\midrule",
    ]
    for name in modes:
        if not store[name]["f1"]:
            continue
        nm = name.replace("_", r"\_")
        if name == "gate_knn_llm":
            nm = r"\textbf{gate\_knn\_llm}"
            cells = [
                f"\\textbf{{{ms(store[name]['f1'])}}}",
                f"\\textbf{{{ms(store[name]['rec'])}}}",
                f"\\textbf{{{msi(store[name]['cost'])}}}",
            ]
        else:
            cells = [ms(store[name]["f1"]), ms(store[name]["rec"]), msi(store[name]["cost"])]
        lines.append(f"{nm} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write("tab_ablation_mean.tex", "\n".join(lines))


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


def table_latency():
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{End-to-end latency from Ollama agentic runs ($n{=}1000$/fold). Escalated-path latency is dominated by local LLM inference.}",
        r"\label{tab:latency}",
        r"\begin{tabular}{crrr}",
        r"\toprule",
        r"Fold & Tier-1 (ms) & Escalated (s) & Mean stack (s) \\",
        r"\midrule",
    ]
    t_all, e_all, m_all = [], [], []
    for f in range(5):
        p = ROOT / f"results/fold{f}/agentic_fold{f}.json"
        d = json.loads(p.read_text())
        m = d.get("metrics", d)
        t = float(m.get("latency_tier1_ms", float("nan")))
        e = float(m.get("latency_escalated_ms", float("nan"))) / 1000.0
        mm = float(m.get("latency_mean_ms", float("nan"))) / 1000.0
        t_all.append(t)
        e_all.append(e)
        m_all.append(mm)
        lines.append(f"{f} & {t:.2f} & {e:.2f} & {mm:.2f} \\\\")
    lines += [
        r"\midrule",
        f"Mean & {np.mean(t_all):.2f} & {np.mean(e_all):.2f} & {np.mean(m_all):.2f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    write("tab_latency.tex", "\n".join(lines))


def table_seed_sweep():
    p = ROOT / "results/seed_sweep/seed_sweep_fold0.json"
    if not p.exists():
        write(
            "tab_seeds.tex",
            "% seed sweep pending — run: python scripts/run_seed_sweep.py --fold 0\n",
        )
        return
    d = json.loads(p.read_text())
    backends = {r.get("stack_backend", "heuristic") for r in d.get("rows", [])}
    backend_note = "Ollama/Mistral" if backends == {"ollama"} else (
        "heuristic" if backends == {"heuristic"} else "mixed backends"
    )
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        rf"\caption{{Multi-seed robustness on Fold~0 ($n{{=}}1000$). Stack backend: {backend_note}.}}",
        r"\label{tab:seeds}",
        r"\begin{tabular}{ccccc}",
        r"\toprule",
        r"Seed & Tier-1 F1 & Stack F1 & Stack Rec. & Cost/1k \\",
        r"\midrule",
    ]
    for r in d.get("rows", []):
        lines.append(
            f"{r['seed']} & {r['tier1_f1']:.3f} & {r['stack_f1']:.3f} & "
            f"{r['stack_recall']:.3f} & {r['stack_cost_per_1000']:.0f} \\\\"
        )
    lines += [
        r"\midrule",
        f"Mean$\\pm$std & {d['tier1_f1_mean']:.3f}$\\pm${d['tier1_f1_std']:.3f} & "
        f"{d['stack_f1_mean']:.3f}$\\pm${d['stack_f1_std']:.3f} & --- & --- \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    write("tab_seeds.tex", "\n".join(lines))




def table_ambiguous_sisfall():
    p = ROOT / "results/kfold_ambiguous/comparison_no_critic.json"
    if not p.exists():
        write("tab_ambiguous_sisfall.tex", "% ambiguous bench pending\n")
        return
    d = json.loads(p.read_text())
    t_rows = {r["fold"]: r for r in d["tier1_only"]}
    s_rows = {r["fold"]: r for r in d["gate_knn_llm"]}
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{SisFall \emph{ambiguous stress benchmark} ($n{=}300$/fold: 100 D18/D19 near-fall + 100 fall + 100 clear ADL). "
        r"Tier-1 F1 collapses on near-fall activities; the full stack recovers performance. "
        r"Cost is clinical safety cost ($10\cdot\mathrm{FN}+\mathrm{FP}$) per window; Esc.\ is System~2 escalation rate.}",
        r"\label{tab:ambiguoussisfall}",
        r"\begin{tabular}{c r ccc ccc cc}",
        r"\toprule",
        r"Fold & $n$ & \multicolumn{3}{c}{Tier-1} & \multicolumn{3}{c}{gate\_knn\_llm} & $\Delta$F1 & Esc. \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r" & & F1 & Rec. & Cost/win & F1 & Rec. & Cost/win & & rate \\",
        r"\midrule",
    ]
    f1t, f1s, rt, rs, ct, cs, esc = [], [], [], [], [], [], []
    for f in range(5):
        t, s = t_rows[f], s_rows[f]
        f1t.append(t["f1"])
        f1s.append(s["f1"])
        rt.append(t["rec"])
        rs.append(s["rec"])
        ct.append(t["cost_win"])
        cs.append(s["cost_win"])
        esc.append(s["esc"])
        lines.append(
            f"{f} & {t['n']} & {t['f1']:.3f} & {t['rec']:.3f} & {t['cost_win']:.2f} & "
            f"{s['f1']:.3f} & {s['rec']:.3f} & {s['cost_win']:.2f} & {s['f1']-t['f1']:+.3f} & {s['esc']:.2f} \\\\"
        )
    lines += [
        r"\midrule",
        f"Mean & 300 & {np.mean(f1t):.3f} & {np.mean(rt):.3f} & {np.mean(ct):.2f} & "
        f"\\textbf{{{np.mean(f1s):.3f}}} & \\textbf{{{np.mean(rs):.3f}}} & \\textbf{{{np.mean(cs):.2f}}} & "
        f"\\textbf{{{np.mean(f1s)-np.mean(f1t):+.3f}}} & {np.mean(esc):.2f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ]
    write("tab_ambiguous_sisfall.tex", "\n".join(lines))





def table_stats():
    """Paired fold-wise statistics: sign consistency + bootstrap CIs."""
    tests_path = ROOT / "results/aggregate_ablation_tests.json"
    if not tests_path.exists():
        write("tab_stats.tex", "% stats pending — run aggregate_folds.py\n")
        return
    tests = {t["metric"]: t for t in json.loads(tests_path.read_text())}

    # Per-fold deltas for sign consistency
    folds = []
    for f in range(5):
        m, meta = load_abl(f)
        t, s = m["tier1_only"], m["gate_knn_llm"]
        folds.append({
            "fold": f,
            "df1": s["f1"] - t["f1"],
            "drec": s["recall"] - t["recall"],
            "dcost": cpk(s, meta) - cpk(t, meta),
        })

    def sign_str(vals, higher_better=True):
        ok = sum(1 for v in vals if (v > 0 if higher_better else v < 0))
        return f"{ok}/{len(vals)}"

    f1_sign = sign_str([r["df1"] for r in folds])
    rec_sign = sign_str([r["drec"] for r in folds])
    cost_sign = sign_str([r["dcost"] for r in folds], higher_better=False)

    def fmt_test(metric, label, higher_better=True):
        t = tests.get(metric) or tests.get("expected_response_cost" if metric == "cost_per_1000" else metric)
        if not t:
            return f"{label} & --- & --- & --- \\\\"
        ci = t.get("bootstrap_ci95", [float("nan"), float("nan")])
        method = t.get("bootstrap_method", "percentile")
        delta = t["mean_delta"]
        if metric in ("cost_per_1000", "expected_response_cost"):
            ci_fmt = f"[{ci[0]:.0f}, {ci[1]:.0f}]"
            delta_fmt = f"{delta:+.0f}"
        else:
            ci_fmt = f"[{ci[0]:.3f}, {ci[1]:.3f}]"
            delta_fmt = f"{delta:+.3f}"
        p = t.get("wilcoxon_pvalue", float("nan"))
        sign = sign_str(
            [r["dcost" if "cost" in metric else ("df1" if metric == "f1" else "drec")] for r in folds],
            higher_better=("cost" not in metric),
        )
        return f"{label} & {delta_fmt} & {ci_fmt} ({method}) & {sign} & {p:.4f} \\\\"

    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Paired statistics for \texttt{gate\_knn\_llm}$-$\texttt{tier1\_only} across five subject-independent folds ($n{=}5$). "
        r"Sign column reports folds with improvement (F1/recall$\uparrow$, cost$\downarrow$). "
        r"With unanimous signs, Wilcoxon $p$ floors at $0.0625$; bootstrap 95\% CIs (BCa when $n{\ge}3$) are primary.}",
        r"\label{tab:stats}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Metric & Mean $\Delta$ & Bootstrap 95\% CI & Sign & Wilcoxon $p$ \\",
        r"\midrule",
        fmt_test("f1", "F1"),
        fmt_test("recall", "Recall"),
        fmt_test("cost_per_1000", "Cost/1000"),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
        r"% Per-fold directional consistency:",
        f"% F1 {f1_sign}, Recall {rec_sign}, Cost {cost_sign}",
        "",
    ]
    write("tab_stats.tex", "\n".join(lines))

    # Also write JSON artifact for briefs
    artifact = {
        "sign_consistency": {"f1": f1_sign, "recall": rec_sign, "cost_per_1000": cost_sign},
        "per_fold": folds,
        "tests": {k: {kk: vv for kk, vv in v.items() if kk != "folds"} for k, v in tests.items()},
    }
    (ROOT / "results/stats_sign_consistency.json").write_text(json.dumps(artifact, indent=2))


def table_astar_conformal():
    p = ROOT / "results/astar/conformal_gate_audit.json"
    if not p.exists():
        write("tab_conformal_gate.tex", "% conformal audit pending\n")
        return
    rows = json.loads(p.read_text())["rows"]
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Split conformal gate audit ($\alpha{=}0.05$): edge-decision FN rate vs.\ cost-calibrated gate escalation.}",
        r"\label{tab:conformalgate}",
        r"\begin{tabular}{c ccc ccc}",
        r"\toprule",
        r"Fold & $q_{\hat{}}$ & Esc.\ (conf.) & Edge FN & F1 (conf.) & Esc.\ (cost) & F1 (cost) \\",
        r"\midrule",
    ]
    sub = [r for r in rows if abs(r["alpha"] - 0.05) < 1e-6]
    for r in sub:
        lines.append(
            f"{r['fold']} & {r['q_hat']:.3f} & {r['conformal_escalation_rate']:.2f} & "
            f"{r['edge_fn_rate']:.4f} & {r['conformal_f1']:.3f} & "
            f"{r['cost_gate_escalation_rate']:.2f} & {r['cost_gate_f1']:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write("tab_conformal_gate.tex", "\n".join(lines))


def table_astar_robustness():
    paths = sorted((ROOT / "results/astar").glob("robustness_fold*.json"))
    if not paths:
        write("tab_robustness.tex", "% robustness pending\n")
        return
    by_cond: dict[str, list[float]] = {}
    for p in paths:
        for r in json.loads(p.read_text())["rows"]:
            by_cond.setdefault(r["condition"], []).append(r["f1"])
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Tier-1 routing robustness (cost gate): mean F1 across folds under IMU perturbations.}",
        r"\label{tab:robustness}",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Condition & Mean F1 \\",
        r"\midrule",
    ]
    for cond, vals in sorted(by_cond.items()):
        cond_tex = cond.replace("_", r"\_")
        lines.append(f"{cond_tex} & {np.mean(vals):.3f} $\\pm$ {np.std(vals):.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write("tab_robustness.tex", "\n".join(lines))


def table_astar_knn_k():
    p = ROOT / "results/astar/knn_k_ablation_fold0.json"
    if not p.exists():
        write("tab_knn_k.tex", "% kNN k ablation pending\n")
        return
    rows = json.loads(p.read_text())["rows"]
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{$k$-NN retrieval ablation ($k \in \{1,5,10\}$, fold~0, heuristic LLM).}",
        r"\label{tab:knnk}",
        r"\begin{tabular}{cccc}",
        r"\toprule",
        r"$k$ & F1 & Recall & Escalation \\",
        r"\midrule",
    ]
    for r in sorted(rows, key=lambda x: x["k"]):
        lines.append(f"{r['k']} & {r['f1']:.3f} & {r['recall']:.3f} & {r['escalation_rate']:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write("tab_knn_k.tex", "\n".join(lines))


def table_astar_fallback():
    paths = sorted((ROOT / "results/astar").glob("partition_fallback_fold*.json"))
    if not paths:
        write("tab_partition_fallback.tex", "% partition fallback pending\n")
        return
    f1s, recs = [], []
    for p in paths:
        d = json.loads(p.read_text())
        f1s.append(d["f1"])
        recs.append(d["recall"])
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Offline partition fallback: single Bayes threshold $\tau^\ast{=}1/(\beta{+}1)$, no escalation ($\beta{=}10$).}",
        r"\label{tab:partitionfallback}",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Metric & Mean $\\pm$ std (5 folds) \\",
        r"\midrule",
        f"F1 & {np.mean(f1s):.3f} $\\pm$ {np.std(f1s):.3f} \\\\",
        f"Recall & {np.mean(recs):.3f} $\\pm$ {np.std(recs):.3f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    write("tab_partition_fallback.tex", "\n".join(lines))


def table_astar_pareto():
    p = ROOT / "results/astar/duty_cycle_pareto.json"
    if not p.exists():
        write("tab_pareto.tex", "% pareto pending\n")
        return
    rows = json.loads(p.read_text())["rows"]
    # Pick representative operating points: min cost, 25/50/75% escalation quantiles
    esc = np.array([r["escalation_rate"] for r in rows])
    cost = np.array([r["expected_response_cost"] for r in rows])
    q_targets = [0.25, 0.50, 0.75]
    picks = []
    for q in q_targets:
        target = float(np.quantile(esc, q))
        idx = int(np.argmin(np.abs(esc - target)))
        picks.append(rows[idx])
    min_idx = int(np.argmin(cost))
    picks.insert(0, rows[min_idx])
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Duty-cycle Pareto operating points (tier-1 routing, pooled folds): clinical cost vs.\ escalation.}",
        r"\label{tab:pareto}",
        r"\begin{tabular}{ccccc}",
        r"\toprule",
        r"Point & $\tau_{\mathrm{low}}$ & $\tau_{\mathrm{high}}$ & Esc.\ rate & Cost \\",
        r"\midrule",
    ]
    labels = ["Min cost", "Q25 esc.", "Q50 esc.", "Q75 esc."]
    for lab, r in zip(labels, picks):
        lines.append(
            f"{lab} & {r['tau_low']:.2f} & {r['tau_high']:.2f} & "
            f"{r['escalation_rate']:.2f} & {r['expected_response_cost']:.0f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write("tab_pareto.tex", "\n".join(lines))


def main():
    table_positioning()
    table_per_fold()
    table_mean_std()
    table_ambiguous_sisfall()
    table_ablation()
    table_llm()
    table_backbone()
    table_latency()
    table_stats()
    table_seed_sweep()
    table_astar_conformal()
    table_astar_robustness()
    table_astar_knn_k()
    table_astar_fallback()
    table_astar_pareto()


if __name__ == "__main__":
    main()
