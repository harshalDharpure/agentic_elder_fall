#!/usr/bin/env python3
"""Run A* experiment suite (Track B science) before paper build.

Experiments:
  1. Split conformal gate audit (all folds, alphas)
  2. Conformal vs cost gate comparison (tier1 routing metrics)
  3. k-NN k ablation {1,5,10}
  4. Sensor orientation ±15° robustness (tier1)
  5. Sampling rate jitter 50-200Hz (tier1)
  6. Partition fallback (offline Bayes threshold, no escalation)
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_fall.agents import ActionAgent, AgenticPipeline, ConfidenceGate, KNNMemory, LLMReasoner
from agentic_fall.data.sisfall import SisFallDataset
from agentic_fall.eval.conformal_gate import ConformalGate
from agentic_fall.eval.metrics import agentic_metrics, binary_metrics
from agentic_fall.eval.protocol import (
    build_gate,
    collect_probs,
    make_fold_split,
    subsample_indices,
    text_embedder,
    verify_dataset_near_falls,
)
from agentic_fall.eval.robustness import resample_window, rotate_window
from agentic_fall.eval.train_loop import evaluate
from agentic_fall.models import build_model
from agentic_fall.utils.config import load_config
from agentic_fall.utils.io import ensure_dir, save_json
from agentic_fall.utils.seed import set_seed


def load_fold_context(fold: int, max_test: int, device: torch.device):
    pcfg = load_config(ROOT / "configs/paper_protocol.yaml")
    tcfg = load_config(ROOT / pcfg["tier1_config"])
    acfg = load_config(ROOT / pcfg["agentic_config"])
    set_seed(int(pcfg["seed"]))
    npz = ROOT / tcfg["processed_dir"] / f"windows_w{tcfg['window']}_h{tcfg['hop']}.npz"
    ds = SisFallDataset(npz, channels=int(tcfg["channels"]))
    verify_dataset_near_falls(ds, pcfg["eval"]["required_activities"])
    split = make_fold_split(
        ds,
        fold=fold,
        n_folds=int(pcfg["splits"]["n_folds"]),
        seed=int(pcfg["seed"]),
        val_fraction=float(pcfg["splits"]["val_fraction"]),
    )
    min_nf = int(pcfg["eval"].get("min_near_fall_per_code", 0) or 0)
    min_per = {c: min_nf for c in pcfg["eval"]["required_activities"]} if min_nf else None
    test_idx = subsample_indices(
        split.test_idx, ds.y, ds.activities, max_test, int(pcfg["seed"]) + 2, min_per_activity=min_per
    )
    cal_idx = list(split.val_idx)[:2000]
    ckpt = ROOT / pcfg["paths"]["checkpoint_dir"] / f"cnn_lstm_attn_fold{fold}.pt"
    model = build_model(
        "cnn_lstm_attn",
        in_channels=int(tcfg["model"]["in_channels"]),
        num_classes=2,
        conv_channels=int(tcfg["model"]["conv_channels"]),
        branch_channels=int(tcfg["model"]["branch_channels"]),
        lstm_hidden=tcfg["model"]["lstm_hidden"],
        attn_heads=int(tcfg["model"]["attn_heads"]),
    )
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    return pcfg, tcfg, acfg, ds, test_idx, cal_idx, model, device


def route_metrics(y_true, routes, preds, esc, cost_fn=10.0, cost_fp=1.0):
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(preds, dtype=int)
    m = binary_metrics(yt, yp)
    m["escalation_rate"] = float(np.mean(esc))
    m["expected_response_cost"] = float(cost_fn * m["fn"] + cost_fp * m["fp"])
    m["n"] = int(len(yt))
    return m


def eval_gate_routing(model, ds, indices, gate, device, perturb=None):
    y_true, preds, routes, esc, ps = [], [], [], [], []
    loader = DataLoader(Subset(ds, indices), batch_size=1, shuffle=False)
    for batch in loader:
        x = batch["x"][0]
        w = x.numpy().T
        if perturb is not None:
            w = perturb(w)
            x = torch.from_numpy(w.T).float()
        with torch.no_grad():
            logits, _ = model(x.unsqueeze(0).to(device))
            p = torch.softmax(logits, dim=-1)[0, 1].item()
        gd = gate.decide(p)
        routes.append(gd.route)
        if gd.route == "ambiguous":
            esc.append(True)
            yp = 1 if p >= 0.5 else 0
        elif gd.route == "fall":
            esc.append(False)
            yp = 1
        else:
            esc.append(False)
            yp = 0
        y_true.append(int(batch["y"][0]))
        preds.append(yp)
        ps.append(p)
    return y_true, preds, routes, esc, ps


def experiment_conformal(folds, alphas, max_test, device, out_dir):
    rows = []
    for fold in folds:
        _, _, acfg, ds, test_idx, cal_idx, model, device = load_fold_context(fold, max_test, device)
        cal_loader = DataLoader(Subset(ds, cal_idx), batch_size=64, shuffle=False)
        p_cal, y_cal = collect_probs(model, cal_loader, device)
        test_loader = DataLoader(Subset(ds, test_idx), batch_size=64, shuffle=False)
        p_test, y_test = collect_probs(model, test_loader, device)
        cost_gate = build_gate(acfg, model=model, ds=ds, cal_idx=cal_idx, device=device, calibrate=True)
        y_cost, pred_cost, _, esc_cost, _ = eval_gate_routing(model, ds, test_idx, cost_gate, device)
        cost_m = route_metrics(y_cost, [], pred_cost, esc_cost)
        for alpha in alphas:
            cg = ConformalGate.calibrate(p_cal, y_cal, alpha=alpha)
            audit = cg.audit_edge_fn(p_test, y_test)
            y_c, pred_c, _, esc_c, _ = eval_gate_routing(model, ds, test_idx, cg, device)
            m = route_metrics(y_c, [], pred_c, esc_c)
            row = {
                "fold": fold,
                "alpha": alpha,
                "q_hat": cg.q_hat,
                "conformal_f1": m["f1"],
                "conformal_recall": m["recall"],
                "conformal_cost": m["expected_response_cost"],
                "conformal_escalation_rate": m["escalation_rate"],
                "edge_fn_rate": audit.edge_fn_rate,
                "edge_n": audit.edge_n,
                "set_size_hist": audit.set_size_hist,
                "cost_gate_f1": cost_m["f1"],
                "cost_gate_escalation_rate": cost_m["escalation_rate"],
            }
            rows.append(row)
            print(
                f"fold{fold} alpha={alpha} q={cg.q_hat:.3f} "
                f"esc={m['escalation_rate']:.2f} edge_FN={audit.edge_fn_rate:.4f} F1={m['f1']:.3f}"
            )
    save_json({"rows": rows}, out_dir / "conformal_gate_audit.json")
    return rows


def experiment_knn_k(fold, ks, max_test, device, out_dir, use_ollama=False):
    pcfg, _, acfg, ds, test_idx, cal_idx, model, device = load_fold_context(fold, max_test, device)
    gate = build_gate(acfg, model=model, ds=ds, cal_idx=cal_idx, device=device, calibrate=True)
    mem_path = ROOT / pcfg["paths"]["memory_dir"] / f"sisfall_fold{fold}.json"
    backend = "ollama" if use_ollama else "heuristic"
    rows = []
    for k in ks:
        memory = KNNMemory(k=k)
        if mem_path.exists():
            memory.load(mem_path)
        reasoner = LLMReasoner(
            backend=backend,
            base_url=acfg["llm"]["base_url"],
            model=acfg["llm"]["model"],
            fallback_heuristic=True,
        )
        pipe = AgenticPipeline(
            model=model,
            gate=gate,
            memory=memory,
            reasoner=reasoner,
            action_agent=ActionAgent(),
            device=device,
            sample_rate_hz=float(acfg["evidence"]["sample_rate_hz"]),
            text_embed_fn=text_embedder(acfg["retrieval"]["embedder"]),
        )
        y_true, y_pred, esc = [], [], []
        for batch in DataLoader(Subset(ds, test_idx), batch_size=1, shuffle=False):
            x = batch["x"][0]
            res = pipe.run(x, x.numpy().T, activity=str(batch["activity"][0]))
            y_true.append(int(batch["y"][0]))
            y_pred.append(1 if res.prediction == "fall" else 0)
            esc.append(res.escalated)
        m = agentic_metrics(
            y_true,
            y_pred,
            esc,
            cost_fn=float(acfg["gate"]["cost_fn"]),
            cost_fp=float(acfg["gate"]["cost_fp"]),
        )
        m["k"] = k
        m["backend"] = backend
        rows.append(m)
        print(f"k={k} backend={backend} F1={m['f1']:.3f} esc={m['escalation_rate']:.2f}")
    save_json({"fold": fold, "rows": rows}, out_dir / f"knn_k_ablation_fold{fold}.json")
    return rows


def experiment_robustness(fold, max_test, device, out_dir):
    _, tcfg, acfg, ds, test_idx, cal_idx, model, device = load_fold_context(fold, max_test, device)
    gate = build_gate(acfg, model=model, ds=ds, cal_idx=cal_idx, device=device, calibrate=True)
    src_hz = float(acfg["evidence"]["sample_rate_hz"])
    conditions = [("clean", lambda w: w)]
    for deg in (10, 15):
        conditions.append(
            (f"rotate_{deg}deg", lambda w, d=deg: rotate_window(w, d, d * 0.5, d * 0.3))
        )
    for hz in (50, 100, 200):
        if hz != src_hz:
            conditions.append(
                (f"resample_{hz}hz", lambda w, h=hz: resample_window(w, src_hz, h))
            )
    rows = []
    for name, fn in conditions:
        y_true, preds, _, esc, _ = eval_gate_routing(model, ds, test_idx, gate, device, perturb=fn)
        m = route_metrics(y_true, [], preds, esc)
        m["condition"] = name
        rows.append(m)
        print(f"robustness {name}: F1={m['f1']:.3f} recall={m['recall']:.3f}")
    save_json({"fold": fold, "rows": rows}, out_dir / f"robustness_fold{fold}.json")
    return rows


def experiment_pareto(folds, max_test, device, out_dir, cost_fn=10.0, cost_fp=1.0):
    """Risk (clinical cost) vs escalation Pareto over gate band width."""
    rows = []
    for fold in folds:
        _, _, acfg, ds, test_idx, cal_idx, model, device = load_fold_context(fold, max_test, device)
        cal_loader = DataLoader(Subset(ds, cal_idx), batch_size=64, shuffle=False)
        p_cal, y_cal = collect_probs(model, cal_loader, device)
        test_loader = DataLoader(Subset(ds, test_idx), batch_size=64, shuffle=False)
        p_test, y_test = collect_probs(model, test_loader, device)
        ps = np.asarray(p_test)
        ys = np.asarray(y_test)
        for lo in np.linspace(0.05, 0.40, 8):
            for hi in np.linspace(0.60, 0.95, 8):
                if lo >= hi:
                    continue
                gate = ConfidenceGate(float(lo), float(hi))
                esc, pred = [], []
                for p in ps:
                    gd = gate.decide(float(p))
                    if gd.route == "ambiguous":
                        esc.append(True)
                        pred.append(1 if p >= 0.5 else 0)
                    elif gd.route == "fall":
                        esc.append(False)
                        pred.append(1)
                    else:
                        esc.append(False)
                        pred.append(0)
                m = route_metrics(ys, [], pred, esc, cost_fn=cost_fn, cost_fp=cost_fp)
                rows.append({
                    "fold": fold,
                    "tau_low": float(lo),
                    "tau_high": float(hi),
                    "escalation_rate": m["escalation_rate"],
                    "expected_response_cost": m["expected_response_cost"],
                    "f1": m["f1"],
                    "recall": m["recall"],
                })
        print(f"pareto fold{fold}: {sum(1 for r in rows if r['fold']==fold)} points")
    save_json({"rows": rows}, out_dir / "duty_cycle_pareto.json")
    return rows


def experiment_partition_fallback(fold, max_test, device, out_dir, beta=10.0):
    _, _, acfg, ds, test_idx, _, model, device = load_fold_context(fold, max_test, device)
    tau = 1.0 / (beta + 1.0)
    y_true, preds, esc = [], [], []
    for batch in DataLoader(Subset(ds, test_idx), batch_size=1, shuffle=False):
        x = batch["x"][0]
        with torch.no_grad():
            logits, _ = model(x.unsqueeze(0).to(device))
            p = torch.softmax(logits, dim=-1)[0, 1].item()
        y_true.append(int(batch["y"][0]))
        preds.append(1 if p >= tau else 0)
        esc.append(False)
    m = route_metrics(y_true, [], preds, esc)
    m["tau_star"] = tau
    m["beta"] = beta
    m["mode"] = "offline_bayes_no_escalation"
    save_json(m, out_dir / f"partition_fallback_fold{fold}.json")
    print(f"partition fallback fold{fold}: F1={m['f1']:.3f} recall={m['recall']:.3f} esc={m['escalation_rate']:.2f}")
    return m


def write_summary(out_dir):
    summary = {"experiments": []}
    for name in (
        "conformal_gate_audit.json",
        "knn_k_ablation_fold0.json",
        "robustness_fold0.json",
        "partition_fallback_fold0.json",
    ):
        p = out_dir / name
        if p.exists():
            summary["experiments"].append({"file": name, "data": json.loads(p.read_text())})
    save_json(summary, out_dir / "ASTAR_EXPERIMENT_SUMMARY.json")
    md = ["# A* Experiment Suite Summary", ""]
    if (out_dir / "conformal_gate_audit.json").exists():
        rows = json.loads((out_dir / "conformal_gate_audit.json").read_text())["rows"]
        md.append("## Conformal gate audit")
        md.append("| fold | alpha | q_hat | esc_rate | edge_FN_rate | F1 |")
        md.append("|---:|---:|---:|---:|---:|---:|")
        for r in rows:
            md.append(
                f"| {r['fold']} | {r['alpha']} | {r['q_hat']:.3f} | "
                f"{r['conformal_escalation_rate']:.2f} | {r['edge_fn_rate']:.4f} | {r['conformal_f1']:.3f} |"
            )
        md.append("")
    (out_dir / "ASTAR_EXPERIMENT_SUMMARY.md").write_text("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/astar")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--max-test", type=int, default=1000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip-conformal", action="store_true")
    ap.add_argument("--skip-knn", action="store_true")
    ap.add_argument("--skip-robustness", action="store_true")
    ap.add_argument("--skip-pareto", action="store_true")
    ap.add_argument("--skip-fallback", action="store_true")
    ap.add_argument("--knn-ollama", action="store_true")
    args = ap.parse_args()

    out_dir = ensure_dir(ROOT / args.out_dir)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    folds = [int(x) for x in args.folds.split(",") if x.strip()]

    if not args.skip_conformal:
        experiment_conformal(folds, alphas=[0.05, 0.01, 0.005], max_test=args.max_test, device=device, out_dir=out_dir)
    if not args.skip_knn:
        experiment_knn_k(0, ks=[1, 5, 10], max_test=args.max_test, device=device, out_dir=out_dir, use_ollama=args.knn_ollama)
    if not args.skip_robustness:
        for f in folds:
            experiment_robustness(f, max_test=args.max_test, device=device, out_dir=out_dir)
    if not args.skip_pareto:
        experiment_pareto(folds, max_test=args.max_test, device=device, out_dir=out_dir)
    if not args.skip_fallback:
        for f in folds:
            experiment_partition_fallback(f, max_test=args.max_test, device=device, out_dir=out_dir)

    write_summary(out_dir)
    print("Wrote", out_dir / "ASTAR_EXPERIMENT_SUMMARY.json")


if __name__ == "__main__":
    main()
