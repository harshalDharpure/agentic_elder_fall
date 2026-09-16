#!/usr/bin/env python3
"""Multi-seed robustness for primary backbone + tier1 vs stack on one fold.

Trains cnn_lstm_attn for each seed, evaluates detector metrics, and (optionally)
runs a fast heuristic-only agentic stack comparison. Full Ollama multi-seed is
expensive; use --with-ollama to enable it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_fall.agents import (
    ActionAgent,
    AgenticPipeline,
    ConfidenceGate,
    KNNMemory,
    LLMReasoner,
)
from agentic_fall.data.sisfall import SisFallDataset
from agentic_fall.eval.metrics import agentic_metrics
from agentic_fall.eval.protocol import (
    build_gate,
    make_fold_split,
    subsample_indices,
    text_embedder,
    verify_dataset_near_falls,
)
from agentic_fall.eval.train_loop import class_pos_weight, evaluate, fit_model
from agentic_fall.models import build_model
from agentic_fall.utils.config import load_config
from agentic_fall.utils.io import ensure_dir, save_json
from agentic_fall.utils.seed import set_seed


def run_stack(model, ds, test_idx, memory_path, device, acfg, gate, backend: str):
    memory = KNNMemory(k=int(acfg["retrieval"]["k"]))
    if Path(memory_path).exists():
        memory.load(memory_path)
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
    y_true, y_pred, esc, acts, lats, ps = [], [], [], [], [], []
    for batch in DataLoader(Subset(ds, test_idx), batch_size=1, shuffle=False):
        x = batch["x"][0]
        res = pipe.run(x, x.numpy().T, activity=str(batch["activity"][0]))
        y_true.append(int(batch["y"][0]))
        y_pred.append(1 if res.prediction == "fall" else 0)
        esc.append(res.escalated)
        acts.append(str(batch["activity"][0]))
        lats.append(res.latency_ms)
        ps.append(res.p_fall)
    return agentic_metrics(
        y_true,
        y_pred,
        esc,
        activities=acts,
        ambiguous_codes=list(acfg["pipeline"]["ambiguous_adl_codes"]),
        latencies_ms=lats,
        p_falls=ps,
        cost_fn=float(acfg["gate"]["cost_fn"]),
        cost_fp=float(acfg["gate"]["cost_fp"]),
    )


def _save_summary(out_dir, fold, seeds, rows, backend: str):
    f1s = [r["tier1_f1"] for r in rows]
    sf1 = [r["stack_f1"] for r in rows]
    summary = {
        "fold": fold,
        "seeds": seeds,
        "n_seeds": len(seeds),
        "tier1_f1_mean": float(np.mean(f1s)),
        "tier1_f1_std": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
        "stack_f1_mean": float(np.mean(sf1)),
        "stack_f1_std": float(np.std(sf1, ddof=1)) if len(sf1) > 1 else 0.0,
        "rows": rows,
        "note": (
            f"Stack backend: {backend}. "
            "Wilcoxon over 5 folds remains coarse; seed std quantifies training variability."
        ),
    }
    save_json(summary, out_dir / f"seed_sweep_fold{fold}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-config", default="configs/paper_protocol.yaml")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seeds", default="42,123,7")
    ap.add_argument("--max-test", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--device", default=None)
    ap.add_argument("--with-ollama", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    pcfg = load_config(ROOT / args.paper_config)
    tcfg = load_config(ROOT / pcfg["tier1_config"])
    acfg = load_config(ROOT / pcfg["agentic_config"])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = ensure_dir(ROOT / "results" / "seed_sweep")

    npz = ROOT / tcfg["processed_dir"] / f"windows_w{tcfg['window']}_h{tcfg['hop']}.npz"
    ds = SisFallDataset(npz, channels=int(tcfg["channels"]))
    verify_dataset_near_falls(ds, pcfg["eval"]["required_activities"])

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    rows = []
    for seed in seeds:
        set_seed(seed)
        split = make_fold_split(
            ds,
            fold=args.fold,
            n_folds=int(pcfg["splits"]["n_folds"]),
            seed=seed,
            val_fraction=float(pcfg["splits"]["val_fraction"]),
        )
        min_nf = int(pcfg["eval"].get("min_near_fall_per_code", 0) or 0)
        min_per = {c: min_nf for c in pcfg["eval"]["required_activities"]} if min_nf else None
        train_idx = subsample_indices(split.train_idx, ds.y, ds.activities, pcfg["train"].get("max_train"), seed)
        val_idx = subsample_indices(split.val_idx, ds.y, ds.activities, pcfg["train"].get("max_val"), seed + 1)
        test_idx = subsample_indices(
            split.test_idx, ds.y, ds.activities, args.max_test, seed + 2, min_per_activity=min_per
        )
        ckpt = out_dir / f"cnn_lstm_attn_fold{args.fold}_seed{seed}.pt"
        model = build_model(
            "cnn_lstm_attn",
            in_channels=int(tcfg["model"]["in_channels"]),
            num_classes=2,
            conv_channels=int(tcfg["model"]["conv_channels"]),
            branch_channels=int(tcfg["model"]["branch_channels"]),
            lstm_hidden=tcfg["model"]["lstm_hidden"],
            attn_heads=int(tcfg["model"]["attn_heads"]),
        )
        if not (args.skip_train and ckpt.exists()):
            cw = class_pos_weight([int(ds.y[i]) for i in train_idx])
            fit_model(
                model,
                DataLoader(Subset(ds, train_idx), batch_size=256, shuffle=True, num_workers=2),
                DataLoader(Subset(ds, val_idx), batch_size=256, shuffle=False, num_workers=2),
                device=device,
                epochs=args.epochs,
                patience=max(5, args.epochs // 4),
                mixup_alpha=0.1,
                checkpoint_path=str(ckpt),
                class_weight=cw,
            )
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        model.to(device)
        det = evaluate(model, DataLoader(Subset(ds, test_idx), batch_size=256, shuffle=False), device)
        det.pop("probs", None)
        det.pop("y_true", None)

        # tier1_only via collapsed gate
        gate = build_gate(acfg, model=model, ds=ds, cal_idx=train_idx, device=device, calibrate=True, max_cal=2000)
        mem = ROOT / pcfg["paths"]["memory_dir"] / f"sisfall_fold{args.fold}.json"
        # For non-42 seeds, reuse fold memory as approximate case bank (text retrieval still helps).
        backend = "ollama" if args.with_ollama else "heuristic"
        stack = run_stack(model, ds, test_idx, mem, device, acfg, gate, backend=backend)
        row = {
            "seed": seed,
            "fold": args.fold,
            "tier1_f1": det["f1"],
            "tier1_recall": det["recall"],
            "stack_backend": backend,
            "stack_f1": stack["f1"],
            "stack_recall": stack["recall"],
            "stack_cost": stack["expected_response_cost"],
            "stack_cost_per_1000": stack["expected_response_cost"] / max(len(test_idx), 1) * 1000.0,
            "n": len(test_idx),
            "tau_low": gate.tau_low,
            "tau_high": gate.tau_high,
        }
        rows.append(row)
        print(row)
        _save_summary(out_dir, args.fold, seeds, rows, backend)

    print("Wrote", out_dir / f"seed_sweep_fold{args.fold}.json")


if __name__ == "__main__":
    main()
