#!/usr/bin/env python3
"""Paired LLM backend compare on KFall ambiguous windows (same IDs)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_fall.agents import ActionAgent, AgenticPipeline, KNNMemory, LLMReasoner
from agentic_fall.data.kfall import KFallDataset
from agentic_fall.eval.metrics import agentic_metrics
from agentic_fall.eval.protocol import build_gate, subsample_indices, text_embedder
from agentic_fall.models import build_model
from agentic_fall.utils.config import load_config
from agentic_fall.utils.io import ensure_dir, save_json
from agentic_fall.utils.seed import set_seed


def _subject_split(ds, fold: int, n_folds: int, seed: int, val_fraction: float):
    folds = ds.fold_indices(n_folds=n_folds, seed=seed)
    train_idx, test_idx = folds[fold]
    train_subj = sorted({str(ds.subjects[i]) for i in train_idx})
    rng = np.random.default_rng(seed + fold)
    rng.shuffle(train_subj)
    n_val = max(1, int(round(len(train_subj) * val_fraction)))
    val_subj = set(train_subj[:n_val])
    tr_subj = set(train_subj[n_val:])
    tr = [i for i in train_idx if str(ds.subjects[i]) in tr_subj]
    va = [i for i in train_idx if str(ds.subjects[i]) in val_subj]
    return tr, va, test_idx


def select_ambiguous(model, ds, indices, device, tau_low, tau_high, max_n, seed):
    model.eval()
    amb = []
    offset = 0
    with torch.no_grad():
        for batch in DataLoader(Subset(ds, indices), batch_size=64, shuffle=False):
            out = model(batch["x"].to(device))
            logits = out[0] if isinstance(out, tuple) else out
            p = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            for j, pj in enumerate(p):
                if tau_low < float(pj) < tau_high:
                    amb.append(indices[offset + j])
            offset += batch["x"].size(0)
    rng = np.random.default_rng(seed)
    if len(amb) > max_n:
        amb = list(rng.choice(amb, size=max_n, replace=False))
    return amb


def eval_backend(pipe, ds, indices, ambiguous_codes):
    y_true, y_pred, esc, acts, lats, ps = [], [], [], [], [], []
    for batch in tqdm(DataLoader(Subset(ds, indices), batch_size=1), desc=str(pipe.reasoner.model)):
        x = batch["x"][0]
        res = pipe.run(x, x.numpy().T, activity=str(batch["activity"][0]), force_escalate=True)
        y_true.append(int(batch["y"][0]))
        y_pred.append(1 if res.prediction == "fall" else 0)
        esc.append(True)
        acts.append(str(batch["activity"][0]))
        lats.append(res.latency_ms)
        ps.append(res.p_fall)
    m = agentic_metrics(
        y_true, y_pred, esc, activities=acts, ambiguous_codes=ambiguous_codes, latencies_ms=lats, p_falls=ps
    )
    m["n"] = len(y_true)
    m["name"] = getattr(pipe.reasoner, "model", pipe.reasoner.backend)
    m["backend"] = pipe.reasoner.backend
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-cases", type=int, default=100)
    ap.add_argument("--backends", default="heuristic,mistral:latest,qwen2.5-coder:7b-instruct")
    args = ap.parse_args()

    kcfg = load_config(ROOT / "configs/tier1_kfall_binary.yaml")
    acfg = load_config(ROOT / "configs/agentic.yaml")
    set_seed(int(kcfg["train"]["seed"]))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    npz = ROOT / kcfg["processed_dir"] / f"windows_transition_w{kcfg['window']}_h{kcfg['hop']}.npz"
    ds = KFallDataset(npz, binary=True)
    tr, va, te = _subject_split(
        ds, args.fold, int(kcfg["splits"]["n_folds"]), int(kcfg["train"]["seed"]), float(kcfg["splits"].get("val_fraction", 0.15))
    )
    te = subsample_indices(te, ds.y, ds.activities, 2000, kcfg["train"]["seed"] + 2, min_per_activity={"D18": 40, "D19": 40})

    ckpt = ROOT / kcfg["checkpoint_dir"] / f"cnn_lstm_attn_binary_fold{args.fold}.pt"
    if not ckpt.exists():
        raise SystemExit(f"Missing {ckpt}; run binary external eval first")
    model = build_model(
        "cnn_lstm_attn",
        in_channels=int(ds.X.shape[-1]),
        num_classes=2,
        conv_channels=int(kcfg["model"]["conv_channels"]),
        branch_channels=int(kcfg["model"]["branch_channels"]),
        lstm_hidden=kcfg["model"]["lstm_hidden"],
        attn_heads=int(kcfg["model"]["attn_heads"]),
    )
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device)

    gate = build_gate(acfg, model=model, ds=ds, cal_idx=tr, device=device, calibrate=True, max_cal=2000)
    amb = select_ambiguous(model, ds, te, device, gate.tau_low, gate.tau_high, args.max_cases, kcfg["train"]["seed"])
    print(f"fold{args.fold}: {len(amb)} ambiguous cases; tau=({gate.tau_low:.3f},{gate.tau_high:.3f})")

    mem_path = ROOT / "data" / "memory" / f"kfall_fold{args.fold}.json"
    memory = KNNMemory(k=int(acfg["retrieval"]["k"]))
    if mem_path.exists():
        memory.load(mem_path)
    embed = text_embedder(acfg["retrieval"]["embedder"])
    codes = list(acfg["pipeline"].get("ambiguous_adl_codes", ["D18", "D19"]))

    rows = []
    for spec in [s.strip() for s in args.backends.split(",") if s.strip()]:
        if spec == "heuristic":
            backend, model_name = "heuristic", "heuristic"
        else:
            backend, model_name = "ollama", spec
        reasoner = LLMReasoner(
            backend=backend,
            base_url=acfg["llm"]["base_url"],
            model=model_name,
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
            text_embed_fn=embed,
        )
        m = eval_backend(pipe, ds, amb, codes)
        m["name"] = model_name
        rows.append(m)
        print(model_name, {k: m[k] for k in ("f1", "recall", "expected_response_cost") if k in m})

    out_dir = ensure_dir(ROOT / "results" / "kfall" / f"fold{args.fold}")
    payload = {"fold": args.fold, "n_cases": len(amb), "case_ids": amb, "rows": rows}
    save_json(payload, out_dir / f"kfall_llm_compare_fold{args.fold}.json")
    save_json(payload, ROOT / "results" / "kfall" / f"kfall_llm_compare_fold{args.fold}.json")
    print("Wrote", out_dir / f"kfall_llm_compare_fold{args.fold}.json")


if __name__ == "__main__":
    main()
