#!/usr/bin/env python3
"""Evaluate the full agentic pipeline on a held-out SisFall fold."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_fall.agents import ActionAgent, AgenticPipeline, KNNMemory, LLMReasoner
from agentic_fall.data.sisfall import SisFallDataset
from agentic_fall.eval.metrics import agentic_metrics
from agentic_fall.eval.protocol import (
    build_gate,
    make_fold_split,
    subsample_indices,
    text_embedder,
    verify_dataset_near_falls,
)
from agentic_fall.models import build_model, kwargs_for_model
from agentic_fall.utils.config import load_config
from agentic_fall.utils.io import ensure_dir, save_json
from agentic_fall.utils.seed import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-config", default="configs/paper_protocol.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--memory", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--backend", default=None)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pcfg = load_config(ROOT / args.paper_config)
    tcfg = load_config(ROOT / pcfg["tier1_config"])
    acfg = load_config(ROOT / pcfg["agentic_config"])
    bcfg = load_config(ROOT / pcfg.get("backbones_config", "configs/backbones.yaml"))
    set_seed(int(pcfg["seed"]))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_name = args.model or pcfg["agentic"].get("primary_backbone", "cnn_lstm_attn")

    npz = ROOT / tcfg["processed_dir"] / f"windows_w{tcfg['window']}_h{tcfg['hop']}.npz"
    ds = SisFallDataset(npz, channels=int(tcfg["channels"]))
    verify_dataset_near_falls(ds, pcfg["eval"]["required_activities"])
    split = make_fold_split(
        ds,
        fold=args.fold,
        n_folds=int(pcfg["splits"]["n_folds"]),
        seed=int(pcfg["seed"]),
        val_fraction=float(pcfg["splits"]["val_fraction"]),
    )
    ev = pcfg["eval"]
    tr = pcfg["train"]
    max_samples = args.max_samples if args.max_samples is not None else ev.get("agentic_max_samples")
    min_nf = int(ev.get("min_near_fall_per_code", 0) or 0)
    min_per = {c: min_nf for c in ev["required_activities"]} if min_nf else None
    # Match ablation calibration subsample (seed / max_train) so gate taus align.
    train_idx = subsample_indices(split.train_idx, ds.y, ds.activities, tr.get("max_train"), pcfg["seed"])
    test_idx = subsample_indices(
        split.test_idx, ds.y, ds.activities, max_samples, pcfg["seed"] + 2, min_per_activity=min_per
    )

    if model_name == "cnn_lstm_attn":
        model = build_model(
            "cnn_lstm_attn",
            in_channels=int(tcfg["model"]["in_channels"]),
            num_classes=2,
            conv_channels=int(tcfg["model"]["conv_channels"]),
            branch_channels=int(tcfg["model"]["branch_channels"]),
            lstm_hidden=tcfg["model"]["lstm_hidden"],
            attn_heads=int(tcfg["model"]["attn_heads"]),
        )
    else:
        zoo_kw = kwargs_for_model(model_name, bcfg)
        model = build_model(model_name, in_channels=int(tcfg["channels"]), num_classes=2, **zoo_kw)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device).eval()

    gate = build_gate(
        acfg,
        model=model,
        ds=ds,
        cal_idx=train_idx,
        device=device,
        calibrate=bool(ev.get("calibrate_gate", True)),
        max_cal=int(ev.get("max_cal", 2000)),
    )
    print(f"Calibrated gate: tau_low={gate.tau_low:.3f}, tau_high={gate.tau_high:.3f}")

    memory = KNNMemory(k=int(acfg["retrieval"]["k"]))
    memory.load(args.memory)

    # Primary end-to-end stack matches ablation mode gate_knn_llm when backend=ollama.
    # heuristic is only for cheap smoke / reasoner ablation (see llm_compare).
    backend = args.backend or pcfg.get("agentic", {}).get("backend") or acfg["llm"]["backend"]
    if backend == "heuristic":
        print(
            "WARNING: backend=heuristic under-reports the full agentic stack; "
            "use --backend ollama for gate_knn_llm-comparable metrics."
        )
    reasoner = LLMReasoner(
        backend=backend,
        base_url=acfg["llm"]["base_url"],
        model=acfg["llm"]["model"],
        temperature=float(acfg["llm"]["temperature"]),
        timeout_s=float(acfg["llm"]["timeout_s"]),
        max_retries=int(acfg["llm"]["max_retries"]),
        fallback_heuristic=bool(acfg["llm"]["fallback_heuristic"]),
    )
    action_agent = ActionAgent(
        impact_severe_g=float(acfg["action"]["impact_severe_g"]),
        impact_moderate_g=float(acfg["action"]["impact_moderate_g"]),
        stillness_severe_s=float(acfg["action"]["stillness_severe_s"]),
        stillness_moderate_s=float(acfg["action"]["stillness_moderate_s"]),
        simulate=bool(acfg["action"]["simulate"]),
    )
    pipe = AgenticPipeline(
        model=model,
        gate=gate,
        memory=memory,
        reasoner=reasoner,
        action_agent=action_agent,
        device=device,
        sample_rate_hz=float(acfg["evidence"]["sample_rate_hz"]),
        freefall_g_threshold=float(acfg["evidence"]["freefall_g_threshold"]),
        stillness_var_threshold=float(acfg["evidence"]["stillness_var_threshold"]),
        text_embed_fn=text_embedder(acfg["retrieval"]["embedder"]),
    )

    y_true, y_pred, escalated, activities, lats, pfalls = [], [], [], [], [], []
    logs = []
    loader = DataLoader(Subset(ds, test_idx), batch_size=1, shuffle=False)
    for batch in tqdm(loader, desc="agentic_eval"):
        x = batch["x"][0]
        win = x.numpy().T
        res = pipe.run(x, win, activity=str(batch["activity"][0]))
        yt = int(batch["y"][0])
        yp = 1 if res.prediction == "fall" else 0
        y_true.append(yt)
        y_pred.append(yp)
        escalated.append(res.escalated)
        activities.append(str(batch["activity"][0]))
        lats.append(res.latency_ms)
        pfalls.append(res.p_fall)
        if len(logs) < 50:
            logs.append(res.to_dict())

    metrics = agentic_metrics(
        y_true,
        y_pred,
        escalated,
        activities=activities,
        ambiguous_codes=list(acfg["pipeline"]["ambiguous_adl_codes"]),
        latencies_ms=lats,
        p_falls=pfalls,
        cost_fn=float(acfg["gate"]["cost_fn"]),
        cost_fp=float(acfg["gate"]["cost_fp"]),
    )
    metrics["tau_low"] = gate.tau_low
    metrics["tau_high"] = gate.tau_high
    metrics["backend"] = backend
    metrics["n"] = len(y_true)
    metrics["model"] = model_name
    metrics["comparable_ablation_mode"] = "gate_knn_llm" if backend == "ollama" else "gate_knn"
    metrics["name"] = f"agentic_{backend}"

    out_dir = ensure_dir(ROOT / f"{pcfg['paths']['results_dir']}/fold{args.fold}")
    out_path = Path(args.out) if args.out else out_dir / f"agentic_fold{args.fold}.json"
    payload = {
        "metrics": metrics,
        "samples": logs,
        "notes": (
            "Primary paper ablation ladder is in ablation_fold*.json. "
            "This end-to-end run uses agentic_max_samples and the selected LLM backend; "
            "ollama ≈ gate_knn_llm, heuristic ≈ gate_knn."
        ),
    }
    save_json(payload, out_path)
    save_json(payload, ROOT / pcfg["paths"]["results_dir"] / f"agentic_fold{args.fold}.json")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
