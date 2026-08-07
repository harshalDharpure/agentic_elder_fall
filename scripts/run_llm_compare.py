#!/usr/bin/env python3
"""Compare LLM backends on the same ambiguous case IDs (paired)."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_fall.agents import ActionAgent, AgenticPipeline, ConfidenceGate, KNNMemory, LLMReasoner
from agentic_fall.data.sisfall import SisFallDataset
from agentic_fall.eval.metrics import agentic_metrics, metrics_to_latex
from agentic_fall.eval.protocol import (
    build_gate,
    make_fold_split,
    subsample_indices,
    text_embedder,
    verify_dataset_near_falls,
)
from agentic_fall.models import build_model
from agentic_fall.utils.config import load_config
from agentic_fall.utils.io import ensure_dir, save_json
from agentic_fall.utils.seed import set_seed


def select_ambiguous_indices(model, ds, indices, device, tau_low, tau_high, max_n, seed):
    model.eval()
    amb = []
    loader = DataLoader(Subset(ds, indices), batch_size=64, shuffle=False)
    offset = 0
    with torch.no_grad():
        for batch in loader:
            logits, _ = model(batch["x"].to(device))
            p = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            for j, pj in enumerate(p):
                if tau_low < float(pj) < tau_high:
                    amb.append(indices[offset + j])
            offset += batch["x"].size(0)
    if not amb:
        scores = []
        offset = 0
        with torch.no_grad():
            for batch in DataLoader(Subset(ds, indices), batch_size=64, shuffle=False):
                logits, _ = model(batch["x"].to(device))
                p = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                for j, pj in enumerate(p):
                    scores.append((abs(float(pj) - 0.5), indices[offset + j]))
                offset += batch["x"].size(0)
        scores.sort()
        amb = [i for _, i in scores[:max_n]]
    rng = np.random.default_rng(seed)
    if len(amb) > max_n:
        amb = list(rng.choice(amb, size=max_n, replace=False))
    return amb


def eval_backend(pipe, ds, indices, ambiguous_codes):
    y_true, y_pred, esc, acts, lats, ps, preds = [], [], [], [], [], [], []
    case_ids = []
    for batch in tqdm(DataLoader(Subset(ds, indices), batch_size=1), desc=str(pipe.reasoner.model)):
        x = batch["x"][0]
        res = pipe.run(x, x.numpy().T, activity=str(batch["activity"][0]), force_escalate=True)
        y_true.append(int(batch["y"][0]))
        y_pred.append(1 if res.prediction == "fall" else 0)
        esc.append(True)
        acts.append(str(batch["activity"][0]))
        lats.append(res.latency_ms)
        ps.append(res.p_fall)
        preds.append(res.prediction)
        case_ids.append(int(batch["index"][0]))
    m = agentic_metrics(
        y_true,
        y_pred,
        esc,
        activities=acts,
        ambiguous_codes=ambiguous_codes,
        latencies_ms=lats,
        p_falls=ps,
    )
    m["n"] = len(y_true)
    m["backend"] = pipe.reasoner.backend
    m["model_name"] = getattr(pipe.reasoner, "model", pipe.reasoner.backend)
    return m, preds, case_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-config", default="configs/paper_protocol.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--memory", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-cases", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--backends", nargs="+", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    pcfg = load_config(ROOT / args.paper_config)
    tcfg = load_config(ROOT / pcfg["tier1_config"])
    acfg = load_config(ROOT / pcfg["agentic_config"])
    set_seed(int(pcfg["seed"]))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_name = args.model or pcfg["agentic"].get("primary_backbone", "cnn_lstm_attn")
    backends = args.backends or list(pcfg["agentic"]["llm_backends"])
    max_cases = args.max_cases if args.max_cases is not None else int(pcfg["eval"]["llm_max_cases"])

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
    test_idx = subsample_indices(
        split.test_idx, ds.y, ds.activities, pcfg["eval"].get("max_test"), pcfg["seed"] + 2
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
        model = build_model(model_name, in_channels=int(tcfg["channels"]), num_classes=2)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device)

    gate = build_gate(
        acfg,
        model=model,
        ds=ds,
        cal_idx=split.train_idx,
        device=device,
        calibrate=True,
        max_cal=int(pcfg["eval"].get("max_cal", 2000)),
    )
    amb_idx = select_ambiguous_indices(
        model, ds, test_idx, device, gate.tau_low, gate.tau_high, max_cases, int(pcfg["seed"])
    )
    print(f"Ambiguous cases selected: {len(amb_idx)} (tau_low={gate.tau_low:.3f}, tau_high={gate.tau_high:.3f})")

    memory = KNNMemory(k=int(acfg["retrieval"]["k"]))
    memory.load(args.memory)
    embed = text_embedder(acfg["retrieval"]["embedder"])
    codes = list(acfg["pipeline"]["ambiguous_adl_codes"])

    rows = []
    pred_matrix = {}
    case_ids = None
    for spec in backends:
        if spec == "heuristic":
            backend, mname = "heuristic", "heuristic"
        else:
            backend, mname = "ollama", spec
        reasoner = LLMReasoner(
            backend=backend,
            base_url=acfg["llm"]["base_url"],
            model=mname,
            temperature=float(acfg["llm"]["temperature"]),
            timeout_s=float(acfg["llm"]["timeout_s"]),
            max_retries=int(acfg["llm"]["max_retries"]),
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
        metrics, preds, cids = eval_backend(pipe, ds, amb_idx, codes)
        metrics["name"] = mname
        rows.append(metrics)
        pred_matrix[mname] = preds
        case_ids = cids
        print(
            json.dumps(
                {
                    k: metrics[k]
                    for k in ("name", "f1", "expected_response_cost", "latency_mean_ms", "n", "n_near_fall")
                    if k in metrics
                },
                indent=2,
            )
        )

    if "heuristic" in pred_matrix:
        base = pred_matrix["heuristic"]
        for name, preds in pred_matrix.items():
            if name == "heuristic":
                continue
            agree = sum(a == b for a, b in zip(base, preds)) / max(1, len(base))
            for r in rows:
                if r["name"] == name:
                    r["agreement_with_heuristic"] = agree

    out_dir = ensure_dir(ROOT / (args.out_dir or f"{pcfg['paths']['results_dir']}/fold{args.fold}"))
    csv_path = out_dir / f"llm_compare_fold{args.fold}.csv"
    keys = sorted({k for r in rows for k in r})
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    (out_dir / f"llm_compare_fold{args.fold}.tex").write_text(
        metrics_to_latex(rows, caption="LLM backend comparison on ambiguous cases")
    )
    payload = {"rows": rows, "case_ids": case_ids}
    save_json(payload, out_dir / f"llm_compare_fold{args.fold}.json")
    save_json(rows, ROOT / pcfg["paths"]["results_dir"] / f"llm_compare_fold{args.fold}.json")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
