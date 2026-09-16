#!/usr/bin/env python3
"""External validation on KFall (binary Fall/ADL) under a locked agentic protocol.

Requires approved KFall extract at data/raw/kfall/.
If missing, writes results/kfall/ACCESS_REQUIRED.json and exits 2.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_fall.agents import (
    ActionAgent,
    AgenticPipeline,
    ConfidenceGate,
    KNNMemory,
    LLMReasoner,
)
from agentic_fall.agents.evidence import serialize_evidence
from agentic_fall.agents.knn_memory import MemoryCase
from agentic_fall.data.kfall import KFallDataset, prepare_kfall
from agentic_fall.eval.metrics import agentic_metrics
from agentic_fall.eval.protocol import build_gate, subsample_indices, text_embedder
from agentic_fall.eval.train_loop import class_pos_weight, evaluate, fit_model
from agentic_fall.features.biomechanics import extract_biomechanics
from agentic_fall.models import build_model, kwargs_for_model
from agentic_fall.utils.config import load_config
from agentic_fall.utils.io import ensure_dir, save_json
from agentic_fall.utils.seed import set_seed


def _subject_split(ds: KFallDataset, fold: int, n_folds: int, seed: int, val_fraction: float):
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


def build_kfall_memory(model, ds, train_idx, device, acfg, mem_path: Path, max_cases: int = 2000):
    memory = KNNMemory(k=int(acfg["retrieval"]["k"]), metric=acfg["retrieval"].get("metric", "cosine"))
    embed = text_embedder(acfg["retrieval"]["embedder"])
    idx = list(train_idx)
    if len(idx) > max_cases:
        rng = np.random.default_rng(0)
        idx = list(rng.choice(idx, size=max_cases, replace=False))
    loader = DataLoader(Subset(ds, idx), batch_size=32, shuffle=False)
    sr = float(acfg["evidence"]["sample_rate_hz"])
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="kfall_memory"):
            x = batch["x"].to(device)
            out = model(x)
            # cnn_lstm_attn returns (logits, emb)
            if isinstance(out, tuple):
                _, emb = out
                emb = emb.cpu().numpy()
            for i in range(x.size(0)):
                win = batch["x"][i].numpy().T
                feats = extract_biomechanics(
                    win,
                    sample_rate_hz=sr,
                    freefall_g_threshold=float(acfg["evidence"]["freefall_g_threshold"]),
                    stillness_var_threshold=float(acfg["evidence"]["stillness_var_threshold"]),
                )
                text = serialize_evidence(feats, activity=str(batch["activity"][i]))
                te = np.asarray(embed(text), dtype=np.float32)
                memory.add(
                    MemoryCase(
                        case_id=f"kfall_{int(batch['index'][i])}",
                        embedding=te.tolist(),
                        label=int(batch["y"][i]),
                        evidence_text=text,
                        features=feats.to_dict(),
                        activity=str(batch["activity"][i]),
                    )
                )
    memory.save(mem_path)
    return memory


def run_mode(model, ds, test_idx, memory_path, device, acfg, mode, gate):
    memory = KNNMemory(k=int(acfg["retrieval"]["k"]))
    if Path(memory_path).exists() and mode in ("gate_knn", "gate_llm", "gate_knn_llm"):
        memory.load(memory_path)
    use_llm = mode in ("gate_llm", "gate_knn_llm")
    use_knn = mode in ("gate_knn", "gate_knn_llm")
    backend = "ollama" if use_llm else "heuristic"
    if mode == "gate_knn":
        backend = "heuristic"
    reasoner = LLMReasoner(
        backend=backend,
        base_url=acfg["llm"]["base_url"],
        model=acfg["llm"]["model"],
        fallback_heuristic=True,
    )
    local_gate = gate
    if mode == "tier1_only":
        local_gate = ConfidenceGate(0.5, 0.5 + 1e-6)
    pipe = AgenticPipeline(
        model=model,
        gate=local_gate,
        memory=memory if use_knn else KNNMemory(),
        reasoner=reasoner,
        action_agent=ActionAgent(),
        device=device,
        sample_rate_hz=float(acfg["evidence"]["sample_rate_hz"]),
        text_embed_fn=text_embedder(acfg["retrieval"]["embedder"]) if use_knn else None,
    )
    y_true, y_pred, esc, acts, lats, ps = [], [], [], [], [], []
    for batch in tqdm(DataLoader(Subset(ds, test_idx), batch_size=1, shuffle=False), desc=mode):
        x = batch["x"][0]
        res = pipe.run(x, x.numpy().T, activity=str(batch["activity"][0]))
        if mode == "gate_only" and res.escalated:
            yp = 1 if res.p_fall >= 0.5 else 0
        else:
            yp = 1 if res.prediction == "fall" else 0
        y_true.append(int(batch["y"][0]))
        y_pred.append(yp)
        esc.append(res.escalated)
        acts.append(str(batch["activity"][0]))
        lats.append(res.latency_ms)
        ps.append(res.p_fall)
    return agentic_metrics(
        y_true,
        y_pred,
        esc,
        activities=acts,
        ambiguous_codes=list(acfg["pipeline"].get("ambiguous_adl_codes", [])),
        latencies_ms=lats,
        p_falls=ps,
        cost_fn=float(acfg["gate"]["cost_fn"]),
        cost_fp=float(acfg["gate"]["cost_fp"]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kfall-config", default="configs/tier1_kfall_binary.yaml")
    ap.add_argument("--agentic-config", default="configs/agentic.yaml")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-test", type=int, default=1000)
    ap.add_argument("--modes", default="tier1_only,gate_knn_llm")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--model", default="harmamba", help="Tier-1 backbone (harmamba|cnn_lstm_attn|...)")
    ap.add_argument("--out-subdir", default=None, help="results/kfall/<subdir>/foldN; default model name")
    args = ap.parse_args()

    kcfg = load_config(ROOT / args.kfall_config)
    acfg = load_config(ROOT / args.agentic_config)
    bcfg = load_config(ROOT / "configs/backbones.yaml") if (ROOT / "configs/backbones.yaml").exists() else {}
    model_name = args.model
    sub = args.out_subdir or model_name
    out_dir = ensure_dir(ROOT / "results" / "kfall" / sub / f"fold{args.fold}")
    raw = ROOT / kcfg["raw_dir"]

    if not raw.exists():
        payload = {
            "status": "ACCESS_REQUIRED",
            "message": (
                "KFall raw data not found. Request access at "
                "https://sites.google.com/view/kfalldataset and extract into data/raw/kfall/"
            ),
            "expected_layout": "data/raw/kfall/SAXX/*.csv (+ optional label/*.xlsx)",
            "next_commands": [
                "python scripts/prepare_kfall.py --config configs/tier1_kfall_binary.yaml --phase transition",
                "python scripts/run_kfall_external.py --fold 0 --device cuda",
            ],
        }
        save_json(payload, out_dir / "ACCESS_REQUIRED.json")
        save_json(payload, ROOT / "results" / "kfall" / "ACCESS_REQUIRED.json")
        print(json.dumps(payload, indent=2))
        raise SystemExit(2)

    set_seed(int(kcfg["train"]["seed"]))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    processed = ROOT / kcfg["processed_dir"]
    npz = processed / f"windows_transition_w{kcfg['window']}_h{kcfg['hop']}.npz"
    if not npz.exists():
        print("Preparing KFall windows…")
        prepare_kfall(
            raw_dir=raw,
            out_dir=processed,
            window=int(kcfg["window"]),
            hop=int(kcfg["hop"]),
            sample_rate_hz=float(kcfg["sample_rate_hz"]),
            rate_threshold=float(kcfg["postfall"]["rate_threshold"]),
            stable_samples=int(kcfg["postfall"]["stable_samples"]),
            phase="transition",
        )

    ds = KFallDataset(npz, binary=True)
    in_ch = int(ds.X.shape[-1])
    tr_idx, va_idx, te_idx = _subject_split(
        ds,
        fold=args.fold,
        n_folds=int(kcfg["splits"]["n_folds"]),
        seed=int(kcfg["train"]["seed"]),
        val_fraction=float(kcfg["splits"].get("val_fraction", 0.15)),
    )
    tr_idx = subsample_indices(tr_idx, ds.y, ds.activities, kcfg["train"].get("max_train"), kcfg["train"]["seed"])
    va_idx = subsample_indices(va_idx, ds.y, ds.activities, kcfg["train"].get("max_val"), kcfg["train"]["seed"] + 1)
    te_idx = subsample_indices(
        te_idx,
        ds.y,
        ds.activities,
        args.max_test,
        kcfg["train"]["seed"] + 2,
        min_per_activity={"D18": 40, "D19": 40},
    )

    ckpt_dir = ensure_dir(ROOT / kcfg["checkpoint_dir"])
    ckpt = ckpt_dir / f"{model_name}_binary_fold{args.fold}.pt"
    zoo_kw = kwargs_for_model(model_name, bcfg)
    if model_name == "cnn_lstm_attn":
        model = build_model(
            "cnn_lstm_attn",
            in_channels=in_ch,
            num_classes=2,
            conv_channels=int(kcfg["model"]["conv_channels"]),
            branch_channels=int(kcfg["model"]["branch_channels"]),
            lstm_hidden=kcfg["model"]["lstm_hidden"],
            attn_heads=int(kcfg["model"]["attn_heads"]),
        )
    else:
        model = build_model(model_name, in_channels=in_ch, num_classes=2, **zoo_kw)
    if not (args.skip_train and ckpt.exists()):
        cw = class_pos_weight([int(ds.y[i]) for i in tr_idx]) if kcfg["train"].get("use_pos_weight", True) else None
        fit_model(
            model,
            DataLoader(Subset(ds, tr_idx), batch_size=int(kcfg["train"]["batch_size"]), shuffle=True, num_workers=2),
            DataLoader(Subset(ds, va_idx), batch_size=256, shuffle=False, num_workers=2),
            device=device,
            epochs=int(kcfg["train"]["epochs"]),
            patience=int(kcfg["train"].get("early_stopping_patience", 10)),
            mixup_alpha=float(kcfg["train"].get("mixup_alpha", 0.1)),
            checkpoint_path=str(ckpt),
            class_weight=cw,
            lr=float(kcfg["train"]["lr"]),
        )
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device)

    mem_path = ensure_dir(ROOT / "data" / "memory") / f"kfall_{model_name}_fold{args.fold}.json"
    if not mem_path.exists():
        build_kfall_memory(model, ds, tr_idx, device, acfg, mem_path)

    gate = build_gate(
        acfg,
        model=model,
        ds=ds,
        cal_idx=tr_idx,
        device=device,
        calibrate=bool(kcfg.get("eval", {}).get("calibrate_gate", True)),
        max_cal=int(kcfg.get("eval", {}).get("max_cal", 2000)),
    )
    det = evaluate(model, DataLoader(Subset(ds, te_idx), batch_size=256, shuffle=False), device)
    det.pop("probs", None)
    det.pop("y_true", None)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    rows = [{"name": model_name, **det}]
    for mode in modes:
        am = run_mode(model, ds, te_idx, str(mem_path), device, acfg, mode, gate)
        am["max_test"] = len(te_idx)
        am["dataset"] = "kfall_binary"
        rows.append({"name": mode, **am})
        print(mode, {k: am[k] for k in ("f1", "recall", "expected_response_cost") if k in am})

    payload = {
        "dataset": "kfall_binary",
        "fold": args.fold,
        "model": model_name,
        "max_test": len(te_idx),
        "n_channels": in_ch,
        "checkpoint": str(ckpt),
        "rows": rows,
    }
    save_json(payload, out_dir / f"kfall_external_fold{args.fold}.json")
    save_json(payload, ROOT / "results" / "kfall" / sub / f"kfall_external_fold{args.fold}.json")
    print(f"Wrote {out_dir / f'kfall_external_fold{args.fold}.json'}")


if __name__ == "__main__":
    main()
