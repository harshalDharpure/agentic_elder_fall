#!/usr/bin/env python3
"""Evaluate gate_knn_llm (NO CRITIC) on KFall ambiguous bench NPZ."""
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

from agentic_fall.agents import ActionAgent, AgenticPipeline, KNNMemory, LLMReasoner
from agentic_fall.agents.confidence_gate import ConfidenceGate
from agentic_fall.data.kfall import KFallDataset
from agentic_fall.eval.metrics import agentic_metrics, binary_metrics
from agentic_fall.eval.protocol import build_gate, subsample_indices, text_embedder
from agentic_fall.models import build_model, kwargs_for_model
from agentic_fall.utils.config import load_config
from agentic_fall.utils.io import ensure_dir, save_json
from agentic_fall.utils.seed import set_seed


def _subject_split(ds, fold, n_folds, seed, val_fraction):
    folds = ds.fold_indices(n_folds=n_folds, seed=seed)
    train_idx, test_idx = folds[fold]
    train_subj = sorted({str(ds.subjects[i]) for i in train_idx})
    rng = np.random.default_rng(seed + fold)
    rng.shuffle(train_subj)
    n_val = max(1, int(round(len(train_subj) * val_fraction)))
    val_subj = set(train_subj[:n_val])
    tr_subj = set(train_subj[n_val:])
    tr = [i for i in train_idx if str(ds.subjects[i]) in tr_subj]
    return tr


def load_bench(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def load_kfall_model(kcfg, ckpt, in_ch, device, model_name: str = "cnn_lstm_attn", bcfg=None):
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
        zoo_kw = kwargs_for_model(model_name, bcfg or {})
        model = build_model(model_name, in_channels=in_ch, num_classes=2, **zoo_kw)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    return model


def make_pipeline(model, gate, memory_path, device, acfg, *, mode: str, backend: str):
    use_knn = mode != "tier1_only"
    memory = KNNMemory(k=int(acfg["retrieval"]["k"]))
    if use_knn and Path(memory_path).exists():
        memory.load(memory_path)
    reasoner = LLMReasoner(
        backend=backend,
        base_url=acfg["llm"]["base_url"],
        model=acfg["llm"]["model"],
        fallback_heuristic=True,
    )
    if mode == "tier1_only":
        gate = ConfidenceGate(0.5, 0.5 + 1e-6)
    return AgenticPipeline(
        model=model,
        gate=gate,
        memory=memory if use_knn else KNNMemory(),
        reasoner=reasoner,
        action_agent=ActionAgent(),
        device=device,
        sample_rate_hz=float(acfg["evidence"]["sample_rate_hz"]),
        text_embed_fn=text_embedder(acfg["retrieval"]["embedder"]) if use_knn else None,
        adjudicator=None,
        allow_action_downgrade=True,
    )


def run_cases(pipe, bench, *, force_ambiguous: bool, mode: str) -> list[dict]:
    rows = []
    X, y = bench["X"], bench["y"]
    acts, buckets = bench["activities"], bench["buckets"]
    for i in tqdm(range(len(y)), desc=f"{mode}"):
        win = np.asarray(X[i], dtype=np.float32)
        x = torch.from_numpy(win.T.copy())
        bucket = str(buckets[i])
        force = mode != "tier1_only" and force_ambiguous and bucket == "ambiguous"
        res = pipe.run(x, win, activity=str(acts[i]), force_escalate=force)
        rows.append(
            {
                "idx": i,
                "bucket": bucket,
                "y": int(y[i]),
                "activity": str(acts[i]),
                "p_fall": float(res.p_fall),
                "route": res.gate.route,
                "escalated": bool(res.escalated),
                "prediction": res.prediction,
                "pred_y": 1 if res.prediction == "fall" else 0,
                "action": str(res.action.action),
                "correct": int((1 if res.prediction == "fall" else 0) == int(y[i])),
            }
        )
    return rows


def summarize(rows, acfg, name):
    yt = [r["y"] for r in rows]
    yp = [r["pred_y"] for r in rows]
    m = agentic_metrics(
        yt,
        yp,
        [r["escalated"] for r in rows],
        activities=[r["activity"] for r in rows],
        ambiguous_codes=list(acfg["pipeline"]["ambiguous_adl_codes"]),
        actions=[r["action"] for r in rows],
        cost_fn=float(acfg["gate"]["cost_fn"]),
        cost_fp=float(acfg["gate"]["cost_fp"]),
    )
    m.update(binary_metrics(yt, yp))
    m["name"] = name
    m["accuracy"] = float(np.mean([r["correct"] for r in rows]))
    m["far"] = float(m["fp"] / max(1, m["fp"] + m["tn"]))
    m["n"] = len(rows)
    m["cost_per_window"] = float(m["expected_response_cost"] / max(1, len(rows)))
    per = {}
    for b in ("strict_fall", "clear_adl", "ambiguous"):
        sub = [r for r in rows if r["bucket"] == b]
        if sub:
            per[b] = {
                "n": len(sub),
                "accuracy": float(np.mean([r["correct"] for r in sub])),
                "escalate_rate": float(np.mean([r["escalated"] for r in sub])),
            }
    m["per_bucket"] = per
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kfall-config", default="configs/tier1_kfall_binary.yaml")
    ap.add_argument("--agentic-config", default="configs/agentic.yaml")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--bench", default=None)
    ap.add_argument("--mode", default="gate_knn_llm", choices=("tier1_only", "gate_knn_llm"))
    ap.add_argument("--backend", default="ollama", choices=("ollama", "heuristic"))
    ap.add_argument("--force-ambiguous", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--model", default="harmamba")
    args = ap.parse_args()

    kcfg = load_config(ROOT / args.kfall_config)
    acfg = load_config(ROOT / args.agentic_config)
    bcfg = load_config(ROOT / "configs/backbones.yaml") if (ROOT / "configs/backbones.yaml").exists() else {}
    set_seed(int(kcfg["train"]["seed"]))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_name = args.model

    bench_path = Path(
        args.bench or ROOT / f"data/kfold_ambiguous_kfall/fold{args.fold}_ambiguous_eval.npz"
    )
    if not bench_path.exists():
        raise SystemExit(f"Missing {bench_path}. Run scripts/build_kfall_ambiguous_bench.py first.")
    bench = load_bench(bench_path)

    npz = ROOT / kcfg["processed_dir"] / f"windows_transition_w{kcfg['window']}_h{kcfg['hop']}.npz"
    ds = KFallDataset(npz, binary=True)
    in_ch = int(ds.X.shape[-1])

    ckpt = ROOT / kcfg["checkpoint_dir"] / f"{model_name}_binary_fold{args.fold}.pt"
    mem = ROOT / "data" / "memory" / f"kfall_{model_name}_fold{args.fold}.json"
    if not mem.exists():
        mem = ROOT / "data" / "memory" / f"kfall_fold{args.fold}.json"
    model = load_kfall_model(kcfg, ckpt, in_ch, device, model_name=model_name, bcfg=bcfg)

    tr_idx = _subject_split(
        ds,
        args.fold,
        int(kcfg["splits"]["n_folds"]),
        int(kcfg["train"]["seed"]),
        float(kcfg["splits"].get("val_fraction", 0.15)),
    )
    tr_idx = subsample_indices(
        tr_idx, ds.y, ds.activities, kcfg["train"].get("max_train"), kcfg["train"]["seed"]
    )
    gate = build_gate(
        acfg,
        model=model,
        ds=ds,
        cal_idx=tr_idx,
        device=device,
        calibrate=bool(kcfg.get("eval", {}).get("calibrate_gate", True)),
        max_cal=int(kcfg.get("eval", {}).get("max_cal", 2000)),
    )

    pipe = make_pipeline(model, gate, str(mem), device, acfg, mode=args.mode, backend=args.backend)
    rows = run_cases(pipe, bench, force_ambiguous=bool(args.force_ambiguous), mode=args.mode)
    metrics = summarize(rows, acfg, args.mode)
    metrics["escalate_rate"] = float(np.mean([r["escalated"] for r in rows]))

    out_default = (
        "results/kfold_ambiguous_kfall_tier1"
        if args.mode == "tier1_only"
        else "results/kfold_ambiguous_kfall"
    )
    out_dir = ensure_dir(ROOT / (args.out_dir or out_default))
    payload = {
        "fold": args.fold,
        "dataset": "kfall",
        "mode": args.mode,
        "backend": args.backend,
        "bench": str(bench_path),
        "metrics": metrics,
        "rows": rows,
        "n_ambiguous": int(sum(1 for r in rows if r["bucket"] == "ambiguous")),
    }
    save_json(payload, out_dir / f"fold{args.fold}_eval.json")

    lines = [
        f"KFall {args.mode} (NO CRITIC) fold {args.fold} backend={args.backend}",
        f"N={metrics['n']} Acc={metrics['accuracy']:.3f} F1={metrics['f1']:.3f} "
        f"Rec={metrics['recall']:.3f} FN={metrics['fn']} FP={metrics['fp']} "
        f"Cost={metrics['expected_response_cost']:.0f} Cost/win={metrics['cost_per_window']:.2f} "
        f"AmbAcc={(metrics.get('per_bucket') or {}).get('ambiguous', {}).get('accuracy', float('nan')):.3f}",
    ]
    (out_dir / f"fold{args.fold}_results.txt").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: metrics[k] for k in metrics if k != "per_bucket"}, indent=2, default=str))


if __name__ == "__main__":
    main()
