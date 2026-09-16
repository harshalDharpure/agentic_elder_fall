#!/usr/bin/env python3
"""Evaluate full agentic stack on a K-fold ambiguous eval NPZ (real SisFall windows)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_fall.agents import (
    ActionAgent,
    AgenticPipeline,
    KNNMemory,
    LLMReasoner,
)
from agentic_fall.agents.confidence_gate import ConfidenceGate
from agentic_fall.data.sisfall import SisFallDataset
from agentic_fall.eval.metrics import agentic_metrics, binary_metrics
from agentic_fall.eval.protocol import build_gate, make_fold_split, subsample_indices, text_embedder
from agentic_fall.models import build_model
from agentic_fall.utils.config import load_config
from agentic_fall.utils.io import ensure_dir, save_json
from agentic_fall.utils.seed import set_seed


def load_bench(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def load_model(pcfg, tcfg, ckpt, device):
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
    return model


def make_pipeline(
    model,
    gate,
    memory_path,
    device,
    acfg,
    *,
    mode: str,
    backend: str,
):
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
    for i in tqdm(range(len(y)), desc="eval"):
        win = np.asarray(X[i], dtype=np.float32)
        x = torch.from_numpy(win.T.copy())
        bucket = str(buckets[i])
        # tier1_only: never force LLM; gate_knn_llm: optionally force ambiguous D18/D19
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
                "adjudicated": bool(res.adjudicated),
                "critic_verdict": res.critic_verdict,
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
    m["critique_usage_rate"] = float(np.mean([r["adjudicated"] for r in rows]))
    per = {}
    for b in ("strict_fall", "clear_adl", "ambiguous"):
        sub = [r for r in rows if r["bucket"] == b]
        if not sub:
            continue
        per[b] = {
            "n": len(sub),
            "accuracy": float(np.mean([r["correct"] for r in sub])),
            "escalate_rate": float(np.mean([r["escalated"] for r in sub])),
            "adjudicated_rate": float(np.mean([r["adjudicated"] for r in sub])),
        }
    m["per_bucket"] = per
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-config", default="configs/paper_protocol.yaml")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--bench", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--memory", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--mode", default="gate_knn_llm", choices=("tier1_only", "gate_knn_llm"))
    ap.add_argument("--backend", default="ollama", choices=("ollama", "heuristic"))
    ap.add_argument("--force-ambiguous", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_default = (
        "results/kfold_ambiguous_tier1"
        if args.mode == "tier1_only"
        else "results/kfold_ambiguous"
    )
    out_dir_rel = args.out_dir or out_default

    pcfg = load_config(ROOT / args.paper_config)
    tcfg = load_config(ROOT / pcfg["tier1_config"])
    acfg = load_config(ROOT / pcfg["agentic_config"])
    set_seed(int(pcfg["seed"]))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    bench_path = Path(
        args.bench or ROOT / f"data/kfold_ambiguous/fold{args.fold}_ambiguous_eval.npz"
    )
    if not bench_path.exists():
        raise SystemExit(f"Missing {bench_path}. Run scripts/build_kfold_ambiguous_bench.py first.")
    bench = load_bench(bench_path)

    ckpt = Path(args.checkpoint) if args.checkpoint else ROOT / pcfg["paths"]["checkpoint_dir"] / f"cnn_lstm_attn_fold{args.fold}.pt"
    memory = args.memory or str(ROOT / pcfg["paths"]["memory_dir"] / f"sisfall_fold{args.fold}.json")
    model = load_model(pcfg, tcfg, ckpt, device)

    npz = ROOT / tcfg["processed_dir"] / f"windows_w{tcfg['window']}_h{tcfg['hop']}.npz"
    ds = SisFallDataset(npz, channels=int(tcfg["channels"]))
    split = make_fold_split(
        ds, fold=args.fold, n_folds=int(pcfg["splits"]["n_folds"]), seed=int(pcfg["seed"]),
        val_fraction=float(pcfg["splits"]["val_fraction"]),
    )
    train_idx = subsample_indices(
        split.train_idx, ds.y, ds.activities, pcfg["train"].get("max_train"), pcfg["seed"]
    )
    gate = build_gate(
        acfg, model=model, ds=ds, cal_idx=train_idx, device=device,
        calibrate=bool(pcfg["eval"].get("calibrate_gate", True)),
        max_cal=int(pcfg["eval"].get("max_cal", 2000)),
    )

    pipe = make_pipeline(
        model, gate, memory, device, acfg, mode=args.mode, backend=args.backend
    )
    rows = run_cases(pipe, bench, force_ambiguous=bool(args.force_ambiguous), mode=args.mode)
    metrics = summarize(rows, acfg, args.mode)
    metrics["escalate_rate"] = float(np.mean([r["escalated"] for r in rows]))
    metrics["n"] = len(rows)
    metrics["cost_per_window"] = float(metrics["expected_response_cost"] / max(1, len(rows)))

    out_dir = ensure_dir(ROOT / out_dir_rel)
    payload = {
        "fold": args.fold,
        "mode": args.mode,
        "backend": args.backend,
        "force_ambiguous": bool(args.force_ambiguous) and args.mode != "tier1_only",
        "bench": str(bench_path),
        "tau_low": pipe.gate.tau_low,
        "tau_high": pipe.gate.tau_high,
        "metrics": metrics,
        "rows": rows,
        "n_ambiguous": int(sum(1 for r in rows if r["bucket"] == "ambiguous")),
    }
    save_json(payload, out_dir / f"fold{args.fold}_eval.json")

    lines = [
        f"{args.mode} (NO CRITIC) — fold {args.fold}  backend={args.backend}",
        f"N={metrics['n']}  Acc={metrics['accuracy']:.3f}  "
        f"F1={metrics['f1']:.3f}  Prec={metrics['precision']:.3f}  Rec={metrics['recall']:.3f}  "
        f"FN={metrics['fn']}  FP={metrics['fp']}  "
        f"FAR={metrics.get('far', metrics.get('fpr', float('nan'))):.3f}  "
        f"Esc={metrics['escalate_rate']:.3f}  "
        f"Cost={metrics['expected_response_cost']:.0f}  "
        f"Cost/win={metrics['cost_per_window']:.2f}  "
        f"GrCost={metrics.get('graded_action_cost', 0):.0f}  "
        f"ambiguous={payload['n_ambiguous']}",
        "",
        "Per bucket:",
    ]
    for b, s in (metrics.get("per_bucket") or {}).items():
        lines.append(
            f"  {b}: n={s['n']} acc={s['accuracy']:.3f} esc_or_adj={s.get('adjudicated_rate', 0):.3f}"
        )
    amb = [r for r in rows if r["bucket"] == "ambiguous"]
    if amb:
        lines.append(
            f"  ambiguous escalate_rate={float(np.mean([r['escalated'] for r in amb])):.3f}"
        )
    (out_dir / f"fold{args.fold}_results.txt").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: metrics[k] for k in metrics if k != "per_bucket"}, indent=2, default=str))
    print(json.dumps(metrics.get("per_bucket"), indent=2))
    print(f"Wrote {out_dir / f'fold{args.fold}_results.txt'}")


if __name__ == "__main__":
    main()
