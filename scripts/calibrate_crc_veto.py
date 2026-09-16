#!/usr/bin/env python3
"""Calibrate CRC veto threshold λ* on subject-held-out val (no test leakage).

Collects (veto_score, label) on windows whose *pre-veto* prediction is fall,
fits an empirical-Bernstein upper bound, and caches λ* plus a route-wise FP
diagnostic (confident-path vs ambiguous).
"""
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

from agentic_fall.agents import (
    ActionAgent,
    AgenticPipeline,
    KNNMemory,
    LLMReasoner,
    build_adjudicator_from_config,
)
from agentic_fall.data.sisfall import SisFallDataset
from agentic_fall.eval.conformal import (
    alpha_break_even,
    area_under_risk_coverage,
    risk_coverage_curve,
    select_lambda_star,
)
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

ALPHAS = (0.01, 0.05, 0.10, 0.20)


def _load_model(pcfg, tcfg, ckpt, device):
    primary = pcfg["agentic"].get("primary_backbone", "cnn_lstm_attn")
    if primary == "cnn_lstm_attn":
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
        model = build_model(primary, in_channels=int(tcfg["channels"]), num_classes=2)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    return model


def build_crc_pipeline(model, memory_path, device, acfg, gate, *, feasible=False, threshold=None, alpha=None):
    memory = KNNMemory(k=int(acfg["retrieval"]["k"]))
    if Path(memory_path).exists():
        memory.load(memory_path)
    reasoner = LLMReasoner(
        backend=acfg["llm"]["backend"],
        base_url=acfg["llm"]["base_url"],
        model=acfg["llm"]["model"],
        fallback_heuristic=True,
    )
    cfg = dict(acfg)
    adj = dict(acfg.get("adjudication") or {})
    adj.update(
        {
            "enabled": True,
            "mode": "crc_veto",
            "actor_backend": "ollama",
            "critic_backend": "heuristic",
            "freeze_label": False,
            "screen_confident_fall": bool(adj.get("screen_confident_fall", True)),
            "veto_threshold": threshold,
            "crc_feasible": bool(feasible),
            "crc_alpha": alpha,
        }
    )
    cfg["adjudication"] = adj
    adjudicator = build_adjudicator_from_config(cfg, reasoner)
    return AgenticPipeline(
        model=model,
        gate=gate,
        memory=memory,
        reasoner=reasoner,
        action_agent=ActionAgent(),
        device=device,
        sample_rate_hz=float(acfg["evidence"]["sample_rate_hz"]),
        text_embed_fn=text_embedder(acfg["retrieval"]["embedder"]),
        adjudicator=adjudicator,
        allow_action_downgrade=True,
    )


def collect_records(pipe: AgenticPipeline, ds, indices, desc: str) -> list[dict]:
    rows = []
    loader = DataLoader(Subset(ds, indices), batch_size=1, shuffle=False)
    for batch in tqdm(loader, desc=desc):
        x = batch["x"][0]
        res = pipe.run(x, x.numpy().T, activity=str(batch["activity"][0]))
        actor = res.actor_prediction or res.prediction
        rows.append(
            {
                "y": int(batch["y"][0]),
                "activity": str(batch["activity"][0]),
                "p_fall": float(res.p_fall),
                "route": res.gate.route,
                "escalated": bool(res.escalated),
                "actor_pred": actor,
                "final_pred": res.prediction,
                "veto_score": float(res.veto_score) if res.veto_score is not None else None,
                "vetoed": bool(res.vetoed),
            }
        )
    return rows


def records_to_eligible(records: list[dict]) -> tuple[list[float], list[int]]:
    scores, labels = [], []
    for r in records:
        if r.get("actor_pred") != "fall":
            continue
        s = r.get("veto_score")
        if s is None:
            continue
        scores.append(float(s))
        labels.append(int(r["y"]))
    return scores, labels


def route_fp_diagnostic(records: list[dict]) -> dict:
    fp = [r for r in records if r["y"] == 0 and r.get("actor_pred") == "fall"]
    n_fp = max(1, len(fp))
    by_route = {}
    for route in ("fall", "ambiguous", "adl"):
        k = sum(1 for r in fp if r["route"] == route)
        by_route[route] = {"n": k, "share": float(k / n_fp) if fp else 0.0}
    return {
        "n_actor_fall_fp": len(fp),
        "n_records": len(records),
        "by_route": by_route,
        "majority_confident_path": bool(fp) and by_route["fall"]["share"] >= 0.5,
    }


def fit_from_records(records: list[dict], acfg: dict, alphas=ALPHAS) -> dict:
    scores, labels = records_to_eligible(records)
    delta = float((acfg.get("adjudication") or {}).get("crc_delta", 0.1))
    method = str((acfg.get("adjudication") or {}).get("crc_method", "bernstein"))
    break_even = alpha_break_even(
        float(acfg["gate"]["cost_fn"]), float(acfg["gate"]["cost_fp"])
    )
    by_alpha = {}
    for a in alphas:
        sel = select_lambda_star(scores, labels, float(a), delta=delta, method=method)
        by_alpha[f"{float(a):.2f}"] = sel
    primary_alpha = float((acfg.get("adjudication") or {}).get("crc_alpha", 0.05))
    primary = by_alpha.get(f"{primary_alpha:.2f}") or next(iter(by_alpha.values()))
    curve = risk_coverage_curve(scores, labels, delta=delta, method=method)
    return {
        "n_eligible_actor_fall": len(scores),
        "n_eligible_true_fall": int(sum(labels)),
        "n_eligible_adl": int(len(labels) - sum(labels)),
        "alpha_break_even": break_even,
        "delta": delta,
        "method": method,
        "primary_alpha": primary_alpha,
        "selection": primary,
        "by_alpha": by_alpha,
        "curve": curve,
        "aurc": area_under_risk_coverage(curve),
        "route_fp_diagnostic": route_fp_diagnostic(records),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-config", default="configs/paper_protocol.yaml")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--memory", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-cal", type=int, default=1000)
    ap.add_argument("--alpha", type=float, default=None)
    args = ap.parse_args()

    pcfg = load_config(ROOT / args.paper_config)
    tcfg = load_config(ROOT / pcfg["tier1_config"])
    acfg = load_config(ROOT / pcfg["agentic_config"])
    if args.alpha is not None:
        acfg.setdefault("adjudication", {})["crc_alpha"] = float(args.alpha)
    set_seed(int(pcfg["seed"]))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

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
    tr = pcfg["train"]
    ev = pcfg["eval"]
    train_idx = subsample_indices(split.train_idx, ds.y, ds.activities, tr.get("max_train"), pcfg["seed"])
    val_idx = subsample_indices(
        split.val_idx, ds.y, ds.activities, args.max_cal or tr.get("max_val"), pcfg["seed"] + 1
    )

    ckpt = Path(args.checkpoint) if args.checkpoint else ROOT / pcfg["paths"]["checkpoint_dir"] / f"cnn_lstm_attn_fold{args.fold}.pt"
    memory_path = args.memory or str(ROOT / pcfg["paths"]["memory_dir"] / f"sisfall_fold{args.fold}.json")
    model = _load_model(pcfg, tcfg, ckpt, device)
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
    # Collect scores with veto disabled (feasible=False).
    pipe = build_crc_pipeline(model, memory_path, device, acfg, gate, feasible=False)
    records = collect_records(pipe, ds, val_idx, desc=f"crc_cal_fold{args.fold}")
    fitted = fit_from_records(records, acfg)
    out_dir = ensure_dir(ROOT / f"{pcfg['paths']['results_dir']}/fold{args.fold}")
    payload = {
        "fold": args.fold,
        "n_val": len(val_idx),
        "tau_low": gate.tau_low,
        "tau_high": gate.tau_high,
        "records": records,
        **fitted,
    }
    out = out_dir / f"veto_calibration_fold{args.fold}.json"
    save_json(payload, out)
    save_json(payload, ROOT / pcfg["paths"]["results_dir"] / out.name)
    sel = fitted["selection"]
    print(json.dumps({k: fitted[k] for k in ("n_eligible_actor_fall", "selection", "route_fp_diagnostic", "aurc", "alpha_break_even")}, indent=2, default=str))
    print(f"Wrote {out} feasible={sel.get('feasible')} lambda_star={sel.get('lambda_star')}")


if __name__ == "__main__":
    main()
