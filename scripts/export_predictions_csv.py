#!/usr/bin/env python3
"""Export per-window prediction CSV for robustness_suite.py (paper protocol)."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_fall.agents import (  # noqa: E402
    ActionAgent,
    AgenticPipeline,
    ConfidenceGate,
    KNNMemory,
    LLMReasoner,
)
from agentic_fall.data.sisfall import SisFallDataset  # noqa: E402
from agentic_fall.eval.protocol import (  # noqa: E402
    build_gate,
    make_fold_split,
    subsample_indices,
    text_embedder,
)
from agentic_fall.features.biomechanics import extract_biomechanics  # noqa: E402
from agentic_fall.models import build_model  # noqa: E402
from agentic_fall.utils.config import load_config  # noqa: E402
from agentic_fall.utils.seed import set_seed  # noqa: E402


def load_ckpt(ckpt, tcfg, device):
    model = build_model(
        "cnn_lstm_attn",
        in_channels=int(tcfg["model"]["in_channels"]),
        num_classes=int(tcfg["num_classes"]),
        conv_channels=int(tcfg["model"]["conv_channels"]),
        branch_channels=int(tcfg["model"]["branch_channels"]),
        lstm_hidden=tcfg["model"]["lstm_hidden"],
        attn_heads=int(tcfg["model"]["attn_heads"]),
        dropout_attn=float(tcfg["model"]["dropout_attn"]),
        dropout_fc=float(tcfg["model"]["dropout_fc"]),
        use_se=bool(tcfg["model"].get("use_se", False)),
    )
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-config", default="configs/paper_protocol.yaml")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/noleak/predictions_fold0.csv")
    ap.add_argument("--full-test-trials", action="store_true")
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--max-rows", type=int, default=0)
    args = ap.parse_args()

    pcfg = load_config(ROOT / args.paper_config)
    tcfg = load_config(ROOT / pcfg["tier1_config"])
    acfg = load_config(ROOT / pcfg["agentic_config"])
    set_seed(int(pcfg["seed"]))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    npz = ROOT / tcfg["processed_dir"] / f"windows_w{tcfg['window']}_h{tcfg['hop']}.npz"
    ds = SisFallDataset(npz, channels=int(tcfg["channels"]))
    split = make_fold_split(
        ds,
        args.fold,
        n_folds=int(pcfg["splits"]["n_folds"]),
        seed=int(pcfg["seed"]),
        val_fraction=float(pcfg["splits"]["val_fraction"]),
    )
    ckpt = ROOT / pcfg["paths"]["checkpoint_dir"] / f"cnn_lstm_attn_fold{args.fold}.pt"
    mem = ROOT / pcfg["paths"]["memory_dir"] / f"sisfall_fold{args.fold}.json"
    model = load_ckpt(ckpt, tcfg, device)
    gate = build_gate(
        acfg,
        model=model,
        ds=ds,
        cal_idx=split.val_idx,
        device=device,
        calibrate=bool(pcfg["eval"].get("calibrate_gate", True)),
        max_cal=int(pcfg["eval"].get("max_cal", 2000)),
    )

    memory = KNNMemory(k=int(acfg["retrieval"]["k"]))
    if mem.exists():
        memory.load(mem)
    backend = "heuristic" if args.skip_llm else "ollama"
    pipe_cfg = acfg.get("pipeline") or {}
    include_activity = bool(pipe_cfg.get("include_activity", False))
    require_embedder = bool(pipe_cfg.get("require_text_embedder", False))
    reasoner = LLMReasoner(
        backend=backend,
        base_url=acfg["llm"]["base_url"],
        model=acfg["llm"]["model"],
        fallback_heuristic=bool(acfg.get("llm", {}).get("fallback_heuristic", False)),
    )
    pipe = AgenticPipeline(
        model=model,
        gate=gate,
        memory=memory,
        reasoner=reasoner,
        action_agent=ActionAgent(),
        device=device,
        sample_rate_hz=float(acfg["evidence"]["sample_rate_hz"]),
        text_embed_fn=text_embedder(
            acfg["retrieval"]["embedder"],
            allow_fallback=not require_embedder,
        ),
        include_activity=include_activity,
        require_text_embedder=require_embedder,
    )
    pipe_primary = AgenticPipeline(
        model=model,
        gate=ConfidenceGate(0.5, 0.5 + 1e-6),
        memory=KNNMemory(),
        reasoner=LLMReasoner(backend="heuristic"),
        action_agent=ActionAgent(),
        device=device,
        sample_rate_hz=float(acfg["evidence"]["sample_rate_hz"]),
        include_activity=include_activity,
    )

    min_per = {
        a: int(pcfg["eval"]["min_near_fall_per_code"])
        for a in pcfg["eval"]["required_activities"]
    }
    if args.full_test_trials:
        test_idx = list(split.test_idx)
    else:
        test_idx = subsample_indices(
            split.test_idx,
            ds.y,
            ds.activities,
            int(pcfg["eval"]["agentic_max_samples"]),
            int(pcfg["seed"]) + 2,
            min_per_activity=min_per,
        )
    max_train = pcfg["train"].get("max_train")
    train_idx = list(split.train_idx)
    if max_train:
        train_idx = train_idx[: int(max_train)]
    # Test first so --max-rows still yields B1/B2-usable rows.
    parts = [
        ("test", list(test_idx)),
        ("calibration", list(split.val_idx)[:2000]),
        ("train", train_idx),
    ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "fold",
        "subject",
        "trial_id",
        "activity_code",
        "t_start",
        "y_true",
        "p",
        "escalated",
        "pred_primary",
        "pred_stack",
        "split",
        "sig_freefall_s",
        "sig_impact_g",
        "sig_stillness_s",
        "sig_tilt_deg",
        "sig_duration_s",
        "sig_mean_acc_g",
        "sig_peak_gyro",
    ]
    hop_s = int(tcfg["hop"]) / float(acfg["evidence"]["sample_rate_hz"])
    n_written = 0
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for split_name, idxs in parts:
            if not idxs:
                continue
            loader = DataLoader(Subset(ds, idxs), batch_size=1, shuffle=False)
            for batch in tqdm(loader, desc=f"fold{args.fold}:{split_name}"):
                x = batch["x"][0]
                win = x.numpy().T
                feats = extract_biomechanics(
                    win,
                    sample_rate_hz=float(acfg["evidence"]["sample_rate_hz"]),
                    freefall_g_threshold=float(acfg["evidence"]["freefall_g_threshold"]),
                    stillness_var_threshold=float(acfg["evidence"]["stillness_var_threshold"]),
                )
                res_p = pipe_primary.run(x, win)
                # Full calibrated-gate stack (LLM or heuristic per --skip-llm).
                res_s = pipe.run(x, win)
                escalated = int(res_s.escalated)
                pred_stack = 1 if res_s.prediction == "fall" else 0
                p_val = float(res_s.p_fall)
                w.writerow(
                    {
                        "fold": args.fold,
                        "subject": str(batch["subject"][0]),
                        "trial_id": str(batch["file_id"][0]),
                        "activity_code": str(batch["activity"][0]),
                        "t_start": float(batch["index"][0]) * hop_s,
                        "y_true": int(batch["y"][0]),
                        "p": p_val,
                        "escalated": escalated,
                        "pred_primary": 1 if res_p.prediction == "fall" else 0,
                        "pred_stack": pred_stack,
                        "split": split_name,
                        "sig_freefall_s": feats.freefall_duration_s,
                        "sig_impact_g": feats.impact_magnitude_g,
                        "sig_stillness_s": feats.post_impact_stillness_s,
                        "sig_tilt_deg": feats.body_tilt_deg,
                        "sig_duration_s": feats.duration_s,
                        "sig_mean_acc_g": feats.mean_acc_g,
                        "sig_peak_gyro": feats.peak_gyro_dps,
                    }
                )
                n_written += 1
                if args.max_rows and n_written >= args.max_rows:
                    print(f"Stopped at max-rows={args.max_rows}")
                    print(f"Wrote {out}")
                    return
    print(f"Wrote {out} ({n_written} rows)")


if __name__ == "__main__":
    main()
