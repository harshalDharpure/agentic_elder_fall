#!/usr/bin/env python3
"""Train Tier-1 detector under the locked paper protocol."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_fall.data.sisfall import SisFallDataset
from agentic_fall.data.kfall import KFallDataset
from agentic_fall.eval.protocol import (
    make_fold_split,
    subsample_indices,
    tune_decision_threshold,
    verify_dataset_near_falls,
)
from agentic_fall.eval.train_loop import class_pos_weight, evaluate, fit_model
from agentic_fall.models import build_model
from agentic_fall.utils.config import load_config
from agentic_fall.utils.io import ensure_dir
from agentic_fall.utils.seed import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tier1_sisfall.yaml")
    ap.add_argument("--paper-config", default="configs/paper_protocol.yaml")
    ap.add_argument("--model", default="cnn_lstm_attn")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-test", type=int, default=None)
    ap.add_argument("--checkpoint-dir", default=None)
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    pcfg = load_config(ROOT / args.paper_config) if (ROOT / args.paper_config).exists() else {}
    set_seed(int(cfg["train"]["seed"]))
    device = torch.device(
        args.device or cfg["train"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )

    dataset_name = cfg["dataset"]
    window = cfg["window"]
    hop = cfg["hop"]
    if dataset_name == "sisfall":
        npz = ROOT / cfg["processed_dir"] / f"windows_w{window}_h{hop}.npz"
        ds = SisFallDataset(npz, channels=int(cfg["channels"]))
        verify_dataset_near_falls(
            ds, pcfg.get("eval", {}).get("required_activities", ["D18", "D19"])
        )
        num_classes = 2
        model_kwargs = dict(cfg["model"])
        in_ch = model_kwargs.pop("in_channels", cfg["channels"])
    else:
        phase = "transition"
        npz = ROOT / cfg["processed_dir"] / f"windows_{phase}_w{window}_h{hop}.npz"
        ds = KFallDataset(npz)
        num_classes = int(cfg["num_classes"])
        model_kwargs = dict(cfg["model"])
        in_ch = model_kwargs.pop("in_channels", cfg["channels"])

    split = make_fold_split(
        ds,
        fold=args.fold,
        n_folds=int(cfg["splits"]["n_folds"]),
        seed=int(cfg["train"]["seed"]),
        val_fraction=float(pcfg.get("splits", {}).get("val_fraction", 0.15)),
    )
    train_idx, val_idx, test_idx = split.train_idx, split.val_idx, split.test_idx

    ptr = pcfg.get("train", {})
    pev = pcfg.get("eval", {})
    max_train = args.max_train if args.max_train is not None else ptr.get("max_train")
    max_test = args.max_test if args.max_test is not None else pev.get("max_test")
    min_nf = int(pev.get("min_near_fall_per_code", 0) or 0)
    min_per = {c: min_nf for c in pev.get("required_activities", ["D18", "D19"])} if min_nf else None

    train_idx = subsample_indices(
        train_idx, ds.y, getattr(ds, "activities", None), max_train, int(cfg["train"]["seed"])
    )
    val_idx = subsample_indices(
        val_idx,
        ds.y,
        getattr(ds, "activities", None),
        ptr.get("max_val"),
        int(cfg["train"]["seed"]) + 1,
    )
    test_idx = subsample_indices(
        test_idx,
        ds.y,
        getattr(ds, "activities", None),
        max_test,
        int(cfg["train"]["seed"]) + 2,
        min_per_activity=min_per,
    )

    use_pos = bool(ptr.get("use_pos_weight", True)) and num_classes == 2
    labels = [int(ds.y[i]) for i in train_idx]
    cw = class_pos_weight(labels) if use_pos else None

    train_loader = DataLoader(
        Subset(ds, train_idx),
        batch_size=min(int(cfg["train"]["batch_size"]), max(1, len(train_idx))),
        shuffle=True,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        Subset(ds, val_idx),
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"]["num_workers"]),
    )
    test_loader = DataLoader(
        Subset(ds, test_idx),
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"]["num_workers"]),
    )

    use_se = bool(model_kwargs.pop("use_se", False))
    if args.model == "cnn_lstm_attn":
        model = build_model(
            args.model,
            in_channels=in_ch,
            num_classes=num_classes,
            use_se=use_se,
            conv_channels=int(model_kwargs.get("conv_channels", 32)),
            branch_channels=int(model_kwargs.get("branch_channels", 16)),
            lstm_hidden=model_kwargs.get("lstm_hidden", [256, 128]),
            attn_heads=int(model_kwargs.get("attn_heads", 4)),
            dropout_attn=float(model_kwargs.get("dropout_attn", 0.1)),
            dropout_fc=float(model_kwargs.get("dropout_fc", 0.5)),
        )
    else:
        model = build_model(args.model, in_channels=in_ch, num_classes=num_classes)

    ckpt_dir = ensure_dir(ROOT / (args.checkpoint_dir or pcfg.get("paths", {}).get("checkpoint_dir") or cfg["checkpoint_dir"]))
    ckpt_path = ckpt_dir / f"{args.model}_fold{args.fold}.pt"
    epochs = args.epochs or int(ptr.get("epochs") or cfg["train"]["epochs"])

    result = fit_model(
        model,
        train_loader,
        val_loader,
        device=device,
        epochs=epochs,
        lr=float(ptr.get("lr") or cfg["train"]["lr"]),
        weight_decay=float(ptr.get("weight_decay") or cfg["train"]["weight_decay"]),
        label_smoothing=float(ptr.get("label_smoothing") or cfg["train"]["label_smoothing"]),
        patience=int(ptr.get("patience") or cfg["train"]["early_stopping_patience"]),
        mixup_alpha=float(ptr.get("mixup_alpha") or cfg["train"].get("mixup_alpha", 0.0)),
        use_amp=bool(ptr.get("amp", cfg["train"].get("amp", True))),
        checkpoint_path=str(ckpt_path),
        class_weight=cw,
    )

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    val_m = evaluate(model, val_loader, device)
    thr, thr_m = tune_decision_threshold(
        val_m.pop("y_true"),
        val_m.pop("probs"),
        mode=str(ptr.get("decision_threshold_mode", "cost")),
        cost_fn=float(pcfg.get("costs", {}).get("cost_fn", 10.0)),
        cost_fp=float(pcfg.get("costs", {}).get("cost_fp", 1.0)),
    )
    test_m = evaluate(model, test_loader, device, threshold=thr)
    test_m.pop("probs", None)
    test_m.pop("y_true", None)
    out = {
        "fold": args.fold,
        "model": args.model,
        "best_val_f1": result.best_f1,
        "decision_threshold": thr,
        "val_at_threshold": thr_m,
        "test": test_m,
        "checkpoint": str(ckpt_path),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
    }
    out_path = ckpt_dir / f"{args.model}_fold{args.fold}_metrics.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
