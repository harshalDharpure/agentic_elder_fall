#!/usr/bin/env python3
"""36-class Bhatti-style Tier-1 context run on KFall transition windows (not main claim)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score, accuracy_score, classification_report

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_fall.data.kfall import KFallDataset
from agentic_fall.eval.train_loop import fit_model
from agentic_fall.models import build_model
from agentic_fall.utils.config import load_config
from agentic_fall.utils.io import ensure_dir, save_json
from agentic_fall.utils.seed import set_seed


def subject_split(ds: KFallDataset, fold: int, n_folds: int, seed: int, val_fraction: float):
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


@torch.no_grad()
def eval_multiclass(model, loader, device, n_classes: int):
    model.eval()
    ys, preds = [], []
    for batch in loader:
        x = batch["x"].to(device)
        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out
        pred = logits.argmax(dim=-1).cpu().numpy()
        preds.extend(pred.tolist())
        ys.extend(batch["y"].tolist())
    y = np.asarray(ys)
    p = np.asarray(preds)
    macro = float(f1_score(y, p, average="macro", labels=list(range(n_classes)), zero_division=0))
    weighted = float(f1_score(y, p, average="weighted", zero_division=0))
    acc = float(accuracy_score(y, p))
    return {
        "accuracy": acc,
        "macro_f1": macro,
        "weighted_f1": weighted,
        "n": int(len(y)),
        "n_classes": n_classes,
        "report": classification_report(y, p, zero_division=0, output_dict=False),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--max-train", type=int, default=40000)
    ap.add_argument("--max-val", type=int, default=5000)
    ap.add_argument("--max-test", type=int, default=4000)
    args = ap.parse_args()

    kcfg = load_config(ROOT / "configs/tier1_kfall.yaml")
    # Use binary-prepared npz which still has multiclass y
    bin_cfg = load_config(ROOT / "configs/tier1_kfall_binary.yaml")
    set_seed(int(kcfg["train"]["seed"]))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    npz = ROOT / bin_cfg["processed_dir"] / f"windows_transition_w{bin_cfg['window']}_h{bin_cfg['hop']}.npz"
    ds = KFallDataset(npz, binary=False)
    n_classes = int(len(ds.class_names) or int(ds.y.max()) + 1)
    in_ch = int(ds.X.shape[-1])
    tr, va, te = subject_split(
        ds, args.fold, int(kcfg["splits"]["n_folds"]), int(kcfg["train"]["seed"]), 0.15
    )
    rng = np.random.default_rng(kcfg["train"]["seed"])
    if len(tr) > args.max_train:
        tr = list(rng.choice(tr, size=args.max_train, replace=False))
    if len(va) > args.max_val:
        va = list(rng.choice(va, size=args.max_val, replace=False))
    if len(te) > args.max_test:
        te = list(rng.choice(te, size=args.max_test, replace=False))

    ckpt_dir = ensure_dir(ROOT / "checkpoints" / "kfall")
    ckpt = ckpt_dir / f"cnn_lstm_attn_multiclass_fold{args.fold}.pt"
    model = build_model(
        "cnn_lstm_attn",
        in_channels=in_ch,
        num_classes=n_classes,
        conv_channels=int(kcfg["model"]["conv_channels"]),
        branch_channels=int(kcfg["model"]["branch_channels"]),
        lstm_hidden=kcfg["model"]["lstm_hidden"],
        attn_heads=int(kcfg["model"]["attn_heads"]),
    )
    fit_model(
        model,
        DataLoader(Subset(ds, tr), batch_size=256, shuffle=True, num_workers=2),
        DataLoader(Subset(ds, va), batch_size=256, shuffle=False, num_workers=2),
        device=device,
        epochs=args.epochs,
        patience=max(5, args.epochs // 4),
        mixup_alpha=0.0,
        checkpoint_path=str(ckpt),
        lr=float(kcfg["train"]["lr"]),
    )
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device)
    metrics = eval_multiclass(model, DataLoader(Subset(ds, te), batch_size=256, shuffle=False), device, n_classes)
    payload = {
        "dataset": "kfall_multiclass_transition",
        "fold": args.fold,
        "note": "Context only — not a claim against Bhatti published 98% multi-class table",
        "class_names": list(ds.class_names),
        "checkpoint": str(ckpt),
        **{k: v for k, v in metrics.items() if k != "report"},
        "report_text": metrics["report"],
    }
    out = ensure_dir(ROOT / "results" / "kfall") / f"kfall_multiclass_fold{args.fold}.json"
    save_json(payload, out)
    print("macro_f1", metrics["macro_f1"], "weighted_f1", metrics["weighted_f1"], "acc", metrics["accuracy"])
    print("Wrote", out)


if __name__ == "__main__":
    main()
