#!/usr/bin/env python3
"""E1 / G1: evaluate frozen backbones on standard SisFall (natural prevalence).

No D18/D19 force-in. Loads existing checkpoints under checkpoints/backbones/.
Gate G1: primary backbone (and preferably all 9) F1 >= 0.95 under Youden threshold.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_fall.data.sisfall import SisFallDataset
from agentic_fall.eval.protocol import make_fold_split, subsample_indices, tune_decision_threshold
from agentic_fall.eval.train_loop import evaluate
from agentic_fall.models import build_model, count_parameters
from agentic_fall.utils.config import load_config
from agentic_fall.utils.io import ensure_dir, save_json
from agentic_fall.utils.seed import set_seed


PAPER_MODELS = [
    "cnn1d",
    "lstm",
    "cnn_lstm",
    "cnn_lstm_attn",
    "tcn",
    "modern_tcn",
    "tsmixer",
    "patchtst",
    "transformer",
]


def prevalence(y: np.ndarray, idx: list[int]) -> float:
    if not idx:
        return 0.0
    return float(np.mean([int(y[i]) for i in idx]))


def eval_one(
    name: str,
    fold: int,
    ds: SisFallDataset,
    split,
    cfg: dict,
    tcfg: dict,
    bcfg: dict,
    device: torch.device,
) -> dict | None:
    ckpt = ROOT / cfg["paths"]["checkpoint_dir"] / f"{name}_fold{fold}.pt"
    if not ckpt.exists():
        print(f"[skip] missing checkpoint {ckpt}")
        return None

    ev = cfg["eval"]
    test_idx = subsample_indices(
        split.test_idx,
        ds.y,
        ds.activities,
        ev.get("max_test"),
        int(cfg["seed"]) + 2 + fold,
        stratified=bool(ev.get("stratified", False)),
        min_per_activity=None,
    )
    val_idx = subsample_indices(
        split.val_idx,
        ds.y,
        ds.activities,
        min(5000, len(split.val_idx)),
        int(cfg["seed"]) + 1 + fold,
        stratified=True,
        min_per_activity=None,
    )

    kw = dict(bcfg.get("model_kwargs", {}).get(name, {}))
    seq_len = int(tcfg["window"])
    if "seq_len" not in kw and name in ("tsmixer", "patchtst"):
        kw["seq_len"] = seq_len
    model = build_model(name, in_channels=int(tcfg["channels"]), num_classes=2, **kw)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device)

    val_loader = DataLoader(Subset(ds, val_idx), batch_size=256, shuffle=False, num_workers=2)
    test_loader = DataLoader(Subset(ds, test_idx), batch_size=256, shuffle=False, num_workers=2)

    rows = {"name": name, "fold": fold, "n_test": len(test_idx), "prevalence": prevalence(ds.y, test_idx)}
    for mode in ("youden", "cost", "f1"):
        val_m = evaluate(model, val_loader, device)
        thr, _ = tune_decision_threshold(
            val_m.pop("y_true"),
            val_m.pop("probs"),
            mode=mode,
            cost_fn=float(cfg["costs"]["cost_fn"]),
            cost_fp=float(cfg["costs"]["cost_fp"]),
        )
        m = evaluate(model, test_loader, device, threshold=thr)
        m.pop("probs", None)
        m.pop("y_true", None)
        prefix = mode
        rows[f"{prefix}_thr"] = thr
        rows[f"{prefix}_f1"] = m.get("f1")
        rows[f"{prefix}_recall"] = m.get("recall")
        rows[f"{prefix}_specificity"] = m.get("specificity")
        rows[f"{prefix}_precision"] = m.get("precision")
        rows[f"{prefix}_accuracy"] = m.get("accuracy")

    rows["params"] = count_parameters(model)
    # Primary reported column for G1 is Youden (literature-comparable)
    rows["f1"] = rows["youden_f1"]
    rows["recall"] = rows["youden_recall"]
    rows["specificity"] = rows["youden_specificity"]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/standard_sisfall.yaml")
    ap.add_argument("--device", default=None)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--include-harmamba", action="store_true")
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    tcfg = load_config(ROOT / cfg["tier1_config"])
    bcfg = load_config(ROOT / cfg["backbones_config"])
    set_seed(int(cfg["seed"]))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    folds = [int(x) for x in str(args.folds).split(",") if x.strip() != ""]

    npz = ROOT / tcfg["processed_dir"] / f"windows_w{tcfg['window']}_h{tcfg['hop']}.npz"
    if not npz.exists():
        raise FileNotFoundError(f"Missing processed SisFall archive: {npz}")
    ds = SisFallDataset(npz, channels=int(tcfg["channels"]))

    models = list(args.models or cfg.get("models") or PAPER_MODELS)
    if args.include_harmamba and "harmamba" not in models:
        models.append("harmamba")

    out_dir = ensure_dir(ROOT / cfg["paths"]["results_dir"])
    all_rows: list[dict] = []

    for fold in folds:
        split = make_fold_split(
            ds,
            fold=fold,
            n_folds=int(cfg["splits"]["n_folds"]),
            seed=int(cfg["seed"]),
            val_fraction=float(cfg["splits"]["val_fraction"]),
        )
        # Natural prevalence on full test subjects (before cap)
        full_prev = prevalence(ds.y, split.test_idx)
        print(f"\n=== Fold {fold} full-test prevalence={full_prev:.4f} n={len(split.test_idx)} ===")
        for name in models:
            print(f"--- {name} ---")
            row = eval_one(name, fold, ds, split, cfg, tcfg, bcfg, device)
            if row is None:
                continue
            row["full_test_prevalence"] = full_prev
            all_rows.append(row)
            print(
                f"  youden F1={row['youden_f1']:.4f}  "
                f"cost F1={row['cost_f1']:.4f}  "
                f"prev={row['prevalence']:.4f}  n={row['n_test']}"
            )
            save_json(row, out_dir / f"{name}_fold{fold}.json")

    if not all_rows:
        print("No results — check checkpoints.")
        sys.exit(1)

    # Aggregate mean ± std over folds per model (Youden)
    by_model: dict[str, list[dict]] = {}
    for r in all_rows:
        by_model.setdefault(r["name"], []).append(r)

    summary = []
    for name, rows in by_model.items():
        f1s = [float(r["youden_f1"]) for r in rows]
        recs = [float(r["youden_recall"]) for r in rows]
        specs = [float(r["youden_specificity"]) for r in rows]
        summary.append(
            {
                "name": name,
                "n_folds": len(rows),
                "f1_mean": float(np.mean(f1s)),
                "f1_std": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
                "recall_mean": float(np.mean(recs)),
                "specificity_mean": float(np.mean(specs)),
                "g1_pass": float(np.mean(f1s)) >= float(cfg.get("g1_f1_threshold", 0.95)),
            }
        )

    summary.sort(key=lambda x: x["f1_mean"], reverse=True)
    save_json({"rows": all_rows, "summary": summary}, out_dir / "standard_protocol_summary.json")

    csv_path = out_dir / "standard_protocol_summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    # LaTeX stub
    tex = out_dir / "tab_standard_protocol.tex"
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Standard SisFall binary protocol (natural prevalence, subject-independent, Youden threshold). Gate G1: F1 $\ge$ 0.95.}",
        r"\label{tab:standardprotocol}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Backbone & F1 & Recall & Spec. \\",
        r"\midrule",
    ]
    for s in summary:
        mark = r"\textbf{" if s["name"] == cfg.get("primary_backbone") else ""
        end = "}" if mark else ""
        nice = s["name"].replace("_", r"\_")
        f1m, f1s = s["f1_mean"], s["f1_std"]
        lines.append(
            f"{mark}{nice}{end} & "
            f"{f1m:.3f}$\\pm${f1s:.3f} & "
            f"{s['recall_mean']:.3f} & {s['specificity_mean']:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    tex.write_text("\n".join(lines))

    print("\n=== G1 SUMMARY (Youden F1) ===")
    thr = float(cfg.get("g1_f1_threshold", 0.95))
    primary = cfg.get("primary_backbone", "cnn_lstm_attn")
    primary_ok = False
    for s in summary:
        flag = "PASS" if s["g1_pass"] else "FAIL"
        print(f"  {s['name']:16s}  F1={s['f1_mean']:.4f}±{s['f1_std']:.4f}  [{flag}]")
        if s["name"] == primary:
            primary_ok = s["g1_pass"]
    print(f"\nPrimary ({primary}) G1: {'PASS' if primary_ok else 'FAIL'} (threshold {thr})")
    print(f"Wrote {out_dir / 'standard_protocol_summary.json'}")
    print(f"Wrote {tex}")
    sys.exit(0 if primary_ok else 2)


if __name__ == "__main__":
    main()
