#!/usr/bin/env python3
"""Fair backbone zoo benchmark under paper_protocol.yaml."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
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
)
from agentic_fall.data.sisfall import SisFallDataset
from agentic_fall.eval.metrics import agentic_metrics, metrics_to_latex
from agentic_fall.eval.protocol import (
    build_gate,
    make_fold_split,
    subsample_indices,
    text_embedder,
    tune_decision_threshold,
    verify_dataset_near_falls,
)
from agentic_fall.eval.train_loop import class_pos_weight, evaluate, fit_model
from agentic_fall.models import build_model, count_parameters
from agentic_fall.utils.config import load_config
from agentic_fall.utils.io import ensure_dir, save_json
from agentic_fall.utils.seed import set_seed


@torch.no_grad()
def measure_latency_ms(model, device, shape=(1, 6, 90), warmup=10, repeats=50) -> float:
    model.eval()
    x = torch.randn(*shape, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / repeats


def run_agentic_on_backbone(model, ds, test_idx, memory_path, device, acfg, backend, gate):
    memory = KNNMemory(k=int(acfg["retrieval"]["k"]))
    if Path(memory_path).exists():
        memory.load(memory_path)
    reasoner = LLMReasoner(backend=backend, fallback_heuristic=True)
    pipe = AgenticPipeline(
        model=model,
        gate=gate,
        memory=memory,
        reasoner=reasoner,
        action_agent=ActionAgent(),
        device=device,
        sample_rate_hz=float(acfg["evidence"]["sample_rate_hz"]),
        text_embed_fn=text_embedder(acfg["retrieval"]["embedder"]),
    )
    y_true, y_pred, esc, acts, lats, ps = [], [], [], [], [], []
    for batch in tqdm(DataLoader(Subset(ds, test_idx), batch_size=1), desc="agentic"):
        x = batch["x"][0]
        res = pipe.run(x, x.numpy().T, activity=str(batch["activity"][0]))
        y_true.append(int(batch["y"][0]))
        y_pred.append(1 if res.prediction == "fall" else 0)
        esc.append(res.escalated)
        acts.append(str(batch["activity"][0]))
        lats.append(res.latency_ms)
        ps.append(res.p_fall)
    return agentic_metrics(
        y_true,
        y_pred,
        esc,
        activities=acts,
        ambiguous_codes=list(acfg["pipeline"]["ambiguous_adl_codes"]),
        latencies_ms=lats,
        p_falls=ps,
        cost_fn=float(acfg["gate"]["cost_fn"]),
        cost_fp=float(acfg["gate"]["cost_fp"]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-config", default="configs/paper_protocol.yaml")
    ap.add_argument("--device", default=None)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--skip-agentic", action="store_true")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--memory", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    pcfg = load_config(ROOT / args.paper_config)
    tcfg = load_config(ROOT / pcfg["tier1_config"])
    acfg = load_config(ROOT / pcfg["agentic_config"])
    bcfg = load_config(ROOT / pcfg["backbones_config"])
    set_seed(int(pcfg["seed"]))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    fold = int(args.fold if args.fold is not None else 0)

    npz = ROOT / tcfg["processed_dir"] / f"windows_w{tcfg['window']}_h{tcfg['hop']}.npz"
    ds = SisFallDataset(npz, channels=int(tcfg["channels"]))
    verify_dataset_near_falls(ds, pcfg["eval"]["required_activities"])

    split = make_fold_split(
        ds,
        fold=fold,
        n_folds=int(pcfg["splits"]["n_folds"]),
        seed=int(pcfg["seed"]),
        val_fraction=float(pcfg["splits"]["val_fraction"]),
    )
    tr = pcfg["train"]
    ev = pcfg["eval"]
    min_nf = int(ev.get("min_near_fall_per_code", 0) or 0)
    min_per = {c: min_nf for c in ev["required_activities"]} if min_nf else None
    train_idx = subsample_indices(split.train_idx, ds.y, ds.activities, tr.get("max_train"), pcfg["seed"])
    val_idx = subsample_indices(split.val_idx, ds.y, ds.activities, tr.get("max_val"), pcfg["seed"] + 1)
    test_idx = subsample_indices(
        split.test_idx, ds.y, ds.activities, ev.get("max_test"), pcfg["seed"] + 2, min_per_activity=min_per
    )

    ckpt_dir = ensure_dir(ROOT / pcfg["paths"]["checkpoint_dir"])
    out_dir = ensure_dir(ROOT / (args.out_dir or f"{pcfg['paths']['results_dir']}/fold{fold}"))
    models = args.models or list(pcfg["models"])
    model_kwargs = bcfg.get("model_kwargs", {})
    in_ch = int(tcfg["channels"])
    seq_len = int(tcfg["window"])

    detector_rows = []
    trained = {}

    for name in models:
        print(f"\n=== Training {name} fold={fold} ===")
        kw = dict(model_kwargs.get(name, {}))
        if "seq_len" not in kw and name in ("tsmixer", "patchtst"):
            kw["seq_len"] = seq_len
        model = build_model(name, in_channels=in_ch, num_classes=2, **kw)
        labels = [int(ds.y[i]) for i in train_idx]
        cw = class_pos_weight(labels) if tr.get("use_pos_weight", True) else None
        train_loader = DataLoader(
            Subset(ds, train_idx),
            batch_size=min(int(tr["batch_size"]), len(train_idx)),
            shuffle=True,
            num_workers=2,
        )
        val_loader = DataLoader(
            Subset(ds, val_idx),
            batch_size=min(int(tr["batch_size"]), len(val_idx)),
            shuffle=False,
            num_workers=2,
        )
        test_loader = DataLoader(
            Subset(ds, test_idx),
            batch_size=min(int(tr["batch_size"]), len(test_idx)),
            shuffle=False,
            num_workers=2,
        )
        ckpt = ckpt_dir / f"{name}_fold{fold}.pt"
        fit_model(
            model,
            train_loader,
            val_loader,
            device=device,
            epochs=int(tr["epochs"]),
            lr=float(tr["lr"]),
            weight_decay=float(tr["weight_decay"]),
            label_smoothing=float(tr["label_smoothing"]),
            patience=int(tr["patience"]),
            mixup_alpha=float(tr["mixup_alpha"]),
            use_amp=bool(tr.get("amp", True)),
            checkpoint_path=str(ckpt),
            class_weight=cw,
        )
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        model.to(device)
        val_m = evaluate(model, val_loader, device)
        thr, _ = tune_decision_threshold(
            val_m.pop("y_true"),
            val_m.pop("probs"),
            mode=str(tr.get("decision_threshold_mode", "cost")),
            cost_fn=float(pcfg["costs"]["cost_fn"]),
            cost_fp=float(pcfg["costs"]["cost_fp"]),
        )
        metrics = evaluate(model, test_loader, device, threshold=thr)
        metrics.pop("probs", None)
        metrics.pop("y_true", None)
        lat = measure_latency_ms(
            model,
            device,
            shape=(1, in_ch, seq_len),
            warmup=int(pcfg["latency"]["warmup"]),
            repeats=int(pcfg["latency"]["repeats"]),
        )
        row = {
            "name": name,
            "params": count_parameters(model),
            "latency_ms": lat,
            "mode": "detector",
            "decision_threshold": thr,
            **metrics,
        }
        detector_rows.append(row)
        trained[name] = model
        print(json.dumps(row, indent=2))

    agentic_rows = []
    if not args.skip_agentic and trained:
        ranked = sorted(detector_rows, key=lambda r: r.get("f1", 0.0), reverse=True)
        top_n = int(pcfg["agentic"]["top_n"])
        top = [r["name"] for r in ranked[:top_n]]
        memory_path = args.memory or str(
            ROOT / pcfg["paths"]["memory_dir"] / f"sisfall_fold{fold}.json"
        )
        agentic_test = subsample_indices(
            test_idx,
            ds.y,
            ds.activities,
            ev.get("agentic_max_samples"),
            pcfg["seed"] + 3,
            min_per_activity=min_per,
        )
        for name in top:
            print(f"\n=== Agentic wrap: {name} ===")
            model = trained[name]
            gate = build_gate(
                acfg,
                model=model,
                ds=ds,
                cal_idx=train_idx,
                device=device,
                calibrate=bool(ev.get("calibrate_gate", True)),
                max_cal=int(ev.get("max_cal", 2000)),
            )
            am = run_agentic_on_backbone(
                model,
                ds,
                agentic_test,
                memory_path,
                device,
                acfg,
                backend=pcfg["agentic"]["backend"],
                gate=gate,
            )
            row = {
                "name": f"{name}+agentic",
                "mode": "agentic",
                "backbone": name,
                "tau_low": gate.tau_low,
                "tau_high": gate.tau_high,
                **am,
            }
            det = next(r for r in detector_rows if r["name"] == name)
            row["detector_f1"] = det["f1"]
            row["delta_f1"] = float(am.get("f1", 0) - det["f1"])
            agentic_rows.append(row)
            print(
                {
                    k: row[k]
                    for k in (
                        "f1",
                        "delta_f1",
                        "expected_response_cost",
                        "escalation_rate",
                        "false_alarm_rate_near_fall",
                        "n_near_fall",
                    )
                    if k in row
                }
            )

    all_rows = detector_rows + agentic_rows
    csv_path = out_dir / f"backbone_compare_fold{fold}.csv"
    keys = sorted({k for r in all_rows for k in r})
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(all_rows)
    (out_dir / f"backbone_compare_fold{fold}.tex").write_text(
        metrics_to_latex(detector_rows, caption="Backbone zoo (detector-only, SisFall)")
    )
    if agentic_rows:
        (out_dir / f"backbone_agentic_fold{fold}.tex").write_text(
            metrics_to_latex(agentic_rows, caption="Top backbones + agentic wrapper")
        )
    save_json(all_rows, out_dir / f"backbone_compare_fold{fold}.json")
    # also mirror to results/ for export_tables compatibility
    mirror = ensure_dir(ROOT / pcfg["paths"]["results_dir"])
    save_json(all_rows, mirror / f"backbone_compare_fold{fold}.json")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
