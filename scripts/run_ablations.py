#!/usr/bin/env python3
"""Run baseline + ablation suite under the locked paper protocol."""
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

from agentic_fall.agents import (
    ActionAgent,
    AgenticPipeline,
    ConfidenceGate,
    KNNMemory,
    LLMReasoner,
)
from agentic_fall.data.sisfall import SisFallDataset
from agentic_fall.eval.metrics import agentic_metrics, binary_metrics, metrics_to_latex
from agentic_fall.eval.protocol import (
    build_gate,
    make_fold_split,
    subsample_indices,
    text_embedder,
    verify_dataset_near_falls,
)
from agentic_fall.eval.train_loop import ThresholdBaseline, class_pos_weight, evaluate, fit_model
from agentic_fall.models import build_model
from agentic_fall.utils.config import load_config
from agentic_fall.utils.io import ensure_dir, save_json
from agentic_fall.utils.seed import set_seed


def eval_threshold(ds, test_idx, thr=2.5):
    base = ThresholdBaseline(thr)
    loader = DataLoader(Subset(ds, test_idx), batch_size=64, shuffle=False)
    ys, preds = [], []
    for batch in loader:
        preds.extend(base.predict_batch(batch["x"]))
        ys.extend(batch["y"].tolist())
    return binary_metrics(ys, preds)


def train_and_eval_model(name, ds, train_idx, val_idx, test_idx, in_ch, device, epochs, ckpt, use_pos_weight):
    model = build_model(name, in_channels=in_ch, num_classes=2)
    cw = class_pos_weight([int(ds.y[i]) for i in train_idx]) if use_pos_weight else None
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=256, shuffle=True, num_workers=2)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=256, shuffle=False, num_workers=2)
    test_loader = DataLoader(Subset(ds, test_idx), batch_size=256, shuffle=False, num_workers=2)
    fit_model(
        model,
        train_loader,
        val_loader,
        device=device,
        epochs=epochs,
        patience=max(5, epochs // 4),
        mixup_alpha=0.1,
        checkpoint_path=str(ckpt),
        class_weight=cw,
    )
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    m = evaluate(model, test_loader, device)
    m.pop("probs", None)
    m.pop("y_true", None)
    return m, model


def run_agentic_variant(model, ds, test_idx, memory_path, device, acfg, mode, gate):
    memory = KNNMemory(k=int(acfg["retrieval"]["k"]))
    if Path(memory_path).exists() and mode in ("gate_knn", "gate_knn_llm", "gate_llm"):
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
    action_agent = ActionAgent()
    embed = text_embedder(acfg["retrieval"]["embedder"]) if use_knn else None

    local_gate = gate
    if mode == "tier1_only":
        local_gate = ConfidenceGate(0.5, 0.5 + 1e-6)

    pipe = AgenticPipeline(
        model=model,
        gate=local_gate,
        memory=memory if use_knn else KNNMemory(),
        reasoner=reasoner,
        action_agent=action_agent,
        device=device,
        sample_rate_hz=float(acfg["evidence"]["sample_rate_hz"]),
        text_embed_fn=embed,
    )

    y_true, y_pred, esc, acts, lats, ps = [], [], [], [], [], []
    loader = DataLoader(Subset(ds, test_idx), batch_size=1, shuffle=False)
    for batch in tqdm(loader, desc=mode):
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
        ambiguous_codes=list(acfg["pipeline"]["ambiguous_adl_codes"]),
        latencies_ms=lats,
        p_falls=ps,
        cost_fn=float(acfg["gate"]["cost_fn"]),
        cost_fp=float(acfg["gate"]["cost_fp"]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-config", default="configs/paper_protocol.yaml")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-test", type=int, default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--memory", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument(
        "--modes",
        default=None,
        help="Comma-separated ablation modes (default: all). "
        "Example: tier1_only,gate_only,gate_knn,gate_knn_llm",
    )
    ap.add_argument(
        "--protocol-tag",
        default=None,
        help="Optional protocol label written into ablation JSON (e.g. full_4000, fast_1000_no_gate_llm).",
    )
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    pcfg = load_config(ROOT / args.paper_config)
    tcfg = load_config(ROOT / pcfg["tier1_config"])
    acfg = load_config(ROOT / pcfg["agentic_config"])
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
    max_test = args.max_test if args.max_test is not None else ev.get("max_test")
    min_nf = int(ev.get("min_near_fall_per_code", 0) or 0)
    min_per = {c: min_nf for c in ev["required_activities"]} if min_nf else None
    train_idx = subsample_indices(split.train_idx, ds.y, ds.activities, tr.get("max_train"), pcfg["seed"])
    val_idx = subsample_indices(split.val_idx, ds.y, ds.activities, tr.get("max_val"), pcfg["seed"] + 1)
    test_idx = subsample_indices(
        split.test_idx, ds.y, ds.activities, max_test, pcfg["seed"] + 2, min_per_activity=min_per
    )

    out_dir = ensure_dir(ROOT / (args.out_dir or f"{pcfg['paths']['results_dir']}/fold{args.fold}"))
    rows = []
    epochs = args.epochs or int(tr["epochs"])
    use_pos = bool(tr.get("use_pos_weight", True))

    if not args.skip_baselines:
        m = eval_threshold(ds, test_idx)
        rows.append({"name": "threshold", **m})
        print("threshold", m)
        for name in ("cnn1d", "lstm", "cnn_lstm"):
            ckpt = out_dir / f"ablation_{name}_fold{args.fold}.pt"
            m, _ = train_and_eval_model(
                name, ds, train_idx, val_idx, test_idx, int(tcfg["channels"]), device, epochs, ckpt, use_pos
            )
            rows.append({"name": name, **m})
            print(name, m)

    primary = pcfg["agentic"].get("primary_backbone", "cnn_lstm_attn")
    ckpt = Path(args.checkpoint) if args.checkpoint else ROOT / pcfg["paths"]["checkpoint_dir"] / f"{primary}_fold{args.fold}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Missing primary checkpoint {ckpt}. Train backbone zoo first or pass --checkpoint."
        )
    model = build_model(primary, in_channels=int(tcfg["channels"]), num_classes=2)
    # rebuild with kwargs if cnn_lstm_attn
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
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device)
    test_loader = DataLoader(Subset(ds, test_idx), batch_size=256, shuffle=False)
    m = evaluate(model, test_loader, device)
    m.pop("probs", None)
    m.pop("y_true", None)
    rows.append({"name": primary, **m})

    memory_path = args.memory or str(ROOT / pcfg["paths"]["memory_dir"] / f"sisfall_fold{args.fold}.json")
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

    default_modes = ("tier1_only", "gate_only", "gate_knn", "gate_llm", "gate_knn_llm")
    if args.modes:
        modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())
        unknown = [m for m in modes if m not in default_modes]
        if unknown:
            raise SystemExit(f"Unknown ablation modes: {unknown}. Allowed: {default_modes}")
    else:
        modes = default_modes

    protocol_tag = args.protocol_tag or (
        f"max_test_{max_test or 'all'}_modes_" + "_".join(modes)
    )
    for mode in modes:
        am = run_agentic_variant(model, ds, test_idx, memory_path, device, acfg, mode, gate)
        am["protocol_tag"] = protocol_tag
        am["max_test"] = int(len(test_idx))
        rows.append({"name": mode, **am})
        print(
            mode,
            {
                k: am[k]
                for k in (
                    "f1",
                    "recall",
                    "escalation_rate",
                    "expected_response_cost",
                    "false_alarm_rate_near_fall",
                    "n_near_fall",
                )
                if k in am
            },
        )

    csv_path = out_dir / f"ablation_fold{args.fold}.csv"
    keys = sorted({k for r in rows for k in r.keys()})
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    latex = metrics_to_latex(rows, caption="Baselines and agentic ablations (SisFall)")
    (out_dir / f"ablation_fold{args.fold}.tex").write_text(latex)
    payload = {
        "protocol_tag": protocol_tag,
        "fold": args.fold,
        "max_test": int(len(test_idx)),
        "modes": list(modes),
        "rows": rows,
    }
    save_json(payload, out_dir / f"ablation_fold{args.fold}.json")
    save_json(payload, ROOT / pcfg["paths"]["results_dir"] / f"ablation_fold{args.fold}.json")
    # sidecar for aggregators / paper footnotes
    save_json(
        {
            "fold": args.fold,
            "protocol_tag": protocol_tag,
            "max_test": int(len(test_idx)),
            "modes": list(modes),
        },
        out_dir / f"protocol_fold{args.fold}.json",
    )
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
