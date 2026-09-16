#!/usr/bin/env python3
"""Run baseline + ablation suite under the locked paper protocol."""
from __future__ import annotations

import argparse
import csv
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
    ConfidenceGate,
    KNNMemory,
    LLMReasoner,
    build_adjudicator_from_config,
)
from agentic_fall.data.sisfall import SisFallDataset
from agentic_fall.eval.metrics import agentic_metrics, binary_metrics, crc_veto_metrics, metrics_to_latex
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


ADJUDICATION_MODES = {
    "gate_knn_actor_only": {
        "use_knn": True,
        "use_llm": False,
        "adj_mode": "actor_only",
        "role_swap": False,
        "actor_backend": "heuristic",
        "critic_backend": "heuristic",
    },
    "gate_knn_actor_critic": {
        "use_knn": True,
        "use_llm": False,
        "adj_mode": "actor_critic",
        "role_swap": False,
        "actor_backend": "heuristic",
        "critic_backend": "heuristic",
    },
    "gate_knn_actor_critic_roleswap": {
        "use_knn": True,
        "use_llm": False,
        "adj_mode": "actor_critic_roleswap",
        "role_swap": True,
        "actor_backend": "heuristic",
        "critic_backend": "heuristic",
    },
    "gate_knn_llm_actor_only": {
        "use_knn": True,
        "use_llm": True,
        "adj_mode": "actor_only",
        "role_swap": False,
        "actor_backend": "ollama",
        "critic_backend": "heuristic",
    },
    # Primary: LLM Actor + constraint Critic (1 LLM call; AutoSCAR-style grounding)
    "gate_knn_llm_actor_critic": {
        "use_knn": True,
        "use_llm": True,
        "adj_mode": "actor_critic",
        "role_swap": False,
        "actor_backend": "ollama",
        "critic_backend": "heuristic",
    },
    # Full LLM Actor+Critic (2 LLM calls)
    "gate_knn_llm_actor_critic_full": {
        "use_knn": True,
        "use_llm": True,
        "adj_mode": "actor_critic",
        "role_swap": False,
        "actor_backend": "ollama",
        "critic_backend": "ollama",
    },
    "gate_knn_llm_actor_critic_roleswap": {
        "use_knn": True,
        "use_llm": True,
        "adj_mode": "actor_critic_roleswap",
        "role_swap": True,
        "actor_backend": "ollama",
        "critic_backend": "heuristic",
    },
    # Q1-safe: same LLM Actor as gate_knn_llm; Critic only action + rationale
    "gate_knn_llm_action_critique": {
        "use_knn": True,
        "use_llm": True,
        "adj_mode": "action_critique",
        "role_swap": False,
        "actor_backend": "ollama",
        "critic_backend": "heuristic",
        "freeze_label": True,
        "screen_confident_fall": True,
    },
    # Contrastive Critic: hard-negative ADLs + action critique on confident falls too
    "gate_knn_llm_contrastive_action_critique": {
        "use_knn": True,
        "use_llm": True,
        "adj_mode": "contrastive_action_critique",
        "role_swap": False,
        "actor_backend": "ollama",
        "critic_backend": "heuristic",
        "freeze_label": True,
        "use_contrastive": True,
        "screen_confident_fall": True,
        "k_pos": 3,
        "k_neg": 3,
    },
    # CRC-calibrated one-directional veto (fall→ADL only when certified)
    "gate_knn_llm_crc_veto": {
        "use_knn": True,
        "use_llm": True,
        "adj_mode": "crc_veto",
        "role_swap": False,
        "actor_backend": "ollama",
        "critic_backend": "heuristic",
        "freeze_label": False,
        "crc": True,
    },
}


def run_agentic_variant(model, ds, test_idx, memory_path, device, acfg, mode, gate):
    adj_spec = ADJUDICATION_MODES.get(mode)
    memory = KNNMemory(k=int(acfg["retrieval"]["k"]))
    knn_modes = ("gate_knn", "gate_knn_llm", "gate_llm") + tuple(ADJUDICATION_MODES)
    if Path(memory_path).exists() and mode in knn_modes:
        memory.load(memory_path)

    if adj_spec is not None:
        use_llm = bool(adj_spec["use_llm"])
        use_knn = bool(adj_spec["use_knn"])
    else:
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

    adjudicator = None
    allow_down = bool((acfg.get("adjudication") or {}).get("allow_action_downgrade", True))
    if adj_spec is not None:
        cfg = dict(acfg)
        cfg["adjudication"] = {
            **(acfg.get("adjudication") or {}),
            "enabled": True,
            "mode": adj_spec["adj_mode"],
            "role_swap": adj_spec["role_swap"],
            "actor_backend": adj_spec.get("actor_backend", backend),
            "critic_backend": adj_spec.get("critic_backend", backend),
            "freeze_label": adj_spec.get("freeze_label", True),
            "veto_threshold": (acfg.get("adjudication") or {}).get("veto_threshold"),
            "crc_feasible": (acfg.get("adjudication") or {}).get("crc_feasible", False),
            "crc_alpha": (acfg.get("adjudication") or {}).get("crc_alpha"),
            "screen_confident_fall": adj_spec.get(
                "screen_confident_fall",
                (acfg.get("adjudication") or {}).get("screen_confident_fall", True),
            ),
            "use_contrastive": adj_spec.get("use_contrastive", False),
            "k_pos": adj_spec.get("k_pos", 3),
            "k_neg": adj_spec.get("k_neg", 3),
        }
        adjudicator = build_adjudicator_from_config(cfg, reasoner)

    pipe = AgenticPipeline(
        model=model,
        gate=local_gate,
        memory=memory if use_knn else KNNMemory(),
        reasoner=reasoner,
        action_agent=action_agent,
        device=device,
        sample_rate_hz=float(acfg["evidence"]["sample_rate_hz"]),
        text_embed_fn=embed,
        adjudicator=adjudicator,
        allow_action_downgrade=allow_down,
    )

    y_true, y_pred, esc, acts, lats, ps = [], [], [], [], [], []
    resp_actions, rationales = [], []
    pred_before, vetoed_flags, windows = [], [], []
    n_adj, n_reject, n_confirm, n_revise, llm_calls = 0, 0, 0, 0, 0
    loader = DataLoader(Subset(ds, test_idx), batch_size=1, shuffle=False)
    for batch in tqdm(loader, desc=mode):
        x = batch["x"][0]
        res = pipe.run(x, x.numpy().T, activity=str(batch["activity"][0]))
        if mode == "gate_only" and res.escalated:
            yp = 1 if res.p_fall >= 0.5 else 0
        else:
            yp = 1 if res.prediction == "fall" else 0
        actor = res.actor_prediction or res.prediction
        y_true.append(int(batch["y"][0]))
        y_pred.append(yp)
        pred_before.append(1 if actor == "fall" else 0)
        vetoed_flags.append(bool(res.vetoed))
        esc.append(res.escalated)
        acts.append(str(batch["activity"][0]))
        lats.append(res.latency_ms)
        ps.append(res.p_fall)
        resp_actions.append(str(res.action.action))
        rationales.append(str(res.rationale))
        windows.append(
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
        if res.adjudicated:
            n_adj += 1
            if res.critic_verdict == "reject":
                n_reject += 1
            elif res.critic_verdict == "confirm":
                n_confirm += 1
            elif res.critic_verdict == "revise":
                n_revise += 1
        llm_calls += int(res.num_llm_calls)
    metrics = agentic_metrics(
        y_true,
        y_pred,
        esc,
        activities=acts,
        ambiguous_codes=list(acfg["pipeline"]["ambiguous_adl_codes"]),
        latencies_ms=lats,
        p_falls=ps,
        cost_fn=float(acfg["gate"]["cost_fn"]),
        cost_fp=float(acfg["gate"]["cost_fp"]),
        actions=resp_actions,
        rationales=rationales,
    )
    adj_cfg = acfg.get("adjudication") or {}
    crc = crc_veto_metrics(
        y_true,
        pred_before,
        y_pred,
        vetoed=vetoed_flags,
        certified_alpha=adj_cfg.get("crc_alpha"),
        lambda_star=adj_cfg.get("veto_threshold"),
        feasible=bool(adj_cfg.get("crc_feasible")),
        aurc=adj_cfg.get("aurc"),
    )
    metrics.update(crc)
    metrics["n_adjudicated"] = n_adj
    metrics["critic_reject"] = n_reject
    metrics["critic_confirm"] = n_confirm
    metrics["critic_revise"] = n_revise
    metrics["total_llm_calls"] = llm_calls
    metrics["mean_llm_calls_escalated"] = float(llm_calls / max(1, n_adj))
    return metrics, windows


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
    ap.add_argument(
        "--out-suffix",
        default="",
        help="Optional suffix for ablation outputs, e.g. '_gate_llm' or '_full4000'. "
        "When empty, writes ablation_fold{N}.json as usual.",
    )
    ap.add_argument(
        "--crc-cal",
        default=None,
        help="Path to veto_calibration_foldN.json (required for gate_knn_llm_crc_veto).",
    )
    ap.add_argument("--crc-alpha", type=float, default=None)
    ap.add_argument(
        "--merge-existing",
        action="store_true",
        help="Merge new mode rows into existing ablation_fold{N}.json (by name) before saving. "
        "Use with --modes gate_llm to add the missing ladder step without wiping other modes.",
    )
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
    else:
        bcfg = load_config(ROOT / pcfg.get("backbones_config", "configs/backbones.yaml"))
        from agentic_fall.models import kwargs_for_model

        zoo_kw = kwargs_for_model(primary, bcfg)
        if zoo_kw:
            model = build_model(primary, in_channels=int(tcfg["channels"]), num_classes=2, **zoo_kw)
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
    allowed_modes = default_modes + tuple(ADJUDICATION_MODES.keys())
    if args.modes:
        modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())
        unknown = [m for m in modes if m not in allowed_modes]
        if unknown:
            raise SystemExit(f"Unknown ablation modes: {unknown}. Allowed: {allowed_modes}")
    else:
        modes = default_modes

    protocol_tag = args.protocol_tag or (
        f"max_test_{max_test or 'all'}_modes_" + "_".join(modes)
    )

    if "gate_knn_llm_crc_veto" in modes:
        cal_path = Path(
            args.crc_cal
            or (out_dir / f"veto_calibration_fold{args.fold}.json")
        )
        if not cal_path.exists():
            raise SystemExit(
                f"CRC veto needs {cal_path}. Run scripts/calibrate_crc_veto.py --fold {args.fold} first."
            )
        cal = json.loads(cal_path.read_text())
        alpha = args.crc_alpha if args.crc_alpha is not None else float(
            (acfg.get("adjudication") or {}).get("crc_alpha", 0.05)
        )
        sel = (cal.get("by_alpha") or {}).get(f"{alpha:.2f}") or cal.get("selection") or {}
        acfg.setdefault("adjudication", {})
        acfg["adjudication"]["crc_alpha"] = alpha
        acfg["adjudication"]["crc_feasible"] = bool(sel.get("feasible"))
        acfg["adjudication"]["veto_threshold"] = sel.get("lambda_star")
        acfg["adjudication"]["aurc"] = cal.get("aurc")
        print(
            f"CRC fold={args.fold} alpha={alpha} feasible={sel.get('feasible')} "
            f"lambda_star={sel.get('lambda_star')} U={sel.get('U')}"
        )

    for mode in modes:
        am, windows = run_agentic_variant(model, ds, test_idx, memory_path, device, acfg, mode, gate)
        am["protocol_tag"] = protocol_tag
        am["max_test"] = int(len(test_idx))
        rows.append({"name": mode, **am})
        if mode == "gate_knn_llm_crc_veto":
            save_json(
                {"fold": args.fold, "windows": windows, "metrics": am},
                out_dir / f"crc_veto_windows_fold{args.fold}.json",
            )
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
                    "false_alarms_per_1000",
                    "n_near_fall",
                    "n_adjudicated",
                    "critic_reject",
                    "critic_confirm",
                    "critic_revise",
                    "total_llm_calls",
                    "graded_action_cost",
                    "emergency_rate_adl",
                    "emergency_rate_near_fall",
                    "rationale_grounding_rate",
                    "veto_rate",
                    "veto_precision",
                    "induced_fn",
                    "lambda_star",
                    "certified_alpha",
                    "crc_feasible",
                )
                if k in am
            },
        )

    suffix = args.out_suffix or ""
    main_json = out_dir / f"ablation_fold{args.fold}{suffix}.json"
    if args.merge_existing:
        # Prefer merging into the canonical (unsuffixed) ablation file when present.
        merge_path = out_dir / f"ablation_fold{args.fold}.json"
        if merge_path.exists():
            existing = json.loads(merge_path.read_text())
            old_rows = existing.get("rows", existing) if isinstance(existing, dict) else existing
            by_name = {r["name"]: r for r in old_rows if isinstance(r, dict) and "name" in r}
            for r in rows:
                by_name[r["name"]] = r
            # Preserve a stable ladder order when possible.
            order = [
                "threshold",
                "cnn1d",
                "lstm",
                "cnn_lstm",
                primary,
                "tier1_only",
                "gate_only",
                "gate_knn",
                "gate_llm",
                "gate_knn_llm",
                "gate_knn_actor_only",
                "gate_knn_actor_critic",
                "gate_knn_actor_critic_roleswap",
                "gate_knn_llm_actor_only",
                "gate_knn_llm_actor_critic",
                "gate_knn_llm_actor_critic_roleswap",
                "gate_knn_llm_action_critique",
                "gate_knn_llm_contrastive_action_critique",
                "gate_knn_llm_crc_veto",
            ]
            merged = [by_name[n] for n in order if n in by_name]
            for n, r in by_name.items():
                if n not in {x["name"] for x in merged}:
                    merged.append(r)
            rows = merged
            # Update protocol tag to reflect newly included modes.
            protocol_tag = args.protocol_tag or (
                (existing.get("protocol_tag") if isinstance(existing, dict) else None)
                or protocol_tag
            )
            if "gate_llm" in {r["name"] for r in rows} and "no_gate_llm" in str(protocol_tag):
                protocol_tag = str(protocol_tag).replace("no_gate_llm", "with_gate_llm")
            main_json = merge_path
            suffix = ""

    csv_path = out_dir / f"ablation_fold{args.fold}{suffix}.csv"
    keys = sorted({k for r in rows for k in r.keys()})
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    latex = metrics_to_latex(rows, caption="Baselines and agentic ablations (SisFall)")
    (out_dir / f"ablation_fold{args.fold}{suffix}.tex").write_text(latex)
    mode_names = [r["name"] for r in rows if r.get("name") in allowed_modes]
    payload = {
        "protocol_tag": protocol_tag,
        "fold": args.fold,
        "max_test": int(len(test_idx)),
        "modes": mode_names or list(modes),
        "rows": rows,
    }
    save_json(payload, main_json)
    save_json(payload, ROOT / pcfg["paths"]["results_dir"] / main_json.name)
    # Always keep a sidecar copy of newly computed modes (useful before merge).
    if suffix or args.merge_existing:
        save_json(
            {
                "protocol_tag": protocol_tag,
                "fold": args.fold,
                "max_test": int(len(test_idx)),
                "modes": list(modes),
                "rows": [r for r in rows if r.get("name") in modes],
            },
            out_dir / f"ablation_fold{args.fold}_modes_{'_'.join(modes)}.json",
        )
    # sidecar for aggregators / paper footnotes
    save_json(
        {
            "fold": args.fold,
            "protocol_tag": protocol_tag,
            "max_test": int(len(test_idx)),
            "modes": payload["modes"],
        },
        out_dir / f"protocol_fold{args.fold}.json",
    )
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
