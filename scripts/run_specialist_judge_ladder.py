#!/usr/bin/env python3
"""Beat-Bhatti ladder: tool-grounded near-fall specialist + val-tuned fusion.

No activity codes. Features = Tier-1 p_fall + kNN vote + biomechanics.
Train specialist on train windows; calibrate blend/threshold/judge on val.
Writes results/noleak_lock/specialist/fold{N}/specialist_fold{N}.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_fall.agents.knn_memory import KNNMemory
from agentic_fall.agents.specialist_judge import (
    FEATURE_NAMES,
    SpecialistJudge,
    calibrate_judge_threshold,
    featurize_row,
    train_specialist,
)
from agentic_fall.data.sisfall import SisFallDataset
from agentic_fall.eval.metrics import agentic_metrics
from agentic_fall.eval.protocol import (
    build_gate,
    make_fold_split,
    subsample_indices,
    text_embedder,
    verify_dataset_near_falls,
)
from agentic_fall.features.biomechanics import extract_biomechanics
from agentic_fall.models import build_model
from agentic_fall.utils.config import load_config
from agentic_fall.utils.io import ensure_dir, save_json
from agentic_fall.utils.seed import set_seed


@torch.no_grad()
def collect_rows(model, ds, idx, device, gate, memory, embed, acfg, *, batch_size: int = 64):
    """Collect per-window features / tier1 / escalation (no LLM, no activity)."""
    model.eval()
    sr = float(acfg["evidence"]["sample_rate_hz"])
    ff_thr = float(acfg["evidence"]["freefall_g_threshold"])
    st_thr = float(acfg["evidence"]["stillness_var_threshold"])
    rows = []
    loader = DataLoader(Subset(ds, idx), batch_size=batch_size, shuffle=False)
    from agentic_fall.agents.evidence import serialize_evidence

    for batch in tqdm(loader, desc="collect", leave=False):
        x = batch["x"].to(device)
        if x.dim() == 3 and x.shape[1] != int(x.shape[-1] if False else x.shape[1]):
            # SisFallDataset yields (C, T); model expects (B, C, T)
            pass
        logits, _emb = model(x)
        probs = F.softmax(logits, dim=-1)
        p_falls = probs[:, 1].detach().cpu().numpy()
        xs_cpu = batch["x"].cpu().numpy()
        ys = batch["y"].cpu().numpy().astype(int)
        acts = [str(a) for a in batch["activity"]]

        for i in range(len(ys)):
            p_fall = float(p_falls[i])
            gate_d = gate.decide(p_fall)
            escalated = gate_d.route == "ambiguous"
            # window as (T, C)
            win = xs_cpu[i].T if xs_cpu[i].shape[0] < xs_cpu[i].shape[1] else xs_cpu[i]
            if xs_cpu[i].ndim == 2 and xs_cpu[i].shape[0] <= 12:
                win = xs_cpu[i].T  # (C,T) -> (T,C)
            feats = extract_biomechanics(
                win, sample_rate_hz=sr, freefall_g_threshold=ff_thr, stillness_var_threshold=st_thr
            )
            text = serialize_evidence(feats, p_fall=p_fall, include_activity=False)
            if embed is not None and len(memory) > 0:
                q = np.asarray(embed(text), dtype=np.float32)
                cases = memory.retrieve(q)
                knn_frac = float(np.mean([c.label for c in cases])) if cases else 0.5
            else:
                knn_frac = 0.5
            t1 = 1 if p_fall >= 0.5 else 0
            if gate_d.route == "fall":
                t1_gate = 1
            elif gate_d.route == "adl":
                t1_gate = 0
            else:
                t1_gate = t1
            rows.append(
                {
                    "y": int(ys[i]),
                    "activity": acts[i],
                    "p_fall": p_fall,
                    "knn_fall_frac": knn_frac,
                    "escalated": bool(escalated),
                    "tier1": int(t1),
                    "tier1_gate": int(t1_gate),
                    "features": feats.to_dict(),
                    "xvec": featurize_row(
                        p_fall=p_fall, knn_fall_frac=knn_frac, feats=feats.to_dict()
                    ).tolist(),
                }
            )
    return rows


def rows_to_xy(rows):
    X = np.asarray([r["xvec"] for r in rows], dtype=np.float64)
    y = np.asarray([r["y"] for r in rows], dtype=np.int64)
    t1 = np.asarray([r["tier1"] for r in rows], dtype=np.int64)
    esc = np.asarray([r["escalated"] for r in rows], dtype=bool)
    p_fall = np.asarray([r["p_fall"] for r in rows], dtype=np.float64)
    acts = [r["activity"] for r in rows]
    return X, y, t1, esc, p_fall, acts


def eval_pred(y, pred, esc, acts, acfg):
    return agentic_metrics(
        y.tolist(),
        pred.tolist(),
        esc.tolist(),
        activities=acts,
        ambiguous_codes=list(acfg["pipeline"].get("ambiguous_adl_codes", [])),
        cost_fn=float(acfg["gate"]["cost_fn"]),
        cost_fp=float(acfg["gate"]["cost_fp"]),
    )


def calibrate_fusion(p_spec, p_fall, y, *, cost_fn=10.0, cost_fp=1.0):
    """Search blend alpha and decision threshold to maximize F1 on val."""
    y = np.asarray(y, dtype=np.int64)
    best = {"f1": -1.0, "alpha": 0.5, "threshold": 0.5, "cost": 1e18}
    for alpha in np.linspace(0.0, 1.0, 21):
        score = alpha * p_spec + (1.0 - alpha) * p_fall
        for thr in np.linspace(0.20, 0.80, 61):
            pred = (score >= thr).astype(np.int64)
            f1 = float(f1_score(y, pred, zero_division=0))
            fp = int(((pred == 1) & (y == 0)).sum())
            fn = int(((pred == 0) & (y == 1)).sum())
            c = cost_fn * fn + cost_fp * fp
            # prefer F1; break ties by lower cost
            if f1 > best["f1"] + 1e-12 or (abs(f1 - best["f1"]) < 1e-12 and c < best["cost"]):
                best = {"f1": f1, "alpha": float(alpha), "threshold": float(thr), "cost": float(c)}
    return best


def calibrate_precision_override(p_spec, tier1, y, esc, *, min_conf=0.55):
    """Only override escalated windows when specialist conf high AND improves val F1."""
    y = np.asarray(y, dtype=np.int64)
    tier1 = np.asarray(tier1, dtype=np.int64)
    esc = np.asarray(esc, dtype=bool)
    conf = np.maximum(p_spec, 1.0 - p_spec)
    y_spec = (p_spec >= 0.5).astype(np.int64)
    base_f1 = float(f1_score(y, tier1, zero_division=0))
    best_t, best_f1, best_pred_rate = 1.01, base_f1, 0.0  # 1.01 => never override
    for t in np.linspace(min_conf, 0.98, 44):
        use = esc & (conf >= t) & (y_spec != tier1)
        pred = tier1.copy()
        pred[use] = y_spec[use]
        f1 = float(f1_score(y, pred, zero_division=0))
        if f1 > best_f1 + 1e-12:
            best_f1, best_t, best_pred_rate = f1, float(t), float(use.mean())
    return best_t, {"f1": best_f1, "conf_threshold": best_t, "override_rate": best_pred_rate, "base_f1": base_f1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-config", default="configs/paper_protocol.yaml")
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-test", type=int, default=None)
    ap.add_argument("--max-train-spec", type=int, default=12000)
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
    tr, ev = pcfg["train"], pcfg["eval"]
    max_test = args.max_test if args.max_test is not None else ev.get("max_test")
    min_nf = int(ev.get("min_near_fall_per_code", 0) or 0)
    min_per = {c: min_nf for c in ev["required_activities"]} if min_nf else None
    train_idx = subsample_indices(split.train_idx, ds.y, ds.activities, tr.get("max_train"), pcfg["seed"])
    # specialist train budget (faster than full 40k)
    train_spec_idx = subsample_indices(
        train_idx, ds.y, ds.activities, args.max_train_spec, pcfg["seed"] + 17
    )
    val_idx = subsample_indices(split.val_idx, ds.y, ds.activities, tr.get("max_val"), pcfg["seed"] + 1)
    test_idx = subsample_indices(
        split.test_idx, ds.y, ds.activities, max_test, pcfg["seed"] + 2, min_per_activity=min_per
    )

    primary = pcfg["agentic"].get("primary_backbone", "cnn_lstm_attn")
    ckpt = ROOT / pcfg["paths"]["checkpoint_dir"] / f"{primary}_fold{args.fold}.pt"
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

    pipe_cfg = acfg.get("pipeline") or {}
    require_embedder = bool(pipe_cfg.get("require_text_embedder", True))
    embed = text_embedder(acfg["retrieval"]["embedder"], allow_fallback=not require_embedder)
    mem_path = ROOT / pcfg["paths"]["memory_dir"] / f"sisfall_fold{args.fold}.json"
    memory = KNNMemory(k=int(acfg["retrieval"]["k"]))
    if mem_path.exists():
        memory.load(mem_path)

    gate = build_gate(
        acfg,
        model=model,
        ds=ds,
        cal_idx=val_idx,
        device=device,
        calibrate=bool(ev.get("calibrate_gate", True)),
        max_cal=int(ev.get("max_cal", 2000)),
    )
    print(f"fold={args.fold} gate τ=({gate.tau_low:.3f},{gate.tau_high:.3f})")

    print(f"Collect train_spec (n={len(train_spec_idx)})…")
    train_rows = collect_rows(model, ds, train_spec_idx, device, gate, memory, embed, acfg)
    print(f"Collect val (n={len(val_idx)})…")
    val_rows = collect_rows(model, ds, val_idx, device, gate, memory, embed, acfg)
    print(f"Collect test (n={len(test_idx)})…")
    test_rows = collect_rows(model, ds, test_idx, device, gate, memory, embed, acfg)

    Xtr, ytr, *_ = rows_to_xy(train_rows)
    Xv, yv, t1v, escv, pfv, actsv = rows_to_xy(val_rows)
    Xt, yt, t1t, esct, pft, actst = rows_to_xy(test_rows)

    clf = train_specialist(Xtr, ytr, seed=int(pcfg["seed"]) + args.fold)
    if escv.sum() >= 40 and len(np.unique(yv[escv])) == 2:
        clf_esc = train_specialist(Xv[escv], yv[escv], seed=int(pcfg["seed"]) + 100 + args.fold)
    else:
        clf_esc = clf

    judge_tmp = SpecialistJudge(clf, conf_threshold=0.0)
    p_spec_v = judge_tmp.predict_proba_fall(Xv)
    p_spec_t = judge_tmp.predict_proba_fall(Xt)

    fuse = calibrate_fusion(
        p_spec_v, pfv, yv,
        cost_fn=float(pcfg["costs"]["cost_fn"]),
        cost_fp=float(pcfg["costs"]["cost_fp"]),
    )
    print("val_fusion", fuse)

    thr_f1, st_f1 = calibrate_judge_threshold(
        clf_esc, Xv, yv, t1v, escv,
        cost_fn=float(pcfg["costs"]["cost_fn"]),
        cost_fp=float(pcfg["costs"]["cost_fp"]),
        metric="f1",
    )
    thr_cost, st_cost = calibrate_judge_threshold(
        clf_esc, Xv, yv, t1v, escv,
        cost_fn=float(pcfg["costs"]["cost_fn"]),
        cost_fp=float(pcfg["costs"]["cost_fp"]),
        metric="cost",
    )
    thr_po, st_po = calibrate_precision_override(p_spec_v, t1v, yv, escv)
    print("judge@f1", st_f1)
    print("judge@cost", st_cost)
    print("prec_override", st_po)

    rows_out = []

    m_bhatti = eval_pred(yt, t1t, np.zeros_like(esct, dtype=bool), actst, acfg)
    rows_out.append({"name": "tier1_only", **m_bhatti})
    print("tier1_only", {k: m_bhatti[k] for k in ("f1", "recall", "expected_response_cost")})

    pred_full = (p_spec_t >= 0.5).astype(np.int64)
    m_full = eval_pred(yt, pred_full, esct, actst, acfg)
    rows_out.append({"name": "specialist_tools_full", **m_full})
    print("specialist_tools_full", {k: m_full[k] for k in ("f1", "recall", "expected_response_cost")})

    # Fixed 0.5 blend (legacy)
    blend05 = 0.5 * p_spec_t + 0.5 * pft
    m_blend05 = eval_pred(yt, (blend05 >= 0.5).astype(np.int64), esct, actst, acfg)
    rows_out.append({"name": "specialist_tier1_blend", **m_blend05})
    print("specialist_tier1_blend", {k: m_blend05[k] for k in ("f1", "recall", "expected_response_cost")})

    # Val-tuned fusion (primary beat-Bhatti candidate)
    score_t = fuse["alpha"] * p_spec_t + (1.0 - fuse["alpha"]) * pft
    pred_fuse = (score_t >= fuse["threshold"]).astype(np.int64)
    m_fuse = eval_pred(yt, pred_fuse, esct, actst, acfg)
    m_fuse["alpha"] = fuse["alpha"]
    m_fuse["threshold"] = fuse["threshold"]
    m_fuse["val_f1"] = fuse["f1"]
    rows_out.append({"name": "specialist_val_fusion", **m_fuse})
    print("specialist_val_fusion", {k: m_fuse[k] for k in ("f1", "recall", "expected_response_cost", "alpha", "threshold")})

    # Fail-closed: use fusion only if val F1 >= tier1 val F1
    t1_val_f1 = float(f1_score(yv, t1v, zero_division=0))
    if fuse["f1"] + 1e-12 >= t1_val_f1:
        pred_safe = pred_fuse
        safe_name_note = "fusion"
    else:
        pred_safe = t1t
        safe_name_note = "tier1_fallback"
    m_safe = eval_pred(yt, pred_safe, esct, actst, acfg)
    m_safe["val_gate"] = safe_name_note
    m_safe["val_fusion_f1"] = fuse["f1"]
    m_safe["val_tier1_f1"] = t1_val_f1
    rows_out.append({"name": "specialist_val_fusion_safe", **m_safe})
    print("specialist_val_fusion_safe", {k: m_safe[k] for k in ("f1", "recall", "expected_response_cost", "val_gate")})

    judge = SpecialistJudge(clf_esc, conf_threshold=thr_f1)
    pred_j, used, _ = judge.decide(Xt, t1t, esct)
    m_j = eval_pred(yt, pred_j, esct, actst, acfg)
    m_j["conf_threshold"] = thr_f1
    m_j["override_rate"] = float(used.mean())
    rows_out.append({"name": "gate_specialist_judge_f1", **m_j})
    print("gate_specialist_judge_f1", {k: m_j[k] for k in ("f1", "recall", "expected_response_cost", "override_rate")})

    judge_c = SpecialistJudge(clf_esc, conf_threshold=thr_cost)
    pred_c, used_c, _ = judge_c.decide(Xt, t1t, esct)
    m_c = eval_pred(yt, pred_c, esct, actst, acfg)
    m_c["conf_threshold"] = thr_cost
    m_c["override_rate"] = float(used_c.mean())
    rows_out.append({"name": "gate_specialist_judge_cost", **m_c})
    print("gate_specialist_judge_cost", {k: m_c[k] for k in ("f1", "recall", "expected_response_cost", "override_rate")})

    # Precision-preserving override (only if val F1 improves)
    conf_t = np.maximum(p_spec_t, 1.0 - p_spec_t)
    y_spec_t = (p_spec_t >= 0.5).astype(np.int64)
    if thr_po <= 1.0:
        use_po = esct & (conf_t >= thr_po) & (y_spec_t != t1t)
        pred_po = t1t.copy()
        pred_po[use_po] = y_spec_t[use_po]
    else:
        use_po = np.zeros_like(esct, dtype=bool)
        pred_po = t1t.copy()
    m_po = eval_pred(yt, pred_po, esct, actst, acfg)
    m_po["conf_threshold"] = thr_po
    m_po["override_rate"] = float(use_po.mean())
    rows_out.append({"name": "gate_specialist_prec_override", **m_po})
    print("gate_specialist_prec_override", {k: m_po[k] for k in ("f1", "recall", "expected_response_cost", "override_rate")})

    # Oracle upper bound
    pred_ub = t1t.copy()
    for i in range(len(yt)):
        if t1t[i] == yt[i]:
            pred_ub[i] = t1t[i]
        elif pred_full[i] == yt[i]:
            pred_ub[i] = pred_full[i]
        else:
            pred_ub[i] = t1t[i]
    m_ub = eval_pred(yt, pred_ub, esct, actst, acfg)
    rows_out.append({"name": "oracle_tier1_or_specialist", **m_ub, "note": "upper_bound_not_for_claim"})

    out_dir = ensure_dir(ROOT / (args.out_dir or f"results/noleak_lock/specialist/fold{args.fold}"))
    payload = {
        "fold": args.fold,
        "protocol": "noleak_specialist_judge_v2",
        "include_activity": False,
        "feature_names": list(FEATURE_NAMES),
        "gate": {"tau_low": gate.tau_low, "tau_high": gate.tau_high},
        "val_fusion": fuse,
        "val_judge_f1": st_f1,
        "val_judge_cost": st_cost,
        "val_prec_override": st_po,
        "max_train_spec": int(len(train_spec_idx)),
        "max_test": int(len(test_idx)),
        "rows": rows_out,
        "beat_bhatti_f1": {
            name: bool(r["f1"] > m_bhatti["f1"] + 1e-9)
            for name, r in ((row["name"], row) for row in rows_out if row["name"] != "tier1_only")
        },
    }
    save_json(payload, out_dir / f"specialist_fold{args.fold}.json")
    keys = sorted({k for r in rows_out for k in r.keys()})
    with (out_dir / f"specialist_fold{args.fold}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)
    print(f"Wrote {out_dir / f'specialist_fold{args.fold}.json'}")
    beat = [n for n, b in payload["beat_bhatti_f1"].items() if b and "oracle" not in n]
    print("BEAT_BHATTI_F1:", beat if beat else "NONE")


if __name__ == "__main__":
    main()
