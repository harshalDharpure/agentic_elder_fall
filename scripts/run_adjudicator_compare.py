#!/usr/bin/env python3
"""G2: compare adjudicators on escalated windows (same biomechanical features).

Trivial floors, heuristic, logistic, GBM, optional LLM backends.
Writes results/adjudicator_compare/ and tab_adjudicator.tex.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_fall.agents.confidence_gate import ConfidenceGate
from agentic_fall.agents.evidence import serialize_evidence
from agentic_fall.agents.llm_reasoner import LLMReasoner
from agentic_fall.data.sisfall import SisFallDataset
from agentic_fall.eval.metrics import binary_metrics
from agentic_fall.eval.protocol import build_gate, make_fold_split, subsample_indices
from agentic_fall.features.biomechanics import extract_biomechanics
from agentic_fall.models import build_model
from agentic_fall.utils.config import load_config
from agentic_fall.utils.io import ensure_dir, save_json
from agentic_fall.utils.seed import set_seed

FEATURE_KEYS = [
    "freefall_duration_s",
    "impact_magnitude_g",
    "post_impact_stillness_s",
    "body_tilt_deg",
    "duration_s",
    "mean_acc_g",
    "peak_gyro_dps",
]


def feats_to_vec(feats) -> np.ndarray:
    d = feats.to_dict()
    return np.array([float(d[k]) for k in FEATURE_KEYS], dtype=np.float64)


@torch.no_grad()
def collect_escalated(
    model,
    gate: ConfidenceGate,
    ds,
    idx: list[int],
    device: torch.device,
    sample_rate: float,
    max_cases: int,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    X, y, meta = [], [], []
    model.eval()
    for i in tqdm(idx, desc="collect escalated"):
        if len(X) >= max_cases:
            break
        batch = ds[i]
        x = batch["x"].unsqueeze(0).to(device)
        logits, _ = model(x)
        p = float(torch.softmax(logits, dim=-1)[0, 1].item())
        dec = gate.decide(p)
        # GateDecision uses route: adl | fall | ambiguous (escalate)
        if getattr(dec, "route", None) != "ambiguous" and not getattr(dec, "escalated", False):
            continue
        win = batch["x"].numpy().T  # (T, C)
        feats = extract_biomechanics(win, sample_rate_hz=sample_rate)
        X.append(feats_to_vec(feats))
        y.append(int(batch["y"]))
        meta.append(
            {
                "idx": int(i),
                "p_fall": p,
                "activity": str(batch["activity"]),
                "evidence": serialize_evidence(feats, activity=str(batch["activity"]), p_fall=p),
                "feats": feats.to_dict(),
            }
        )
    return np.asarray(X), np.asarray(y, dtype=np.int64), meta


def score_preds(y_true, y_pred) -> dict:
    return binary_metrics(list(y_true), list(y_pred))


def always_fall(n: int) -> np.ndarray:
    return np.ones(n, dtype=np.int64)


def always_adl(n: int) -> np.ndarray:
    return np.zeros(n, dtype=np.int64)


def random_pred(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=n)


def majority_pred(y_train: np.ndarray, n: int) -> np.ndarray:
    maj = int(np.round(y_train.mean())) if len(y_train) else 1
    return np.full(n, maj, dtype=np.int64)


def _label_to_bin(label) -> int:
    if hasattr(label, "prediction"):
        label = label.prediction
    elif hasattr(label, "label"):
        label = label.label
    if isinstance(label, tuple):
        label = label[0]
    if isinstance(label, dict):
        label = label.get("prediction") or label.get("label", "fall")
    return 1 if str(label).lower().startswith("fall") else 0


def heuristic_pred(meta: list[dict]) -> np.ndarray:
    reasoner = LLMReasoner(backend="heuristic", fallback_heuristic=True)
    out = []
    for m in meta:
        from agentic_fall.features.biomechanics import BiomechanicalFeatures

        feats = BiomechanicalFeatures(**m["feats"])
        res = reasoner.reason(m["evidence"], feats, m["p_fall"], cases=[])
        out.append(_label_to_bin(res))
    return np.asarray(out, dtype=np.int64)


def train_sklearn(X_tr, y_tr, X_te, kind: str):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_tr)
    Xte = scaler.transform(X_te)
    if kind == "logreg":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    else:
        clf = GradientBoostingClassifier(random_state=42)
    clf.fit(Xtr, y_tr)
    return clf.predict(Xte)


def llm_pred(meta: list[dict], backend: str, max_n: int) -> np.ndarray:
    reasoner = LLMReasoner(backend=backend, fallback_heuristic=True)
    out = []
    for m in tqdm(meta[:max_n], desc=f"llm:{backend}"):
        from agentic_fall.features.biomechanics import BiomechanicalFeatures

        feats = BiomechanicalFeatures(**m["feats"])
        res = reasoner.reason(m["evidence"], feats, m["p_fall"], cases=[])
        out.append(_label_to_bin(res))
    while len(out) < len(meta):
        out.append(out[-1] if out else 0)
    return np.asarray(out[: len(meta)], dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-config", default="configs/paper_protocol.yaml")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-cases", type=int, default=400)
    ap.add_argument("--llm-backends", nargs="*", default=[])
    ap.add_argument("--llm-max", type=int, default=80)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--model", default=None, help="Backbone name; inferred from checkpoint filename if omitted")
    args = ap.parse_args()

    pcfg = load_config(ROOT / args.paper_config)
    tcfg = load_config(ROOT / pcfg["tier1_config"])
    acfg = load_config(ROOT / pcfg["agentic_config"])
    bcfg = load_config(ROOT / pcfg["backbones_config"])
    set_seed(int(pcfg["seed"]))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    fold = int(args.fold)

    npz = ROOT / tcfg["processed_dir"] / f"windows_w{tcfg['window']}_h{tcfg['hop']}.npz"
    ds = SisFallDataset(npz, channels=int(tcfg["channels"]))
    split = make_fold_split(
        ds,
        fold=fold,
        n_folds=int(pcfg["splits"]["n_folds"]),
        seed=int(pcfg["seed"]),
        val_fraction=float(pcfg["splits"]["val_fraction"]),
    )
    ev = pcfg["eval"]
    min_nf = int(ev.get("min_near_fall_per_code", 0) or 0)
    min_per = {c: min_nf for c in ev.get("required_activities", [])} if min_nf else None
    test_idx = subsample_indices(
        split.test_idx,
        ds.y,
        ds.activities,
        ev.get("max_test"),
        pcfg["seed"] + 2,
        min_per_activity=min_per,
    )
    # use a larger pool for escalation collection
    pool = subsample_indices(
        split.test_idx, ds.y, ds.activities, 4000, pcfg["seed"] + 9, min_per_activity=min_per
    )

    name = args.model or pcfg.get("agentic", {}).get("primary_backbone", "cnn_lstm_attn")
    ckpt = Path(args.checkpoint) if args.checkpoint else ROOT / pcfg["paths"]["checkpoint_dir"] / f"{name}_fold{fold}.pt"
    # Infer architecture from checkpoint stem: {model}_fold{N}.pt
    if args.checkpoint:
        stem = Path(args.checkpoint).stem  # e.g. cnn_lstm_attn_fold0
        if "_fold" in stem:
            name = stem.rsplit("_fold", 1)[0]
    if not ckpt.exists():
        name = "cnn_lstm_attn"
        ckpt = ROOT / "checkpoints" / "standard" / f"{name}_fold{fold}.pt"
        if not ckpt.exists():
            ckpt = ROOT / pcfg["paths"]["checkpoint_dir"] / f"{name}_fold{fold}.pt"
    print(f"Loading model={name} ckpt={ckpt}")
    kw = dict(bcfg.get("model_kwargs", {}).get(name, {}))
    model = build_model(name, in_channels=int(tcfg["channels"]), num_classes=2, **kw)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device)

    gate = build_gate(
        acfg,
        model=model,
        ds=ds,
        cal_idx=split.train_idx,
        device=device,
        calibrate=True,
        max_cal=int(ev.get("max_cal", 2000)),
    )

    X, y, meta = collect_escalated(
        model,
        gate,
        ds,
        pool,
        device,
        sample_rate=float(acfg["evidence"]["sample_rate_hz"]),
        max_cases=int(args.max_cases),
    )
    if len(y) < 40:
        raise RuntimeError(f"Too few escalated cases ({len(y)}); widen pool or gate.")

    # train/test split on escalated set
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(y))
    n_tr = max(20, int(0.6 * len(y)))
    tr, te = perm[:n_tr], perm[n_tr:]
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    meta_te = [meta[i] for i in te]

    rows = []
    for label, preds in [
        ("always_fall", always_fall(len(yte))),
        ("always_adl", always_adl(len(yte))),
        ("random", random_pred(len(yte), 42)),
        ("majority_train", majority_pred(ytr, len(yte))),
        ("heuristic", heuristic_pred(meta_te)),
    ]:
        m = score_preds(yte, preds)
        rows.append({"name": label, "n": len(yte), **m})

    for kind in ("logreg", "gbm"):
        try:
            preds = train_sklearn(Xtr, ytr, Xte, kind)
            m = score_preds(yte, preds)
            rows.append({"name": kind, "n": len(yte), **m})
        except Exception as e:  # noqa: BLE001
            rows.append({"name": kind, "n": len(yte), "error": str(e), "f1": 0.0})

    for backend in args.llm_backends:
        try:
            preds = llm_pred(meta_te, backend, args.llm_max)
            # if llm_max < len, score only first llm_max
            k = min(args.llm_max, len(yte))
            m = score_preds(yte[:k], preds[:k])
            rows.append({"name": f"llm:{backend}", "n": k, **m})
        except Exception as e:  # noqa: BLE001
            rows.append({"name": f"llm:{backend}", "error": str(e), "f1": 0.0, "n": 0})

    out_dir = ensure_dir(ROOT / "results" / "adjudicator_compare")
    payload = {
        "fold": fold,
        "backbone": name,
        "n_escalated": int(len(y)),
        "n_test": int(len(yte)),
        "feature_keys": FEATURE_KEYS,
        "rows": rows,
    }
    save_json(payload, out_dir / f"fold{fold}.json")

    # G2 verdict
    gbm_f1 = next((r["f1"] for r in rows if r["name"] == "gbm"), 0.0)
    llm_f1s = [r["f1"] for r in rows if str(r["name"]).startswith("llm:") and "f1" in r]
    best_llm = max(llm_f1s) if llm_f1s else None
    verdict = {
        "gbm_f1": gbm_f1,
        "best_llm_f1": best_llm,
        "g2": (
            "llm_wins"
            if best_llm is not None and best_llm > gbm_f1 + 0.03
            else ("tie_or_gbm" if best_llm is not None else "gbm_only_no_llm")
        ),
    }
    save_json(verdict, out_dir / f"fold{fold}_g2_verdict.json")

    tex = out_dir / "tab_adjudicator.tex"
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Adjudicator comparison on escalated windows (matched features). Gate G2.}",
        r"\label{tab:adjudicator}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Adjudicator & $n$ & F1 \\",
        r"\midrule",
    ]
    for r in rows:
        nm = str(r["name"]).replace("_", r"\_")
        lines.append(f"{nm} & {r.get('n',0)} & {r.get('f1',0):.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    tex.write_text("\n".join(lines))

    print(json.dumps(payload, indent=2))
    print("G2 verdict:", verdict)


if __name__ == "__main__":
    main()
