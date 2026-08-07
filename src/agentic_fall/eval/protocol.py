"""Shared Q1 evaluation protocol — one source of truth for splits / caps / gate."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ..agents.confidence_gate import ConfidenceGate
from ..data.sisfall import assert_required_activities, inventory_activity_counts
from ..eval.metrics import binary_metrics


@dataclass
class FoldSplit:
    train_idx: list[int]
    val_idx: list[int]
    test_idx: list[int]
    fold: int


def subject_held_out_val(
    train_idx: Sequence[int],
    subjects: Sequence[str],
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Split train windows by holding out a fraction of *train subjects* for val."""
    rng = np.random.default_rng(seed)
    train_subjects = sorted({str(subjects[i]) for i in train_idx})
    if len(train_subjects) < 2:
        # degenerate: fall back to window split
        idx = list(train_idx)
        rng.shuffle(idx)
        n_val = max(1, int(val_fraction * len(idx)))
        return idx[n_val:], idx[:n_val]

    rng.shuffle(train_subjects)
    n_val_subj = max(1, int(round(val_fraction * len(train_subjects))))
    n_val_subj = min(n_val_subj, len(train_subjects) - 1)
    val_subj = set(train_subjects[:n_val_subj])
    tr_subj = set(train_subjects[n_val_subj:])
    new_train = [i for i in train_idx if str(subjects[i]) in tr_subj]
    new_val = [i for i in train_idx if str(subjects[i]) in val_subj]
    if not new_train or not new_val:
        idx = list(train_idx)
        rng.shuffle(idx)
        n_val = max(1, int(val_fraction * len(idx)))
        return idx[n_val:], idx[:n_val]
    return new_train, new_val


def make_fold_split(
    ds,
    fold: int,
    n_folds: int = 5,
    seed: int = 42,
    val_fraction: float = 0.15,
) -> FoldSplit:
    folds = ds.fold_indices(n_folds=n_folds, seed=seed)
    train_idx, test_idx = folds[fold]
    train_idx, val_idx = subject_held_out_val(
        train_idx, ds.subjects, val_fraction=val_fraction, seed=seed + fold
    )
    # Leakage check
    tr_s = {str(ds.subjects[i]) for i in train_idx}
    va_s = {str(ds.subjects[i]) for i in val_idx}
    te_s = {str(ds.subjects[i]) for i in test_idx}
    if tr_s & te_s or va_s & te_s:
        raise RuntimeError("Subject leakage between train/val and test")
    if tr_s & va_s:
        raise RuntimeError("Subject leakage between train and val")
    return FoldSplit(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, fold=fold)


def subsample_indices(
    indices: Sequence[int],
    y: np.ndarray,
    activities: np.ndarray | None,
    n: int | None,
    seed: int,
    *,
    stratified: bool = True,
    min_per_activity: dict[str, int] | None = None,
) -> list[int]:
    """Cap an index list with optional label stratification and forced activity quotas."""
    indices = list(indices)
    if n is None or len(indices) <= n:
        out = list(indices)
    else:
        rng = np.random.default_rng(seed)
        if stratified:
            pos = [i for i in indices if int(y[i]) == 1]
            neg = [i for i in indices if int(y[i]) == 0]
            n_pos = min(len(pos), n // 2)
            n_neg = min(len(neg), n - n_pos)
            out = list(rng.choice(pos, n_pos, replace=False)) + list(
                rng.choice(neg, n_neg, replace=False)
            )
            rng.shuffle(out)
        else:
            out = list(rng.choice(indices, size=n, replace=False))

    if min_per_activity and activities is not None:
        have = {code: 0 for code in min_per_activity}
        for i in out:
            a = str(activities[i])
            if a in have:
                have[a] += 1
        pool = set(indices)
        selected = set(out)
        rng = np.random.default_rng(seed + 17)
        for code, need in min_per_activity.items():
            deficit = need - have.get(code, 0)
            if deficit <= 0:
                continue
            candidates = [
                i for i in pool if i not in selected and str(activities[i]) == code
            ]
            if not candidates:
                continue
            take = min(deficit, len(candidates))
            picked = list(rng.choice(candidates, size=take, replace=False))
            # Replace random non-priority samples if over budget
            if n is not None and len(out) + take > n:
                replaceable = [
                    i
                    for i in out
                    if str(activities[i]) not in min_per_activity
                ]
                rng.shuffle(replaceable)
                for j, new_i in enumerate(picked):
                    if j < len(replaceable):
                        out.remove(replaceable[j])
                        selected.discard(replaceable[j])
                    out.append(new_i)
                    selected.add(new_i)
            else:
                out.extend(picked)
                selected.update(picked)
        rng.shuffle(out)
    return out


def verify_dataset_near_falls(ds, required: Sequence[str] = ("D18", "D19")) -> dict[str, int]:
    counts = inventory_activity_counts(ds.activities)
    assert_required_activities(counts, list(required))
    return counts


def collect_probs(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[float], list[int]]:
    model.to(device).eval()
    ps, ys = [], []
    with torch.no_grad():
        for batch in loader:
            logits, _ = model(batch["x"].to(device))
            p = torch.softmax(logits, dim=-1)
            if p.size(1) == 2:
                ps.extend(p[:, 1].cpu().tolist())
            else:
                ps.extend(p.max(dim=1).values.cpu().tolist())
            ys.extend(batch["y"].tolist())
    return ps, ys


def tune_decision_threshold(
    y_true: list[int],
    p_fall: list[float],
    *,
    mode: str = "cost",
    cost_fn: float = 10.0,
    cost_fp: float = 1.0,
    grid: int = 101,
) -> tuple[float, dict[str, float]]:
    """Select probability threshold on validation predictions."""
    yt = np.asarray(y_true, dtype=int)
    pp = np.asarray(p_fall, dtype=float)
    best_t = 0.5
    best_score = -1e18
    best_m: dict[str, float] = {}
    for t in np.linspace(0.05, 0.95, grid):
        yp = (pp >= t).astype(int)
        m = binary_metrics(yt, yp)
        if mode == "youden":
            score = m["sensitivity"] + m["specificity"] - 1.0
        elif mode == "f1":
            score = m["f1"]
        else:  # cost-sensitive: lower cost is better
            cost = cost_fn * m["fn"] + cost_fp * m["fp"]
            score = -float(cost)
        if score > best_score:
            best_score = score
            best_t = float(t)
            best_m = m
    return best_t, best_m


def build_gate(
    acfg: dict[str, Any],
    model: torch.nn.Module | None = None,
    ds=None,
    cal_idx: Sequence[int] | None = None,
    device: torch.device | None = None,
    *,
    calibrate: bool | None = None,
    max_cal: int = 2000,
) -> ConfidenceGate:
    gate = ConfidenceGate(float(acfg["gate"]["tau_low"]), float(acfg["gate"]["tau_high"]))
    do_cal = acfg["gate"].get("calibrate", True) if calibrate is None else calibrate
    if do_cal and model is not None and ds is not None and cal_idx is not None:
        device = device or torch.device("cpu")
        model.to(device)
        idx = list(cal_idx)[:max_cal]
        loader = DataLoader(Subset(ds, idx), batch_size=64, shuffle=False)
        ps, ys = collect_probs(model, loader, device)
        gate.calibrate(
            ps,
            ys,
            cost_fn=float(acfg["gate"]["cost_fn"]),
            cost_fp=float(acfg["gate"]["cost_fp"]),
        )
    return gate


def load_paper_protocol(root: Path, path: str | Path = "configs/paper_protocol.yaml") -> dict:
    from ..utils.config import load_config

    return load_config(root / path)


def text_embedder(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer

        st = SentenceTransformer(model_name)
        return lambda t: st.encode(t, normalize_embeddings=True)
    except Exception:
        rng = np.random.default_rng(0)
        W = rng.normal(size=(64, 384)).astype(np.float32)

        def embed(text: str):
            v = np.zeros(64, dtype=np.float32)
            for i, ch in enumerate(text.encode("utf-8")[:512]):
                v[i % 64] += ch / 255.0
            z = W.T @ v
            return z / (np.linalg.norm(z) + 1e-8)

        return embed
