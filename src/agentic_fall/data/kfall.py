from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .splits import subject_folds
from .windowing import sliding_windows
from ..utils.io import ensure_dir, save_json


def segment_postfall(
    acc: np.ndarray,
    sample_rate_hz: float = 100.0,
    rate_threshold: float = 1.0,
    stable_samples: int = 10,
) -> tuple[int, int]:
    """Bhatti et al. Algorithm 1: return (t_peak, t_stable) indices.

    acc: (T, 3) acceleration in m/s^2.
    Rate of change uses dt = 1/sample_rate.
    """
    if acc.shape[0] < 3:
        return 0, max(0, acc.shape[0] - 1)
    dt = 1.0 / sample_rate_hz
    dacc = np.diff(acc, axis=0) / dt
    rate_max = np.max(np.abs(dacc), axis=1)
    t_peak = int(np.argmax(rate_max))
    count = 0
    j = t_peak
    while j < len(rate_max) and count < stable_samples:
        if rate_max[j] < rate_threshold:
            count += 1
        else:
            count = 0
        j += 1
    t_stable = min(j, acc.shape[0] - 1)
    return t_peak, t_stable


def _load_kfall_csv(path: Path) -> np.ndarray:
    """Load KFall CSV; return (T, C) with at least AccXYZ (optionally GyroXYZ)."""
    df = pd.read_csv(path)
    acc_keys = []
    for axis in ("x", "y", "z"):
        found = None
        for name in df.columns:
            nl = name.lower().replace("_", "").replace(" ", "")
            if nl in (f"acc{axis}", f"acceleration{axis}") or nl == f"a{axis}":
                found = name
                break
        if found is None:
            break
        acc_keys.append(found)
    if len(acc_keys) != 3:
        if df.shape[1] >= 5:
            acc = df.iloc[:, 2:5].to_numpy(dtype=np.float32)
        else:
            raise ValueError(f"Cannot find accel columns in {path}")
    else:
        acc = df[acc_keys].to_numpy(dtype=np.float32)

    gyr_keys = []
    for axis in ("x", "y", "z"):
        found = None
        for name in df.columns:
            nl = name.lower().replace("_", "").replace(" ", "")
            if nl in (f"gyr{axis}", f"gyro{axis}", f"gyroscope{axis}") or nl == f"g{axis}":
                found = name
                break
        if found is None:
            break
        gyr_keys.append(found)
    if len(gyr_keys) == 3:
        gyr = df[gyr_keys].to_numpy(dtype=np.float32)
        return np.concatenate([acc, gyr], axis=1).astype(np.float32)
    return acc.astype(np.float32)


def parse_kfall_stem(stem: str) -> tuple[str, int | None, int | None]:
    """Parse Kaggle/official stems like ``S06T20R01`` → (activity, task_id, trial_id).

    KFall task IDs: 1–19 ADL, 20–34 falls (F01–F15). Older layouts may use ``F01_…``.
    """
    m = re.match(r"^S?\d*T(\d+)R(\d+)$", stem, flags=re.IGNORECASE)
    if m:
        task_id = int(m.group(1))
        trial_id = int(m.group(2))
        # Official KFall: task IDs 20–34 = F01–F15; 1–19 and 35–36 = ADL
        if 20 <= task_id <= 34:
            activity = f"F{task_id - 19:02d}"
        elif task_id >= 35:
            activity = f"D{task_id - 15:02d}"  # 35→D20, 36→D21
        else:
            activity = f"D{task_id:02d}"
        return activity, task_id, trial_id
    if "_" in stem:
        activity = stem.split("_")[0]
    else:
        activity = stem[:3]
    return activity, None, None


def _lookup_label_onset_impact(
    label_map: dict[str, pd.DataFrame],
    subject: str,
    activity: str,
    n_samples: int,
    task_id: int | None = None,
    trial_id: int | None = None,
) -> tuple[int | None, int | None]:
    """Best-effort onset/impact from KFall label workbooks."""
    candidates = []
    for key, df in label_map.items():
        if subject.lower() in key.lower() or key.lower() in subject.lower():
            candidates.append(df)
    if not candidates:
        candidates = list(label_map.values())

    for df in candidates:
        # Forward-fill task codes (KFall label sheets leave blanks within a block)
        work = df.copy()
        cols = {str(c).lower(): c for c in work.columns}
        task_col = next((cols[c] for c in cols if "task" in c), None)
        trial_col = next((cols[c] for c in cols if "trial" in c), None)
        onset_col = next((cols[c] for c in cols if "onset" in c), None)
        impact_col = next((cols[c] for c in cols if "impact" in c), None)
        if task_col is not None:
            work[task_col] = work[task_col].ffill()
        for _, row in work.iterrows():
            if trial_id is not None and trial_col is not None and pd.notna(row[trial_col]):
                try:
                    if int(row[trial_col]) != int(trial_id):
                        continue
                except Exception:
                    continue
            task_val = str(row[task_col]) if task_col is not None else ""
            ok = False
            if activity and activity.upper() in task_val.upper():
                ok = True
            if task_id is not None and f"({task_id})" in task_val.replace(" ", ""):
                ok = True
            if task_id is not None and re.search(rf"\({task_id}\)", task_val):
                ok = True
            if not ok and activity.upper() not in task_val.upper():
                continue
            onset = int(row[onset_col]) if onset_col is not None and pd.notna(row[onset_col]) else None
            impact = int(row[impact_col]) if impact_col is not None and pd.notna(row[impact_col]) else None
            if onset is not None:
                onset = int(np.clip(onset, 0, n_samples - 1))
            if impact is not None:
                impact = int(np.clip(impact, 0, n_samples - 1))
            return onset, impact
    return None, None


def prepare_kfall(
    raw_dir: str | Path,
    out_dir: str | Path,
    window: int = 90,
    hop: int = 10,
    sample_rate_hz: float = 100.0,
    rate_threshold: float = 1.0,
    stable_samples: int = 10,
    phase: str = "transition",
    use_g: bool = True,
) -> Path:
    """Prepare KFall windows.

    Expects layout::
        raw_dir/
          SAXX/*.csv
          label/*.xlsx   (optional)

    Acc assumed in g when use_g=True; Algorithm-1 threshold is applied in m/s^2
    (``rate_threshold * 9.81``).
    """
    raw_dir = Path(raw_dir)
    out_dir = ensure_dir(out_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"KFall raw dir not found: {raw_dir}. "
            "Request access at https://sites.google.com/view/kfalldataset "
            "and place data under data/raw/kfall/"
        )

    # Algorithm-1 threshold is in m/s^2 when accel stored in g
    thr_ms2 = float(rate_threshold) * (9.81 if use_g else 1.0)
    csv_files = sorted(raw_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files under {raw_dir}")

    label_map: dict[str, pd.DataFrame] = {}
    for xlsx in raw_dir.rglob("*.xlsx"):
        try:
            label_map[xlsx.stem] = pd.read_excel(xlsx)
        except Exception:
            continue

    windows: list[np.ndarray] = []
    labels: list[int] = []
    binary_labels: list[int] = []
    subjects: list[str] = []
    activities: list[str] = []
    phases: list[str] = []
    act2id: dict[str, int] = {}

    for path in tqdm(csv_files, desc="prepare_kfall"):
        subject = path.parent.name
        stem = path.stem
        activity, task_id, trial_id = parse_kfall_stem(stem)
        is_fall = activity.upper().startswith("F") or (
            task_id is not None and 20 <= task_id <= 34
        )
        try:
            sig = _load_kfall_csv(path)
        except Exception:
            continue
        if sig.shape[0] < window:
            continue
        acc = sig[:, :3]

        if is_fall:
            onset_lbl, impact_lbl = _lookup_label_onset_impact(
                label_map,
                subject,
                activity,
                acc.shape[0],
                task_id=task_id,
                trial_id=trial_id,
            )
            acc_ms2 = acc * (9.81 if use_g else 1.0)
            t_peak, t_stable = segment_postfall(
                acc_ms2,
                sample_rate_hz=sample_rate_hz,
                rate_threshold=thr_ms2,
                stable_samples=stable_samples,
            )
            impact = impact_lbl if impact_lbl is not None else t_peak
            onset = (
                onset_lbl
                if onset_lbl is not None
                else max(0, impact - int(0.5 * sample_rate_hz))
            )
            if phase == "pre":
                seg = sig[onset:impact]
                ph = "pre"
            elif phase == "post":
                seg = sig[impact:t_stable]
                ph = "post"
            else:
                seg = sig[onset:t_stable]
                ph = "transition"
        else:
            seg = sig
            ph = "adl"

        if seg.shape[0] < 5:
            continue
        chunks = sliding_windows(seg, window=window, hop=hop)
        if activity not in act2id:
            act2id[activity] = len(act2id)
        aid = act2id[activity]
        n = chunks.shape[0]
        windows.append(chunks)
        labels.extend([aid] * n)
        binary_labels.extend([1 if is_fall else 0] * n)
        subjects.extend([subject] * n)
        activities.extend([activity] * n)
        phases.extend([ph] * n)

    if not windows:
        raise RuntimeError("No KFall windows produced — check raw layout.")

    X = np.concatenate(windows, axis=0).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    y_bin = np.asarray(binary_labels, dtype=np.int64)
    out_path = out_dir / f"windows_{phase}_w{window}_h{hop}.npz"
    np.savez_compressed(
        out_path,
        X=X,
        y=y,
        y_binary=y_bin,
        subjects=np.asarray(subjects),
        activities=np.asarray(activities),
        phases=np.asarray(phases),
        class_names=np.asarray(sorted(act2id, key=lambda a: act2id[a])),
    )
    save_json(
        {
            "n_windows": int(X.shape[0]),
            "window": window,
            "hop": hop,
            "phase": phase,
            "channels": int(X.shape[-1]),
            "n_classes": len(act2id),
            "class_to_id": act2id,
            "n_falls_binary": int((y_bin == 1).sum()),
            "n_adl_binary": int((y_bin == 0).sum()),
            "rate_threshold_ms2": thr_ms2,
            "stable_samples": stable_samples,
            "used_label_xlsx": bool(label_map),
        },
        out_dir / f"meta_{phase}_w{window}_h{hop}.json",
    )
    return out_path


class KFallDataset(Dataset):
    def __init__(
        self,
        npz_path: str | Path,
        indices: list[int] | None = None,
        binary: bool = False,
    ):
        data = np.load(npz_path, allow_pickle=True)
        self.X = data["X"]
        self.y_multiclass = data["y"]
        self.y_binary = data["y_binary"] if "y_binary" in data.files else None
        self.binary = binary
        if binary:
            if self.y_binary is None:
                # derive from class names if available
                class_names = list(data["class_names"]) if "class_names" in data.files else []
                mask = fall_class_mask(class_names)
                self.y = np.array([int(mask[int(yi)]) for yi in self.y_multiclass], dtype=np.int64)
            else:
                self.y = self.y_binary
        else:
            self.y = self.y_multiclass
        self.subjects = data["subjects"]
        self.activities = data["activities"]
        self.phases = data["phases"] if "phases" in data.files else np.array(["?"] * len(self.y))
        self.class_names = list(data["class_names"]) if "class_names" in data.files else []
        self.indices = list(range(len(self.y))) if indices is None else list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict[str, Any]:
        idx = self.indices[i]
        x = torch.from_numpy(self.X[idx]).float().transpose(0, 1)
        return {
            "x": x,
            "y": int(self.y[idx]),
            "subject": str(self.subjects[idx]),
            "activity": str(self.activities[idx]),
            "phase": str(self.phases[idx]),
            "index": idx,
        }

    def fold_indices(self, n_folds: int = 5, seed: int = 42):
        subj = [str(s) for s in self.subjects]
        has_fall: dict[str, int] = {}
        for s, a in zip(subj, self.activities):
            has_fall[s] = has_fall.get(s, 0) or int(str(a).upper().startswith("F"))
        strata = ["fall" if has_fall[s] else "adl_only" for s in subj]
        folds = subject_folds(subj, n_folds=n_folds, seed=seed, strata=strata)
        out = []
        for train_s, test_s in folds:
            tr, te = set(train_s), set(test_s)
            train_idx = [i for i, s in enumerate(subj) if s in tr]
            test_idx = [i for i, s in enumerate(subj) if s in te]
            out.append((train_idx, test_idx))
        return out


def fall_class_mask(class_names: list[str]) -> np.ndarray:
    """Boolean mask over classes that are falls (name starts with F)."""
    return np.array([str(c).upper().startswith("F") for c in class_names], dtype=bool)


def multiclass_logits_to_p_fall(logits: torch.Tensor, fall_mask: np.ndarray) -> torch.Tensor:
    """Sum softmax probability mass over fall classes → p(fall) for the gate."""
    probs = torch.softmax(logits, dim=-1)
    idx = torch.as_tensor(np.where(fall_mask)[0], device=logits.device, dtype=torch.long)
    if idx.numel() == 0:
        return probs.max(dim=-1).values
    return probs.index_select(-1, idx).sum(dim=-1)
