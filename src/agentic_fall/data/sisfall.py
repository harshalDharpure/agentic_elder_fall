from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .splits import subject_folds
from .windowing import sliding_windows
from ..utils.io import ensure_dir, save_json

# SisFall raw values are integers; scale to physical units (from dataset paper).
ADXL345_SCALE = (2 * 16) / (2**13)  # g
ITG3200_SCALE = (2 * 2000) / (2**16)  # deg/s

FILE_RE = re.compile(r"^(D|F)(\d+)_([A-Z]{2}\d+)_R(\d+)\.txt$")


def _parse_filename(name: str) -> dict[str, Any] | None:
    m = FILE_RE.match(name)
    if not m:
        return None
    kind, code_num, subject, trial = m.groups()
    activity = f"{kind}{int(code_num):02d}"
    return {
        "activity": activity,
        "is_fall": kind == "F",
        "subject": subject,
        "trial": int(trial),
    }


def read_sisfall_file(path: Path) -> np.ndarray:
    """Return (T, 6) float32: ADXL345 xyz + ITG3200 xyz in physical units."""
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip().rstrip(";")
            if not line:
                continue
            parts = [p.strip() for p in line.split(",") if p.strip() != ""]
            if len(parts) < 6:
                continue
            vals = [float(parts[i]) for i in range(6)]
            rows.append(vals)
    arr = np.asarray(rows, dtype=np.float32)
    arr[:, 0:3] *= ADXL345_SCALE
    arr[:, 3:6] *= ITG3200_SCALE
    return arr


DEFAULT_REQUIRED_ACTIVITIES = ("D18", "D19")


def inventory_activity_counts(activities: list[str] | np.ndarray) -> dict[str, int]:
    from collections import Counter

    return dict(Counter(str(a) for a in activities))


def assert_required_activities(
    activity_counts: dict[str, int],
    required: list[str] | tuple[str, ...] = DEFAULT_REQUIRED_ACTIVITIES,
) -> None:
    missing = [c for c in required if activity_counts.get(c, 0) <= 0]
    if missing:
        raise RuntimeError(
            f"Processed SisFall is missing required near-fall activities {missing}. "
            f"Present counts={activity_counts}. Re-run prepare_sisfall without "
            f"--max-files (or with force-include of {list(required)})."
        )


def _force_include_activities(
    all_files: list[Path],
    selected: list[Path],
    required_activities: list[str] | tuple[str, ...],
) -> list[Path]:
    """Ensure required activity codes are present when subsampling files."""
    have = set()
    for p in selected:
        meta = _parse_filename(p.name)
        if meta:
            have.add(meta["activity"])
    need = [c for c in required_activities if c not in have]
    if not need:
        return selected
    extra: list[Path] = []
    selected_set = set(selected)
    for code in need:
        for p in all_files:
            if p in selected_set:
                continue
            meta = _parse_filename(p.name)
            if meta and meta["activity"] == code:
                extra.append(p)
                selected_set.add(p)
    return sorted(set(selected) | set(extra), key=lambda p: str(p))


def prepare_sisfall(
    raw_dir: str | Path,
    out_dir: str | Path,
    window: int = 90,
    hop: int = 10,
    max_files: int | None = None,
    required_activities: list[str] | tuple[str, ...] | None = DEFAULT_REQUIRED_ACTIVITIES,
) -> Path:
    """Window SisFall recordings into a packed .npz archive."""
    raw_dir = Path(raw_dir)
    out_dir = ensure_dir(out_dir)
    required_activities = list(required_activities or [])

    files = sorted(raw_dir.rglob("*.txt"))
    files = [p for p in files if p.name.lower() != "readme.txt"]
    all_files = list(files)
    if max_files is not None and len(files) > max_files:
        # Round-robin across subjects, alternating fall/ADL when possible
        by_subj: dict[str, dict[str, list[Path]]] = {}
        for p in files:
            kind = "fall" if p.name.startswith("F") else "adl"
            by_subj.setdefault(p.parent.name, {"fall": [], "adl": []})[kind].append(p)
        for sub in by_subj.values():
            sub["fall"].sort()
            sub["adl"].sort()
        files = []
        keys = sorted(by_subj)
        idx = {k: {"fall": 0, "adl": 0} for k in keys}
        toggle = 0
        while len(files) < max_files and keys:
            for k in keys:
                prefer = "fall" if toggle % 2 == 0 else "adl"
                other = "adl" if prefer == "fall" else "fall"
                picked = False
                for kind in (prefer, other):
                    i = idx[k][kind]
                    bucket = by_subj[k][kind]
                    if i < len(bucket):
                        files.append(bucket[i])
                        idx[k][kind] = i + 1
                        picked = True
                        break
                if len(files) >= max_files:
                    break
            keys = [
                k
                for k in keys
                if idx[k]["fall"] < len(by_subj[k]["fall"])
                or idx[k]["adl"] < len(by_subj[k]["adl"])
            ]
            toggle += 1
        files = sorted(files, key=lambda p: str(p))
        if required_activities:
            files = _force_include_activities(all_files, files, required_activities)

    windows: list[np.ndarray] = []
    labels: list[int] = []
    subjects: list[str] = []
    activities: list[str] = []
    file_ids: list[str] = []

    for path in tqdm(files, desc="prepare_sisfall"):
        meta = _parse_filename(path.name)
        if meta is None:
            continue
        try:
            sig = read_sisfall_file(path)
        except Exception:
            continue
        if sig.shape[0] < 10:
            continue
        chunks = sliding_windows(sig, window=window, hop=hop)
        n = chunks.shape[0]
        windows.append(chunks)
        labels.extend([1 if meta["is_fall"] else 0] * n)
        subjects.extend([meta["subject"]] * n)
        activities.extend([meta["activity"]] * n)
        file_ids.extend([path.name] * n)

    if not windows:
        raise RuntimeError("No SisFall windows produced — check raw_dir layout.")

    X = np.concatenate(windows, axis=0).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    activity_counts = inventory_activity_counts(activities)
    if required_activities:
        assert_required_activities(activity_counts, required_activities)

    out_path = out_dir / f"windows_w{window}_h{hop}.npz"
    np.savez_compressed(
        out_path,
        X=X,
        y=y,
        subjects=np.asarray(subjects),
        activities=np.asarray(activities),
        file_ids=np.asarray(file_ids),
    )
    save_json(
        {
            "n_windows": int(X.shape[0]),
            "window": window,
            "hop": hop,
            "channels": int(X.shape[-1]),
            "n_falls": int((y == 1).sum()),
            "n_adl": int((y == 0).sum()),
            "n_subjects": len(set(subjects)),
            "n_files": len(set(file_ids)),
            "activity_counts": activity_counts,
            "required_activities": required_activities,
        },
        out_dir / f"meta_w{window}_h{hop}.json",
    )
    return out_path


class SisFallDataset(Dataset):
    def __init__(
        self,
        npz_path: str | Path,
        indices: list[int] | None = None,
        channels: int = 6,
    ):
        data = np.load(npz_path, allow_pickle=True)
        self.X = data["X"][:, :, :channels]
        self.y = data["y"]
        self.subjects = data["subjects"]
        self.activities = data["activities"]
        self.file_ids = data["file_ids"]
        self.indices = list(range(len(self.y))) if indices is None else list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict[str, Any]:
        idx = self.indices[i]
        x = torch.from_numpy(self.X[idx]).float()  # (T, C)
        x = x.transpose(0, 1)  # (C, T) for Conv1d
        return {
            "x": x,
            "y": int(self.y[idx]),
            "subject": str(self.subjects[idx]),
            "activity": str(self.activities[idx]),
            "file_id": str(self.file_ids[idx]),
            "index": idx,
        }

    def fold_indices(self, n_folds: int = 5, seed: int = 42) -> list[tuple[list[int], list[int]]]:
        subj = [str(s) for s in self.subjects]
        # Stratify by whether the subject has any fall windows (SA falls vs SE ADL-only)
        has_fall = {}
        for s, y in zip(subj, self.y):
            has_fall[s] = has_fall.get(s, 0) or int(y)
        strata = ["fall" if has_fall[s] else "adl_only" for s in subj]
        folds = subject_folds(subj, n_folds=n_folds, seed=seed, strata=strata)
        out = []
        for train_s, test_s in folds:
            train_set, test_set = set(train_s), set(test_s)
            train_idx = [i for i, s in enumerate(subj) if s in train_set]
            test_idx = [i for i, s in enumerate(subj) if s in test_set]
            out.append((train_idx, test_idx))
        return out
