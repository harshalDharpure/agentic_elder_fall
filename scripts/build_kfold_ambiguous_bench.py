#!/usr/bin/env python3
"""Build an eval dataset from SisFall K-fold test splits (real windows only).

Copies windows from the existing processed SisFall NPZ using the paper's
subject-independent fold split. Guarantees ambiguous near-falls are present:

  - n_ambiguous D18/D19 windows (default 100 = 50 D18 + 50 D19)
  - n_fall clear falls
  - n_adl clear ADLs (non D18/D19)

No synthetic IMU is generated.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_fall.data.sisfall import SisFallDataset
from agentic_fall.eval.protocol import make_fold_split
from agentic_fall.utils.io import ensure_dir, save_json

AMBIGUOUS_CODES = ("D18", "D19")


def _sample(idxs: list[int], n: int, rng: np.random.Generator) -> list[int]:
    if not idxs:
        return []
    if len(idxs) <= n:
        return list(idxs)
    return list(rng.choice(idxs, size=n, replace=False))


def build_fold_subset(
    ds: SisFallDataset,
    fold: int,
    *,
    n_ambiguous: int = 100,
    n_fall: int = 100,
    n_adl: int = 100,
    seed: int = 42,
    n_folds: int = 5,
    val_fraction: float = 0.15,
) -> tuple[dict, list[dict]]:
    rng = np.random.default_rng(seed + fold)
    split = make_fold_split(
        ds, fold=fold, n_folds=n_folds, seed=seed, val_fraction=val_fraction
    )
    test = list(split.test_idx)

    d18 = [i for i in test if str(ds.activities[i]) == "D18"]
    d19 = [i for i in test if str(ds.activities[i]) == "D19"]
    falls = [i for i in test if int(ds.y[i]) == 1]
    clear_adl = [
        i
        for i in test
        if int(ds.y[i]) == 0 and str(ds.activities[i]) not in AMBIGUOUS_CODES
    ]

    half = n_ambiguous // 2
    rem = n_ambiguous - half
    pick_d18 = _sample(d18, half, rng)
    pick_d19 = _sample(d19, rem, rng)
    # If one code is short, backfill from the other.
    need = n_ambiguous - (len(pick_d18) + len(pick_d19))
    if need > 0:
        pool = [i for i in (d18 + d19) if i not in set(pick_d18 + pick_d19)]
        pick_d18 += _sample(pool, need, rng)

    amb = pick_d18 + pick_d19
    # Avoid overlapping falls already taken (none should be)
    pick_fall = _sample(falls, n_fall, rng)
    pick_adl = _sample(clear_adl, n_adl, rng)

    chosen = list(amb) + list(pick_fall) + list(pick_adl)
    rng.shuffle(chosen)

    X = np.stack([np.asarray(ds.X[i], dtype=np.float32) for i in chosen], axis=0)
    y = np.asarray([int(ds.y[i]) for i in chosen], dtype=np.int64)
    subjects = np.asarray([str(ds.subjects[i]) for i in chosen])
    activities = np.asarray([str(ds.activities[i]) for i in chosen])
    file_ids = np.asarray([str(ds.file_ids[i]) for i in chosen])
    buckets = []
    for i in chosen:
        a = str(ds.activities[i])
        if a in AMBIGUOUS_CODES:
            buckets.append("ambiguous")
        elif int(ds.y[i]) == 1:
            buckets.append("strict_fall")
        else:
            buckets.append("clear_adl")
    buckets_arr = np.asarray(buckets)

    manifest = []
    for j, i in enumerate(chosen):
        manifest.append(
            {
                "idx": j,
                "source_index": int(i),
                "case_id": f"fold{fold}_{j:04d}",
                "bucket": buckets[j],
                "y": int(y[j]),
                "activity": str(activities[j]),
                "subject": str(subjects[j]),
                "file_id": str(file_ids[j]),
                "source": "sisfall_kfold",
                "fold": fold,
            }
        )

    payload = {
        "X": X,
        "y": y,
        "subjects": subjects,
        "activities": activities,
        "file_ids": file_ids,
        "buckets": buckets_arr,
        "source_indices": np.asarray(chosen, dtype=np.int64),
        "fold": np.asarray([fold] * len(chosen), dtype=np.int64),
    }
    meta = {
        "fold": fold,
        "n": len(chosen),
        "n_ambiguous": int((buckets_arr == "ambiguous").sum()),
        "n_strict_fall": int((buckets_arr == "strict_fall").sum()),
        "n_clear_adl": int((buckets_arr == "clear_adl").sum()),
        "n_d18": int(sum(1 for a in activities if a == "D18")),
        "n_d19": int(sum(1 for a in activities if a == "D19")),
        "n_fall_label": int((y == 1).sum()),
        "n_adl_label": int((y == 0).sum()),
        "test_pool_size": len(test),
        "requested": {
            "n_ambiguous": n_ambiguous,
            "n_fall": n_fall,
            "n_adl": n_adl,
        },
        "cases": manifest,
    }
    return payload, meta


def write_txt(meta: dict, path: Path) -> None:
    lines = [
        f"K-fold ambiguous eval set — fold {meta['fold']}",
        "=" * 72,
        f"n={meta['n']}  ambiguous={meta['n_ambiguous']} "
        f"(D18={meta['n_d18']} D19={meta['n_d19']})  "
        f"strict_fall={meta['n_strict_fall']}  clear_adl={meta['n_clear_adl']}",
        f"labels: fall={meta['n_fall_label']} ADL={meta['n_adl_label']}",
        f"source: SisFall K-fold test split (no synthetic windows)",
        "",
        f"{'case_id':<16} {'bucket':<12} {'y':>2} {'act':<5} {'subject':<8} file_id",
        "-" * 90,
    ]
    for c in meta["cases"]:
        lines.append(
            f"{c['case_id']:<16} {c['bucket']:<12} {c['y']:>2} {c['activity']:<5} "
            f"{c['subject']:<8} {c['file_id']}"
        )
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="data/processed/sisfall/windows_w90_h10.npz")
    ap.add_argument("--out-dir", default="data/kfold_ambiguous")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--n-ambiguous", type=int, default=100, help="Total D18+D19 (default 100)")
    ap.add_argument("--n-fall", type=int, default=100)
    ap.add_argument("--n-adl", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = ensure_dir(ROOT / args.out_dir)
    ds = SisFallDataset(ROOT / args.npz, channels=6)
    folds = [int(x) for x in args.folds.split(",") if x.strip() != ""]

    summary = []
    for fold in folds:
        payload, meta = build_fold_subset(
            ds,
            fold,
            n_ambiguous=args.n_ambiguous,
            n_fall=args.n_fall,
            n_adl=args.n_adl,
            seed=args.seed,
        )
        assert meta["n_ambiguous"] >= min(args.n_ambiguous, meta["n_ambiguous"])
        npz_path = out_dir / f"fold{fold}_ambiguous_eval.npz"
        np.savez_compressed(npz_path, **payload)
        save_json(meta, out_dir / f"fold{fold}_manifest.json")
        write_txt(meta, out_dir / f"fold{fold}_ambiguous_eval.txt")
        summary.append(
            {
                "fold": fold,
                "n": meta["n"],
                "n_ambiguous": meta["n_ambiguous"],
                "n_d18": meta["n_d18"],
                "n_d19": meta["n_d19"],
                "n_strict_fall": meta["n_strict_fall"],
                "n_clear_adl": meta["n_clear_adl"],
                "npz": str(npz_path.relative_to(ROOT)),
            }
        )
        print(
            f"fold{fold}: n={meta['n']} ambiguous={meta['n_ambiguous']} "
            f"(D18={meta['n_d18']} D19={meta['n_d19']}) "
            f"fall={meta['n_strict_fall']} adl={meta['n_clear_adl']} -> {npz_path}"
        )

    save_json({"folds": summary, "ambiguous_codes": list(AMBIGUOUS_CODES)}, out_dir / "summary.json")
    # Convenient combined txt for fold0
    print(f"Wrote {out_dir}/summary.json")


if __name__ == "__main__":
    main()
