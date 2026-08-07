#!/usr/bin/env python3
"""Prepare KFall windows (drop data into data/raw/kfall/ after access approval)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_fall.data.kfall import prepare_kfall
from agentic_fall.utils.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tier1_kfall.yaml")
    ap.add_argument("--phase", default="transition", choices=["pre", "post", "transition"])
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    raw = ROOT / cfg["raw_dir"]
    if not raw.exists():
        print(
            "KFall not found.\n"
            "1) Request access: https://sites.google.com/view/kfalldataset\n"
            "2) Extract into: data/raw/kfall/\n"
            "   Expected: subject folders with CSV + optional label xlsx\n"
        )
        raise SystemExit(1)

    path = prepare_kfall(
        raw_dir=raw,
        out_dir=ROOT / cfg["processed_dir"],
        window=int(cfg["window"]),
        hop=int(cfg["hop"]),
        sample_rate_hz=float(cfg["sample_rate_hz"]),
        rate_threshold=float(cfg["postfall"]["rate_threshold"]),
        stable_samples=int(cfg["postfall"]["stable_samples"]),
        phase=args.phase,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
