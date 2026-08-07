#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_fall.data.sisfall import prepare_sisfall
from agentic_fall.utils.config import load_config


def main():
    ap = argparse.ArgumentParser(description="Prepare SisFall windows")
    ap.add_argument("--config", default="configs/tier1_sisfall.yaml")
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument(
        "--required-activities",
        nargs="*",
        default=["D18", "D19"],
        help="Fail if these activity codes are absent after prepare",
    )
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    raw = ROOT / cfg["raw_dir"]
    out = ROOT / cfg["processed_dir"]
    path = prepare_sisfall(
        raw_dir=raw,
        out_dir=out,
        window=int(cfg["window"]),
        hop=int(cfg["hop"]),
        max_files=args.max_files,
        required_activities=list(args.required_activities),
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
