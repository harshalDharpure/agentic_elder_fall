#!/usr/bin/env python3
"""Symlink or verify SisFall raw data under data/raw/sisfall."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

DEFAULT_SRC = Path(
    "/DATA/tauseef_2121cs04/agentic-ai/Fall-detection-Using-Agentic-AI/"
    "fallsage/datasets/raw/SisFall"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--dst", type=Path, default=Path("data/raw/sisfall"))
    args = ap.parse_args()

    dst = args.dst.resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        print(f"Already present: {dst}")
    else:
        if not args.src.exists():
            raise SystemExit(
                f"SisFall not found at {args.src}. "
                "Download from http://sistemic.udea.edu.co/ and place under data/raw/sisfall/"
            )
        os.symlink(args.src.resolve(), dst)
        print(f"Linked {args.src} -> {dst}")

    n = len(list(dst.rglob("*.txt")))
    print(f"Found {n} txt files")


if __name__ == "__main__":
    main()
