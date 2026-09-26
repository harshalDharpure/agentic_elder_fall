#!/usr/bin/env python3
"""Lightweight manuscript guards for Week-0 factual claims."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX_FILES = [
    ROOT / "paper" / "main.tex",
    ROOT / "overleaf_rcdp" / "main.tex",
]

FORBIDDEN = [
    (re.compile(r"sudden\s+sit", re.I), "D19 must not be labelled 'sudden sit'"),
    (re.compile(r"sudden\s+sitting", re.I), "D19 must not be labelled 'sudden sitting'"),
]

REQUIRED = [
    (re.compile(r"gently\s+jump", re.I), "D19 correct label 'gently jump' missing"),
    (re.compile(r"elderly", re.I), "elderly disclosure missing"),
    (re.compile(r"D05|D11", re.I), "elderly-feasible proxy D05/D11 commitment missing"),
]


def check_file(path: Path) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return [f"missing file: {path}"]
    text = path.read_text(encoding="utf-8", errors="replace")
    for rx, msg in FORBIDDEN:
        if rx.search(text):
            errors.append(f"{path.name}: {msg}")
    for rx, msg in REQUIRED:
        if not rx.search(text):
            errors.append(f"{path.name}: {msg}")
    return errors


def main() -> None:
    errs: list[str] = []
    for p in TEX_FILES:
        errs.extend(check_file(p))
    if errs:
        print("check_manuscript FAILED:")
        for e in errs:
            print(f"  - {e}")
        sys.exit(1)
    print("check_manuscript OK: D19 label + elderly disclosure present in both TeX trees.")


if __name__ == "__main__":
    main()
