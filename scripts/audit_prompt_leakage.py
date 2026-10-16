#!/usr/bin/env python3
"""Fail if System-2 prompts / evidence still contain activity codes.

Scans:
  - logged prompt text files under --prompts
  - ablation / agentic result JSON ``windows[*].evidence`` / ``evidence_text``
  - kNN memory JSON ``evidence_text`` fields

Exit code 1 if any leak is found.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_fall.agents.evidence import evidence_contains_activity_code  # noqa: E402


def scan_text(label: str, text: str, hits: list[str]) -> None:
    if evidence_contains_activity_code(text):
        snippet = text.replace("\n", " ")[:160]
        hits.append(f"{label}: {snippet}")


def scan_prompts(folder: Path, hits: list[str]) -> None:
    if not folder.exists():
        return
    for p in folder.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".txt", ".json", ".log", ".md", ""}:
            continue
        try:
            text = p.read_text(errors="ignore")
        except OSError:
            continue
        if p.suffix.lower() == ".json":
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                scan_text(str(p), text, hits)
                continue
            blob = json.dumps(obj)
            scan_text(str(p), blob, hits)
        else:
            scan_text(str(p), text, hits)


def scan_result_json(path: Path, hits: list[str]) -> None:
    try:
        obj = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return
    windows = obj.get("windows") or obj.get("logs") or []
    if isinstance(windows, list):
        for i, w in enumerate(windows):
            if not isinstance(w, dict):
                continue
            for key in ("evidence", "evidence_text", "prompt", "user_prompt"):
                if key in w and isinstance(w[key], str):
                    scan_text(f"{path.name}:{key}[{i}]", w[key], hits)
    # memory-style list
    if isinstance(obj, list):
        for i, c in enumerate(obj[:500]):  # sample
            if isinstance(c, dict) and "evidence_text" in c:
                scan_text(f"{path.name}:evidence_text[{i}]", c["evidence_text"], hits)


def scan_memory(path: Path, hits: list[str], max_cases: int = 0) -> None:
    cases = json.loads(path.read_text())
    n = len(cases) if max_cases <= 0 else min(len(cases), max_cases)
    for i in range(n):
        scan_text(f"{path.name}[{i}]", cases[i].get("evidence_text", ""), hits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=str, default="", help="Folder of logged System-2 prompts")
    ap.add_argument("--results", nargs="*", default=[], help="Result JSON files to scan")
    ap.add_argument("--memory-dir", type=str, default="", help="Memory dir to scan")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        assert evidence_contains_activity_code("- Activity context code: D18")
        assert evidence_contains_activity_code("code F05 nearby")
        assert not evidence_contains_activity_code("- Free-fall duration: 0.2s")
        print("self-test OK")
        return

    hits: list[str] = []
    if args.prompts:
        scan_prompts(Path(args.prompts), hits)
    for r in args.results:
        scan_result_json(Path(r), hits)
    if args.memory_dir:
        md = Path(args.memory_dir)
        for p in sorted(md.glob("*.json")):
            if "with_activity" in p.name or p.name.endswith(".bak"):
                continue
            scan_memory(p, hits)

    if hits:
        print(f"LEAKAGE: {len(hits)} hit(s)")
        for h in hits[:40]:
            print(" ", h)
        if len(hits) > 40:
            print(f"  ... and {len(hits) - 40} more")
        raise SystemExit(1)
    print("OK: no activity-code leakage found in scanned inputs")


if __name__ == "__main__":
    main()
