#!/usr/bin/env python3
"""Export summary LaTeX/CSV tables from results/*.json."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--out-prefix", default="paper_tables")
    args = ap.parse_args()

    rows = []
    paths = list(sorted(args.results_dir.glob("*.json")))
    paths += list(sorted(args.results_dir.glob("fold*/*.json")))
    for path in paths:
        if path.name.startswith("aggregate_"):
            continue
        data = json.loads(path.read_text())
        if isinstance(data, list):
            for r in data:
                r = dict(r)
                r["source_file"] = str(path.relative_to(args.results_dir))
                rows.append(r)
        elif isinstance(data, dict) and "rows" in data:
            for r in data["rows"]:
                r = dict(r)
                r["source_file"] = str(path.relative_to(args.results_dir))
                rows.append(r)
        elif isinstance(data, dict) and "metrics" in data:
            r = dict(data["metrics"])
            r["name"] = path.stem
            r["source_file"] = str(path.relative_to(args.results_dir))
            rows.append(r)

    if not rows:
        print("No result JSON found.")
        return

    out_csv = args.results_dir / f"{args.out_prefix}.csv"
    keys = sorted({k for r in rows for k in r})
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
