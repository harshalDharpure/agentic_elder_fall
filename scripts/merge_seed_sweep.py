#!/usr/bin/env python3
"""Merge partial seed sweep rows into seed_sweep_fold0.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

OLLAMA_42 = {
    "seed": 42,
    "fold": 0,
    "tier1_f1": 0.7469586374695864,
    "tier1_recall": 0.7156177156177156,
    "stack_backend": "ollama",
    "stack_f1": 0.9087048832271762,
    "stack_recall": 0.9976689976689976,
    "stack_cost": 95.0,
    "stack_cost_per_1000": 95.0,
    "n": 1000,
    "tau_low": 0.07105263157894737,
    "tau_high": 0.55,
}


def main():
    out = ROOT / "results/seed_sweep/seed_sweep_fold0.json"
    by_seed: dict[int, dict] = {42: OLLAMA_42}

    if out.exists():
        for r in json.loads(out.read_text()).get("rows", []):
            if int(r["seed"]) in (43, 44):
                by_seed[int(r["seed"])] = r

    for p in sys.argv[1:]:
        d = json.loads(Path(p).read_text())
        for r in d.get("rows", [d]):
            if "seed" in r and int(r["seed"]) in (42, 43, 44):
                by_seed[int(r["seed"])] = r

    rows = [by_seed[s] for s in sorted(by_seed)]
    f1s = [r["tier1_f1"] for r in rows]
    sf1 = [r["stack_f1"] for r in rows]
    summary = {
        "fold": 0,
        "seeds": [r["seed"] for r in rows],
        "n_seeds": len(rows),
        "tier1_f1_mean": float(np.mean(f1s)),
        "tier1_f1_std": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
        "stack_f1_mean": float(np.mean(sf1)),
        "stack_f1_std": float(np.std(sf1, ddof=1)) if len(sf1) > 1 else 0.0,
        "rows": rows,
        "note": "Ollama/Mistral stack; seeds 42,43,44.",
    }
    out.write_text(json.dumps(summary, indent=2))
    print("Wrote", out, "seeds:", summary["seeds"])


if __name__ == "__main__":
    main()
