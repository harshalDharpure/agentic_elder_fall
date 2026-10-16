#!/usr/bin/env python3
"""Strip activity codes from kNN memory evidence_text and re-embed.

Existing memories were built with ``Activity context code: Dxx/Fxx`` in the
free-text evidence. Those strings leak into System-2 prompts via case
summaries. This script rewrites every memory JSON in place (with .bak backup)
so evidence_text has no activity codes and embeddings match the scrubbed text.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_fall.agents.evidence import (  # noqa: E402
    evidence_contains_activity_code,
    strip_activity_from_evidence,
)


def get_text_embedder(model_name: str):
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(model_name)

    def embed_batch(texts: list[str]) -> np.ndarray:
        return np.asarray(
            st.encode(texts, normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )

    return embed_batch


def scrub_file(path: Path, embed_batch, batch_size: int = 256, dry_run: bool = False) -> dict:
    cases = json.loads(path.read_text())
    dirty = 0
    texts: list[str] = []
    for c in cases:
        raw = c.get("evidence_text", "")
        if evidence_contains_activity_code(raw):
            dirty += 1
        clean = strip_activity_from_evidence(raw)
        c["evidence_text"] = clean
        texts.append(clean)

    if dry_run:
        return {"path": str(path), "n": len(cases), "dirty": dirty, "dry_run": True}

    bak = path.with_suffix(path.suffix + ".with_activity.bak")
    if not bak.exists():
        shutil.copy2(path, bak)

    for i in tqdm(range(0, len(texts), batch_size), desc=path.name, leave=False):
        chunk = texts[i : i + batch_size]
        emb = embed_batch(chunk)
        for j, row in enumerate(emb):
            cases[i + j]["embedding"] = row.tolist()

    path.write_text(json.dumps(cases))
    # verify
    still = sum(1 for c in cases if evidence_contains_activity_code(c["evidence_text"]))
    return {"path": str(path), "n": len(cases), "dirty": dirty, "still_dirty": still}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--memory-dir",
        default="data/memory",
        help="Directory containing *fold*.json memories",
    )
    ap.add_argument(
        "--pattern",
        default="*.json",
        help="Glob under memory-dir (skip .bak)",
    )
    ap.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    mem_dir = ROOT / args.memory_dir
    paths = sorted(
        p
        for p in mem_dir.glob(args.pattern)
        if p.is_file() and not p.name.endswith(".bak") and "with_activity" not in p.name
    )
    # Prefer real files over symlinks that point into the same dir (avoid double-scrub).
    real_seen: set[Path] = set()
    unique: list[Path] = []
    for p in paths:
        try:
            rp = p.resolve()
        except OSError:
            rp = p
        if rp in real_seen:
            continue
        real_seen.add(rp)
        unique.append(p)

    print(f"Found {len(unique)} memory files under {mem_dir}")
    embed_batch = None if args.dry_run else get_text_embedder(args.embedder)
    for p in unique:
        info = scrub_file(p, embed_batch, batch_size=args.batch_size, dry_run=args.dry_run)
        print(info)
        if not args.dry_run and info.get("still_dirty", 0) > 0:
            raise SystemExit(f"scrub failed for {p}")


if __name__ == "__main__":
    main()
