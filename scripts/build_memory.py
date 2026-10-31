#!/usr/bin/env python3
"""Build k-NN case memory from a trained Tier-1 model + evidence text embeddings."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_fall.agents.evidence import serialize_evidence
from agentic_fall.agents.knn_memory import KNNMemory, MemoryCase, public_case_id
from agentic_fall.data.sisfall import SisFallDataset
from agentic_fall.eval.protocol import make_fold_split, text_embedder
from agentic_fall.features.biomechanics import extract_biomechanics
from agentic_fall.models import build_model, kwargs_for_model
from agentic_fall.utils.config import load_config
from agentic_fall.utils.io import ensure_dir
from agentic_fall.utils.seed import set_seed


def memory_training_indices(ds, tcfg, pcfg, fold, max_cases):
    """Use the final fit split, never the outer train+validation pool."""
    if max_cases < 1:
        raise ValueError("max_cases must be positive")
    seed = int(tcfg["train"]["seed"])
    split = make_fold_split(
        ds, fold=fold, n_folds=int(tcfg["splits"]["n_folds"]),
        seed=seed, val_fraction=float(pcfg.get("splits", {}).get("val_fraction", 0.15)),
    )
    indices = list(split.train_idx)
    if len(indices) > max_cases:
        indices = np.random.default_rng(seed).choice(indices, max_cases, replace=False).tolist()
    return indices, split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier1-config", default="configs/tier1_sisfall.yaml")
    ap.add_argument("--agentic-config", default="configs/agentic.yaml")
    ap.add_argument("--paper-config", default="configs/paper_protocol.yaml")
    ap.add_argument("--out", default=None, help="Explicit memory output; avoids shared fold aliases")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--model", default="cnn_lstm_attn")
    ap.add_argument("--max-cases", type=int, default=5000)
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--allow-embedder-fallback",
        action="store_true",
        help="Permit random-projection embedder if sentence-transformers missing",
    )
    args = ap.parse_args()

    tcfg = load_config(ROOT / args.tier1_config)
    acfg = load_config(ROOT / args.agentic_config)
    pcfg = load_config(ROOT / args.paper_config)
    bcfg = load_config(ROOT / "configs/backbones.yaml") if (ROOT / "configs/backbones.yaml").exists() else {}
    set_seed(int(tcfg["train"]["seed"]))
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    pipe_cfg = acfg.get("pipeline") or {}
    include_activity = bool(pipe_cfg.get("include_activity", False))
    require_embedder = bool(pipe_cfg.get("require_text_embedder", True)) and not args.allow_embedder_fallback

    npz = ROOT / tcfg["processed_dir"] / f"windows_w{tcfg['window']}_h{tcfg['hop']}.npz"
    ds = SisFallDataset(npz, channels=int(tcfg["channels"]))
    train_idx, split = memory_training_indices(ds, tcfg, pcfg, args.fold, args.max_cases)

    if args.model == "cnn_lstm_attn":
        model = build_model(
            "cnn_lstm_attn",
            in_channels=int(tcfg["model"]["in_channels"]),
            num_classes=int(tcfg["num_classes"]),
            conv_channels=int(tcfg["model"]["conv_channels"]),
            branch_channels=int(tcfg["model"]["branch_channels"]),
            lstm_hidden=tcfg["model"]["lstm_hidden"],
            attn_heads=int(tcfg["model"]["attn_heads"]),
            dropout_attn=float(tcfg["model"]["dropout_attn"]),
            dropout_fc=float(tcfg["model"]["dropout_fc"]),
            use_se=bool(tcfg["model"].get("use_se", False)),
        )
    else:
        zoo_kw = kwargs_for_model(args.model, bcfg)
        model = build_model(args.model, in_channels=int(tcfg["channels"]), num_classes=2, **zoo_kw)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device).eval()

    embed_text = text_embedder(
        acfg["retrieval"]["embedder"],
        allow_fallback=not require_embedder,
    )
    memory = KNNMemory(k=int(acfg["retrieval"]["k"]), metric=acfg["retrieval"]["metric"])
    loader = DataLoader(Subset(ds, train_idx), batch_size=64, shuffle=False, num_workers=2)

    sr = float(acfg["evidence"]["sample_rate_hz"])
    with torch.no_grad():
        for batch in tqdm(loader, desc="build_memory"):
            x = batch["x"].to(device)
            for i in range(x.size(0)):
                # (C,T) -> (T,C)
                win = batch["x"][i].numpy().T
                feats = extract_biomechanics(
                    win,
                    sample_rate_hz=sr,
                    freefall_g_threshold=float(acfg["evidence"]["freefall_g_threshold"]),
                    stillness_var_threshold=float(acfg["evidence"]["stillness_var_threshold"]),
                )
                text = serialize_evidence(
                    feats,
                    activity=str(batch["activity"][i]),
                    include_activity=include_activity,
                )
                te = np.asarray(embed_text(text), dtype=np.float32)
                raw_id = f"{batch['file_id'][i]}_{int(batch['index'][i])}"
                memory.add(
                    MemoryCase(
                        case_id=public_case_id(raw_id),
                        embedding=te.tolist(),
                        label=int(batch["y"][i]),
                        evidence_text=text,
                        features=feats.to_dict(),
                        # Kept for hard-neg filter only; never emitted in summary().
                        activity=str(batch["activity"][i]),
                    )
                )
    out_path = ROOT / args.out if args.out else ROOT / acfg["retrieval"]["memory_dir"] / f"sisfall_fold{args.fold}.json"
    ensure_dir(out_path.parent)
    memory.save(out_path)
    manifest = {
        "fold": args.fold, "model": args.model, "checkpoint": str(Path(args.checkpoint).resolve()),
        "paper_config": str(args.paper_config), "source": str(npz),
        "memory_indices": train_idx,
        "fit_subjects": sorted({str(ds.subjects[i]) for i in split.train_idx}),
        "validation_subjects": sorted({str(ds.subjects[i]) for i in split.val_idx}),
        "test_subjects": sorted({str(ds.subjects[i]) for i in split.test_idx}),
        "memory_subjects": sorted({str(ds.subjects[i]) for i in train_idx}),
    }
    out_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Saved {len(memory)} cases -> {out_path}")


if __name__ == "__main__":
    main()
