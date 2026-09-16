#!/usr/bin/env python3
"""Build k-NN case memory from a trained Tier-1 model + evidence text embeddings."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_fall.agents.evidence import serialize_evidence
from agentic_fall.agents.knn_memory import KNNMemory, MemoryCase
from agentic_fall.data.sisfall import SisFallDataset
from agentic_fall.features.biomechanics import extract_biomechanics
from agentic_fall.models import build_model, kwargs_for_model
from agentic_fall.utils.config import load_config
from agentic_fall.utils.io import ensure_dir
from agentic_fall.utils.seed import set_seed


def get_text_embedder(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer

        st = SentenceTransformer(model_name)
        def embed(text: str) -> np.ndarray:
            return st.encode(text, normalize_embeddings=True)
        return embed
    except Exception as e:
        print(f"sentence-transformers unavailable ({e}); using random projection fallback")
        rng = np.random.default_rng(0)
        W = rng.normal(size=(64, 384)).astype(np.float32)

        def embed(text: str) -> np.ndarray:
            # bag-of-bytes hash embedding
            v = np.zeros(64, dtype=np.float32)
            for i, ch in enumerate(text.encode("utf-8")[:512]):
                v[i % 64] += (ch / 255.0)
            z = W.T @ v
            z = z / (np.linalg.norm(z) + 1e-8)
            return z
        return embed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier1-config", default="configs/tier1_sisfall.yaml")
    ap.add_argument("--agentic-config", default="configs/agentic.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--model", default="cnn_lstm_attn")
    ap.add_argument("--max-cases", type=int, default=5000)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    tcfg = load_config(ROOT / args.tier1_config)
    acfg = load_config(ROOT / args.agentic_config)
    bcfg = load_config(ROOT / "configs/backbones.yaml") if (ROOT / "configs/backbones.yaml").exists() else {}
    set_seed(int(tcfg["train"]["seed"]))
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    npz = ROOT / tcfg["processed_dir"] / f"windows_w{tcfg['window']}_h{tcfg['hop']}.npz"
    ds = SisFallDataset(npz, channels=int(tcfg["channels"]))
    folds = ds.fold_indices(n_folds=int(tcfg["splits"]["n_folds"]), seed=int(tcfg["train"]["seed"]))
    train_idx, _ = folds[args.fold]
    if len(train_idx) > args.max_cases:
        rng = np.random.default_rng(tcfg["train"]["seed"])
        train_idx = list(rng.choice(train_idx, size=args.max_cases, replace=False))

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

    embed_text = get_text_embedder(acfg["retrieval"]["embedder"])
    memory = KNNMemory(k=int(acfg["retrieval"]["k"]), metric=acfg["retrieval"]["metric"])
    loader = DataLoader(Subset(ds, train_idx), batch_size=64, shuffle=False, num_workers=2)

    sr = float(acfg["evidence"]["sample_rate_hz"])
    with torch.no_grad():
        for batch in tqdm(loader, desc="build_memory"):
            x = batch["x"].to(device)
            _, emb = model(x)
            emb = emb.cpu().numpy()
            for i in range(x.size(0)):
                # (C,T) -> (T,C)
                win = batch["x"][i].numpy().T
                feats = extract_biomechanics(
                    win,
                    sample_rate_hz=sr,
                    freefall_g_threshold=float(acfg["evidence"]["freefall_g_threshold"]),
                    stillness_var_threshold=float(acfg["evidence"]["stillness_var_threshold"]),
                )
                text = serialize_evidence(feats, activity=str(batch["activity"][i]))
                # blend text embedding with model emb (concat then will use text primarily)
                te = np.asarray(embed_text(text), dtype=np.float32)
                # if dims differ, use text only for retrieval
                q = te
                memory.add(
                    MemoryCase(
                        case_id=f"{batch['file_id'][i]}_{int(batch['index'][i])}",
                        embedding=q.tolist(),
                        label=int(batch["y"][i]),
                        evidence_text=text,
                        features=feats.to_dict(),
                        activity=str(batch["activity"][i]),
                    )
                )

    out_dir = ensure_dir(ROOT / acfg["retrieval"]["memory_dir"])
    out_path = out_dir / f"sisfall_fold{args.fold}.json"
    memory.save(out_path)
    print(f"Saved {len(memory)} cases -> {out_path}")


if __name__ == "__main__":
    main()
