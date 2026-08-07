#!/usr/bin/env python3
"""Stage-1 sensor–language contrastive alignment (lightweight)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_fall.alignment import SensorEncoder, caption_from_window, info_nce
from agentic_fall.data.sisfall import SisFallDataset
from agentic_fall.utils.config import load_config
from agentic_fall.utils.io import ensure_dir, save_checkpoint
from agentic_fall.utils.seed import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tier1_sisfall.yaml")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--max-samples", type=int, default=8000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    set_seed(int(cfg["train"]["seed"]))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    npz = ROOT / cfg["processed_dir"] / f"windows_w{cfg['window']}_h{cfg['hop']}.npz"
    ds = SisFallDataset(npz, channels=int(cfg["channels"]))
    train_idx, _ = ds.fold_indices(seed=int(cfg["train"]["seed"]))[args.fold]
    train_idx = train_idx[: args.max_samples]
    loader = DataLoader(Subset(ds, train_idx), batch_size=args.batch_size, shuffle=True, num_workers=2)

    # Frozen text encoder
    try:
        from sentence_transformers import SentenceTransformer

        st = SentenceTransformer(args.embedder, device=str(device))
        text_dim = st.get_sentence_embedding_dimension()
    except Exception as e:
        print(f"Falling back to hash text encoder ({e})")
        st = None
        text_dim = 384

    enc = SensorEncoder(in_channels=int(cfg["channels"]), embed_dim=text_dim).to(device)
    opt = torch.optim.Adam(enc.parameters(), lr=args.lr)

    enc.train()
    for epoch in range(1, args.epochs + 1):
        total = 0.0
        n = 0
        for batch in tqdm(loader, desc=f"align {epoch}"):
            x = batch["x"].to(device)
            captions = [
                caption_from_window(batch["x"][i], sample_rate_hz=float(cfg["sample_rate_hz"]))
                for i in range(x.size(0))
            ]
            if st is not None:
                with torch.no_grad():
                    tz = st.encode(captions, convert_to_tensor=True, normalize_embeddings=True)
                    tz = tz.to(device)
            else:
                # deterministic hash embedding
                vecs = []
                for cap in captions:
                    v = torch.zeros(text_dim)
                    for i, ch in enumerate(cap.encode()[:512]):
                        v[i % text_dim] += ch / 255.0
                    vecs.append(v / (v.norm() + 1e-8))
                tz = torch.stack(vecs).to(device)

            sz = enc(x)
            loss = info_nce(sz, tz)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * x.size(0)
            n += x.size(0)
        print(f"epoch {epoch}: loss={total / max(1, n):.4f}")

    out = ensure_dir(ROOT / "checkpoints" / "alignment") / f"sensor_encoder_fold{args.fold}.pt"
    save_checkpoint({"model": enc.state_dict(), "embed_dim": text_dim}, out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
