from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from agentic_fall.vision.metrics_util import cost_metrics
from agentic_fall.vision.urfall import PROC, load_manifest, parse_vlm_label
from agentic_fall.vision.vlm_zeroshot import DEFAULT_MODEL, PROMPT, _generate_one, _load_qwen


class FallVLMDataset(Dataset):
    def __init__(self, conversations: list[dict], processor, split: str = "train"):
        self.rows = [c for c in conversations if c["split"] == split]
        self.processor = processor

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        image = Image.open(row["image"]).convert("RGB")
        answer = "fall" if row["label"] == 1 else "adl"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": PROMPT},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text, "image": image, "label": row["label"], "id": row["id"]}


def _collate(batch, processor):
    texts = [b["text"] for b in batch]
    images = [b["image"] for b in batch]
    batch_out = processor(text=texts, images=images, return_tensors="pt", padding=True)
    labels = batch_out["input_ids"].clone()
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is not None:
        labels[labels == pad_id] = -100
    batch_out["labels"] = labels
    return batch_out


def train_lora(
    model_id: str = DEFAULT_MODEL,
    epochs: int = 3,
    batch_size: int = 1,
    grad_accum: int = 8,
    lr: float = 1e-4,
    out_dir: Path | None = None,
) -> dict:
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader

    from agentic_fall.vision.vlm_zeroshot import _model_device

    out_dir = out_dir or (Path(__file__).resolve().parents[3] / "checkpoints" / "vision" / "qwen2vl_lora")
    out_dir.mkdir(parents=True, exist_ok=True)

    conv = json.loads((PROC / "vlm_conversations.json").read_text())
    processor, model = _load_qwen(model_id)
    # enable grads for LoRA (device_map models may freeze)
    model.train()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = FallVLMDataset(conv, processor, split="train")

    def collate(batch):
        return _collate(batch, processor)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate)

    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
    step = 0
    model.train()
    device = _model_device(model)
    for ep in range(epochs):
        running = 0.0
        opt.zero_grad()
        pbar = tqdm(loader, desc=f"lora-ep{ep+1}")
        for batch in pbar:
            batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / grad_accum
            loss.backward()
            running += float(out.loss.item())
            step += 1
            if step % grad_accum == 0:
                opt.step()
                opt.zero_grad()
            pbar.set_postfix(loss=running / max(1, pbar.n))
        if step % grad_accum != 0:
            opt.step()
            opt.zero_grad()
        print(f"[lora] epoch {ep+1} mean_loss={running / max(1, len(loader)):.4f}")

    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)

    # free base before eval reload
    del model
    torch.cuda.empty_cache()

    metrics = eval_lora(model_id=model_id, adapter_dir=out_dir)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


@torch.no_grad()
def eval_lora(
    model_id: str = DEFAULT_MODEL,
    adapter_dir: Path | None = None,
    out_path: Path | None = None,
) -> dict:
    from peft import PeftModel

    adapter_dir = adapter_dir or (
        Path(__file__).resolve().parents[3] / "checkpoints" / "vision" / "qwen2vl_lora"
    )
    processor, base = _load_qwen(model_id)
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()

    manifest = load_manifest()
    samples = [s for s in manifest["samples"] if s["split"] == "test"]
    ys, ps, texts = [], [], []
    for s in tqdm(samples, desc="lora-eval"):
        mid = s["frame_paths"][len(s["frame_paths"]) // 2]
        try:
            text = _generate_one(processor, model, mid)
        except Exception as e:
            text = f"ERROR:{e}"
        pred = parse_vlm_label(text)
        ys.append(int(s["label"]))
        ps.append(pred)
        texts.append({"id": s["sample_id"], "text": text, "pred": pred, "label": s["label"]})

    metrics = cost_metrics(ys, ps)
    metrics["model"] = model_id
    metrics["mode"] = "lora"
    metrics["adapter"] = str(adapter_dir)
    metrics["n_test"] = len(ys)

    out_path = out_path or (
        Path(__file__).resolve().parents[3] / "results" / "vision" / "lora_preds.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"metrics": metrics, "preds": texts}, indent=2))
    print("[lora-eval]", metrics)
    return metrics
