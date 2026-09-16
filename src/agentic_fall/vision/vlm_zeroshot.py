from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from agentic_fall.vision.metrics_util import cost_metrics
from agentic_fall.vision.urfall import load_manifest, parse_vlm_label

PROMPT = (
    "Look at this indoor camera frame of a person. "
    "Answer with exactly one word: fall or adl. "
    "Use fall if the person has fallen or is lying on the floor after a fall. "
    "Use adl for normal daily activity (standing, sitting, walking, lying in bed intentionally)."
)

DEFAULT_MODEL = "Qwen/Qwen2-VL-2B-Instruct"


def _load_qwen(model_id: str, device: str = "cuda"):
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    if not torch.cuda.is_available():
        model = model.to(device)
    model.eval()
    return processor, model


def _model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def _generate_one(processor, model, image_path: str, max_new_tokens: int = 16) -> str:
    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    device = _model_device(model)
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = out[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def run_zeroshot(
    model_id: str = DEFAULT_MODEL,
    split: str = "test",
    limit: int | None = None,
    out_path: Path | None = None,
) -> dict:
    manifest = load_manifest()
    samples = [s for s in manifest["samples"] if s["split"] == split]
    if limit is not None:
        samples = samples[:limit]

    processor, model = _load_qwen(model_id)
    ys, ps, texts = [], [], []
    for s in tqdm(samples, desc="zeroshot-vlm"):
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
    metrics["mode"] = "zeroshot"
    metrics["n_test"] = len(ys)

    out_path = out_path or (
        Path(__file__).resolve().parents[3] / "results" / "vision" / "zeroshot_preds.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"metrics": metrics, "preds": texts}, indent=2))
    print("[zeroshot]", metrics)
    return metrics
