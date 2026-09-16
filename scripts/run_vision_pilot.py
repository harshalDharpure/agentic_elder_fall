#!/usr/bin/env python3
"""End-to-end UR Fall VLM go/no-go pilot."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_fall.vision.cnn_baseline import train_cnn_baseline
from agentic_fall.vision.metrics_util import summarize_go_nogo
from agentic_fall.vision.urfall import prepare_urfall
from agentic_fall.vision.vlm_lora import train_lora
from agentic_fall.vision.vlm_zeroshot import run_zeroshot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-prepare", action="store_true")
    ap.add_argument("--skip-cnn", action="store_true")
    ap.add_argument("--skip-zeroshot", action="store_true")
    ap.add_argument("--skip-lora", action="store_true")
    ap.add_argument("--cnn-epochs", type=int, default=15)
    ap.add_argument("--lora-epochs", type=int, default=3)
    ap.add_argument("--model-id", default="Qwen/Qwen2-VL-2B-Instruct")
    args = ap.parse_args()

    out = ROOT / "results" / "vision"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log = {"started_at": time.strftime("%Y-%m-%d %H:%M:%S"), "steps": {}}

    if not args.skip_prepare:
        print("=== prepare UR Fall ===")
        p = prepare_urfall()
        log["steps"]["prepare"] = str(p)

    cnn_m = zeroshot_m = lora_m = None

    if not args.skip_cnn:
        print("=== CNN baseline ===")
        cnn_m = train_cnn_baseline(epochs=args.cnn_epochs)
        log["steps"]["cnn"] = cnn_m
        (out / "cnn_metrics.json").write_text(json.dumps(cnn_m, indent=2))
    else:
        p = out / "cnn_metrics.json"
        if p.exists():
            cnn_m = json.loads(p.read_text())

    if not args.skip_zeroshot:
        print("=== Zero-shot VLM ===")
        zeroshot_m = run_zeroshot(model_id=args.model_id)
        log["steps"]["zeroshot"] = zeroshot_m
        (out / "zeroshot_metrics.json").write_text(json.dumps(zeroshot_m, indent=2))
    else:
        p = out / "zeroshot_metrics.json"
        if p.exists():
            zeroshot_m = json.loads(p.read_text())

    if not args.skip_lora:
        print("=== LoRA VLM ===")
        lora_m = train_lora(model_id=args.model_id, epochs=args.lora_epochs)
        log["steps"]["lora"] = lora_m
        (out / "lora_metrics.json").write_text(json.dumps(lora_m, indent=2))
    else:
        p = out / "lora_metrics.json"
        if p.exists():
            lora_m = json.loads(p.read_text())

    if cnn_m and zeroshot_m and lora_m:
        verdict = summarize_go_nogo(cnn_m, zeroshot_m, lora_m)
        log["verdict"] = verdict
        (out / "pilot_go_nogo.json").write_text(json.dumps(verdict, indent=2))
        md = out / "PILOT_VERDICT.md"
        md.write_text(
            f"# Vision VLM Pilot Verdict\n\n"
            f"**Decision: {verdict['decision']}**\n\n"
            f"| Model | F1 | Recall | Spec | Cost/1k |\n"
            f"|---|---:|---:|---:|---:|\n"
            f"| CNN ResNet18 | {cnn_m['f1']:.3f} | {cnn_m['recall']:.3f} | {cnn_m['specificity']:.3f} | {cnn_m['cost_per_1000']:.1f} |\n"
            f"| Zero-shot VLM | {zeroshot_m['f1']:.3f} | {zeroshot_m['recall']:.3f} | {zeroshot_m['specificity']:.3f} | {zeroshot_m['cost_per_1000']:.1f} |\n"
            f"| LoRA VLM | {lora_m['f1']:.3f} | {lora_m['recall']:.3f} | {lora_m['specificity']:.3f} | {lora_m['cost_per_1000']:.1f} |\n\n"
            f"- beat_cnn: {verdict['beat_cnn']}\n"
            f"- beat_zeroshot: {verdict['beat_zeroshot']}\n"
            f"- criteria: `{json.dumps(verdict['criteria'])}`\n"
        )
        print("=== VERDICT:", verdict["decision"], "===")

    log["elapsed_sec"] = time.time() - t0
    (out / "pilot_run_log.json").write_text(json.dumps(log, indent=2))
    print("done in", log["elapsed_sec"], "s")


if __name__ == "__main__":
    main()
