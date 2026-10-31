#!/usr/bin/env python3
"""Isolated SisFall backbone-swap campaign. Never runs KFall or edits defaults."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from datetime import datetime, timezone

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2))


def run_step(command, log, threads):
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), PYTHONUNBUFFERED="1")
    print("Running:", " ".join(map(str, command)), flush=True)
    with Path(log).open("w") as stream:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f"Step exited {completed.returncode}; inspect {log}")


def aggregate(run_dir, models, folds, smoke):
    summary = {"smoke_only": smoke, "folds": folds, "models": {}}
    for model in models:
        metrics = [json.loads((run_dir / model / "checkpoints" / f"{model}_fold{fold}_metrics.json").read_text())
                   for fold in folds]
        values = [row["test"]["f1"] for row in metrics]
        summary["models"][model] = {
            "f1_mean": statistics.mean(values),
            "f1_sample_std": statistics.stdev(values) if len(values) > 1 else None,
            "per_fold": metrics,
        }
    if all(model in summary["models"] for model in ("harmamba", "cnn_lstm_attn")):
        summary["mamba_minus_cnn_f1_percentage_points"] = 100 * (
            summary["models"]["harmamba"]["f1_mean"] - summary["models"]["cnn_lstm_attn"]["f1_mean"])
    if not smoke:
        summary["agentic"] = {}
        for model in models:
            by_mode = {}
            for fold in folds:
                payload = json.loads((run_dir / model / f"fold{fold}" / f"ablation_fold{fold}.json").read_text())
                for row in payload.get("rows", []) if isinstance(payload, dict) else payload:
                    if row.get("f1") is not None:
                        by_mode.setdefault(row["name"], []).append(row["f1"])
            summary["agentic"][model] = {
                mode: {"n_folds": len(values), "f1_mean": statistics.mean(values),
                       "f1_sample_std": statistics.stdev(values) if len(values) > 1 else None}
                for mode, values in by_mode.items()
            }
    write_json(run_dir / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "smoke", "full"), default="preflight")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--models", nargs="+", choices=("harmamba", "cnn_lstm_attn"),
                        default=["cnn_lstm_attn", "harmamba"])
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    folds = [int(value) for value in args.folds.split(",")]
    if not folds or len(set(folds)) != len(folds) or any(fold not in range(5) for fold in folds):
        parser.error("folds must be unique members of 0,1,2,3,4")
    if len(set(args.models)) != len(args.models) or args.threads < 1:
        parser.error("models must be unique and threads positive")
    smoke = args.stage == "smoke"
    if smoke:
        folds = [folds[0]]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = (ROOT / (args.out_dir or f"results/sisfall_mamba/{args.stage}_{timestamp}")).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        parser.error(f"Refusing to overwrite nonempty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    sources = [ROOT / path for path in ("configs/tier1_sisfall.yaml", "configs/paper_protocol.yaml", "configs/agentic.yaml", "configs/backbones.yaml")]
    tier, paper, agentic, _ = [yaml.safe_load(path.read_text()) for path in sources]
    data = ROOT / tier["processed_dir"] / f"windows_w{tier['window']}_h{tier['hop']}.npz"
    manifest = {
        "status": "preflight", "stage": args.stage, "smoke_only": smoke, "device": args.device,
        "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "models": args.models, "folds": folds, "data_exists": data.exists(),
        "data_sha256": sha256(data) if data.exists() else None,
        "source_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
        "code_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in
                        [ROOT / "scripts" / name for name in ("run_sisfall_mamba.py", "train_tier1.py", "build_memory.py", "run_ablations.py")] +
                        sorted((ROOT / "src" / "agentic_fall").rglob("*.py"))},
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)),
        "notes": ["Existing custom HARMamba-style bidirectional SSM, not official mamba-ssm.",
                  "Fixed 90-sample trial-labelled windows retained to isolate backbone swap.",
                  "No event-level, prospective, or universal-generalization claims.",
                  "No KFall commands; original configurations/checkpoints/results are preserved."],
    }
    errors = []
    if not data.exists():
        errors.append(f"Missing processed SisFall: {data}")
    if args.device == "cuda" and not torch.cuda.is_available():
        errors.append("CUDA unavailable; full GPU campaign cannot start. CPU smoke is supported.")
    if args.stage != "smoke":
        if importlib.util.find_spec("sentence_transformers") is None:
            errors.append("sentence-transformers is missing; real retrieval embeddings are required.")
        try:
            import requests
            response = requests.get(agentic["llm"]["base_url"] + "/api/tags", timeout=3)
            response.raise_for_status()
            available = [entry["name"] for entry in response.json().get("models", [])]
            manifest["ollama_models"] = available
            manifest["ollama_model_details"] = response.json().get("models", [])
            model = agentic["llm"]["model"]
            if model not in available and model + ":latest" not in available:
                errors.append(f"Required local LLM {model!r} is not installed.")
        except Exception as error:
            errors.append(f"Ollama unavailable: {type(error).__name__}: {error}")
    manifest["errors"] = errors
    manifest["status"] = "blocked" if errors else "ready"
    write_json(run_dir / "manifest.json", manifest)
    print(json.dumps({"run_dir": str(run_dir), "status": manifest["status"], "errors": errors}, indent=2), flush=True)
    if errors:
        return 2
    if args.stage == "preflight":
        return 0
    manifest["status"] = "running"
    write_json(run_dir / "manifest.json", manifest)
    try:
        for model in args.models:
            model_dir = run_dir / model
            model_dir.mkdir()
            tc, pc, ac = [yaml.safe_load(yaml.safe_dump(config)) for config in (tier, paper, agentic)]
            tc["train"]["device"] = args.device
            tc["train"]["num_workers"] = 0
            ac["pipeline"]["include_activity"] = False
            ac["pipeline"]["require_text_embedder"] = True
            ac["llm"]["fallback_heuristic"] = False
            ac["evidence"]["sample_rate_hz"] = tc["sample_rate_hz"]
            ac["retrieval"]["memory_dir"] = str(model_dir / "memory")
            pc["agentic"]["primary_backbone"] = model
            pc["agentic"]["backend"] = "ollama"
            pc["paths"] = {"results_dir": str(model_dir), "checkpoint_dir": str(model_dir / "checkpoints"), "memory_dir": str(model_dir / "memory")}
            if smoke:
                tc["train"]["batch_size"] = 32
                pc["train"].update(epochs=1, max_train=256, max_val=128, patience=1, amp=False)
                pc["eval"].update(max_test=128, agentic_max_samples=128, min_near_fall_per_code=0)
            tier_path, paper_path, agentic_path = [model_dir / name for name in ("tier1.yaml", "protocol.yaml", "agentic.yaml")]
            pc["tier1_config"], pc["agentic_config"] = str(tier_path), str(agentic_path)
            for path, value in ((tier_path, tc), (paper_path, pc), (agentic_path, ac)):
                path.write_text(yaml.safe_dump(value, sort_keys=False))
            for fold in folds:
                manifest["current_model"], manifest["current_fold"] = model, fold
                manifest["current_step"] = "training"
                write_json(run_dir / "manifest.json", manifest)
                fold_dir = model_dir / f"fold{fold}"
                fold_dir.mkdir()
                checkpoint = model_dir / "checkpoints" / f"{model}_fold{fold}.pt"
                command = [sys.executable, "scripts/train_tier1.py", "--config", str(tier_path),
                           "--paper-config", str(paper_path), "--model", model, "--fold", str(fold),
                           "--device", args.device, "--checkpoint-dir", str(checkpoint.parent)]
                run_step(command, fold_dir / "train.log", args.threads)
                if smoke:
                    continue
                manifest["current_step"] = "memory"
                write_json(run_dir / "manifest.json", manifest)
                memory = model_dir / "memory" / f"sisfall_fold{fold}.json"
                run_step([sys.executable, "scripts/build_memory.py", "--tier1-config", str(tier_path),
                          "--paper-config", str(paper_path), "--agentic-config", str(agentic_path),
                          "--checkpoint", str(checkpoint), "--fold", str(fold), "--model", model,
                          "--device", args.device, "--out", str(memory)], fold_dir / "memory.log", args.threads)
                manifest["current_step"] = "agentic_evaluation"
                write_json(run_dir / "manifest.json", manifest)
                run_step([sys.executable, "scripts/run_ablations.py", "--paper-config", str(paper_path),
                          "--fold", str(fold), "--checkpoint", str(checkpoint), "--memory", str(memory),
                          "--device", args.device, "--skip-baselines", "--modes",
                          "tier1_only,gate_only,gate_knn,gate_knn_llm", "--out-dir", str(fold_dir)],
                         fold_dir / "ablation.log", args.threads)
        summary = aggregate(run_dir, args.models, folds, smoke)
        manifest["status"] = "smoke_complete" if smoke else "complete"
        manifest["current_step"] = "finished"
        manifest["summary"] = str(run_dir / "summary.json")
        print(json.dumps({"status": manifest["status"], "summary": manifest["summary"],
                          "f1": {model: row["f1_mean"] for model, row in summary["models"].items()}}, indent=2))
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(run_dir / "manifest.json", manifest)
        raise
    write_json(run_dir / "manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
