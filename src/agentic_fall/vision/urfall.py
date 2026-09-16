from __future__ import annotations

import json
import re
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
RAW_ZIPS = ROOT / "data" / "raw" / "urfall" / "zips"
RAW_RGB = ROOT / "data" / "raw" / "urfall" / "rgb"
PROC = ROOT / "data" / "processed" / "urfall"


@dataclass
class StripSample:
    sample_id: str
    sequence_id: str
    label: int  # 1=fall, 0=adl
    split: Literal["train", "test"]
    frame_paths: list[str]


def _seq_ids() -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for i in range(1, 31):
        out.append((f"fall-{i:02d}", 1))
    for i in range(1, 41):
        out.append((f"adl-{i:02d}", 0))
    return out


def extract_zips(force: bool = False, require_all: bool = False) -> dict[str, Path]:
    RAW_RGB.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, Path] = {}
    missing: list[str] = []
    for seq_id, _ in _seq_ids():
        zip_name = f"{seq_id}-cam0-rgb.zip"
        zpath = RAW_ZIPS / zip_name
        dest = RAW_RGB / seq_id
        if not zpath.exists() or zpath.stat().st_size == 0:
            missing.append(seq_id)
            continue
        if dest.exists() and any(dest.rglob("*.png")) and not force:
            mapping[seq_id] = dest
            continue
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(dest)
        mapping[seq_id] = dest
    if missing:
        print(f"[urfall] missing zips for {len(missing)} sequences: {missing[:5]}...")
    if require_all and missing:
        raise FileNotFoundError(f"Missing {len(missing)} UR Fall zips")
    if len(mapping) < 20:
        raise RuntimeError(f"Too few sequences extracted ({len(mapping)}); abort prepare")
    return mapping


def _list_frames(seq_dir: Path) -> list[Path]:
    frames = sorted(seq_dir.rglob("*.png"))
    if not frames:
        frames = sorted(seq_dir.rglob("*.jpg"))
    return frames


def _pick_indices(n: int, k: int, label: int) -> list[int]:
    """Evenly sample k frames; for falls prefer the later half (posture on floor)."""
    if n <= 0:
        return []
    if label == 1 and n >= 8:
        lo, hi = n // 2, n - 1
    else:
        lo, hi = 0, n - 1
    if hi <= lo:
        return [min(lo, n - 1)] * k
    pos = np.linspace(lo, hi, num=k, dtype=int)
    return [int(p) for p in pos]


def build_strips(
    frames_per_strip: int = 4,
    strips_per_seq: int = 3,
    seed: int = 42,
    test_frac: float = 0.3,
    available: dict[str, Path] | None = None,
) -> list[StripSample]:
    rng = np.random.default_rng(seed)
    seqs = [(s, y) for s, y in _seq_ids() if available is None or s in available]
    # stratified sequence split on available only
    falls = [s for s, y in seqs if y == 1]
    adls = [s for s, y in seqs if y == 0]
    rng.shuffle(falls)
    rng.shuffle(adls)
    n_fall_test = max(1, int(round(len(falls) * test_frac)))
    n_adl_test = max(1, int(round(len(adls) * test_frac)))
    test_set = set(falls[:n_fall_test] + adls[:n_adl_test])

    samples: list[StripSample] = []
    for seq_id, label in seqs:
        frames = _list_frames(RAW_RGB / seq_id)
        if len(frames) < frames_per_strip:
            continue
        split: Literal["train", "test"] = "test" if seq_id in test_set else "train"
        # multiple slightly shifted strips
        for s_i in range(strips_per_seq):
            shift = int(s_i * max(1, len(frames) // (strips_per_seq * 4)))
            idxs = _pick_indices(len(frames) - shift, frames_per_strip, label)
            idxs = [min(i + shift, len(frames) - 1) for i in idxs]
            paths = [str(frames[i].resolve()) for i in idxs]
            samples.append(
                StripSample(
                    sample_id=f"{seq_id}_s{s_i}",
                    sequence_id=seq_id,
                    label=label,
                    split=split,
                    frame_paths=paths,
                )
            )
    return samples


def write_manifest(samples: list[StripSample]) -> Path:
    PROC.mkdir(parents=True, exist_ok=True)
    manifest = {
        "n_samples": len(samples),
        "n_train": sum(1 for s in samples if s.split == "train"),
        "n_test": sum(1 for s in samples if s.split == "test"),
        "n_fall_train": sum(1 for s in samples if s.split == "train" and s.label == 1),
        "n_fall_test": sum(1 for s in samples if s.split == "test" and s.label == 1),
        "samples": [asdict(s) for s in samples],
    }
    path = PROC / "strips_manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    # VLM conversation JSON (train only for finetune; test for eval prompts)
    conv = []
    for s in samples:
        mid = s.frame_paths[len(s.frame_paths) // 2]
        ans = "fall" if s.label == 1 else "adl"
        conv.append(
            {
                "id": s.sample_id,
                "sequence_id": s.sequence_id,
                "split": s.split,
                "label": s.label,
                "image": mid,
                "images": s.frame_paths,
                "conversations": [
                    {
                        "from": "human",
                        "value": (
                            "<image>\nLook at this indoor camera frame of a person. "
                            "Answer with exactly one word: fall or adl. "
                            "Use fall if the person has fallen or is lying on the floor after a fall. "
                            "Use adl for normal daily activity (standing, sitting, walking, lying in bed intentionally)."
                        ),
                    },
                    {"from": "gpt", "value": ans},
                ],
            }
        )
    (PROC / "vlm_conversations.json").write_text(json.dumps(conv, indent=2))
    return path


def prepare_urfall() -> Path:
    available = extract_zips()
    samples = build_strips(available=available)
    if not samples:
        raise RuntimeError("No strip samples built from UR Fall RGB")
    return write_manifest(samples)


def load_manifest(path: Path | None = None) -> dict:
    path = path or (PROC / "strips_manifest.json")
    return json.loads(path.read_text())


_FALL_RE = re.compile(r"\bfall\b", re.I)
_ADL_RE = re.compile(r"\b(adl|not[\s-]?fall|no[\s-]?fall|daily)\b", re.I)


def parse_vlm_label(text: str) -> int:
    t = (text or "").strip().lower()
    if t.startswith("fall") or _FALL_RE.search(t[:40]):
        # prefer explicit adl if both appear early
        if t.startswith("adl") or t.startswith("not"):
            return 0
        return 1
    if t.startswith("adl") or _ADL_RE.search(t[:60]):
        return 0
    if "lying on the floor" in t or "fallen" in t:
        return 1
    return 0


def load_strip_image(paths: list[str], size: int = 224) -> Image.Image:
    """Stack middle frame (CNN) — keep simple for pilot."""
    mid = paths[len(paths) // 2]
    img = Image.open(mid).convert("RGB")
    return img.resize((size, size), Image.BILINEAR)


if __name__ == "__main__":
    p = prepare_urfall()
    print("wrote", p)
