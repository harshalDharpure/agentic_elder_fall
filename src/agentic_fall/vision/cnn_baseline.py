from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

from agentic_fall.vision.metrics_util import cost_metrics
from agentic_fall.vision.urfall import load_manifest


class StripDataset(Dataset):
    def __init__(self, samples: list[dict], train: bool):
        self.samples = samples
        if train:
            self.tf = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(0.1, 0.1, 0.1),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.tf = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        mid = s["frame_paths"][len(s["frame_paths"]) // 2]
        img = Image.open(mid).convert("RGB")
        x = self.tf(img)
        y = int(s["label"])
        return x, y


def _build_resnet18(num_classes: int = 2) -> nn.Module:
    try:
        weights = models.ResNet18_Weights.DEFAULT
        m = models.resnet18(weights=weights)
    except Exception:
        m = models.resnet18(pretrained=True)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


@torch.no_grad()
def _predict(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    ys, ps, probs = [], [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        p = torch.softmax(logits, dim=-1)[:, 1]
        pred = (p >= 0.5).long().cpu().numpy()
        ys.extend(y.numpy().tolist())
        ps.extend(pred.tolist())
        probs.extend(p.cpu().numpy().tolist())
    return ys, ps, probs


def train_cnn_baseline(
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 1e-4,
    seed: int = 42,
    out_dir: Path | None = None,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = load_manifest()
    train_s = [s for s in manifest["samples"] if s["split"] == "train"]
    test_s = [s for s in manifest["samples"] if s["split"] == "test"]
    train_loader = DataLoader(StripDataset(train_s, True), batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(StripDataset(test_s, False), batch_size=batch_size, shuffle=False, num_workers=2)

    model = _build_resnet18().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    best_f1 = -1.0
    best_state = None
    for ep in range(epochs):
        model.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        yt, yp, _ = _predict(model, test_loader, device)
        m = cost_metrics(yt, yp)
        print(f"[cnn] epoch {ep+1}/{epochs} loss={np.mean(losses):.4f} test_f1={m['f1']:.3f}")
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    yt, yp, probs = _predict(model, test_loader, device)
    metrics = cost_metrics(yt, yp)
    metrics["model"] = "resnet18"
    metrics["n_test"] = len(yt)

    out_dir = out_dir or (Path(__file__).resolve().parents[3] / "checkpoints" / "vision")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "resnet18_urfall.pt"
    torch.save({"state_dict": model.state_dict(), "metrics": metrics}, ckpt)
    pred_path = out_dir.parent.parent / "results" / "vision" / "cnn_preds.json"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_path.write_text(
        json.dumps({"y_true": yt, "y_pred": yp, "p_fall": probs, "metrics": metrics}, indent=2)
    )
    print("[cnn] saved", ckpt, metrics)
    return metrics
