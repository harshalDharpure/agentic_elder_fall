from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from ..eval.metrics import binary_metrics
from ..utils.io import save_checkpoint


@dataclass
class TrainResult:
    best_f1: float
    history: list[dict]
    checkpoint_path: str


def mixup(x: torch.Tensor, y: torch.Tensor, alpha: float):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def make_weighted_sampler(labels: list[int] | np.ndarray) -> WeightedRandomSampler:
    y = np.asarray(labels, dtype=np.int64)
    classes, counts = np.unique(y, return_counts=True)
    freq = {int(c): float(n) for c, n in zip(classes, counts)}
    weights = np.array([1.0 / freq[int(yi)] for yi in y], dtype=np.float64)
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


def class_pos_weight(labels: list[int] | np.ndarray) -> torch.Tensor:
    y = np.asarray(labels, dtype=np.int64)
    n_neg = max(1, int((y == 0).sum()))
    n_pos = max(1, int((y == 1).sum()))
    # For BCEWithLogits / CE weight on positive class via weight vector
    w0 = 1.0
    w1 = n_neg / n_pos
    return torch.tensor([w0, w1], dtype=torch.float32)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: nn.Module,
    scaler: GradScaler | None,
    mixup_alpha: float = 0.0,
) -> float:
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        x, y_a, y_b, lam = mixup(x, y, mixup_alpha)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with autocast():
                logits, _ = model(x)
                loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, _ = model(x)
            loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
            loss.backward()
            optimizer.step()
        total += float(loss.item()) * x.size(0)
        n += x.size(0)
    return total / max(1, n)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float | None = None,
) -> dict:
    model.eval()
    ys, preds, probs = [], [], []
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"]
        logits, _ = model(x)
        p = torch.softmax(logits, dim=-1)
        if p.size(1) == 2:
            p_fall = p[:, 1].cpu()
            probs.extend(p_fall.tolist())
            if threshold is None:
                pred = p.argmax(dim=-1).cpu()
            else:
                pred = (p_fall >= float(threshold)).long()
        else:
            probs.extend(p.max(dim=1).values.cpu().tolist())
            pred = p.argmax(dim=-1).cpu()
        ys.extend(y.tolist())
        preds.extend(pred.tolist())
    m = binary_metrics(ys, preds)
    m["n"] = len(ys)
    if threshold is not None:
        m["decision_threshold"] = float(threshold)
    m["probs"] = probs  # callers may pop if serializing
    m["y_true"] = ys
    return m


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 80,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.05,
    patience: int = 12,
    mixup_alpha: float = 0.0,
    use_amp: bool = True,
    checkpoint_path: str = "checkpoints/best.pt",
    class_weight: torch.Tensor | None = None,
) -> TrainResult:
    if class_weight is not None:
        criterion = nn.CrossEntropyLoss(
            weight=class_weight.to(device), label_smoothing=label_smoothing
        )
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler() if (use_amp and device.type == "cuda") else None

    best_f1 = -1.0
    bad = 0
    history = []
    model.to(device)

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(
            model, train_loader, optimizer, device, criterion, scaler, mixup_alpha
        )
        val = evaluate(model, val_loader, device)
        # strip large arrays from history
        val_row = {k: v for k, v in val.items() if k not in ("probs", "y_true")}
        scheduler.step()
        row = {"epoch": epoch, "loss": loss, **val_row}
        history.append(row)
        print(
            f"epoch {epoch:03d}  loss={loss:.4f}  "
            f"f1={val['f1']:.4f}  acc={val['accuracy']:.4f}  "
            f"rec={val['recall']:.4f}  spec={val['specificity']:.4f}"
        )
        if val["f1"] > best_f1:
            best_f1 = val["f1"]
            bad = 0
            save_checkpoint(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "metrics": val_row,
                },
                checkpoint_path,
            )
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    return TrainResult(best_f1=best_f1, history=history, checkpoint_path=checkpoint_path)


class ThresholdBaseline:
    """Simple peak-acceleration threshold classifier."""

    def __init__(self, thr_g: float = 2.5):
        self.thr_g = thr_g

    def predict_batch(self, x: torch.Tensor) -> list[int]:
        # x: (B, C, T), accel on first 3 channels in g
        acc = x[:, :3, :]
        svm = torch.sqrt((acc ** 2).sum(dim=1) + 1e-8)
        peak = svm.max(dim=1).values
        return (peak >= self.thr_g).long().tolist()
