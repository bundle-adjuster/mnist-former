"""Training loop, metrics logging, checkpoints (no plotting)."""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from mnist_former.config import ModelConfig, TrainConfig, config_to_dict
from mnist_former.data import get_dataloaders
from mnist_former.model import FashionMNISTViT


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return (pred == y).float().mean().item()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits, y) * bs
        n += bs
    return total_loss / max(n, 1), total_acc / max(n, 1)


def _append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def save_checkpoint(
    path: Path,
    model: nn.Module,
    model_config: ModelConfig,
    train_config: TrainConfig,
    epoch: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
    }
    torch.save(payload, path)


def train(
    output_dir: str | Path,
    model_config: ModelConfig | None = None,
    train_config: TrainConfig | None = None,
    device: torch.device | None = None,
    on_epoch_end: Callable[[int, dict[str, Any]], None] | None = None,
) -> Path:
    """
    Train FashionMNISTViT. Writes metrics.jsonl, checkpoints/best.pt, checkpoints/last.pt.
    Optional on_epoch_end(epoch, metrics_dict) for tests or extra hooks.
    Returns path to output directory.
    """
    model_config = model_config or ModelConfig()
    train_config = train_config or TrainConfig()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out / "checkpoints"
    metrics_path = out / "metrics.jsonl"
    grad_norm_path = out / "grad_norms.jsonl"

    set_seed(train_config.seed)
    dev = device or torch.device(
        train_config.device
        if train_config.device == "cuda" and torch.cuda.is_available()
        else "cpu"
    )
    if train_config.device == "cuda" and not torch.cuda.is_available():
        dev = torch.device("cpu")

    train_loader, val_loader, _ = get_dataloaders(train_config)
    model = FashionMNISTViT(model_config).to(dev)
    criterion = nn.CrossEntropyLoss()
    opt = AdamW(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )

    best_val = -1.0
    global_step = 0

    for epoch in range(1, train_config.epochs + 1):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        n_train = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{train_config.epochs}")
        for x, y in pbar:
            x = x.to(dev)
            y = y.to(dev)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            if train_config.log_grad_norm_every is not None:
                if global_step % train_config.log_grad_norm_every == 0:
                    sq = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            sq += float(p.grad.data.norm(2).item() ** 2)
                    gn = sq**0.5
                    _append_jsonl(
                        grad_norm_path,
                        {"epoch": epoch, "step": global_step, "grad_norm_l2": gn},
                    )

            opt.step()
            global_step += 1
            bs = x.size(0)
            train_loss += loss.item() * bs
            train_acc += accuracy(logits, y) * bs
            n_train += bs
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= max(n_train, 1)
        train_acc /= max(n_train, 1)
        val_loss, val_acc = evaluate(model, val_loader, dev, criterion)

        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        _append_jsonl(metrics_path, row)

        if on_epoch_end is not None:
            on_epoch_end(epoch, row)

        save_checkpoint(
            ckpt_dir / "last.pt",
            model,
            model_config,
            train_config,
            epoch,
        )
        if val_acc > best_val:
            best_val = val_acc
            save_checkpoint(
                ckpt_dir / "best.pt",
                model,
                model_config,
                train_config,
                epoch,
            )

    # Save full config snapshot for notebooks
    cfg_path = out / "run_config.json"
    cfg_path.write_text(
        json.dumps(config_to_dict(model_config, train_config), indent=2),
        encoding="utf-8",
    )

    return out
