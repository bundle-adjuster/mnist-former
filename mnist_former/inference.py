"""Load checkpoints and run predictions (no visualization)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn

from mnist_former.config import ModelConfig, TrainConfig, model_from_dict, train_from_dict
from mnist_former.model import FashionMNISTViT


def load_checkpoint(
    path: str | Path,
    device: torch.device | None = None,
) -> Tuple[FashionMNISTViT, ModelConfig, TrainConfig, int]:
    path = Path(path)
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(path, map_location=dev, weights_only=False)
    model_config = model_from_dict(payload["model_config"])
    train_config = train_from_dict(payload["train_config"])
    model = FashionMNISTViT(model_config).to(dev)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    epoch = int(payload.get("epoch", 0))
    return model, model_config, train_config, epoch


@torch.no_grad()
def predict(
    model: nn.Module,
    x: torch.Tensor,
    return_probs: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    """
    x: (B, 1, 28, 28) on same device as model.
    Returns class indices (B,) or (indices, probs).
    """
    logits = model(x)
    probs = torch.softmax(logits, dim=-1)
    pred = logits.argmax(dim=-1)
    if return_probs:
        return pred, probs
    return pred


@torch.no_grad()
def predict_with_attention(
    model: FashionMNISTViT,
    x: torch.Tensor,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Returns (logits, per_layer_attention) for visualization in notebooks.
    Attention tensors are (B, num_heads, T, T) per layer.
    """
    return model.forward_with_attention(x, return_attention=True)


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Returns (average loss, accuracy) using CrossEntropyLoss."""
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        bs = x.size(0)
        total_loss += loss.item() * bs
        correct += (logits.argmax(dim=-1) == y).sum().item()
        n += bs
    return total_loss / max(n, 1), correct / max(n, 1)
