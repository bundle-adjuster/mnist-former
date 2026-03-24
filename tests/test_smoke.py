"""CPU smoke: one forward pass and one training step."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from mnist_former.config import ModelConfig, TrainConfig
from mnist_former.model import FashionMNISTViT
from mnist_former.train import evaluate, set_seed, train


def test_forward_and_attention_shapes():
    set_seed(0)
    cfg = ModelConfig()
    m = FashionMNISTViT(cfg)
    x = torch.randn(2, 1, 28, 28)
    logits, aw = m.forward_with_attention(x, return_attention=True)
    assert logits.shape == (2, 10)
    assert aw is not None
    assert len(aw) == cfg.n_layers
    assert aw[0].shape[0] == 2  # batch
    assert aw[0].ndim == 4  # B, heads, T, T


def test_one_train_step():
    set_seed(0)
    cfg = ModelConfig()
    m = FashionMNISTViT(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    x = torch.randn(4, 1, 28, 28)
    y = torch.randint(0, 10, (4,))
    opt.zero_grad(set_to_none=True)
    loss = criterion(m(x), y)
    loss.backward()
    opt.step()
    assert not torch.isnan(loss)


def test_train_short_run(tmp_path: Path):
    mc = ModelConfig()
    tc = TrainConfig(epochs=1, batch_size=32, num_workers=0, val_fraction=0.1)
    train(
        output_dir=tmp_path,
        model_config=mc,
        train_config=tc,
        device=torch.device("cpu"),
    )
    assert (tmp_path / "metrics.jsonl").is_file()
    assert (tmp_path / "checkpoints" / "best.pt").is_file()
    assert (tmp_path / "run_config.json").is_file()


def test_evaluate_loader():
    set_seed(0)
    cfg = ModelConfig()
    m = FashionMNISTViT(cfg)
    x = torch.randn(8, 1, 28, 28)
    y = torch.randint(0, 10, (8,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    loss, acc = evaluate(m, loader, torch.device("cpu"), nn.CrossEntropyLoss())
    assert 0.0 <= acc <= 1.0
    assert loss >= 0.0
