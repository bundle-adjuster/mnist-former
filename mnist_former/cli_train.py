"""CLI entry point for training."""

from __future__ import annotations

import argparse
import contextlib
from pathlib import Path

import torch

from mnist_former.config import ModelConfig, TrainConfig
from mnist_former.data import get_dataloaders
from mnist_former.model import FashionMNISTViT
from mnist_former.profiling import profile_training_steps
from mnist_former.train import set_seed, train


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train Fashion-MNIST ViT (N=2 default)")
    p.add_argument(
        "--output-dir",
        type=str,
        default="runs/exp",
        help="Directory for checkpoints, metrics.jsonl, run_config.json",
    )
    p.add_argument("--epochs", type=int, default=None, help="Override TrainConfig.epochs")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--device",
        type=str,
        default=None,
        choices=("cpu", "cuda"),
        help="Force device (default: cuda if available else cpu)",
    )
    p.add_argument(
        "--profile",
        action="store_true",
        help="Run a short torch.profiler segment before training; writes under output-dir/profile",
    )
    p.add_argument(
        "--log-grad-norm-every",
        type=int,
        default=None,
        help="If set, append grad L2 norms to grad_norms.jsonl every N global steps",
    )
    p.add_argument(
        "--nvtx",
        action="store_true",
        help="Emit NVTX ranges via torch.autograd.profiler.emit_nvtx() (CUDA only; use with Nsight Systems)",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    model_config = ModelConfig()
    train_config = TrainConfig()
    if args.epochs is not None:
        train_config.epochs = args.epochs
    if args.batch_size is not None:
        train_config.batch_size = args.batch_size
    if args.lr is not None:
        train_config.lr = args.lr
    if args.seed is not None:
        train_config.seed = args.seed
    if args.device is not None:
        train_config.device = args.device
    if args.log_grad_norm_every is not None:
        train_config.log_grad_norm_every = args.log_grad_norm_every

    out = Path(args.output_dir)
    dev = torch.device(
        train_config.device
        if train_config.device == "cuda" and torch.cuda.is_available()
        else "cpu"
    )
    if train_config.device == "cuda" and not torch.cuda.is_available():
        dev = torch.device("cpu")

    if args.profile:
        set_seed(train_config.seed)
        train_loader, _, _ = get_dataloaders(train_config)
        model = FashionMNISTViT(model_config).to(dev)
        criterion = torch.nn.CrossEntropyLoss()
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )
        x, y = next(iter(train_loader))
        x = x.to(dev)
        y = y.to(dev)

        def train_step() -> None:
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()

        prof_dir = out / "profile"
        profile_training_steps(dev, train_step, prof_dir)
        print(f"Profiler output written under {prof_dir.resolve()}")

    if args.nvtx and dev.type != "cuda":
        raise SystemExit("--nvtx requires CUDA (use --device cuda)")
    nvtx_ctx = (
        torch.autograd.profiler.emit_nvtx()
        if args.nvtx
        else contextlib.nullcontext()
    )
    with nvtx_ctx:
        train(
            output_dir=out,
            model_config=model_config,
            train_config=train_config,
            device=dev,
        )
    print(f"Done. Artifacts in {out.resolve()}")


if __name__ == "__main__":
    main()
