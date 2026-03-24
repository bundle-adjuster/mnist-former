"""CLI entry point for evaluation / inference."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mnist_former.config import TrainConfig
from mnist_former.data import get_dataloaders
from mnist_former.inference import evaluate_loader, load_checkpoint, predict


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate Fashion-MNIST ViT checkpoint")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    p.add_argument(
        "--split",
        type=str,
        default="test",
        choices=("val", "test"),
        help="Which split to evaluate (uses same transforms as training)",
    )
    p.add_argument(
        "--predictions-csv",
        type=str,
        default=None,
        help="If set, save rows: index, label, pred, prob_max",
    )
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, train_config, epoch = load_checkpoint(args.checkpoint, device=dev)

    tc = TrainConfig(seed=args.seed, batch_size=args.batch_size)
    _, val_loader, test_loader = get_dataloaders(tc)
    loader = val_loader if args.split == "val" else test_loader

    loss, acc = evaluate_loader(model, loader, dev)
    print(f"checkpoint_epoch={epoch} split={args.split} loss={loss:.4f} acc={acc:.4f}")

    if args.predictions_csv:
        model.eval()
        rows: list[tuple[int, int, int, float]] = []
        idx = 0
        with torch.no_grad():
            for x, y in loader:
                x = x.to(dev)
                y = y.to(dev)
                pred, probs = predict(model, x, return_probs=True)
                prob_max = probs.max(dim=-1).values
                for i in range(x.size(0)):
                    rows.append(
                        (
                            idx,
                            int(y[i].item()),
                            int(pred[i].item()),
                            float(prob_max[i].item()),
                        )
                    )
                    idx += 1
        path = Path(args.predictions_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["index", "label", "pred", "prob_max"])
            w.writerows(rows)
        print(f"Wrote {len(rows)} rows to {path.resolve()}")


if __name__ == "__main__":
    main()
