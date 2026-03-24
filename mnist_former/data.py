"""Fashion-MNIST dataloaders with train/validation split."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from mnist_former.config import TrainConfig


def get_dataloaders(
    train_config: TrainConfig,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns train_loader, val_loader, test_loader.
    Validation is a random subset of the official training set.
    """
    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),  # Fashion-MNIST channel stats
        ]
    )

    train_full = datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=tf,
    )
    test_set = datasets.FashionMNIST(
        root="./data",
        train=False,
        download=True,
        transform=tf,
    )

    g = torch.Generator().manual_seed(train_config.seed)
    val_n = int(len(train_full) * train_config.val_fraction)
    train_n = len(train_full) - val_n
    train_subset, val_subset = random_split(
        train_full,
        [train_n, val_n],
        generator=g,
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader
