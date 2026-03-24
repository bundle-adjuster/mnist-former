"""Hyperparameter and run configuration dataclasses."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class ModelConfig:
    """Vision Transformer architecture for 28×28 Fashion-MNIST."""

    image_size: int = 28
    patch_size: int = 4
    num_classes: int = 10
    d_model: int = 32
    nhead: int = 1
    n_layers: int = 2
    dim_feedforward: int = 64
    dropout: float = 0.1
    num_patches: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        object.__setattr__(
            self, "num_patches", (self.image_size // self.patch_size) ** 2
        )


@dataclass
class TrainConfig:
    """Training loop and optimizer settings."""

    seed: int = 42
    epochs: int = 15
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.01
    val_fraction: float = 0.1
    num_workers: int = 2
    log_grad_norm_every: int | None = None  # e.g. 50; None disables grad norm logging
    device: str = "cuda"  # or "cpu"; CLI may override


def config_to_dict(model: ModelConfig, train: TrainConfig) -> dict:
    return {"model": asdict(model), "train": asdict(train)}


def model_from_dict(d: dict) -> ModelConfig:
    skip = {"num_patches"}  # recomputed in __post_init__
    allowed = {f.name for f in ModelConfig.__dataclass_fields__.values()}
    kwargs = {k: v for k, v in d.items() if k in allowed and k not in skip}
    return ModelConfig(**kwargs)


def train_from_dict(d: dict) -> TrainConfig:
    allowed = {f.name for f in TrainConfig.__dataclass_fields__.values()}
    kwargs = {k: v for k, v in d.items() if k in allowed}
    return TrainConfig(**kwargs)
