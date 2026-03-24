"""Vision Transformer for Fashion-MNIST with optional attention outputs."""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from mnist_former.config import ModelConfig


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, need_weights: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        x_norm = self.norm1(x)
        attn_out, attn_w = self.attn(
            x_norm,
            x_norm,
            x_norm,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_w


class FashionMNISTViT(nn.Module):
    """Patch embedding + class token + positional encoding + N transformer blocks."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        c = 1
        p = config.patch_size
        d = config.d_model
        n_patches = config.num_patches

        self.patch_embed = nn.Conv2d(c, d, kernel_size=p, stride=p)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, d))
        self.pos_drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d,
                    config.nhead,
                    config.dim_feedforward,
                    config.dropout,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, config.num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_attention(x, return_attention=False)
        return logits

    def forward_with_attention(
        self,
        x: torch.Tensor,
        return_attention: bool = True,
    ) -> Tuple[torch.Tensor, List[torch.Tensor] | None]:
        """
        x: (B, 1, 28, 28)
        Returns logits (B, num_classes) and list of attention weight tensors per layer
        (each (B, num_heads, T, T)) or None if return_attention is False.
        """
        b = x.shape[0]
        d = self.config.d_model

        x = self.patch_embed(x)  # (B, d, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches, d)

        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        attn_weights: List[torch.Tensor] | None = [] if return_attention else None
        for blk in self.blocks:
            x, aw = blk(x, need_weights=return_attention)
            if return_attention and attn_weights is not None and aw is not None:
                attn_weights.append(aw)

        x = self.norm(x)
        logits = self.head(x[:, 0])
        return logits, attn_weights
