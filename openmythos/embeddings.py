"""Embeddings and normalisation primitives.

- ``RMSNorm``: root-mean-square layer normalisation (no mean centring, no bias),
  the standard choice for modern decoder LMs.
- ``TokenEmbedding``: thin wrapper around ``nn.Embedding`` so the prelude owns
  exactly one clearly-named object.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return self.weight * x


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # Small-init keeps early logits calm; std = 1/sqrt(d) is conventional.
        nn.init.normal_(self.embed.weight, mean=0.0, std=d_model ** -0.5)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.embed(ids))
