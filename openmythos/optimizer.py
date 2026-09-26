"""Optimizer factory: AdamW with selective weight decay.

Decay applies to matrices only; norms, embeddings and biases are exempt, the
standard recipe for transformer training.
"""

from __future__ import annotations

import torch

from .config import TrainConfig


def build_optimizer(model: torch.nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # 1-D tensors are norms; embeddings are exempt by name.
        if p.ndim == 1 or "embed" in name or "prelude" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    use_fused = torch.cuda.is_available()
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=cfg.betas, fused=use_fused)
