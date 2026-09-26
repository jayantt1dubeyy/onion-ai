"""Learning-rate schedule: linear warmup then cosine decay to ``min_lr``."""

from __future__ import annotations

import math

from .config import TrainConfig


class WarmupCosine:
    def __init__(self, optimizer, cfg: TrainConfig) -> None:
        self.opt = optimizer
        self.warmup = cfg.warmup_steps
        self.total = cfg.max_steps
        self.base = cfg.lr
        self.floor = cfg.min_lr
        self.step(0)

    def _factor(self, step: int) -> float:
        if step < self.warmup:
            return (step + 1) / max(1, self.warmup)
        # Cosine decay reaches min_lr exactly at the final step (max_steps - 1).
        t = (step - self.warmup) / max(1, self.total - 1 - self.warmup)
        t = min(1.0, t)
        return self.floor / self.base + 0.5 * (1 - self.floor / self.base) * (
            1 + math.cos(math.pi * t)
        )

    def step(self, step: int) -> float:
        lr = self.base * self._factor(step)
        for g in self.opt.param_groups:
            g["lr"] = lr
        return lr
