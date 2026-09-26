"""The recurrent block: Prelude -> RecurrentBlock x N -> Coda.

The core architectural idea: a *single* transformer block whose weights are
shared across ``n_recurrent`` iterations. Depth (in compute steps) grows
without growing the parameter count -- the bet being that iterative
refinement buys representational power more cheaply than wider layers.

Two engineering consequences, handled explicitly here:

1. **Truncated BPTT.** Backpropagating through all N iterations costs O(N)
   activation memory. ``grad_recurrence_steps`` caps how many iterations carry
   gradients; the state is detached between chunks. Invariant (checked in
   config): ``1 <= grad_recurrence_steps <= n_recurrent``.

2. **KV caches during generation.** Each iteration has its own key/value
   projections over a *different* input distribution (the state evolves), so
   caches cannot be shared across iterations. Generation therefore keeps one
   cache per iteration depth -- a list of ``n_recurrent`` (k, v) tuples.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .attention import GroupedQueryAttention
from .config import ModelConfig
from .embeddings import RMSNorm
from .moe import SparseMoE, SwiGLU


class RecurrentBlock(nn.Module):
    """One shared transformer block: attn -> FFN/MoE, pre-norm residuals."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = GroupedQueryAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        if cfg.use_moe:
            self.ffn = SparseMoE(cfg.d_model, cfg.d_ff, cfg.moe_num_experts,
                                 cfg.moe_top_k, cfg.dropout)
        else:
            self.ffn = SwiGLU(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor,
               tuple[torch.Tensor, torch.Tensor] | None]:
        # Attention sub-layer (pre-norm residual)
        a, cache = self.attn(self.attn_norm(x), cache=cache, start_pos=start_pos)
        x = x + a
        # FFN sub-layer (pre-norm residual)
        h = self.ffn_norm(x)
        if isinstance(self.ffn, SparseMoE):
            f, aux = self.ffn(h)
        else:
            f, aux = self.ffn(h), torch.zeros((), device=x.device, dtype=x.dtype)
        return x + f, aux, cache


class RecurrentStack(nn.Module):
    """Applies one RecurrentBlock for ``n_recurrent`` iterations."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.block = RecurrentBlock(cfg)

    def forward(
        self,
        x: torch.Tensor,
        caches: list | None = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, list | None]:
        n = self.cfg.n_recurrent
        k = self.cfg.grad_recurrence_steps
        total_aux = torch.zeros((), device=x.device, dtype=x.dtype)
        new_caches = [] if caches is not None else None

        for i in range(n):
            cache = caches[i] if caches is not None else None
            # Truncated BPTT: detach the carried state every k iterations so
            # gradients never span more than k iterations. The detach is a
            # no-op for the first chunk (i == 0).
            if self.training and i % k == 0 and i > 0:
                x = x.detach()
            if self.cfg.grad_checkpoint and self.training:
                x, aux, cache = checkpoint(
                    self.block, x, cache, start_pos, use_reentrant=False
                )
            else:
                x, aux, cache = self.block(x, cache=cache, start_pos=start_pos)
            total_aux = total_aux + aux
            if new_caches is not None:
                new_caches.append(cache)

        return x, total_aux, new_caches
