"""OpenMythosModel: Prelude -> RecurrentStack -> Coda.

- **Prelude**: token embeddings.
- **RecurrentStack**: one shared block applied ``n_recurrent`` times.
- **Coda**: final RMSNorm + linear head (optionally tied to the embeddings).

The forward pass returns logits; ``forward_loss`` is the standard
next-token cross-entropy plus the MoE load-balancing auxiliary loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .embeddings import RMSNorm, TokenEmbedding
from .recurrent import RecurrentStack


class OpenMythosModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.prelude = TokenEmbedding(cfg.vocab_size, cfg.d_model, cfg.dropout)
        self.stack = RecurrentStack(cfg)
        self.coda_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.prelude.embed.weight

    # -- parameter count -------------------------------------------------
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # -- forward ----------------------------------------------------------
    def forward(
        self,
        ids: torch.Tensor,
        caches: list | None = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, list | None]:
        """Returns (logits, aux_loss, new_caches)."""
        if ids.shape[1] > self.cfg.max_seq_len:
            raise ValueError("input longer than max_seq_len")
        x = self.prelude(ids)
        x, aux, new_caches = self.stack(x, caches=caches, start_pos=start_pos)
        logits = self.lm_head(self.coda_norm(x))
        return logits, aux, new_caches

    def forward_loss(
        self, ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Next-token cross-entropy. Returns (total_loss, ce_loss, aux_loss)."""
        logits, aux, _ = self.forward(ids[:, :-1])
        targets = ids[:, 1:].contiguous()
        ce = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
        )
        total = ce + self.cfg.moe_aux_loss_coef * aux
        return total, ce.detach(), aux.detach()

    # -- generation helpers -----------------------------------------------
    def init_caches(self) -> list:
        """One (empty) KV cache slot per recurrent iteration; see recurrent.py."""
        return [None] * self.cfg.n_recurrent
