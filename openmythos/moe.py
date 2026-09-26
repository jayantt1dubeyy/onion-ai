"""Feed-forward blocks: dense SwiGLU and sparse mixture-of-experts.

- ``SwiGLU``: the standard gated FFN, ``down(silu(gate(x)) * up(x))``.
- ``SparseMoE``: routes each token to ``top_k`` of ``num_experts`` SwiGLU
  experts. Returns ``(output, aux_loss)`` where ``aux_loss`` is the
  Switch-Transformer load-balancing loss that discourages router collapse.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class SparseMoE(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_experts: int = 8,
                 top_k: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        assert top_k <= num_experts
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLU(d_model, d_ff, dropout) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, L, D)
        B, L, D = x.shape
        logits = self.gate(x)                       # (B, L, E)
        probs = F.softmax(logits, dim=-1)

        top_w, top_i = probs.topk(self.top_k, dim=-1)   # (B, L, K)
        top_w = top_w / (top_w.sum(dim=-1, keepdim=True) + 1e-9)  # renormalise

        out = torch.zeros_like(x)
        # Dispatch per expert. Simple loop is fine at research scale; a batched
        # implementation would matter for production.
        for e, expert in enumerate(self.experts):
            mask = (top_i == e)                       # (B, L, K) bool
            if not mask.any():
                continue
            w = (top_w * mask).sum(dim=-1)             # (B, L)
            sel = w > 0
            out[sel] += w[sel].unsqueeze(-1) * expert(x[sel])

        # Load-balancing auxiliary loss (Switch Transformer, simplified):
        # penalises the gap between the fraction of tokens routed to each
        # expert and the mean routing probability for that expert.
        with torch.no_grad():
            assign = torch.zeros_like(probs).scatter_(-1, top_i, 1.0).sum(-1)  # (B,L,E)
            density = assign.mean(dim=(0, 1))          # fraction of tokens per expert
        mean_prob = probs.mean(dim=(0, 1))
        aux_loss = (density * mean_prob).sum() * self.num_experts
        return out, aux_loss
