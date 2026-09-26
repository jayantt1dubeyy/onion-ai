"""Grouped-query attention with rotary position embeddings.

- ``RotaryEmbedding`` precomputes cos/sin tables and applies the LLaMA-style
  half-split rotation.
- ``GroupedQueryAttention`` projects to ``n_heads`` query heads but only
  ``n_kv_heads`` key/value heads, repeating the latter. The actual attention
  goes through ``torch.nn.functional.scaled_dot_product_attention``, which
  dispatches to FlashAttention on CUDA when the shapes allow and falls back
  to a memory-efficient or math implementation otherwise.
- KV caching: pass ``cache=(k, v)`` from the previous step and get the updated
  cache back. The caller owns cache *storage*; see ``recurrent.py`` for why a
  recurrent model needs one cache per iteration depth.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int,
                 theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)          # (L, D/2)
        emb = torch.cat([freqs, freqs], dim=-1)   # (L, D)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def apply(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """Rotate q/k of shape (B, H, L, D); positions start at ``start_pos``.

        LLaMA-style half-split rotation: the first D/2 dims are the real part,
        the last D/2 the imaginary part of a complex vector rotated by the
        position angles.
        """
        d = x.shape[-1]
        L = x.shape[-2]
        if start_pos + L > self.cos.shape[0]:
            raise ValueError(
                f"sequence length {start_pos + L} exceeds RoPE table "
                f"({self.cos.shape[0]})"
            )
        # Table stores [freqs | freqs]; the two halves are identical, so take one.
        cos = self.cos[start_pos:start_pos + L, : d // 2].to(dtype=x.dtype)
        sin = self.sin[start_pos:start_pos + L, : d // 2].to(dtype=x.dtype)
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, L, D/2)
        sin = sin.unsqueeze(0).unsqueeze(0)
        x1, x2 = x[..., : d // 2], x[..., d // 2:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        self.dropout = cfg.dropout

        d = cfg.d_model
        self.q_proj = nn.Linear(d, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, d, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, cfg.max_seq_len, cfg.rope_theta)

    def forward(
        self,
        x: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = self.rope.apply(q, start_pos)
        k = self.rope.apply(k, start_pos)

        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        # The cache always stores the kv-head tensors (before head repetition);
        # repetition is deterministic so it is redone on every call.
        new_cache = (k.detach(), v.detach()) if not self.training else (k, v)

        kr = k.repeat_interleave(self.n_rep, dim=1) if self.n_rep > 1 else k
        vr = v.repeat_interleave(self.n_rep, dim=1) if self.n_rep > 1 else v

        # is_causal=True is only valid when Q and K have equal length (SDPA
        # aligns the mask top-left otherwise, which is wrong for decoding).
        if cache is None:
            y = F.scaled_dot_product_attention(
                q, kr, vr, is_causal=True,
                dropout_p=self.dropout if self.training else 0.0,
            )
        elif q.shape[2] == 1:
            # Single decode query: the whole key history is its past.
            y = F.scaled_dot_product_attention(q, kr, vr, is_causal=False)
        else:
            # Chunked extension with a cache: explicit bottom-right causal mask.
            Lq, Lk = q.shape[2], kr.shape[2]
            qi = torch.arange(Lq, device=x.device)[:, None]
            kj = torch.arange(Lk, device=x.device)[None, :]
            mask = kj <= qi + (Lk - Lq)  # (Lq, Lk), True = attend
            y = F.scaled_dot_product_attention(q, kr, vr, attn_mask=mask)
        y = y.transpose(1, 2).contiguous().view(B, L, -1)
        return self.o_proj(y), new_cache
