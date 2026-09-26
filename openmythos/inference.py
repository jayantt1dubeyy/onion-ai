"""Text generation with per-iteration KV caches.

The recurrent wrinkle, handled here: each of the ``n_recurrent`` iterations
attends over a different input distribution (the state evolves between
iterations), so key/value caches cannot be shared across iterations. We keep
a *list* of caches -- one (k, v) tuple per iteration -- and thread it through
every decode step. Memory cost is ``n_recurrent`` x the usual KV cache, which
is the honest price of recurrent generation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import OpenMythosModel
from .tokenizer import ByteTokenizer


@torch.no_grad()
def generate(model: OpenMythosModel, tok: ByteTokenizer, prompt: str,
             max_new_tokens: int = 64, temperature: float = 0.8,
             top_k: int = 50, device: str = "cpu",
             seed: int | None = None) -> str:
    model.eval()
    if seed is not None:
        torch.manual_seed(seed)

    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    caches = model.init_caches()
    pos = 0

    # Prefill the prompt in one forward pass, seeding every iteration's cache.
    logits, _, caches = model(ids, caches=caches, start_pos=pos)
    pos += ids.shape[1]
    out = []

    for _ in range(max_new_tokens):
        next_logits = logits[:, -1, :]
        if temperature <= 0:
            nxt = next_logits.argmax(dim=-1)
        else:
            probs = F.softmax(next_logits / temperature, dim=-1)
            if top_k > 0:
                v, ix = probs.topk(top_k, dim=-1)
                probs = torch.zeros_like(probs).scatter_(-1, ix, v)
                probs = probs / probs.sum(dim=-1, keepdim=True)
            nxt = torch.multinomial(probs, num_samples=1).squeeze(-1)
        out.append(nxt.item())
        if nxt.item() == tok.eos_id:
            break
        nxt = nxt.unsqueeze(0)
        logits, _, caches = model(nxt, caches=caches, start_pos=pos)
        pos += 1

    return tok.decode(out)
