# Patch: sliding-window attention interleave (Gemma-style 5:1 local:global)

**Why:** The recurrent block re-processes every token up to 16×. Full O(T²)
attention on *every* loop iteration is where the FLOP budget dies. Gemma 2/3
use a 5:1 interleave — 5 local sliding-window layers per 1 global layer —
with no measured quality loss. For OpenMythos: **local windows inside the
recurrent loop, global attention outside it** (prelude/coda stay global).

**Against:** `kyegomez/OpenMythos`, file `open_mythos/main.py`.

## Changes

### 1. `MythosConfig` — add:

```python
sliding_window: int = 1024   # 0 = full attention everywhere (default off)
```

### 2. `GQAttention.__init__` — accept a per-layer window:

```python
def __init__(self, cfg: MythosConfig, window: int = 0):
    ...
    self.window = window
```

### 3. `GQAttention.forward` — enforce the window:

- **flash-attn path:** pass `window_size=(self.window, 0)` to
  `flash_attn_func` when `self.window > 0` (left-windowed causal attention;
  supported natively by flash-attn ≥ 2.1).
- **Fallback path:** after building the additive causal `mask`, add a
  band mask: positions `j < i - window + 1` get `-inf`. I.e.
  `mask = torch.minimum(mask, band_mask)`.

### 4. `TransformerBlock.__init__` — thread the window through:

```python
def __init__(self, cfg, use_moe=False, window: int = 0):
    self.attn = GQAttention(cfg, window=window)
```

### 5. `RecurrentBlock.__init__` — the recurrent block gets the window:

The recurrent block is the hot loop: construct its `TransformerBlock` with
`window=cfg.sliding_window` (1024). Prelude and coda blocks keep `window=0`
(full global attention) — this is the 5:1-style interleave adapted to the
Prelude → Loop → Coda topology.

### 6. KV-cache interaction

`GQAttention` already keeps per-iteration caches (`cache_key`). With a 1024
window, cache entries older than the window can be evicted on decode —
optional memory win; correctness does not depend on it (masking handles it).

## Verification
- With `sliding_window=0`, numerics must be bit-identical to unpatched
  (regression test on fixed seed).
- With `sliding_window=1024`: forward pass OK; benchmark tok/s vs full —
  expect ~2–4× faster loop iterations at 4K context.
- Ablation: compare perplexity at 50B tokens, windowed vs full, before
  committing to the full 1B run.
