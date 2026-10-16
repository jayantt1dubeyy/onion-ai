# Patch: QK-Norm (RMSNorm on Q/K before softmax)

**Why:** Qwen3 added RMSNorm on queries and keys pre-softmax specifically for
training stability. A *looped* recurrent block re-processes tokens up to 16× —
attention-logit blowup across iterations is the #1 instability risk. This is the
single highest-EV architectural import in the 1B plan.

**Against:** `kyegomez/OpenMythos`, file `open_mythos/main.py`, class `GQAttention`
(line ~177). `RMSNorm` already exists in the same file (line ~89) — reuse it.

## Changes

### 1. `GQAttention.__init__` — add two norms (after `self.wo = ...`):

```python
self.q_norm = RMSNorm(self.head_dim)
self.k_norm = RMSNorm(self.head_dim)
```

`head_dim = dim // n_heads` (128 at dim=2048/16 heads). Per-head norm, applied
independently to each head's slice — matches Qwen3 (RMSNorm over the head dim).

### 2. `GQAttention.forward` — normalize after RoPE, before attention:

```python
q = apply_rope(q, freqs_cis)
k = apply_rope(k, freqs_cis)

# QK-Norm (patch): stabilize logits across recurrent iterations
q = self.q_norm(q)
k = self.k_norm(k)
```

Place this **before** the KV-cache write so cached K is already normalized
(keeps prefill/decode consistent), and before the flash-attn / fallback branch.

### 3. Config flag (optional, for ablations):

Add to `MythosConfig`:
```python
qk_norm: bool = True
```
and gate the two lines above on `cfg.qk_norm` (store as `self.qk_norm` in
`__init__`). Default True; set False to ablate.

## Verification
- Param delta: +2 × head_dim (256 params) — negligible.
- Smoke test: forward pass on random ids, check no NaN in logits; confirm
  `q_norm`/`k_norm` appear in `model.state_dict()`.
- Ablation: train 5B tokens with/without at small scale; expect lower loss
  variance and no logit blowup (track max |QKᵀ| per loop iteration in logs).
