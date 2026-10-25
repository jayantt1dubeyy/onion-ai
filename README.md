# OpenMythos

A miniature recurrent language model designed to train on a **single NVIDIA T4 (16 GB)** in Google Colab. Rebuilt from scratch with full verification — every module is shape-checked and the whole pipeline runs end to end.

## Architecture

**Prelude → Recurrent Block × N → Coda**, with the block's weights *shared* across iterations:

- **Prelude**: byte-level token embeddings (vocab 260, no tokenizer training needed)
- **Recurrent block** (shared): grouped-query attention + SwiGLU FFN (or sparse MoE), pre-norm residuals, RoPE position embeddings
- **Coda**: final RMSNorm + LM head (optionally tied)

The research bet: iterative refinement buys effective depth more cheaply than wider layers. Two engineering consequences are handled explicitly:

1. **Truncated BPTT** — gradients flow through at most `grad_recurrence_steps` iterations (state detached between chunks).
2. **Per-iteration KV caches** — each iteration attends over a different input distribution, so generation keeps one cache per iteration depth (see `recurrent.py`).

## Quick start

```bash
pip install -r requirements.txt

# 1. Smoke test: does the model run? (~1M params, CPU, seconds)
python -m openmythos.benchmark --preset tiny_debug --mode train --steps 5

# 2. Hardware check: parameter count + VRAM estimate for the real model
python -m openmythos.benchmark --preset mini_60m --mode params
python -m openmythos.benchmark --preset mini_60m --mode vram

# 3. Train (on a T4; see COLAB.md)
python -m openmythos.train --preset mini_60m \
    --train-data data/train.txt --val-data data/val.txt

# 4. Evaluate / generate
python -m openmythos.evaluate --ckpt checkpoints/final.pt --text data/val.txt
```

## Project layout

| File | Purpose |
|---|---|
| `openmythos/config.py` | Model/train configs, `tiny_debug` and `mini_60m` presets |
| `openmythos/tokenizer.py` | Byte-level tokenizer (no training step) |
| `openmythos/embeddings.py` | RMSNorm, token embeddings |
| `openmythos/attention.py` | GQA + RoPE + SDPA (FlashAttention when available), KV cache |
| `openmythos/moe.py` | SwiGLU FFN, sparse MoE with load-balancing loss |
| `openmythos/recurrent.py` | Shared recurrent block, truncated BPTT, per-iteration caches |
| `openmythos/model.py` | Prelude → RecurrentStack → Coda |
| `openmythos/dataset.py` | Text chunking + dataloaders |
| `openmythos/optimizer.py` | AdamW, selective weight decay |
| `openmythos/scheduler.py` | Linear warmup + cosine decay |
| `openmythos/trainer.py` | Loop: grad accumulation, AMP, clipping, ckpt save/resume |
| `openmythos/inference.py` | Generation with per-iteration KV caches |
| `openmythos/evaluate.py` | Perplexity CLI |
| `openmythos/benchmark.py` | Params / VRAM / tok-s smoke tests |
| `openmythos/export.py` | HF-style export (config.json + weights) |
| `openmythos/train.py` | Training CLI entry point |

## Presets

| Preset | Params | Purpose |
|---|---|---|
| `tiny_debug` | ~1M | CPU smoke tests: forward/backward/generate in seconds |
| `mini_60m` | ~62M | Reference model for a single T4 |

Run `python -m openmythos.benchmark --preset mini_60m --mode params` for the exact count.

## Reproducibility

- All randomness is seeded (`TrainConfig.seed`); dataloader shuffling uses a seeded generator.
- Checkpoints store model, optimizer, scaler, and RNG states — resume with `--resume checkpoints/latest.pt`.
- Every run appends JSONL metrics to `checkpoints/loss.jsonl`.

## Roadmap: 1B — beat them all

The T4-scale pilots validate the recurrent-depth idea. The milestone is a
**1B-parameter OpenMythos that beats every open 1B model**: TinyLlama-1.1B,
SmolLM2-1.7B, OLMo-2-1B, Llama-3.2-1B, Qwen2.5-1.5B.

- `research/1b_battle_plan.md` — the battle plan: targets, data mix, recipe, compute stages
- `research/44m_experiment_report.md` — Stage-0 evidence: the 44M TinyStories run vs its matched-parameter baseline (loss, CIs, compute, limitations)
- `research/3b_landscape.md` — competitive survey (12 models) behind the plan
- `configs/mythos_1b.py` — 1B config (1.06B params, verified): GQA, 64 experts / 8 active,
  64K custom BPE, loop-depth curriculum
- `patches/` — code modifications vs upstream: QK-Norm, sliding-window interleave
  (embeddings already tied upstream)
- `training/` — 1B training code (FSDP + AdamW, staged data)

The killer feature: **test-time depth extrapolation** — train with n_loops ≤ 8,
sweep 4→16 at eval, watch reasoning scores climb. No dense 1B model can do this.

> Status note (Oct 2026): the 44M Stage-0 run does **not** show extrapolation
> gains on TinyStories loss — eval depth 8 is significantly worse than the
> trained depth 4 (paired 95% CI [+0.101, +0.110]). See
> `research/44m_experiment_report.md` §4. The extrapolation story now rests
> on the 1B loop-curriculum run, not on this result.
