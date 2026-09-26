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
