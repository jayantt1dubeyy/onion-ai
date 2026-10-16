# 1B Battle Plan — beating every open 1B model

**Mission:** train a 1B OpenMythos that beats TinyLlama-1.1B, SmolLM2-1.7B,
OLMo-2-1B, Llama-3.2-1B, and Qwen2.5-1.5B on the standard benchmarks.
**Status:** planning + Stage 0 running. **Config:** `configs/mythos_1b.py`
(1.06B params, verified). **Landscape:** `research/3b_landscape.md`.

---

## 1. The targets (from the landscape survey)

| Model | Params | Tokens | What we must beat |
|---|---|---|---|
| TinyLlama-1.1B | 1.1B | 3T | The overtraining pioneer; weakest target, first baseline |
| OLMo-2-1B | 1B | 4T | Fully open recipe; the honest baseline |
| Llama-3.2-1B | 1.2B | 9T | Pruned+distilled from larger Llama; strong instruct |
| Qwen2.5-1.5B | 1.5B | 18T | 18T tokens; the data-scale hammer |
| SmolLM2-1.7B | 1.7B | 11T | Best fully-open recipe; beats Qwen2.5-1.5B per its paper |

Key evals to win: **MMLU, GSM8K, MATH, ARC-Challenge, HellaSwag, HumanEval/MBPP.**
Harness: `lm-evaluation-harness`, base checkpoint, same few-shot setups as the
reported numbers. No cherry-picking.

## 2. Why we can win (the honest version)

Nobody at 1B wins on architecture — all targets are dense GQA transformers.
They win on data × tokens × post-training. Our edge is the **only different
compute lever in the field**: recurrent depth = adaptive test-time compute.
Google tried recurrence (RecurrentGemma-2B, MMLU 38.4) and failed — so the bar
is *recurrence done right*:

1. **QK-Norm** (`patches/qk_norm.md`) — the stability fix recurrence needs.
2. **GQA + sliding-window loop** (`patches/sliding_window.md`) — don't burn
   FLOPs on full attention × 16 iterations.
3. **Qwen3-style MoE** — 64 fine-grained experts, 8 active, no shared expert,
   global-batch load-balancing loss. Same active FLOPs as dense, more capacity.
4. **Loop-depth curriculum** — train n_loops 2 → 4 → 8 (anneal); then the
   killer demo: **test-time depth extrapolation**, sweeping n_loops 4→8→16 at
   eval and watching reasoning scores climb. No dense 1B model can do this.

## 3. Data (SmolLM3-shaped, Nemotron-upgraded)

- **Stage 1 (0–60%):** 80% web (Nemotron-CC-HQ + FineWeb-Edu + DCLM-Baseline),
  12% code (Stack-Edu), 5% math (FineMath4+), 3% synthetic QA
- **Stage 2 (60–90%):** 70% web / 15% code / 12% math (+MegaMath)
- **Anneal (90–100%):** 55% web / 20% code / 20% math + reasoning/instruction
  data (OpenMathInstruct-style) + long-context data
- **Tokenizer:** custom 64K BPE trained on the Stage-1 mix (not GPT-2's —
  a 50K vocab wastes the embedding budget on tokens we never see)

## 4. Training recipe (copy the winners)

- Optimizer: **AdamW (0.9, 0.95)**, weight decay 0.1
- Schedule: **Warmup-Stable-Decay** — warmup ~2k steps, peak LR ~4e-4,
  decay over final 10% (lets us extend training without recommitting)
- Batch: **~2M tokens/step** (settled science at this scale)
- Precision: bf16 + FSDP
- Budget: **Stage 1 proof = 100B+ tokens** → full run **2–4T tokens**
  (Chinchilla-optimal for 1B is ~20B; winners overtrain 100–200× past it)

## 5. Post-training ladder (non-negotiable — this is where MATH points live)

SFT on synthetic conversational data → DPO/APO → **RLVR/GRPO on verifiable
math+code rewards**. Phi-4-mini's MATH-500 67.6 comes from exactly this ladder.
Budget it as part of the run, not an afterthought.

## 6. Compute stages (honest)

| Stage | What | Tokens | Compute | Status |
|---|---|---|---|---|
| 0 | 44M pipeline validation (Kaggle T4) | 120M | free | **running** |
| 1 | 1B proof run: beat TinyLlama/DCLM-1B, demo loop extrapolation | 100B+ | ~1–2K H100-hrs | needs cluster/grant |
| 2 | Full 1B: staged data + post-training ladder | 2–4T | ~10–20K H100-hrs | needs partnership |

Stage 1 is the milestone that earns Stage 2. A 1B proof run cannot happen on
the Kaggle free tier (100B tokens ≈ months on a T4) — options: academic
compute grants, lab partnerships, or staged cloud spend.

## 7. Eval protocol

- During training: perplexity on held-out FineWeb-Edu + MATH-500 every 5B tokens
- Loop sweep: at each checkpoint, eval n_loops ∈ {2, 4, 8, 16} on GSM8K —
  the extrapolation curve is our signature result
- Final: full lm-eval suite vs the five targets, same harness version,
  results committed to `evals/` with the exact commands to reproduce
