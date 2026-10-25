# OpenMythos 44M — Experiment Report (Stage 0)

**Date:** 16 October 2026 · **Status:** complete, single seed · **Scale:** 44M params, TinyStories, one Kaggle T4

**One-line verdict.** Shared-weight recurrence helps enormously at matched
parameter count (held-out loss 3.38 → 1.48 from loop depth 1 → 4), but
**loop-depth extrapolation beyond the trained depth does not help at this
scale** — depth 8 is significantly *worse* than depth 4. Any efficiency claim
must be stated in compute (tokens, GPU-hours), not parameters: at depth 4 the
loop block runs 4× per token, so the 44M model costs roughly 4× the loop-block
compute of its own 44M unlooped baseline.

> **Codebase note.** This run used the `kyegomez/OpenMythos` package
> (pip `open-mythos`), not the from-scratch code in this repo. This repo's
> 1B roadmap builds on this result; the report lives here because this is
> the project's research home. Checkpoints and eval scripts are local (see
> §7); the numbers below come from the run logs and eval JSONs, not from
> memory.

## 1. Question

Can shared-weight recurrent computation (one block, iterated) buy effective
depth more efficiently than a plain stack of layers? The test: train one
model with 4 loop iterations, then evaluate the *same weights* at loop
depths 1, 2, 4, 8, 12, 16. Depth 1 is the matched-parameter baseline — the
same 44M parameters with recurrence switched off.

## 2. Model and training

| Item | Value (as logged by the run) |
|---|---|
| Parameters | 43,974,722 (run-printed) |
| Architecture | Prelude (6 layers) → shared recurrent block → Coda (2 layers) |
| Loop iterations (trained) | 4 |
| Attention | Grouped-query attention (GQA) + RoPE |
| Feed-forward | Sparse MoE, 8 experts |
| Tokenizer | GPT-2 BPE, vocab 50,257 |
| Recurrent stability | Spectral radius ρ(A) max = 0.3679 (< 1, contractive) |
| Data | TinyStories (streamed) |
| Token budget | ~120M tokens (119.2M logged at step 4,850) |
| Hardware | Kaggle Tesla T4, 15 GB, fp16 autocast |
| Optimizer steps | 4,850 (~2.5 GPU-hours; 152 min logged) |
| Throughput | ~13.1k tokens/s, steady |
| LR schedule | Warmup to 3.00e-04, cosine decay to ~0 (3.95e-08 at step 4,850) |
| Gradient clipping | 1.0 |

**Training loss:** 9.5242 (step 50) → 1.5263 (step 4,850). The curve is noisy
throughout (typical of small-batch TinyStories training) but trends down
monotonically in the large.

**Generation quality over training** (fixed prompt:
*"Once upon a time there was a little girl named Lily."*, 60 new tokens,
temperature 0.8, top-k 50):

- Step 500: *"She wanted to sit down the table in the sky."* — barely coherent.
- Step 4,500: *"She loved to play with her toys in the park. One day, Lily
  and her friends decided to have a picnic in the park. They brought a big
  basket full of yummy food."* — fluent, on-topic, grammatical.

## 3. Evaluation methodology

- **Held-out set:** 256 TinyStories sequences × 256 tokens, fixed across all
  depths (same sequences, same weights — the comparison is paired).
- **Metrics:** mean cross-entropy loss and perplexity (exp of loss).
- **Uncertainty:** per-sequence bootstrap 95% CIs for each depth, plus
  *paired* per-sequence CIs for depth-to-depth differences (the paired
  intervals are what license "significantly better/worse" claims).
- **Generation check:** one fixed prompt, greedy-ish sampling, quoted
  verbatim below — qualitative only, not scored.
- **What was NOT measured:** no downstream benchmarks (ARC, HellaSwag,
  etc.), no human evaluation, no latency/throughput benchmark at inference.

## 4. Results: loop-depth sweep (the baseline comparison)

| Eval depth | Loss | 95% CI (bootstrap) | Perplexity |
|---:|---:|---:|---:|
| 1 (baseline: no recurrence) | 3.380 | [3.343, 3.418] | 29.38 |
| 2 | 2.414 | [2.384, 2.446] | 11.18 |
| **4 (trained depth)** | **1.484** | **[1.454, 1.516]** | **4.41** |
| 8 | 1.590 | [1.559, 1.621] | 4.90 |
| 12 | 1.591 | [1.560, 1.624] | 4.91 |
| 16 | 1.591 | [1.560, 1.623] | 4.91 |

**Paired difference, depth 4 → 8:** +0.106 loss, 95% CI [0.101, 0.110] —
depth 8 is significantly worse, not just noisier. Depths 12 and 16 are
identical to each other (1.591): past the trained depth, extra loops
saturate, they do not compound.

**Generations at eval depth** (same prompt as §2):

- Depth 1: *"Lily liked to dress up for. One, she got a call dressed and
  even a a bow from the mom."* — broken grammar, drifting.
- Depth 4: *"She loved to go to the theater with her mom and dad. One day,
  they went to a big theater far away. Lily was very excited…"* — fluent,
  coherent narrative.
- Depth 8: *"She had a very delicate dress that was very special to her.
  She loved it very much and kept it very special."* — grammatical but
  repetitive and confused; matches the loss regression.

**Reading.** Recurrence is doing real work: at identical parameters, depth
4 beats depth 1 by 1.90 loss points (perplexity 29.4 → 4.4). But the gain
is a *trained-depth* phenomenon. Nothing in this run supports the roadmap's
"test-time depth extrapolation" claim at 44M scale on TinyStories loss:
extrapolation costs +0.11 loss and then flatlines. Whether a loop-depth
*curriculum* (training across depths, as the 1B plan proposes) changes this
is an open hypothesis, not an established property.

## 5. Compute, honestly

| Metric | Value |
|---|---|
| Parameters | 43,974,722 |
| Training tokens | ~120M |
| Training time | ~2.5 Tesla T4-hours |
| Throughput | ~13.1k tokens/s |
| Rough 6ND upper bound | ~3 × 10¹⁶ FLOPs (see caveat) |

**Caveats (this is the reviewer's point, stated plainly).** (a) The 6ND
figure is an *upper bound*: with sparse MoE, only a fraction of experts is
active per token, so true training FLOPs are lower, and they were not
directly measured. The measured compute metrics are the tokens and the
GPU-hours. (b) Parameter count is not compute here: at depth 4, each token
passes through the shared block four times, so inference costs roughly 4×
the loop-block compute of the depth-1 baseline *at the same parameter
count*. Any "44M beats X" claim that compares only parameter counts is
invalid for this architecture. (c) No external model was retrained as a
baseline; depth 1 is an internal, matched-parameter control, not a tuned
competitor.

## 6. Limitations

1. **Single seed.** The headline numbers come from one training run.
   A second seed is the main open gap (needs GPU time); until it exists,
   treat the second decimal as suggestive, not settled.
2. **TinyStories only.** Simple children's stories with a small vocabulary.
   Nothing here licenses claims about general language modelling.
3. **Loss/perplexity only.** No downstream task scores, no human eval.
4. **No tuned external baseline.** Depth 1 controls for parameters, not for
   "best comparably-sized standard transformer" — a tuned dense baseline
   at the same compute could differ.
5. **MoE routing unanalyzed.** Expert utilization and load balancing were
   not measured; a collapsed router would change the interpretation.
6. **Generations are one fixed prompt, qualitative.** They illustrate, they
   do not evaluate.
7. **Extrapolation claim contradicted at this scale.** See §4. The 1B
   roadmap's extrapolation story needs its own evidence (loop curriculum),
   not this run's.

## 7. Artifacts and reproducibility

- Final checkpoint: `mythos44m_final.pt` (+ step 1000/2000/3000/4000
  checkpoints), training log `train.log`, Kaggle run log — local,
  `openmythos-kaggle/run-output/`.
- Eval: held-out sequences `eval_seqs_256.pt`; sweep + bootstrap scripts
  (`eval_loop_sweep.py`, `bootstrap_ci.py`); results JSONs
  (`loop_sweep_results.json`, `loop_sweep_ci.json`, `loop_sweep_perseq.json`).
- The 44M run is Stage 0 of the 1B roadmap (`research/1b_battle_plan.md`).
  Next stages: second seed, then the 1B config (`configs/mythos_1b.py`,
  1.06B params) with a loop-depth curriculum — the experiment that would
  actually test extrapolation.

## 8. What would change these conclusions

- A second seed agreeing within CIs → the depth-4 optimum is real.
- A tuned dense baseline at matched *compute* (not params) → the
  efficiency story becomes comparative instead of internal.
- A loop-curriculum run where eval depth > trained depth *improves* loss →
  the extrapolation hypothesis revives.
- Downstream evals (even simple ones: story completion accuracy, BLiMP-style
  probes) → the loss gains mean something beyond TinyStories.
