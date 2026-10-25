# Addendum: MiniMythos 125M zero-shot results

**Provenance.** Numbers below are taken from Jayant's own July 2026
one-page summary and slides ("MiniMythos 125M — A Case for Recurrent
Virtual Depth", ONI CHAIN Research Notebook, v1.0). They are **not**
independently verified runs: unlike the 44M report, the summary gives no
seeds, no confidence intervals, and no eval code. Treat every figure as
reported, not reproduced. This is a second data point alongside
`44m_experiment_report.md`, not a replacement for it.

## Results (zero-shot accuracy, %)

| Model | HellaSwag | PIQA |
|---|---:|---:|
| GPT-2 Small (124M) | 28.5 | 62.8 |
| TinyLlama-120M | 31.5 | 64.2 |
| MiniMythos (8 loops) | 29.5 | 50.0 |
| MiniMythos (16 loops) | 30.0 | 50.0 |

Baselines for ARC-Challenge are shown in the summary (GPT-2 Small 21.0,
TinyLlama-120M 23.5) but **no MiniMythos ARC number is given** — that gap
is flagged here, not filled.

## What the numbers say, precisely

- **HellaSwag:** MiniMythos improves with loops (29.5 → 30.0 from 8 to 16)
  and beats GPT-2 Small (28.5) at both loop counts — but it does **not**
  beat TinyLlama-120M (31.5). The "breakthrough" is vs GPT-2 Small only.
- **PIQA:** flat 50.0% at both loop counts, far below both baselines
  (62.8 / 64.2). Loops add no factual recall — the summary's own
  "knowledge ceiling" framing.
- **Effective depth:** 7 physical layers → 22 effective layers at 16
  loops, per the summary's definition (not independently derived here).

## Reconciliation with the 44M TinyStories result

The two results do not contradict each other, but they do narrow the
claim:

- 44M, TinyStories *loss*, fixed-depth training: extrapolating past the
  trained depth (4 → 8) **hurt** significantly (paired CI [+0.101, +0.110]).
- 125M, zero-shot *accuracy*, Loop-Agnostic Training (per the summary):
  going 8 → 16 loops **helped** HellaSwag (+0.5pt) and held PIQA flat.

Two differences could explain this, both stated as hypotheses, not
findings: (1) scale (125M vs 44M); (2) training recipe — MiniMythos
claims loop-agnostic training, which is designed to stabilize hidden
states across depths, while the 44M run trained at a single fixed depth.
The experiment that would settle it: a loop-curriculum run at one scale
with both loss and zero-shot accuracy reported — which is exactly what
the 1B plan proposes.

## Caveats carried into the record

- No CIs, seeds, or eval code in the summary → no significance claims.
- PIQA at 50.0% is exactly chance level: PIQA is a 2-choice task, so
  "stable at 50.0%" does not show robustness — it shows the model has
  no PIQA knowledge to lose. The summary's "stability" framing should
  not be repeated without this qualifier.
- "7 → 22 effective layers" is the summary's arithmetic, not a measured
  quantity.
