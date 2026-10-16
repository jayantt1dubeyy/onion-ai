# The ~3B Language Model Landscape (2026)
**Research for:** training a 3B OpenMythos model (Recurrent-Depth Transformer: Prelude → looped Recurrent Block → Coda, MoE FFN, MLA/GQA attention — [kyegomez/OpenMythos](https://github.com/kyegomez/OpenMythos)) that can beat the best models in its class.
**Compiled:** 16 October 2026. All eval numbers are reported with their source; base vs instruct checkpoints are distinguished.

---

## 1. Model-by-model survey

### 1.1 Llama 3.2 3B (Meta, Sep 2024)
- **Architecture:** Dense decoder-only transformer, 3.21B params (28 layers, d=3072, 24 Q heads / 8 KV heads GQA), SwiGLU, RoPE, RMSNorm, **tied embeddings**, 128K context. 128K-token BPE vocabulary.
- **Pretraining data:** Up to **9T tokens** of "a new mix of publicly available online data" (Meta does not disclose composition). Knowledge cutoff Dec 2023. Notably, the 1B/3B sizes were derived by **pruning larger Llama models and distilling** on those 9T tokens (per the SmolLM2 paper's account).
- **Recipe:** Not disclosed (optimizer/schedule/batch unknown publicly).
- **Post-training:** SFT + RLHF.
- **Evals (Instruct, Meta's published table):** MMLU 5-shot **63.4**, GSM8K 8-shot CoT **77.7**, MATH 0-shot **48.0**, ARC-Challenge **78.6**, HellaSwag **69.8**, GPQA **32.8**, IFEval **77.4**, BFCL-v2 **67.0**. (Open LLM Leaderboard reruns: MMLU 62.98, GSM8K 77.94.)
- **License/openness:** Llama 3.2 Community License (custom; commercial use allowed with restrictions). Weights only — no code, no data.
- Sources: [Meta Llama 3.2 model card](https://github.com/meta-llama/llama-models/blob/rc-0.1.4.dev3/models/llama3_2/MODEL_CARD.md), [eval table mirror](https://huggingface.co/context-labs/meta-llama-Llama-3.2-3B-Instruct-FP16)

### 1.2 Qwen2.5-3B (Alibaba, Sep 2024)
- **Architecture:** Dense, **3.09B** (2.77B non-embedding), 36 layers, GQA **16 Q / 2 KV heads**, SwiGLU, RMSNorm, RoPE, QKV bias, **tied embeddings**, 32K native context (128K for Instruct via YaRN + Dual Chunk Attention). BBPE tokenizer, **151,669** vocab.
- **Pretraining data:** **18T tokens** for the series (up from 7T in Qwen2), with aggressive quality filtering (Qwen2-Instruct used as a data scorer), domain rebalancing, and **synthetic math/code injection**.
- **Recipe:** Not fully disclosed; the family scales one recipe 0.5B→72B.
- **Post-training (best-documented open recipe of its generation):** 3 stages — SFT on **1M+ examples** → offline RL (**DPO**, ~150K pairs) → online RL (**GRPO** with multi-criteria reward model). Specialized branches (Math, Coder, QwQ) reuse the backbone with different post-training mixes.
- **Evals (Base, from the Qwen3 technical report):** MMLU **65.62**, MMLU-Redux **63.68**, GSM8K **79.08**, MATH **42.64**, MBPP **54.6**, GPQA **26.26**. (Instruct: MMLU-Redux 64.4, MATH-500 67.2.)
- **License/openness:** Apache 2.0. Weights only.
- Sources: [Qwen2.5-3B HF card](https://huggingface.co/Qwen/Qwen2.5-3B), [Qwen2.5 tech report](https://arxiv.org/abs/2412.15115) (arXiv:2412.15115)

### 1.3 Qwen3-4B (Alibaba, Apr 2025) — the current dense model to beat
- **Architecture:** Dense, 36 layers, **32/8 GQA**, **tied embeddings**, 128K native context. Changes vs Qwen2.5: **QK-Norm added** (RMSNorm on Q/K before softmax, for training stability), QKV bias removed. Same 151,669 tokenizer.
- **Pretraining data:** **36T tokens**, 119 languages, in **3 stages**: (1) 30T+ tokens at 4K context — broad LM; (2) STEM/code/reasoning enhancement; (3) 32K long-context extension.
- **Post-training:** Hybrid **thinking / non-thinking** modes; SFT + RL.
- **Evals (Base, Qwen3 report):** MMLU **72.99**, MMLU-Redux **72.79**, MMLU-Pro **50.58**, GSM8K **87.79**, MATH **54.10**, MBPP **67.0**, GPQA **36.87**, BBH **72.59**. Beats Qwen2.5-7B-Base on 50%+ of benchmarks despite being ~45% smaller — the clearest proof that data scale + recipe now dominate raw size.
- **License/openness:** Apache 2.0. Weights only.
- Sources: [Qwen3 technical report](https://arxiv.org/pdf/2505.09388) (arXiv:2505.09388), [Qwen3 blog](https://qwenlm.github.io/blog/qwen3/)

### 1.4 Phi-4-mini 3.8B (Microsoft, Feb 2025)
- **Architecture:** Dense decoder-only, **3.8B**, **GQA** (new vs Phi-3.5-mini), **200K vocabulary** (expanded for multilinguality), **tied embeddings**, 128K context.
- **Pretraining data:** **5T tokens** — high-quality web **plus a heavy synthetic-data recipe emphasizing math and code**. The synthetic mix is the stated reason it matches models 2× its size on reasoning tasks.
- **Recipe (partially disclosed):** Trained Nov–Dec 2024 on **512× A100-80G for 21 days**.
- **Post-training:** SFT; a reasoning-enhanced variant trained further matches DeepSeek-R1-Distill-Qwen-7B / Llama-8B on reasoning.
- **Evals (Instruct, from the Qwen3 report's comparison table):** MMLU-Redux **67.9**, IFEval strict **68.6**, MATH-500 **67.6**, AIME'24 **8.1**, Arena-Hard **32.8** — the strongest math scores per parameter in this survey.
- **License/openness:** MIT. Weights only.
- Sources: [Phi-4-mini technical report](https://arxiv.org/abs/2503.01743) (arXiv:2503.01743), [model card mirror](https://github.com/khimaros/aimbot/blob/HEAD/research/data/cards/unsloth_Phi-4-mini-instruct-GGUF.md)

### 1.5 Gemma 2 2B (Google, Jun 2024)
- **Architecture:** 2.59B total (**2.02B non-embedding**), 26 layers, d=2304, 8 heads / 4 KV heads, head dim 256, FFN 9216 (approx GeGLU), **256K vocabulary** (inherited from Gemini — embeddings alone are ~590M params). Signature features: **interleaved local sliding-window (4096) + global attention every other layer**, GQA, **pre- AND post-RMSNorm** on every sub-layer, **logit soft-capping** (tanh, cap 50 attn / 30 final), RoPE, 8K context. **Distilled from a larger teacher** during pretraining.
- **Pretraining data:** **2T tokens**.
- **Post-training:** SFT + RLHF **+ knowledge distillation** from Gemma-2-27B-IT.
- **Evals (PT 2B, Google):** MMLU 5-shot **51.3**, HellaSwag **73.0**, ARC-c **55.4**, GSM8K **23.9**, MATH **15.0**, HumanEval **17.7**, MBPP **29.6**.
- **License/openness:** Gemma License (custom; commercial use permitted with use-policy restrictions). Weights only.
- Sources: [Gemma 2 technical report](http://arxiv.org/pdf/2408.00118v2) (arXiv:2408.00118), [HF model card](https://huggingface.co/google/gemma-2-2b)

### 1.6 Gemma 3 4B (Google, Mar 2025)
- **Architecture:** 4B, **5:1 local-to-global attention** (1024-token local windows, global layers span 128K), 128K context, multimodal (SigLIP vision encoder on 4B+), 140+ languages.
- **Pretraining data:** **4T tokens** (1B: 2T, 12B: 12T, 27B: 14T) — web + code + math + images.
- **Post-training:** Knowledge distillation from larger models + multi-objective RL (RLHF + ground-truth outcome rewards). QAT variants for int4/FP8.
- **Evals (Base, Qwen3 report):** MMLU **59.51**, GSM8K **43.97**, MATH **26.10**, MBPP **46.40**.
- **License/openness:** Gemma License. Weights only.
- Sources: [Gemma 3 technical report](https://arxiv.org/abs/2503.19786) (arXiv:2503.19786), [HF model card](https://huggingface.co/google/gemma-3-4b-it-qat-q4_0-unquantized)

### 1.7 SmolLM2-1.7B (HuggingFace, Feb 2025) — the fully-open recipe reference
- **Architecture:** Llama2-style, 24 layers, d=2048, 32 heads, FFN 8192, SwiGLU, RoPE (θ=10,000), **tied embeddings**, 2K context extended to 8K. Custom tokenizer, **49,152** vocab (trained on 70% FineWeb-Edu / 15% Cosmopedia-v2 / 8% OpenWebMath / 5% StarCoder / 2% StackOverflow).
- **Pretraining data:** **~11T tokens** (overtrained ~2 epochs over collected data), **multi-stage with performance-driven mix interventions**: Stage 1 (0–6T): 60% FineWeb-Edu + 40% DCLM web, 10% StarCoderData code; later stages progressively upsample math/code (FineMath, Stack-Edu, InfiMM-WebMath, OpenWebMath) during annealing. Released **FineMath (54B tokens)** and **Stack-Edu (125B)** as new open datasets.
- **Recipe (fully disclosed):** nanotron framework, **256× H100**, **AdamW (β 0.9/0.95)**, **Warmup-Stable-Decay (WSD)** schedule — 2,000-step warmup, peak LR **5e-4**, decay to 0 over final 10% of steps; **2M tokens/batch**; ~**$250K** of compute.
- **Post-training:** SmolTalk (950M-token synthetic conversational dataset) SFT + DPO.
- **Evals:** Outperforms Qwen2.5-1.5B and Llama-3.2-1B (per the paper).
- **License/openness:** Apache 2.0. **Fully open: weights, datasets, configs, training logs.**
- Sources: [SmolLM2 paper](https://arxiv.org/pdf/2502.02737) (arXiv:2502.02737)

### 1.8 SmolLM3-3B (HuggingFace, Jul 2025) — the fully-open 3B to beat
- **Architecture:** **3.08B**, decoder-only, **GQA + NoPE** (no positional encoding on 3 of 4 attention head groups — ablated as better for long context), 64K trained context → 128K via YaRN.
- **Pretraining data:** **11.2T tokens**, 3 stages: S1 (0–8T): web 85% (FineWeb-Edu, DCLM, FineWeb2, FineWeb2-HQ), code 12% (Stack v2), math 3%; S2 (8–10T): web 75%, code 15% (+Stack-Edu), math 10% (FineMath4+, InfiWebMath4+, **MegaMath** incl. synthetic rewrites); S3 decay (10–11.1T): web 63%, code 24%, math 13% (+OpenMathReasoning). **Mid-training: 140B reasoning tokens.** Trained on **384× H100 for 24 days**.
- **Post-training:** SFT + **Anchored Preference Optimization (APO)**; dual-mode reasoning (`/think` on by default, `/no_think` off).
- **Evals:** Outperforms Llama-3.2-3B and Qwen2.5-3B; **competitive with Qwen3-4B and Gemma-3-4B** (per HF).
- **License/openness:** Apache 2.0. **Fully open: weights, public data mixture, training configs, logs, intermediate checkpoints.**
- Sources: [SmolLM3 blog/paper](https://huggingface.co/blog/smollm3), [model card](https://github.com/cezary13k/miscellany/blob/HEAD/agents/model_summary/examples/SmolLM3-3B.md)

### 1.9 OLMo-2 family (AI2, Nov 2024–Mar 2025) — the science-first reference
- **Architecture:** Standard dense transformer. 1B: 16L/d2048/16 heads; 7B: 32L/d4096/32 heads; 13B: 40L/d5120/40 heads; 32B: 64L/d5120/40 heads. 4K context.
- **Pretraining data:** **Stage 1:** OLMo-Mix-1124 (**3.9T tokens**, ~1 epoch for 7B; 5T for 13B). **Stage 2 (mid-training):** Dolmino-Mix-1124 (**843B** high-quality tokens — 50% academic/Q&A/instruction/math) with **model souping** (3 independent runs merged for 7B; 4 for 13B).
- **Recipe:** AdamW; the defining feature is **full artifact release** (code, data, checkpoints, logs, exact batch order reconstructable).
- **Post-training:** **Tulu 3** recipe — SFT → DPO → RLVR/PPO (verifiable-reward RL, e.g. GRPO on GSM8K/MATH-style tasks).
- **Evals:** 7B/13B competitive with Llama-3.1-8B-class models.
- **License/openness:** Apache 2.0. **The most open frontier recipe in existence.**
- Sources: [OLMo-2-1124-13B HF card](https://huggingface.co/allenai/OLMo-2-1124-13B), [OLMo-2-0425-1B HF card](https://huggingface.co/allenai/OLMo-2-0425-1B)

### 1.10 TinyLlama 1.1B (2023–2024) — the overtraining pioneer
- **Architecture:** Llama2-style, 22 layers, d=2048, 32 heads / 4 KV groups (GQA), SwiGLU, 2K context, Llama-2 tokenizer.
- **Pretraining data:** **3T tokens** (~3 epochs over ~950B: SlimPajama + StarCoderData at **7:3** NL:code ratio; GitHub deduped between sources).
- **Recipe (fully open):** FSDP + FlashAttention-2 + fused layernorm/SwiGLU/CE/RoPE kernels → **24K tok/s/GPU on A100-40G (56% MFU)**; 16× A100-40G, 90 days; **2M-token batches**, LR **4e-4 cosine** with 2,000-step warmup.
- **License/openness:** Apache 2.0. Code + checkpoints + logs open.
- **Why it matters:** Proved that massively overtraining a small model on repeated data keeps improving it — the empirical basis for SmolLM2/3's 11T-token runs.
- Sources: [TinyLlama GitHub](https://github.com/jzhang38/TinyLlama), [TinyLlama paper](https://arxiv.org/pdf/2401.02385) (arXiv:2401.02385)

### 1.11 DCLM / DataComp-LM (Apple + UW, 2024) — the dataset, not just a model
- **DCLM-BASELINE:** **3.8T-token** web dataset (4.1T with StarCoder + ProofPile2 mixed in): Resiliparse HTML extraction → Bloom-filter dedup → **fastText quality classifier** (trained on OpenWebText2 + ELI5). Competition scales include **1B-1x: 1.4B params / 28.8B tokens** and **1B-5x: 1.4B / 144B** — the "DCLM-1.4B" reference models.
- **Key result:** 7B trained on **2.6T** DCLM-BASELINE → **64% MMLU (5-shot)**, SOTA among open-data models, near Mistral-7B-v0.3 (63%) and Llama-3-8B (66%) at up to **6.6× less compute**. Vs Llama-2-7B: +5pp MMLU at 7× less compute.
- **License/openness:** Open dataset + framework ([datacomp.ai/dclm](https://datacomp.ai/dclm)).
- Sources: [DCLM paper](https://arxiv.org/pdf/2406.11794) (arXiv:2406.11794)

### 1.12 Also notable (2025–2026)
- **Ministral-3-3B-Instruct (Mistral, Dec 2025):** 3.3B vision-language model (Mistral3 arch); language backbone relevant — **tied embeddings**, 256K context, Tekken tokenizer (131K vocab), GQA. Weights open under Mistral license; training details undisclosed. Sources: [vLLM recipe](https://github.com/vllm-project/recipes/blob/HEAD/Mistral/Ministral-3-Instruct.md)
- **DeepSeek-R1-Distill-Qwen-1.5B (Jan 2025):** reasoning-distilled 1.5B; relevant as a post-training target style (distilling R1 traces into small models works — Phi-4-mini's reasoning variant plays the same game).
- **RecurrentGemma-2B (Google, 2024):** Griffin (recurrent + local attention hybrid), 2B — MMLU 38.4, GSM8K 13.4. The closest *architectural* cousin to OpenMythos's recurrence idea, and a cautionary data point: naive recurrence underperformed transformers at the time. Source: [HF card](https://huggingface.co/google/recurrentgemma-2b)

---

## 2. Comparison table

| Model | Params | Train tokens | Attention | FFN | Context | Vocab | Standout eval (source) | License |
|---|---|---|---|---|---|---|---|---|
| Llama 3.2 3B | 3.21B | 9T | GQA | dense SwiGLU | 128K | 128K | MMLU 63.4, GSM8K 77.7 (Meta) | Llama Community |
| Qwen2.5-3B | 3.09B | 18T | GQA 16/2 | dense SwiGLU | 32K→128K | 151.7K | MMLU 65.6, GSM8K 79.1 (Qwen3 rpt) | Apache 2.0 |
| **Qwen3-4B** | 4B | **36T** | GQA 32/8 +QK-Norm | dense SwiGLU | 128K | 151.7K | MMLU 73.0, GSM8K 87.8, MATH 54.1 (Qwen3 rpt) | Apache 2.0 |
| Phi-4-mini | 3.8B | 5T | GQA | dense | 128K | 200K | MATH-500 67.6, IFEval 68.6 (Qwen3 rpt) | MIT |
| Gemma 2 2B | 2.59B | 2T | GQA + SWA/global interleave | GeGLU | 8K | 256K | MMLU 51.3, HumanEval 17.7 (Google) | Gemma |
| Gemma 3 4B | 4B | 4T | 5:1 local/global SWA | GeGLU | 128K | 256K | MMLU 59.5, GSM8K 44.0 (Qwen3 rpt) | Gemma |
| SmolLM2-1.7B | 1.7B | 11T | GQA-ish, RoPE | dense SwiGLU | 8K | 49K | beats Qwen2.5-1.5B, Llama-3.2-1B (paper) | Apache 2.0 |
| **SmolLM3-3B** | 3.08B | **11.2T** | GQA + NoPE | dense SwiGLU | 64K→128K | ~128K? | ≈ Qwen3-4B / Gemma-3-4B (HF) | Apache 2.0 |
| OLMo-2 1B | 1B | 4T | MHA-ish dense | dense SwiGLU | 4K | 50K | — (7B ≈ Llama-3.1-8B class) | Apache 2.0 |
| TinyLlama 1.1B | 1.1B | 3T | GQA 32/4 | dense SwiGLU | 2K | 32K | overtraining pioneer | Apache 2.0 |
| DCLM-1.4B | 1.4B | 29–144B | dense | dense | 2K | 50K | data-efficiency reference | Open |

**Reading the table:** the models that win (Qwen3-4B, SmolLM3-3B, Phi-4-mini) are separated not by architecture — all are dense GQA transformers — but by **token volume × data quality × post-training**. Nobody in this weight class wins on architecture alone.

---

## 3. Dataset landscape for training a 3B model from scratch (2026)

1. **FineWeb-Edu (HuggingFace, 1.3T tokens)** — web text filtered by an educational-quality classifier. The default high-quality web base; used at 60–85% of the mix by SmolLM2/3. Mature, trusted, slightly dated (2024 crawl).
2. **DCLM-Baseline (3.8T)** — model-based fastText filtering on Common Crawl. Best *demonstrated* token-efficiency among open datasets (64% MMLU at 7B/2.6T). SmolLM3 still uses it in the web blend.
3. **Nemotron-CC (NVIDIA, Jan 2025, 6.3T)** — classifier-ensemble filtering **plus ~2T tokens of synthetic rephrasing** (QA pairs, distilled content, knowledge lists) to recover the ~90% of Common Crawl that heuristic filters discard. **Nemotron-CC-HQ beats DCLM by +5.6 MMLU** (8B/1T setting); the full 6.3T matches DCLM with **4× more unique tokens** — the best open option for *long-horizon* (multi-trillion-token) training without excessive repetition. Pipeline integrated into NeMo Curator.
4. **FineWeb2 / FineWeb2-HQ (2025)** — multilingual web (1,000+ languages); the HQ subset is the current best open non-English base. SmolLM3 uses both.
5. **Math:** FineMath4+ (evolved from the 54B FineMath), MegaMath (with Qwen-Q&A and Pro synthetic rewrites), InfiWebMath, OpenWebMath — synthetic math is now load-bearing (Phi-4-mini, SmolLM3).
6. **Code:** Stack-Edu (125B, educational filter on The Stack v2), StarCoder2, plus PRs/notebooks/issues.
7. **2026 consensus mix shape** (from SmolLM3, the freshest fully-open 3B): start **web-heavy (85%)**, then *stage* the quality upward — code 12→24%, math 3→13% — and finish with an **annealing/decay phase on the highest-quality + reasoning/instruction data**. Synthetic data at every stage, not just post-training.

**Verdict:** For a from-scratch 3B in 2026, the best-regarded open foundation is **Nemotron-CC (or its HQ subset) + FineWeb-Edu + DCLM-Baseline** for web, **Stack-Edu** for code, **FineMath4+/MegaMath** for math, staged SmolLM3-style. Nemotron-CC is the single biggest upgrade over the 2024 recipe because it solves the repetition problem for long runs.

---

## 4. Training-recipe consensus (what OLMo-2, SmolLM2, TinyLlama, DCLM agree on)

- **Optimizer:** AdamW, β=(0.9, 0.95), weight decay 0.1, grad-clip 1.0. Unanimous.
- **LR schedule:** warmup (hundreds–2,000 steps) → long stable phase → decay to ~0. SmolLM2 uses **Warmup-Stable-Decay** (peak 5e-4, decay over final 10%); TinyLlama uses cosine (4e-4); OLMo-1 used linear to 0.1× peak. **WSD is the modern default** — it lets you extend training without committing to a total step count upfront.
- **Batch size:** **~2M tokens/batch** (TinyLlama, SmolLM2). Large batches are settled science for stability at scale.
- **Precision:** bf16 mixed precision (fp16 on older hardware), full-precision gradient reduction / optimizer states.
- **Tokenizer:** 32K–65K vocab for English-centric small models (SmolLM2's custom 49K is the sweet-spot example; Qwen's 151K and Gemma's 256K are multilingual taxes a 3B model pays in embedding params). **Train a custom BPE on your actual data mix.**
- **Tied embeddings:** used by Llama-3.2-3B, Qwen3-4B, SmolLM2, Ministral-3-3B. At 3B scale with a 50K vocab and d=3072, untying costs ~300M params (10% of the budget) for no measured gain — tie them.
- **Curriculum:** multi-stage data mix with math/code upsampled late + a high-quality annealing phase. **Model souping** (OLMo-2: merge 3–4 independent late-stage runs) is nearly free points.
- **Post-training ladder:** SFT (synthetic conversational, e.g. SmolTalk-style) → DPO/APO → **RLVR/GRPO on verifiable math/code rewards**. This ladder is where Phi-4-mini and Qwen3 get their MATH/AIME scores; skipping it leaves 10–20 points on reasoning benchmarks.
- **Overtraining works:** Chinchilla-optimal for 3B is ~60B tokens; the winners train 100–500× past that (11T, 18T, 36T). Data repetition up to ~4–5 epochs is fine (SmolLM2); Nemotron-CC exists precisely to push past that limit.

---

## 5. Recommendations: how a 3B OpenMythos beats them

**The strategic picture first:** every winner in this class is a *dense GQA transformer*. Nobody wins on architecture — they win on data × tokens × post-training. OpenMythos's recurrent depth is the only genuinely different compute lever in the field (RecurrentGemma tried recurrence and failed — MMLU 38.4 at 2B — which means the bar is "recurrence done right," not recurrence per se). The plan below plays to that while copying everything the winners agree on.

### 5.1 Architecture modifications (concrete)
1. **Add QK-Norm.** Qwen3 introduced RMSNorm on Q/K pre-softmax specifically for training stability, and a *looped* recurrent block is exactly where attention-logit blowups will happen across iterations. This is the single highest-EV architectural import.
2. **Use GQA, not MLA, for the 3B.** MLA's KV-compression shines at 70B+ inference; at 3B it adds training instability surface for no needed win. GQA 32/8 (Qwen3-4B's exact ratio) is the proven choice.
3. **Adopt Gemma-style 5:1 local:global attention interleave** (1024-token sliding windows, every 5th layer global). The recurrent block re-processes tokens up to 8× — full attention on every loop iteration is where your FLOPs go to die. Local windows inside the loop + global layers outside it preserves long-range ability at a fraction of the cost. Combine with **logit soft-capping** (Gemma 2: tanh cap 50/30) for loop stability.
4. **Fix the MoE the Qwen3 way.** Current OpenMythos: 8 experts, 2 active + 1 shared. Qwen3's ablated conclusion: **fine-grained experts (128 total, 8 active), NO shared expert, global-batch load-balancing loss** to force specialization. At 3B, try 64 experts / 8 active with the load-balancing loss — more total capacity, same active FLOPs.
5. **Tie embeddings; custom 64K BPE.** Saves ~10% of the param budget; train the tokenizer on the actual mix (70% web / 15% code / 10% math / 5% synthetic, mirroring SmolLM2's recipe).
6. **Keep the depth-extrapolation property as the killer feature:** train with n_loops=4, then at eval time sweep n_loops 4→8→16 on reasoning benchmarks. If accuracy rises with loops on MATH/GSM8K, you have a story no dense model can tell: *test-time compute scaling inside a 3B model*.

### 5.2 Training modifications (concrete)
7. **Loop-depth curriculum.** Don't train at full loop depth from step 0 — start n_loops=1–2 (cheap, stable), ramp to 4 by ~20% of training, optionally 8 in the anneal phase. This mirrors the staged data curriculum and avoids burning FLOPs on deep recurrence before the representations are worth iterating on.
8. **Data mix (SmolLM3-shaped, Nemotron-upgraded):** Stage 1: 80% web (Nemotron-CC-HQ + FineWeb-Edu + DCLM), 12% code (Stack-Edu), 5% math (FineMath4+), 3% synthetic QA. Stage 2: shift to 70/15/12 web/code/math (+MegaMath). Anneal: 55/20/20 + reasoning/instruction data (OpenMathReasoning-style) + a small % of long-context data. Total budget: as many tokens as compute allows — **2T minimum to be taken seriously, 4T+ to compete** (OLMo-2-1B did 4T).
9. **WSD schedule, 2M-token batches, AdamW(0.9,0.95), peak LR ~4e-4** — copy SmolLM2 exactly; it's the best fully-open recipe at this scale.
10. **Mid-training + souping (OLMo-2 playbook):** after the main run, do 2–3 short continuations on Dolmino-style high-quality mix with different seeds and **merge them**. Nearly free benchmark points.
11. **Post-training ladder is non-negotiable:** SFT on synthetic conversational data → DPO/APO → **RLVR/GRPO on verifiable math+code** (this is literally where Phi-4-mini's MATH-500 67.6 comes from). Budget 100B+ tokens of mid/post-training, not an afterthought.
12. **Ablate cheap, commit expensive:** use DCLM's competition scales (1.4B params / 29–144B tokens) for architecture ablations (loop curriculum, expert counts, SWA ratios), then do ONE full 3B run. SmolLM3 ablated on 3B/50–100B-token runs before committing to 11.2T.

### 5.3 The honest compute note
SmolLM3 cost **384 H100 × 24 days ≈ 220K H100-hours** (roughly $400K+ at cloud prices). A Kaggle-free-tier run cannot beat Qwen3-4B — the honest near-term target is a **1B-scale proof** (100B+ tokens on FineWeb-Edu + FineMath + Stack-Edu) that beats TinyLlama/DCLM-1B baselines and demonstrates the loop-depth extrapolation effect. That result is what earns the compute partnership for the real 3B run.

---

## 6. Sources (primary first)
- Meta Llama 3.2 model card: https://github.com/meta-llama/llama-models/blob/rc-0.1.4.dev3/models/llama3_2/MODEL_CARD.md
- Qwen2.5 tech report (arXiv:2412.15115): https://arxiv.org/abs/2412.15115
- Qwen3 tech report (arXiv:2505.09388): https://arxiv.org/pdf/2505.09388
- Phi-4-mini tech report (arXiv:2503.01743): https://arxiv.org/abs/2503.01743
- Gemma 2 tech report (arXiv:2408.00118): http://arxiv.org/pdf/2408.00118v2
- Gemma 3 tech report (arXiv:2503.19786): https://arxiv.org/abs/2503.19786
- SmolLM2 paper (arXiv:2502.02737): https://arxiv.org/pdf/2502.02737
- SmolLM3 blog: https://huggingface.co/blog/smollm3
- OLMo-2 HF cards: https://huggingface.co/allenai/OLMo-2-1124-13B · https://huggingface.co/allenai/OLMo-2-0425-1B
- TinyLlama repo + paper: https://github.com/jzhang38/TinyLlama · https://arxiv.org/pdf/2401.02385
- DCLM paper (arXiv:2406.11794): https://arxiv.org/pdf/2406.11794 · https://datacomp.ai/dclm
- kyegomez/OpenMythos: https://github.com/kyegomez/OpenMythos
- Eval tables: [Llama 3.2 3B instruct evals](https://huggingface.co/context-labs/meta-llama-Llama-3.2-3B-Instruct-FP16) · [Gemma 2 2B card](https://huggingface.co/google/gemma-2-2b) · [RecurrentGemma-2B](https://huggingface.co/google/recurrentgemma-2b) · [Ministral-3 vLLM recipe](https://github.com/vllm-project/recipes/blob/HEAD/Mistral/Ministral-3-Instruct.md)
