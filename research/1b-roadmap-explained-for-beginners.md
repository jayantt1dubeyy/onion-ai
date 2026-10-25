# OpenMythos 1B Roadmap — The Big Target, Explained for Beginners

This is a beginner explainer for this project, in plain language with one simple analogy. It is not a replacement for the technical paper or report.

**How to read the honesty labels:**

- **Verified** = the number or result is in a file Jayant actually produced, and it was checked here.
- **Reported, not verified** = the number comes from Jayant's own summary or slides. No seeds, no confidence intervals, and no test code were given with it. Treat it as "he says so", not "proven".
- **Plan only** = an idea or a target. It has not happened yet.

---


**In one line:** A plan to build a 1-billion-parameter recurrent model that beats the best open models of similar size. It is a plan, not a finished model.

**Analogy:** It is like a training plan for a marathon. The plan exists. The marathon has not been run yet.

**What Jayant actually did (verified as a plan):**

- He wrote a roadmap and a model configuration. The configuration is listed as 1.06 billion parameters, verified as a count, not as a trained model.
- The design includes grouped-query attention, 64 experts with 8 active at a time, a 64,000-token vocabulary, and a loop-depth curriculum. That last part means: train across different loop depths, instead of only one depth.
- The target list includes TinyLlama-1.1B, SmolLM2-1.7B, OLMo-2-1B, Llama-3.2-1B, and Qwen2.5-1.5B.

**The real result:**

- There is no 1B result yet. No trained 1B model, no 1B test scores, and no 1B training run are claimed here.
- The 44M run is Stage 0. It is a pilot, not the 1B model.

**What is NOT proved / still open:**

- A free Kaggle T4 cannot deliver a 1B training run of this kind. That needs a cluster or a grant, which is a resource question, not just a code question.
- The "killer feature" in the roadmap, test-time depth extrapolation, is still unproven after the 44M result. The 1B loop-curriculum run is the experiment that would actually test it.
- No training cost, training time, or expected score is given here, because no 1B run exists to measure.
