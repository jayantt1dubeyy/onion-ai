# OpenMythos 44M — A Small Recurrent Language Model, Explained for Beginners

This is a beginner explainer for this project, in plain language with one simple analogy. It is not a replacement for the technical paper or report.

**How to read the honesty labels:**

- **Verified** = the number or result is in a file Jayant actually produced, and it was checked here.
- **Reported, not verified** = the number comes from Jayant's own summary or slides. No seeds, no confidence intervals, and no test code were given with it. Treat it as "he says so", not "proven".
- **Plan only** = an idea or a target. It has not happened yet.

---


**In one line:** Jayant trained a small language model that reuses one block of brain-like machinery again and again, instead of stacking many different blocks.

**Analogy:** Imagine reading a page, then reading it again to understand it better, using the same eyes. The model does something like that. It loops over the same block.

**What Jayant actually did (verified):**

- He trained a model with 43,974,722 parameters. People round that to 44M.
- He trained it on TinyStories. Those are simple children's stories.
- He used one Kaggle Tesla T4 computer. Training took about 2.5 GPU-hours and about 120 million tokens.
- He trained it to loop 4 times. Then he tested the same model at loop depths 1, 2, 4, 8, 12, and 16.
- Important detail: this run used an existing OpenMythos package, not Jayant's from-scratch code. The report says that clearly.

**The real result (verified):**

- Loop depth 1, which means no looping, had a loss of 3.380. Lower loss is better.
- Loop depth 4, the depth it was trained at, had a loss of 1.484. That is much better.
- Loop depth 8 had a loss of 1.590. That is worse than depth 4.
- The difference from depth 4 to depth 8 was +0.106, with a 95% confidence interval of [+0.101, +0.110]. In plain words: going past the trained depth made it worse, and the gap is too big to be just noise.
- Depths 12 and 16 stayed at 1.591. More loops stopped helping.

**What is NOT proved / still open:**

- This was one training run, with one seed. A second seed is still missing. Until then, treat the second decimal place as suggestive, not settled.
- It was tested on TinyStories only. That does not prove anything about general language, science, or code.
- It measured loss and perplexity only. It did not measure downstream test scores or human judgment.
- The big idea, "test-time depth extrapolation", failed at this scale, on this task. More loops at test time did not help. That does not kill the idea forever. It means the idea needs a different training recipe, or a bigger scale, to be tested again.
