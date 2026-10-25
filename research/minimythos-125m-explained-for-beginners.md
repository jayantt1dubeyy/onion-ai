# MiniMythos 125M — An Earlier, Bigger Model, Explained for Beginners

This is a beginner explainer for this project, in plain language with one simple analogy. It is not a replacement for the technical paper or report.

**How to read the honesty labels:**

- **Verified** = the number or result is in a file Jayant actually produced, and it was checked here.
- **Reported, not verified** = the number comes from Jayant's own summary or slides. No seeds, no confidence intervals, and no test code were given with it. Treat it as "he says so", not "proven".
- **Plan only** = an idea or a target. It has not happened yet.

---


**In one line:** An earlier recurrent model, about 125 million parameters, reported as improving on one reasoning test when it looped more, but staying flat on a factual test.

**Analogy:** Think of a student who gets better at logic puzzles when given more time to think, but still cannot recall facts they never learned. More thinking time helps reasoning. It does not create knowledge.

**What Jayant actually did — and the honesty label:**

- **Reported, not verified.** These numbers come from Jayant's own July 2026 one-page summary and slides. They are not independently verified runs.
- The summary gives no seeds, no confidence intervals, and no test code. So no claim about statistical significance can be made here.
- The model is described as having 7 physical layers, becoming 22 effective layers at 16 loops. That is the summary's arithmetic, not a measured quantity checked here.

**The reported result (reported only):**

- HellaSwag, a test about finishing a sensible story or situation: MiniMythos scored 29.5% at 8 loops and 30.0% at 16 loops.
- For comparison, GPT-2 Small scored 28.5% and TinyLlama-120M scored 31.5%.
- So MiniMythos beat GPT-2 Small, but it did not beat TinyLlama-120M.
- PIQA, a test about physical common sense: MiniMythos scored 50.0% at both 8 loops and 16 loops.
- For comparison, GPT-2 Small scored 62.8% and TinyLlama-120M scored 64.2%.
- PIQA has 2 choices per question. So 50.0% is exactly chance level. It is like guessing on a coin flip. "Stable at 50%" does not mean robust. It means the model had no PIQA knowledge to lose.

**What is NOT proved / still open:**

- Nothing here is independently verified. Do not quote these numbers as proven results.
- No MiniMythos ARC-Challenge number was given in the summary. That gap is left open, not filled in.
- The summary's framing is "knowledge ceiling": loops help reasoning, but not factual recall. That is a plausible story, not a proven explanation.

**How this fits with the 44M result:**

- They do not directly contradict each other. They are different scales and different measures.
- At 44M, on TinyStories loss, going past the trained depth hurt.
- At 125M, on HellaSwag accuracy, going from 8 to 16 loops reportedly helped a little.
- The honest picture right now is: extrapolation might depend on scale and task. It is not simply true or false. The test that would settle it is a loop-curriculum run that reports both loss and test accuracy at one scale.
