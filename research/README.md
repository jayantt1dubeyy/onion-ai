# Research

Competitive landscape for the 3B milestone.

- `3b_landscape.md` — survey of the best 2–4B models (in progress): architectures,
  training data, recipes, eval scores; what we borrow and what we change.
- `44m_experiment_report.md` — Stage-0 evidence: the 44M TinyStories run vs its
  matched-parameter (depth-1) baseline — loss and bootstrap CIs, training
  compute, samples, and limitations. Headline: recurrence helps hugely at
  matched params; loop extrapolation beyond trained depth does not (at 44M).
- `minimythos_125m_addendum.md` — second data point, reported (not
  independently verified): MiniMythos 125M zero-shot HellaSwag 29.5% → 30.0%
  from 8 to 16 loops, PIQA flat at 50.0% (chance). Extrapolation looks
  scale- and task-dependent, not simply true or false.
