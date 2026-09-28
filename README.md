# Understanding On-Policy Distillation: A Mechanistic Interpretability Perspective via Sparse Crosscoders

[**Project page**](https://yzc-666.github.io/understanding-opd-crosscoders/) · **Paper** (coming soon) · [**Crosscoder training code**](crosscoder)

![Overview](docs/static/images/overview.png)

**TL;DR.** This work reveals that on-policy distillation improves reasoning primarily by reshaping how
students use existing representations, offering a mechanistic account of what stronger teachers actually
teach.

## Key findings

- **The swap readout.** We read each student checkpoint on its own through a sparse crosscoder, by
  placing its activation in both student slots and keeping the teacher's activation fixed. This
  measures how training changes the student's use of every feature, even for checkpoints the
  crosscoder has never seen.
- **OPD creates no new features.** Across three OPD settings, no feature is gained or lost, the
  teacher's own features are not passed on, and over 98% of the student's frequently used features
  change their firing rate by less than 20%.
- **Large changes sit at decision tokens.** Features for words such as *Wait*, *Hmm*, and *So*
  are strongly over-represented among the features OPD changes most.
- **The SFT warm-up reweights too.** It adds no features. Imposing its feature change on a directly
  distilled student, without changing any weights, recovers most of the warm-up's benefit, whereas the
  same change on shuffled features does not.

## Code

- [`crosscoder/`](crosscoder): building the shared token stream, caching activations, and training the
  heterogeneous BatchTopK crosscoders with the paper's hyperparameters.
- Coming soon: the swap readout, the feature statistics and decision-token analysis, the decomposition of
  the warm-up's reweighting, the feature-level intervention, and scripts that reproduce every figure.

## Citation

A BibTeX entry will be added when the paper is public.
