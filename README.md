<div align="center">

# Understanding On-Policy Distillation

### A Mechanistic Interpretability Perspective via Sparse Crosscoders

[![Project Page](https://img.shields.io/badge/Project-Page-6F8F76?style=for-the-badge&logo=githubpages&logoColor=white)](https://yzc-666.github.io/understanding-opd-crosscoders/)
[![Paper](https://img.shields.io/badge/Paper-coming%20soon-B9B3AC?style=for-the-badge&logo=arxiv&logoColor=white)](#citation)
[![Code](https://img.shields.io/badge/Code-crosscoder%20training-D8704F?style=for-the-badge&logo=github&logoColor=white)](crosscoder)

![Python](https://img.shields.io/badge/Python-3.10%2B-3B3B3B?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-3B3B3B?logo=pytorch&logoColor=white)

</div>

![Overview](docs/static/images/overview.png)

> **TL;DR.** This work reveals that on-policy distillation improves reasoning primarily by reshaping how
> students use existing representations, offering a mechanistic account of what stronger teachers actually
> teach.

## Crosscoder Training Code

The code to train the crosscoders is in **[`crosscoder/`](crosscoder)**. It builds the shared token
stream, caches the activations of the student, the OPD student, and the teacher, and trains the
heterogeneous BatchTopK crosscoders with the paper's hyperparameters. See
[`crosscoder/README.md`](crosscoder/README.md) for details.

```bash
cd crosscoder && pip install -e .

# 1. shared token stream (200M OpenThoughts + 200M RedPajama tokens)
python -m dictionary_learning.scripts.prepare_opd_crosscoder_tokens \
  --tokenizer /path/to/student --output-dir DATA/tokens

# 2. activations of the three models at a middle layer
bash scripts/collect_activations.sh --token-dir DATA/tokens --cache-dir DATA/cache \
  --student /path/to/student --student-layer 14 \
  --opd /path/to/opd_student --opd-layer 14 \
  --teacher /path/to/teacher --teacher-layer 14

# 3. three-model crosscoder
python -m dictionary_learning.scripts.train_opd_crosscoders \
  --cache-root DATA/cache --output-root DATA/runs
```

## Key Findings

- **The swap readout.** We read each student checkpoint on its own through a sparse crosscoder, by
  placing its activation in both student slots and keeping the teacher's activation fixed. This
  measures how training changes the student's use of every feature, even for checkpoints the
  crosscoder has never seen.
- **OPD creates no new features.** Across three OPD settings, no feature is gained or lost, the
  teacher's own features are not passed on, and over 98% of the student's frequently used features
  change their firing rate by less than 20%.
- **Large changes sit at decision tokens.** Features for words such as *Wait*, *Hmm*, and *So*
  are strongly over-represented among the features OPD changes most.
- **The SFT warm-up reweights too.** It adds no features. It does part of OPD's work in advance and
  changes features in ways that OPD alone would not, and imposing its feature change on a directly
  distilled student, without changing any weights, recovers most of the warm-up's benefit.

## Repository Structure

```
understanding-opd-crosscoders/
├── crosscoder/                    crosscoder training
│   ├── dictionary_learning/       crosscoder model, activation cache, trainer
│   │   └── scripts/               token stream, activation caching, training
│   └── scripts/collect_activations.sh
└── docs/                          project page
```

## Citation

A BibTeX entry will be added when the paper is public.

## Acknowledgements

The crosscoder code builds on
[science-of-finetuning/crosscoder_learning](https://github.com/science-of-finetuning/crosscoder_learning)
and [saprmarks/dictionary_learning](https://github.com/saprmarks/dictionary_learning).
