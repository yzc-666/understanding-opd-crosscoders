# Crosscoder training

Code to train the sparse crosscoders of the paper: one heterogeneous BatchTopK crosscoder per setting,
trained jointly on the base student, the OPD student, and the teacher, whose hidden sizes may differ.

## Install

```bash
cd crosscoder
pip install -e .
```

## 1. Build the shared token stream

Every model of a crosscoder reads the same token ids, so all models must share one tokenizer family.
The stream mixes 200M tokens of OpenThoughts-114k (prompts through the chat template) and 200M tokens of
RedPajama-Data-1T-Sample, cut into sequences of 512 tokens.

```bash
python -m dictionary_learning.scripts.prepare_opd_crosscoder_tokens \
  --tokenizer /path/to/student \
  --output-dir DATA/tokens
```

## 2. Cache activations

Cache the residual stream at the output of a middle block for each model. The script checks that every
model uses the vocabulary of the token stream and splits the shards over GPUs.

```bash
bash scripts/collect_activations.sh \
  --token-dir DATA/tokens --cache-dir DATA/cache \
  --student /path/to/student --student-layer 14 \
  --opd     /path/to/opd_student --opd-layer 14 \
  --teacher /path/to/teacher --teacher-layer 14 \
  --gpus 8
```

The paper reads block 14 of 28 for the DeepSeek-R1-Distill models, JustRL, Skywork-OR1, and
Qwen3-1.7B-Base, and block 18 of 36 for Qwen3-4B-Base-GRPO (counting from 0). The cache is stored in
bfloat16 and takes about 2 bytes × 400M tokens × hidden size per model, e.g. 1.2 TB for a 1536-dimensional
model.

## 3. Train

```bash
python -m dictionary_learning.scripts.train_opd_crosscoders \
  --cache-root DATA/cache --output-root DATA/runs
```

The defaults train the three-model crosscoder (`--combinations base-opd-teacher`) with the paper's
hyperparameters; `base-teacher`, `opd-teacher`, and `base-opd` train two-model crosscoders. Add
`--resume-latest` to continue an interrupted run from its last checkpoint. Each run writes
`DATA/runs/<combination>-batchtopk-seed42/` with `model_final.pt`, full-state checkpoints, validation
logs, and `run_config.json`.

| Hyperparameter | Value |
|---|---|
| Dictionary size | 32,768 |
| Sparsity | BatchTopK, k = 50 on average per token |
| Inference threshold | estimated from step 1,000 |
| Batch size | 4,096 token positions |
| Optimizer | Adam, β = (0.9, 0.999) |
| Learning rate | 2e-4 / √(32,768 / 16,384) ≈ 1.41e-4, 1,000 warm-up steps, no decay |
| Training length | one pass over the training tokens |
| Auxiliary dead-feature loss | weight 1/32 |
| Held-out data | 4 shards spread evenly over the stream |
| Activation normalization | per model, centered and scaled to unit total variance |
| Precision, seed | float32, 42 |

## Load a trained crosscoder

```python
from dictionary_learning import BatchTopKHeterogeneousCrossCoder

crosscoder = BatchTopKHeterogeneousCrossCoder.from_pretrained(
    "DATA/runs/base-opd-teacher-batchtopk-seed42/model_final.pt", device="cuda"
)
# h_base, h_opd, h_teacher: raw activations of the three models on the same tokens, [tokens, hidden size]
codes = crosscoder.encode([h_base, h_opd, h_teacher])
```

## Acknowledgements

This code builds on [science-of-finetuning/crosscoder_learning](https://github.com/science-of-finetuning/crosscoder_learning),
which extends [saprmarks/dictionary_learning](https://github.com/saprmarks/dictionary_learning). It keeps
only what the paper's crosscoders need and is released under the MIT license of the original code
(see `LICENSE`).
