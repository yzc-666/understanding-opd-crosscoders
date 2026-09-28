#!/usr/bin/env bash
# Cache student / OPD student / teacher activations on one token stream.
#
#   bash scripts/collect_activations.sh \
#     --token-dir TOKENS --cache-dir CACHE \
#     --student /path/to/student --student-layer 14 \
#     --opd     /path/to/opd     --opd-layer 14 \
#     --teacher /path/to/teacher --teacher-layer 18 \
#     [--roles student,opd,teacher] [--gpus 8] [--batch 64]
#
# Caches are written to CACHE/{base,opd,teacher}, the layout that
# train_opd_crosscoders.py --cache-root expects.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON=${PYTHON:-python}
GPUS=8
BATCH=64
ROLE_LIST=student,opd,teacher

while [[ $# -gt 0 ]]; do
  case $1 in
    --token-dir)     TOKEN_DIR=$2; shift 2 ;;
    --cache-dir)     CACHE_DIR=$2; shift 2 ;;
    --student)       STUDENT_MODEL=$2; shift 2 ;;
    --student-layer) STUDENT_LAYER=$2; shift 2 ;;
    --opd)           OPD_MODEL=$2; shift 2 ;;
    --opd-layer)     OPD_LAYER=$2; shift 2 ;;
    --teacher)       TEACHER_MODEL=$2; shift 2 ;;
    --teacher-layer) TEACHER_LAYER=$2; shift 2 ;;
    --roles)         ROLE_LIST=$2; shift 2 ;;
    --gpus)          GPUS=$2; shift 2 ;;
    --batch)         BATCH=$2; shift 2 ;;
    -h|--help)       sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
: "${TOKEN_DIR:?--token-dir is required}" "${CACHE_DIR:?--cache-dir is required}"
IFS=',' read -r -a ROLES <<< "$ROLE_LIST"

stream_family=$("$PYTHON" -m dictionary_learning.scripts.tokenizer_family --token-dir "$TOKEN_DIR" --short)
for role in "${ROLES[@]}"; do
  model=$(eval "echo \${${role^^}_MODEL:?--$role is required}")
  family=$("$PYTHON" -m dictionary_learning.scripts.tokenizer_family --model "$model" --short)
  if [[ "$family" != "$stream_family" ]]; then
    echo "$role uses vocabulary $family, but $TOKEN_DIR was built with $stream_family" >&2
    exit 2
  fi
done

# The student fills the slot that the trainer calls "base".
slot_of () { case $1 in student) echo base ;; *) echo "$1" ;; esac; }

for role in "${ROLES[@]}"; do
  model=$(eval "echo \${${role^^}_MODEL}")
  layer=$(eval "echo \${${role^^}_LAYER:?--$role-layer is required}")
  slot=$(slot_of "$role")
  args=(--role "$slot" --model "$model" --layer "$layer" --token-dir "$TOKEN_DIR"
        --output-dir "$CACHE_DIR/$slot" --batch-size "$BATCH" --strict-tokenizer-check)
  echo "=== $role -> $CACHE_DIR/$slot (layer $layer, $GPUS GPUs)"
  pids=()
  for ((rank = 0; rank < GPUS; rank++)); do
    CUDA_VISIBLE_DEVICES=$rank "$PYTHON" -m dictionary_learning.scripts.collect_opd_crosscoder_activations \
      "${args[@]}" --device-map cuda:0 --rank "$rank" --world-size "$GPUS" &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  "$PYTHON" -m dictionary_learning.scripts.collect_opd_crosscoder_activations "${args[@]}" --finalize-only
done
