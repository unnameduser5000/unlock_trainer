#!/usr/bin/env bash
set -euo pipefail

TRAIN_MANIFEST="${TRAIN_MANIFEST:-data/sft_requests/tinyllama_rotten_tomatoes128_label_train64_prompt24_lr3e4_balanced/requests.jsonl}"
EVAL_MANIFEST="${EVAL_MANIFEST:-data/sft_requests/tinyllama_rotten_tomatoes128_label_val256_prompt24_balanced/requests.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-debug_runs/server_label_sweeps/$(date +%Y%m%d-%H%M%S)}"
MODEL_NAME="${MODEL_NAME:-tinyllama}"
NUM_CHUNKS="${NUM_CHUNKS:-3}"
TRAIN_CHUNKS="${TRAIN_CHUNKS:-all}"
TRAIN_LIMIT="${TRAIN_LIMIT:-64}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-1}"
EVAL_LIMIT="${EVAL_LIMIT:-256}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-float32}"
LORA_RANK="${LORA_RANK:-4}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_TARGETS="${LORA_TARGETS:-q_proj,v_proj}"
ALPHA="${ALPHA:-0.5}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.1}"
SEED="${SEED:-20260531}"
LRS="${LRS:-1e-5 3e-5 1e-4 3e-4}"

mkdir -p "${OUTPUT_ROOT}"

for lr in ${LRS}; do
  run_dir="${OUTPUT_ROOT}/lr_${lr}"
  echo "========================================================================"
  echo "Running lr=${lr} -> ${run_dir}"
  python tools/sim/run_bpfree_lora_label_experiment.py \
    --model_name "${MODEL_NAME}" \
    --train_manifest "${TRAIN_MANIFEST}" \
    --eval_manifest "${EVAL_MANIFEST}" \
    --output_dir "${run_dir}" \
    --num_chunks "${NUM_CHUNKS}" \
    --train_chunks "${TRAIN_CHUNKS}" \
    --train_limit "${TRAIN_LIMIT}" \
    --train_epochs "${TRAIN_EPOCHS}" \
    --eval_limit "${EVAL_LIMIT}" \
    --learning_rate "${lr}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --lora_rank "${LORA_RANK}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_targets "${LORA_TARGETS}" \
    --alpha "${ALPHA}" \
    --label_smoothing "${LABEL_SMOOTHING}" \
    --seed "${SEED}"
done

echo "========================================================================"
echo "Sweep finished: ${OUTPUT_ROOT}"
