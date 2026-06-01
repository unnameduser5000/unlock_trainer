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
GRAD_CLIP="${GRAD_CLIP:-1.0}"
LORA_RANK="${LORA_RANK:-4}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_TARGETS="${LORA_TARGETS:-q_proj,v_proj}"
ALPHA="${ALPHA:-0.5}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.1}"
TRAIN_SCHEDULE="${TRAIN_SCHEDULE:-fifo}"
PIPELINE_WINDOW="${PIPELINE_WINDOW:-1}"
OPTIMIZER="${OPTIMIZER:-adamw}"
SGD_MOMENTUM="${SGD_MOMENTUM:-0.0}"
SGD_DAMPENING="${SGD_DAMPENING:-0.0}"
SGD_WEIGHT_DECAY="${SGD_WEIGHT_DECAY:-0.0}"
SGD_NESTEROV="${SGD_NESTEROV:-false}"
SEED="${SEED:-20260531}"
LRS="${LRS:-1e-5 3e-5 1e-4 3e-4}"

mkdir -p "${OUTPUT_ROOT}"

for lr in ${LRS}; do
  run_dir="${OUTPUT_ROOT}/lr_${lr}"
  echo "========================================================================"
  echo "Running lr=${lr} -> ${run_dir}"
  extra_args=()
  if [[ "${SGD_NESTEROV}" == "true" || "${SGD_NESTEROV}" == "1" ]]; then
    extra_args+=(--sgd_nesterov)
  fi

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
    --grad_clip "${GRAD_CLIP}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --lora_rank "${LORA_RANK}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_targets "${LORA_TARGETS}" \
    --train_schedule "${TRAIN_SCHEDULE}" \
    --pipeline_window "${PIPELINE_WINDOW}" \
    --optimizer "${OPTIMIZER}" \
    --sgd_momentum "${SGD_MOMENTUM}" \
    --sgd_dampening "${SGD_DAMPENING}" \
    --sgd_weight_decay "${SGD_WEIGHT_DECAY}" \
    --alpha "${ALPHA}" \
    --label_smoothing "${LABEL_SMOOTHING}" \
    --seed "${SEED}" \
    "${extra_args[@]}"
done

echo "========================================================================"
echo "Sweep finished: ${OUTPUT_ROOT}"
