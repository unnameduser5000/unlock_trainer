#!/usr/bin/env bash
set -euo pipefail

TRAIN_MANIFEST="${TRAIN_MANIFEST:-data/sft_requests/tinyllama_agnews128_label_train512_seed20260531/requests.jsonl}"
EVAL_MANIFEST="${EVAL_MANIFEST:-data/sft_requests/tinyllama_agnews128_label_eval256_seed20260531/requests.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-debug_runs/server_window_compare/$(date +%Y%m%d-%H%M%S)}"

MODEL_NAME="${MODEL_NAME:-tinyllama}"
NUM_CHUNKS="${NUM_CHUNKS:-3}"
TRAIN_CHUNKS="${TRAIN_CHUNKS:-all}"
TRAIN_LIMIT="${TRAIN_LIMIT:-512}"
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
OPTIMIZER="${OPTIMIZER:-adamw}"
SGD_MOMENTUM="${SGD_MOMENTUM:-0.0}"
SGD_DAMPENING="${SGD_DAMPENING:-0.0}"
SGD_WEIGHT_DECAY="${SGD_WEIGHT_DECAY:-0.0}"
SGD_NESTEROV="${SGD_NESTEROV:-false}"
SEED="${SEED:-20260531}"
LRS="${LRS:-1e-4}"
PIPELINE_WINDOWS="${PIPELINE_WINDOWS:-1 3}"

mkdir -p "${OUTPUT_ROOT}"

for window in ${PIPELINE_WINDOWS}; do
  if [[ "${window}" == "1" ]]; then
    schedule="fifo"
    case_name="fifo_window1"
  else
    schedule="stage_window"
    case_name="stage_window${window}"
  fi
  case_root="${OUTPUT_ROOT}/${case_name}"

  echo "========================================================================"
  echo "Window case: ${case_name}"
  echo "  schedule=${schedule} pipeline_window=${window}"
  echo "  output=${case_root}"

  TRAIN_MANIFEST="${TRAIN_MANIFEST}" \
  EVAL_MANIFEST="${EVAL_MANIFEST}" \
  OUTPUT_ROOT="${case_root}" \
  MODEL_NAME="${MODEL_NAME}" \
  NUM_CHUNKS="${NUM_CHUNKS}" \
  TRAIN_CHUNKS="${TRAIN_CHUNKS}" \
  TRAIN_LIMIT="${TRAIN_LIMIT}" \
  TRAIN_EPOCHS="${TRAIN_EPOCHS}" \
  EVAL_LIMIT="${EVAL_LIMIT}" \
  DEVICE="${DEVICE}" \
  DTYPE="${DTYPE}" \
  GRAD_CLIP="${GRAD_CLIP}" \
  LORA_RANK="${LORA_RANK}" \
  LORA_ALPHA="${LORA_ALPHA}" \
  LORA_TARGETS="${LORA_TARGETS}" \
  ALPHA="${ALPHA}" \
  LABEL_SMOOTHING="${LABEL_SMOOTHING}" \
  TRAIN_SCHEDULE="${schedule}" \
  PIPELINE_WINDOW="${window}" \
  OPTIMIZER="${OPTIMIZER}" \
  SGD_MOMENTUM="${SGD_MOMENTUM}" \
  SGD_DAMPENING="${SGD_DAMPENING}" \
  SGD_WEIGHT_DECAY="${SGD_WEIGHT_DECAY}" \
  SGD_NESTEROV="${SGD_NESTEROV}" \
  SEED="${SEED}" \
  LRS="${LRS}" \
  bash tools/sim/run_bpfree_lora_label_sweep.sh
done

echo "========================================================================"
python tools/report/summarize_label_controls.py \
  "${OUTPUT_ROOT}" \
  --output_csv "${OUTPUT_ROOT}/summary_table.csv"
echo "========================================================================"
echo "Window compare finished: ${OUTPUT_ROOT}"
echo "Summary table: ${OUTPUT_ROOT}/summary_table.csv"
