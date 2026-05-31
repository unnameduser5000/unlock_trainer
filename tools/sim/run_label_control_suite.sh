#!/usr/bin/env bash
set -euo pipefail

TRAIN_MANIFEST="${TRAIN_MANIFEST:-data/sft_requests/tinyllama_agnews128_label_train512_seed20260531/requests.jsonl}"
EVAL_MANIFEST="${EVAL_MANIFEST:-data/sft_requests/tinyllama_agnews128_label_eval256_seed20260531/requests.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-debug_runs/server_label_controls/$(date +%Y%m%d-%H%M%S)}"

MODEL_NAME="${MODEL_NAME:-tinyllama}"
TRAIN_LIMIT="${TRAIN_LIMIT:-512}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-1}"
EVAL_LIMIT="${EVAL_LIMIT:-256}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-float32}"
LRS="${LRS:-3e-5 1e-4 3e-4}"
LORA_RANK="${LORA_RANK:-4}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_TARGETS="${LORA_TARGETS:-q_proj,v_proj}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.1}"
SEED="${SEED:-20260531}"
RUN_CURRENT="${RUN_CURRENT:-0}"

run_case() {
  local name="$1"
  local num_chunks="$2"
  local train_chunks="$3"
  local alpha="$4"
  local case_root="${OUTPUT_ROOT}/${name}"

  echo "========================================================================"
  echo "Control: ${name}"
  echo "  num_chunks=${num_chunks} train_chunks=${train_chunks} alpha=${alpha}"
  echo "  output=${case_root}"

  TRAIN_MANIFEST="${TRAIN_MANIFEST}" \
  EVAL_MANIFEST="${EVAL_MANIFEST}" \
  OUTPUT_ROOT="${case_root}" \
  MODEL_NAME="${MODEL_NAME}" \
  NUM_CHUNKS="${num_chunks}" \
  TRAIN_CHUNKS="${train_chunks}" \
  TRAIN_LIMIT="${TRAIN_LIMIT}" \
  TRAIN_EPOCHS="${TRAIN_EPOCHS}" \
  EVAL_LIMIT="${EVAL_LIMIT}" \
  DEVICE="${DEVICE}" \
  DTYPE="${DTYPE}" \
  LRS="${LRS}" \
  LORA_RANK="${LORA_RANK}" \
  LORA_ALPHA="${LORA_ALPHA}" \
  LORA_TARGETS="${LORA_TARGETS}" \
  ALPHA="${alpha}" \
  LABEL_SMOOTHING="${LABEL_SMOOTHING}" \
  SEED="${SEED}" \
  bash tools/sim/run_bpfree_lora_label_sweep.sh
}

mkdir -p "${OUTPUT_ROOT}"

run_case "full_lora_upper_bound" "1" "all" "1.0"
run_case "terminal_chunk_ce" "3" "2" "1.0"
run_case "bpfree_ce_only" "3" "all" "1.0"

if [[ "${RUN_CURRENT}" == "1" ]]; then
  run_case "bpfree_alpha05_current" "3" "all" "0.5"
fi

echo "========================================================================"
echo "Control suite finished: ${OUTPUT_ROOT}"
