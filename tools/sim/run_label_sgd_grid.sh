#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-debug_runs/server_sgd_grid/$(date +%Y%m%d-%H%M%S)}"

TRAIN_MANIFEST="${TRAIN_MANIFEST:-data/sft_requests/tinyllama_agnews128_label_train512_seed20260531/requests.jsonl}"
EVAL_MANIFEST="${EVAL_MANIFEST:-data/sft_requests/tinyllama_agnews128_label_eval256_seed20260531/requests.jsonl}"
MODEL_NAME="${MODEL_NAME:-tinyllama}"
NUM_CHUNKS="${NUM_CHUNKS:-3}"
TRAIN_CHUNKS="${TRAIN_CHUNKS:-all}"
TRAIN_LIMIT="${TRAIN_LIMIT:-512}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-1}"
EVAL_LIMIT="${EVAL_LIMIT:-256}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-float32}"
GRAD_CLIP="${GRAD_CLIP:-0.0}"
LORA_RANK="${LORA_RANK:-4}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_TARGETS="${LORA_TARGETS:-q_proj,v_proj}"
ALPHA="${ALPHA:-0.5}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.1}"
TRAIN_SCHEDULE="${TRAIN_SCHEDULE:-stage_window}"
PIPELINE_WINDOW="${PIPELINE_WINDOW:-3}"
SEED="${SEED:-20260531}"

LRS="${LRS:-1e-4 3e-4 1e-3 3e-3 1e-2}"
SGD_MOMENTUMS="${SGD_MOMENTUMS:-0.0 0.5 0.9}"
SGD_DAMPENINGS="${SGD_DAMPENINGS:-0.0}"
SGD_WEIGHT_DECAYS="${SGD_WEIGHT_DECAYS:-0.0}"
SGD_NESTEROVS="${SGD_NESTEROVS:-false}"

mkdir -p "${OUTPUT_ROOT}"

for momentum in ${SGD_MOMENTUMS}; do
  for dampening in ${SGD_DAMPENINGS}; do
    for weight_decay in ${SGD_WEIGHT_DECAYS}; do
      for nesterov in ${SGD_NESTEROVS}; do
        if [[ "${nesterov}" == "true" ]]; then
          if [[ "${momentum}" == "0" || "${momentum}" == "0.0" ]]; then
            echo "Skipping invalid Nesterov case with momentum=${momentum}"
            continue
          fi
        fi
        if [[ "${nesterov}" == "true" && "${dampening}" != "0" && "${dampening}" != "0.0" ]]; then
          echo "Skipping invalid Nesterov case with dampening=${dampening}"
          continue
        fi
        case_name="sgd_m${momentum}_d${dampening}_wd${weight_decay}_n${nesterov}"
        case_root="${OUTPUT_ROOT}/${case_name}"

        echo "========================================================================"
        echo "SGD case: ${case_name}"
        echo "  lrs=${LRS} grad_clip=${GRAD_CLIP} schedule=${TRAIN_SCHEDULE} window=${PIPELINE_WINDOW}"
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
        TRAIN_SCHEDULE="${TRAIN_SCHEDULE}" \
        PIPELINE_WINDOW="${PIPELINE_WINDOW}" \
        OPTIMIZER="sgd" \
        SGD_MOMENTUM="${momentum}" \
        SGD_DAMPENING="${dampening}" \
        SGD_WEIGHT_DECAY="${weight_decay}" \
        SGD_NESTEROV="${nesterov}" \
        SEED="${SEED}" \
        LRS="${LRS}" \
        bash tools/sim/run_bpfree_lora_label_sweep.sh
      done
    done
  done
done

echo "========================================================================"
python tools/report/summarize_label_controls.py \
  "${OUTPUT_ROOT}" \
  --output_csv "${OUTPUT_ROOT}/summary_table.csv"
echo "========================================================================"
echo "SGD grid finished: ${OUTPUT_ROOT}"
echo "Summary table: ${OUTPUT_ROOT}/summary_table.csv"
