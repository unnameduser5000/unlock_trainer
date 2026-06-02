#!/usr/bin/env bash
set -euo pipefail

# Server-side wrapper for TinyLlama BP-free LoRA training shards.
# The base chunk weights are frozen and only LoRA adapter tensors are trainable
# inside each Android TrainingModule step. Cross-stage traffic remains forward-only.

MODEL_NAME="${MODEL_NAME:-tinyllama}"
NUM_CHUNKS="${NUM_CHUNKS:-4}"
CHUNK_IDX="${CHUNK_IDX:-0,1}"
SEQ_LEN="${SEQ_LEN:-64}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TRANSPORT_DTYPE="${TRANSPORT_DTYPE:-float16}"
BELIEF_TRANSPORT_MODE="${BELIEF_TRANSPORT_MODE:-full}"
ARTIFACT_PREFIX="${ARTIFACT_PREFIX:-tinyllama_lora}"
ARTIFACT_SUFFIX="${ARTIFACT_SUFFIX:-}"
OUTPUT_DIR="${OUTPUT_DIR:-model}"
ALPHA="${ALPHA:-0.5}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.1}"
LORA_RANK="${LORA_RANK:-4}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_TARGETS="${LORA_TARGETS:-q_proj,v_proj}"
LORA_INIT_STD="${LORA_INIT_STD:-0.01}"
ENABLE_XNNPACK="${ENABLE_XNNPACK:-0}"
DUMP_JOINT_GRAPH="${DUMP_JOINT_GRAPH:-0}"

EXTRA_ARGS=()
if [[ "${ENABLE_XNNPACK}" == "1" ]]; then
  EXTRA_ARGS+=(--enable_xnnpack)
fi
if [[ "${DUMP_JOINT_GRAPH}" == "1" ]]; then
  EXTRA_ARGS+=(--dump_joint_graph)
fi

python tools/export/sid_export_mobile.py \
  --model_name "${MODEL_NAME}" \
  --num_chunks "${NUM_CHUNKS}" \
  --chunk_idx "${CHUNK_IDX}" \
  --seq_len "${SEQ_LEN}" \
  --batch_size "${BATCH_SIZE}" \
  --transport_dtype "${TRANSPORT_DTYPE}" \
  --belief_transport_mode "${BELIEF_TRANSPORT_MODE}" \
  --alpha "${ALPHA}" \
  --label_smoothing "${LABEL_SMOOTHING}" \
  --artifact_prefix "${ARTIFACT_PREFIX}" \
  --artifact_suffix "${ARTIFACT_SUFFIX}" \
  --output_dir "${OUTPUT_DIR}" \
  --lora_rank "${LORA_RANK}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_targets "${LORA_TARGETS}" \
  --lora_init_std "${LORA_INIT_STD}" \
  "${EXTRA_ARGS[@]}"
