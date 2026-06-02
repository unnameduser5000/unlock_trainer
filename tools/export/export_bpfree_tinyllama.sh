#!/usr/bin/env bash
set -euo pipefail

# Server-side wrapper for BP-free TinyLlama training shards.
# BP-free here means no cross-chunk backward-gradient traffic. Each exported
# shard is still a TrainingModule artifact with chunk-local loss/backward.
#
# Example:
#   CHUNK_IDX=0,1 OUTPUT_DIR=model bash tools/export/export_bpfree_tinyllama.sh

MODEL_NAME="${MODEL_NAME:-tinyllama}"
NUM_CHUNKS="${NUM_CHUNKS:-4}"
CHUNK_IDX="${CHUNK_IDX:-0,1}"
SEQ_LEN="${SEQ_LEN:-64}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TRANSPORT_DTYPE="${TRANSPORT_DTYPE:-float16}"
BELIEF_TRANSPORT_MODE="${BELIEF_TRANSPORT_MODE:-full}"
ARTIFACT_PREFIX="${ARTIFACT_PREFIX:-tinyllama}"
ARTIFACT_SUFFIX="${ARTIFACT_SUFFIX:-}"
OUTPUT_DIR="${OUTPUT_DIR:-model}"
ALPHA="${ALPHA:-0.5}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.1}"
ENABLE_XNNPACK="${ENABLE_XNNPACK:-0}"

EXTRA_ARGS=()
if [[ "${ENABLE_XNNPACK}" == "1" ]]; then
  EXTRA_ARGS+=(--enable_xnnpack)
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
  "${EXTRA_ARGS[@]}"
