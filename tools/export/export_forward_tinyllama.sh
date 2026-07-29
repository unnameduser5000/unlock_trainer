#!/usr/bin/env bash
set -euo pipefail

# Server-side convenience wrapper for the current two-phone mobile pipeline.
# Override any value as an environment variable, for example:
#   CHUNK_IDX=-1 OUTPUT_DIR=exported_pte bash tools/export/export_forward_tinyllama.sh

MODEL_NAME="${MODEL_NAME:-tinyllama}"
NUM_CHUNKS="${NUM_CHUNKS:-4}"
CHUNK_IDX="${CHUNK_IDX:-0,1}"
SEQ_LEN="${SEQ_LEN:-64}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TRANSPORT_DTYPE="${TRANSPORT_DTYPE:-float16}"
ARTIFACT_PREFIX="${ARTIFACT_PREFIX:-tinyllama}"
ARTIFACT_SUFFIX="${ARTIFACT_SUFFIX:-_inf}"
OUTPUT_DIR="${OUTPUT_DIR:-model}"

python tools/export/sid_export_forward_mobile.py \
  --model_name "${MODEL_NAME}" \
  --num_chunks "${NUM_CHUNKS}" \
  --chunk_idx "${CHUNK_IDX}" \
  --seq_len "${SEQ_LEN}" \
  --batch_size "${BATCH_SIZE}" \
  --transport_dtype "${TRANSPORT_DTYPE}" \
  --relay_only \
  --artifact_prefix "${ARTIFACT_PREFIX}" \
  --artifact_suffix "${ARTIFACT_SUFFIX}" \
  --output_dir "${OUTPUT_DIR}"
