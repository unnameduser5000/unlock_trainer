#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
IDLE_CONFIRMATIONS="${IDLE_CONFIRMATIONS:-5}"
IDLE_POLL_SECONDS="${IDLE_POLL_SECONDS:-60}"
TARGET_GPUS=(0 1 2 3)

cd "$REPO_ROOT"

target_gpus_are_idle() {
  local active_uuids gpu_index gpu_uuid
  active_uuids="$(
    nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader \
      | sed '/^[[:space:]]*$/d' \
      | sort -u
  )"
  [[ -z "$active_uuids" ]] && return 0

  for gpu_index in "${TARGET_GPUS[@]}"; do
    gpu_uuid="$(
      nvidia-smi --id="$gpu_index" --query-gpu=uuid \
        --format=csv,noheader | tr -d '[:space:]'
    )"
    if grep -qxF "$gpu_uuid" <<<"$active_uuids"; then
      return 1
    fi
  done
  return 0
}

wait_for_stable_idle() {
  local confirmations=0
  while (( confirmations < IDLE_CONFIRMATIONS )); do
    if target_gpus_are_idle; then
      confirmations=$((confirmations + 1))
      echo "[$(date --iso-8601=seconds)] GPU 0-3 idle check ${confirmations}/${IDLE_CONFIRMATIONS}"
    else
      confirmations=0
      echo "[$(date --iso-8601=seconds)] GPU 0-3 busy; waiting"
    fi
    if (( confirmations < IDLE_CONFIRMATIONS )); then
      sleep "$IDLE_POLL_SECONDS"
    fi
  done
}

run_experiment() {
  local label="$1"
  local launcher="$2"
  local config="$3"
  wait_for_stable_idle
  echo "[$(date --iso-8601=seconds)] starting $label"
  "$PYTHON_BIN" "$launcher" --config "$config" --resume
  echo "[$(date --iso-8601=seconds)] completed $label"
}

run_experiment E4.1 \
  experiments/e4_throughput/run_e4_1_cpu_transport_scaling.py \
  experiments/e4_throughput/configs/e4_1_scaling.json

run_experiment E4.2a \
  experiments/e4_throughput/run_e4_2_cpu_transport.py \
  experiments/e4_throughput/configs/e4_2a_batch_geometry.json

run_experiment E4.2b \
  experiments/e4_throughput/run_e4_2_cpu_transport.py \
  experiments/e4_throughput/configs/e4_2b_low_batch.json

run_experiment E4.4 \
  experiments/e4_throughput/run_e4_4_overhead_decomposition.py \
  experiments/e4_throughput/configs/e4_4_overhead_decomposition.json

run_experiment E4.3 \
  experiments/e4_throughput/run_e4_3_mobile_network_sensitivity.py \
  experiments/e4_throughput/configs/e4_3_network_sensitivity.json

echo "[$(date --iso-8601=seconds)] E4 formal queue complete"
