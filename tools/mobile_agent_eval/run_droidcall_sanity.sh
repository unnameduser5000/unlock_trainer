#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

OUTPUT_DIR="${1:-debug_runs/droidcall_tinyllama_sanity_20260624}"
SEED="${SEED:-20260624}"

python tools/mobile_agent_eval/prepare_droidcall_openai_jsonl.py \
  --train data/droidcall/DroidCall_train.jsonl \
  --test data/droidcall/DroidCall_test.jsonl \
  --api_catalog data/droidcall/api.jsonl \
  --output data/droidcall/droidcall_openai_style.jsonl \
  --n_api 4 \
  --seed "$SEED"

mkdir -p "$OUTPUT_DIR"

python -m tools.mobile_agent_eval.run_mobile_actions_lora_sft \
  --data data/droidcall/droidcall_openai_style.jsonl \
  --output_dir "$OUTPUT_DIR" \
  --train_limit 512 \
  --eval_limit 100 \
  --gen_eval_limit 100 \
  --epochs 1 \
  --seed "$SEED" \
  --lora_init_seed "$SEED" \
  --log_interval 25
