# Server Multi-GPU Stage Training

Use this when phone training is too slow and each phone-like stage should map to
a different server GPU.

Main speed-oriented script:

- `tools/sim/run_bpfree_lora_pipeline_dist.py`

Fallback/debug script:

- `tools/sim/run_bpfree_lora_pipeline_multigpu.py`

Both are real PyTorch training. They are not the lightweight queue simulator.
Use the distributed script first on the server; the fallback script uses CPU
`multiprocessing.Queue` handoff and is mainly for debugging.

## What It Runs

The script starts one Python process per stage:

```text
stage0 process on cuda:0
stage1 process on cuda:1
stage2 process on cuda:2
```

Each stage process:

- loads the model;
- injects LoRA adapters;
- keeps only its chunk, final norm, and lm head on its GPU;
- owns its local optimizer;
- executes local CE/KD loss;
- runs local backward and optimizer step for `mode=train`;
- forwards detached hidden state to the next stage.

The cross-stage graph is still forward-only:

```text
stage0(request i) -> stage1(request i) -> stage2(request i)
```

There is no cross-stage backward pass.

In the distributed runner, boundary hidden states move through
`torch.distributed.send/recv` between adjacent ranks. With `--dist_backend nccl`
and `--stage_devices cuda:0,cuda:1,cuda:2`, this is GPU tensor communication
handled by NCCL instead of the CPU queue fallback.

## Example

```bash
python tools/sim/run_bpfree_lora_pipeline_dist.py \
  --model_name tinyllama \
  --train_manifest data/sft_requests/tinyllama_agnews128_label_train512_seed20260531/requests.jsonl \
  --eval_manifest data/sft_requests/tinyllama_agnews128_label_eval256_seed20260531/requests.jsonl \
  --output_dir debug_runs/server_multigpu/agnews_3gpu_terminal \
  --num_chunks 3 \
  --stage_devices cuda:0,cuda:1,cuda:2 \
  --train_limit 512 \
  --eval_limit 256 \
  --train_epochs 1 \
  --learning_rate 1e-4 \
  --optimizer adamw \
  --belief_transport_mode terminal \
  --alpha 1.0 \
  --label_smoothing 0.0 \
  --dtype bfloat16
```

For a quick smoke:

```bash
python tools/sim/run_bpfree_lora_pipeline_dist.py \
  --model_name tinyllama \
  --train_manifest data/sft_requests/tinyllama_agnews128_label_train512_seed20260531/requests.jsonl \
  --eval_manifest data/sft_requests/tinyllama_agnews128_label_eval256_seed20260531/requests.jsonl \
  --output_dir debug_runs/server_multigpu/smoke \
  --num_chunks 3 \
  --stage_devices cuda:0,cuda:1,cuda:2 \
  --train_limit 4 \
  --eval_limit 4 \
  --skip_eval_after
```

## Important Parameters

- `--stage_devices`: one device per stage. This is the explicit server mapping
  that replaces phone device assignment.
- `--dist_backend`: defaults to `nccl`, which is the intended same-machine
  multi-GPU backend.
- `--master_port`: change this if another distributed job is already using the
  default rendezvous port.
- The distributed runner uses blocking point-to-point send/recv for natural
  backpressure. The older `--max_buffered_per_stage` queue parameter only
  applies to the CPU Queue fallback runner.
- `--belief_transport_mode terminal`: current CE-only AG News path. Intermediate
  stages do not send full vocab log-probs; the terminal stage returns log-probs
  for metrics.
- `--belief_transport_mode full`: sends full log-probs across stages and enables
  KL when `--alpha < 1`.
- `--train_chunks`: defaults to `all`; can be set to `2` for terminal-only local
  training controls.
- `--dtype`: use `bfloat16` or `float16` on GPUs that support it.

## Outputs

The output directory contains:

- `eval_before.csv`
- `train.csv`
- `eval_after.csv`
- `*.stage_metrics.csv`
- `summary.json`

The stage metrics CSV records real GPU execution timings:

- `stage_id`
- `device`
- `local_loss`
- `queue_wait_ms`
- `h2d_ms`
- `execute_ms`
- `optimizer_ms`
- `cpu_transfer_ms`
- `cuda_peak_memory_allocated`
- `cuda_peak_memory_reserved`

These files are the right input for later placement/planner work. The first
goal is to make the server produce real multi-GPU training traces; only after
that should we optimize GPU placement.
