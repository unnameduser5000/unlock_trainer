# Server Scheduler Lab

This is the server-side harness for aggressive scheduling and recovery
experiments. It is not the phone FIFO runner and it is not the fixed
`rank0 -> rank1 -> rank2` distributed runner.

Main script:

- `tools/sim/run_bpfree_scheduler_lab.py`

## Purpose

The phone path has already shown that the BP-free forward-only stage pipeline can
train. The scheduler lab is for questions that are too slow or risky to answer
on phones:

- which task should run next when recovery work competes with fresh work;
- whether a failed request should retry only the failed stage or restart from
  stage 0;
- what happens when a stage fails after applying a local optimizer step but
  before returning its boundary tensor;
- how replicas for one stage affect recovery and throughput;
- how often duplicate local updates happen under at-least-once recovery.

## Architecture

```text
central scheduler
  dispatches StageTask(request_id, stage_id, attempt, input_state)
      to stage workers

stage workers
  each owns one stage chunk on one GPU
  run real local forward/loss/backward/optimizer
  return boundary tensor or a failure event
```

Boundary tensors are held by the scheduler as CPU tensors. That is deliberate in
this lab: recovery policies need cached stage boundaries that can be retried or
rerouted to another worker. Once a policy is chosen, the transport layer can be
optimized separately.

## Basic Run

One worker per stage:

```bash
python tools/sim/run_bpfree_scheduler_lab.py \
  --model_name tinyllama \
  --manifest data/sft_requests/tinyllama_agnews128_label_train512_seed20260531/requests.jsonl \
  --output_dir debug_runs/scheduler_lab/fifo_no_failure \
  --num_chunks 3 \
  --stage_devices cuda:0,cuda:1,cuda:2 \
  --limit 64 \
  --max_inflight 6 \
  --scheduler_policy fifo \
  --recovery_policy retry_stage \
  --belief_transport_mode terminal \
  --alpha 1.0 \
  --label_smoothing 0.0 \
  --learning_rate 1e-4 \
  --dtype bfloat16
```

With a stage-2 replica:

```bash
python tools/sim/run_bpfree_scheduler_lab.py \
  --model_name tinyllama \
  --manifest data/sft_requests/tinyllama_agnews128_label_train512_seed20260531/requests.jsonl \
  --output_dir debug_runs/scheduler_lab/stage2_replica \
  --num_chunks 3 \
  --workers 0:cuda:0,1:cuda:1,2:cuda:2,2:cuda:3 \
  --limit 64 \
  --max_inflight 8 \
  --scheduler_policy recovery_first \
  --recovery_policy retry_stage \
  --belief_transport_mode terminal \
  --alpha 1.0 \
  --label_smoothing 0.0 \
  --learning_rate 1e-4 \
  --dtype bfloat16
```

Replica warning: replicas for the same stage currently own independent local
LoRA state. There is no parameter averaging or central optimizer in this first
lab. This is intentional for exposing the consistency problem; do not interpret
replica speedups as quality-equivalent until a synchronization policy is added.

## Failure Injection

Fail one specific stage task before execution:

```bash
python tools/sim/run_bpfree_scheduler_lab.py \
  --model_name tinyllama \
  --manifest data/sft_requests/tinyllama_agnews128_label_train512_seed20260531/requests.jsonl \
  --output_dir debug_runs/scheduler_lab/fail_before_retry_stage \
  --num_chunks 3 \
  --stage_devices cuda:0,cuda:1,cuda:2 \
  --limit 16 \
  --failure_mode once \
  --failure_stage 1 \
  --failure_seq 5 \
  --failure_point before_execute \
  --recovery_policy retry_stage
```

Fail after a local update:

```bash
python tools/sim/run_bpfree_scheduler_lab.py \
  --model_name tinyllama \
  --manifest data/sft_requests/tinyllama_agnews128_label_train512_seed20260531/requests.jsonl \
  --output_dir debug_runs/scheduler_lab/fail_after_update \
  --num_chunks 3 \
  --stage_devices cuda:0,cuda:1,cuda:2 \
  --limit 16 \
  --failure_mode once \
  --failure_stage 1 \
  --failure_seq 5 \
  --failure_point after_update \
  --recovery_policy retry_stage
```

The second case is intentionally dangerous: the failed stage may have already
changed local LoRA weights but lost its output boundary. The ledger exposes this
as `update_applied=true` on a failure event, so the policy can be judged rather
than hidden.

## Key Parameters

- `--scheduler_policy fifo`: recovered tasks go to the back of their stage queue.
- `--scheduler_policy recovery_first`: recovered tasks go to the front.
- `--recovery_policy retry_stage`: retry the failed stage from the same input
  boundary.
- `--recovery_policy retry_from_zero`: discard cached boundaries for that request
  and restart from stage 0.
- `--recovery_policy skip`: mark the request failed.
- `--max_attempts`: cap retries per stage.
- `--workers`: explicit worker pool with replicas, `STAGE:DEVICE`.
- `--max_inflight`: admission window controlled by the central scheduler.

## Outputs

Each run writes:

- `scheduler_results.csv`: request-level completion/failure rows.
- `scheduler_stage_metrics.csv`: real stage timing, memory, update, and failure
  metrics.
- `scheduler_ledger.csv`: every admit, dispatch, success, failure, and retry
  decision.
- `scheduler_summary.json`: aggregate accuracy/loss/throughput and policy
  metadata.

The ledger is the most important artifact for recovery experiments.
