# PipeDream exploratory baseline

This directory is isolated from the official E4 runners. It implements a
straight, continuous 1F1B baseline with:

- one warmup at the start and one cooldown at the end of the full train stream;
- stage-local optimizer updates, without a global flush between logical windows;
- LoRA weight stashing through `torch.func.functional_call`;
- local gradient coalescing so `physical_batch * accumulation_steps` matches the
  synchronous Exact-BP effective optimizer batch;
- the same pinned-CPU/Gloo hidden and hidden-gradient transport used by the E4
  Exact-BP CPU-communication runner.

The optimization semantics are asynchronous and are not equivalent to
synchronous Exact-BP or PyTorch `Schedule1F1B`. Only trainable LoRA tensors are
versioned because all base-model parameters are frozen.

Files:

- `experiments.shared.baselines.pipedream_cpu`: continuous runner and weight-version ledger;
- `evaluate_pipeline_lora_state.py`: evaluator for E4 hidden-state manifests;
- `plot_schedule_semantics.py`: observed GPipe/1F1B/PipeDream action timeline.

The first pilot results are under
`results/e4_throughput/raw/e4_pipedream_quality_pilot_v1` and
`results/e4_throughput/raw/e4_pipedream_schedule_semantics_v2`. They are
exploratory because one of the three GPUs was shared with another process.
