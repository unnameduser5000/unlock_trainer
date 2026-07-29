# E2: GPU Memory Exchange

This directory contains the E2 protocol, launch matrix, result checks, and
report generation. Training and transport implementations are shared with the
other experiments.

- BP-free execution: `sg_exe_trainer.runtime.bpfree.cpu_runner`
- GPipe and Sync 1F1B execution: `sg_exe_trainer.runtime.exactbp.cpu_runner`
- PipeDream baseline: `experiments.shared.baselines.pipedream_cpu`
- Fixed protocol: `configs/gpu_memory_v1.json`

The protocol uses TinyLlama LoRA on three L40 GPUs, fixes the effective
batch at 32, and measures `(b,m)=(1,32),(2,16),(4,8),(8,4)`. Terminal output
log-probabilities are disabled. Each job runs in a fresh process; `b8_m4` has
three repetitions and the other geometries have one.

Run the complete matrix:

```bash
python experiments/e2_memory/run_gpu_memory.py \
  --data-root /path/to/artifact-data \
  --output-root results/e2_memory
```

Run a four-method, one-window check:

```bash
python experiments/e2_memory/run_gpu_memory.py \
  --data-root /path/to/artifact-data \
  --output-root results/e2_memory_quick \
  --geometries b8_m4 \
  --smoke
```

Build the tables and figures from a completed result root:

```bash
python experiments/e2_memory/build_report.py \
  --raw-root results/e2_memory \
  --output-dir results/e2_memory_report
```
