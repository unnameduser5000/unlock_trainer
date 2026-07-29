# E4 throughput experiments

E4 owns benchmark geometry, subprocess orchestration, and report generation.
The training implementations live outside this directory.

## Experiment matrix

| Point | Question | Launcher | Configuration |
|---|---|---|---|
| E4.1 | 2/3/4-stage scaling | `run_e4_1_cpu_transport_scaling.py` | `configs/e4_1_scaling.json` |
| E4.2a | Fixed-B=32 batch geometry | `run_e4_2_cpu_transport.py` | `configs/e4_2a_batch_geometry.json` |
| E4.2b | Low-batch microbatch sweep | `run_e4_2_cpu_transport.py` | `configs/e4_2b_low_batch.json` |
| E4.3 | Synthetic mobile-link sensitivity | `run_e4_3_mobile_network_sensitivity.py` | `configs/e4_3_network_sensitivity.json` |
| E4.4 | Synchronized overhead decomposition | `run_e4_4_overhead_decomposition.py` | `configs/e4_4_overhead_decomposition.json` |
| E4.4-S | Late-window steady-state trace | `run_e4_4_steady_trace.py` | `configs/e4_4_steady_trace.json` |

## Supplementary diagnostics

| Point | Scope | Use |
|---|---|---|
| E4.1-G | Direct GPU/NCCL BP-free versus exact-BP scaling | Runtime scaling diagnostic |
| E4.4-N | Nsight CUDA compute/copy activity | External CUDA activity trace |

E4.1-G uses `configs/e4_1_gpu_scaling.json` and
`run_e4_1_gpu_scaling.py`. It does not replace the resource-matched CPU/Gloo
CPU/Gloo matrix. E4.4-N wraps the E4.2 `b8_m4` command definitions with Nsight;
it does not maintain a second method-command implementation.

The CPU/Gloo configurations compare BP-free, GPipe, synchronous 1F1B, and
PipeDream with `OMP_NUM_THREADS=4`, the same pinned-memory send/receive budgets,
and the same receive-prepost depth.

## Implementations

The implementations are:

- BP-free CPU/Gloo: `sg_exe_trainer.runtime.bpfree.cpu_runner`;
- BP-free GPU/NCCL: `sg_exe_trainer.runtime.bpfree.gpu_runner`;
- exact-BP GPipe/1F1B: `sg_exe_trainer.runtime.exactbp.cpu_runner`;
- pinned CPU/Gloo transport: `sg_exe_trainer.runtime.transport.cpu`;
- PipeDream baseline: `experiments.shared.baselines.pipedream_cpu`.

The `pipedream_async/` directory contains standalone baseline evaluation and
report scripts; its configurations are under `pipedream_async/configs/`.

## Measurement design

E4.1-E4.3 report unsynchronized training throughput from fresh worker
processes. Their summaries include synchronized maximum wall time, full-run
throughput, transport byte budgets, and per-method optimizer progress.

E4.3 applies deterministic sender-side pacing:

```text
delay_ms = one_way_latency_ms + bytes * 8 / (bandwidth_mbps * 1000) + jitter
```

This is a controlled sensitivity model, not a packet-level emulator. It does
not modify host network settings.

E4.4 synchronizes CUDA at action boundaries and provides a mechanism-level
breakdown. E4.4-N leaves action tracing disabled and reads external CUDA
activity timestamps. E4.1-E4.3 provide the end-to-end throughput measurements.

## Commands

```bash
python experiments/e4_throughput/run_e4_1_cpu_transport_scaling.py --dry-run
python experiments/e4_throughput/run_e4_2_cpu_transport.py --config experiments/e4_throughput/configs/e4_2a_batch_geometry.json --dry-run
python experiments/e4_throughput/run_e4_3_mobile_network_sensitivity.py --dry-run
python experiments/e4_throughput/run_e4_4_overhead_decomposition.py --dry-run
python experiments/e4_throughput/run_e4_1_gpu_scaling.py --dry-run
python experiments/e4_throughput/gpu_timeline/run_nsys_timeline.py --dry-run
bash experiments/e4_throughput/run_formal_queue.sh
```

`build_e4_formal_figures.py` builds the E4.1-E4.3 tables and figures.
`run_e4_4_steady_trace.py` drives the E4.4 trace, analysis, and figure chain.
