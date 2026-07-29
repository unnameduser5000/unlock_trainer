# Measurement Methodology

This document defines the measurements shared by the memory, throughput, and
recovery experiments.

## E2 GPU Memory

E2 uses TinyLlama LoRA over three stages with effective batch `B=32` and the
following physical-batch/accumulation geometries:

```text
(b,m) = (1,32), (2,16), (4,8), (8,4)
```

The four methods are BP-free, synchronous 1F1B, GPipe, and PipeDream. Each job
runs in a fresh process. The `b8_m4` geometry uses three repetitions; the other
geometries use one.

Reported memory includes:

- the CUDA allocator peak;
- saved non-leaf tensor peaks;
- model, optimizer, and communication-buffer components;
- PipeDream trainable-version storage.

These components are independently observed peaks and are not added together
to reconstruct the allocator peak. Terminal output log-probabilities are
disabled for every method.

## E4 Throughput

E4.1-E4.3 measure end-to-end training in fresh worker processes with:

| Setting | Value |
| --- | --- |
| Python garbage collection in measured loop | disabled |
| `torch.cuda.empty_cache()` in measured loop | disabled |
| activation tracker | disabled |
| PyTorch/autograd profiler | disabled |
| evaluation during training | disabled |
| checkpointing and recovery | disabled |
| progress printing | disabled or fixed |
| timer | synchronized maximum worker wall time |

Summaries distinguish:

- `full_run_throughput_per_s`: complete execution including fill and drain;
- `steady_state_throughput_per_s`: execution after fill and before drain;
- `warmup_or_fill_ms`: startup and pipeline fill;
- `drain_ms`: pipeline tail;
- `fill_drain_overhead_ms`: sum of fill and drain.

E4.3 applies deterministic sender-side pacing:

```text
delay_ms = one_way_latency_ms + bytes * 8 / (bandwidth_mbps * 1000) + jitter
```

The pacing model controls message delay and bandwidth without changing host
network settings.

E4.4 synchronizes CUDA at action boundaries to separate compute, transfer,
pacing, and wait intervals. Its timings explain the runtime critical path;
E4.1-E4.3 provide the end-to-end throughput values.

## E5 Recovery

Recovery timing starts when Stage 1 becomes available and ends when the
terminal stage reaches the same target window for both methods. Reports include:

- stage commits present at rejoin;
- remaining stage-windows;
- replayed boundary or request bytes;
- durable and volatile recovery state;
- terminal-aligned recovery latency.

The transient-outage experiments keep worker processes and their memory alive.
Durable-outbox experiments additionally write boundary and checkpoint state.
Process-loss recovery is a separate failure model.
