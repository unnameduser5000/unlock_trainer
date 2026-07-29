# E5 formal v2

This directory contains the E5 outage protocol, method-specific runners,
comparison builders, and plotting tools. Shared checkpoint, journal, replay,
and commit implementations live in `src/sg_exe_trainer/runtime/recovery` and
are tested under `tests/runtime/recovery`.

## Recovery comparison

Persisting a stage-boundary hidden tensor is useful to both exact BP and
BP-free: either method can resume downstream forward computation from that
boundary. The difference measured by E5 is the state still owed upstream.

- Exact BP keeps upstream work provisional until its downstream gradient has
  returned. A strong exact-BP baseline may retain activations and weight
  versions, or trade that storage for forward recomputation.
- BP-free may durably commit the upstream local optimizer window before the
  downstream stage recovers. It retains the exact versioned boundary but owes
  no upstream backward pass for that request.

The comparison reports durable bytes, committed local windows, remaining
catch-up computation, and time to the same terminal commit count. Both methods
can resume downstream work from a saved boundary; E5 isolates the upstream
work that remains after the outage.

## State contract

`state_contract.py` provides two small persistence components:

1. `DurableBoundaryOutbox` writes a CPU tensor snapshot before publishing an
   atomic JSON commit marker. Every boundary is keyed by run, adjacent stages,
   update window, and physical microbatch, and records the producer version.
2. `StageCommitLedger` records one immutable optimizer commit per stage and
   update window. Repeating the same record is idempotent; changing it is an
   error.

The outbox has a pending-window capacity. Reaching it raises
`OutboxCapacityError`, which signals backpressure instead of silently dropping
or recomputing a boundary.

`window_journal.py` adds the publication order used by the BP-free runner:

```text
begin window
  -> snapshot each exact output boundary in memory
  -> local optimizer step succeeds
  -> publish every boundary payload
  -> publish the immutable stage commit marker
```

Recovery reads go through `CommittedBoundaryReader`. A tensor file without its
producer's stage commit is incomplete state and cannot be consumed. The
capacity check happens in `begin`, before model work or an optimizer update, so
backpressure cannot strand an already-applied update.

`checkpoint_store.py` writes an atomic filesystem checkpoint containing
trainable parameters, optimizer state, and RNG state. The payload and semantic
tensor state have independent SHA-256 fingerprints. Reusing a stage/version
key with different state is an error rather than an overwrite.

`runtime/recovery/runtime_adapter.py` integrates the recovery journal with the
BP-free stage. It captures exact body outputs for selected windows and
publishes them only after the optimizer step and durable checkpoint. The
outage runner combines its `skip_p2p_policy` with the downstream idle and
catch-up schedule.

`toy_outage_oracle.py` is a deterministic three-stage CPU contract test. It
compares normal window-major execution with an outage schedule in which all
stage-0 windows commit before stages 1 and 2 catch up. With exact versioned
boundaries and unchanged per-stage sample order, every final stage parameter
matches the fault-free execution exactly. At rejoin, prefix commits are
non-zero while terminal commits remain zero.

`protocol.py` defines a controlled stage-1 service outage.
During the outage, stage 0 uses the journaled outbox, stage 1 is idle, and stage
2 is blocked. It exposes two catch-up policies over the same durable state:

- `drain_first`: stage 1 drains the complete backlog before stage 2 starts.
- `window_streamed`: stage 1 publishes one durable window commit at a time;
  stage 2 waits for that exact commit and immediately consumes the associated
  boundary. The two stages therefore overlap after the first window becomes
  ready without allowing stage 2 to read provisional state.

Both policies deliberately reject other failed stages and undersized outboxes
instead of silently changing the experiment semantics.

`event_log.py` writes one atomic event file per rank and sequence number using
the host monotonic clock. The primary recovery intervals are stage rejoin to
stage-1 catch-up, all-stage catch-up, and live-P2P resume. There is no ambiguous
`global completion` metric.

`run_bpfree_outage.py` is the three-rank runner. It keeps one NCCL process
group alive but models stage 1 as an unavailable service: stage 0 diverts
selected windows to the durable outbox while stages 1 and 2 idle. The default
entry uses drain-first behavior. `run_bpfree_streamed_outage.py`
selects window-streamed catch-up without duplicating the runner. This is a
controlled service outage, not an OS-process or NCCL-rank crash.

`catchup_stream.py` implements the streamed dependency: only an immutable
`StageCommitLedger` marker releases a downstream window. Boundary-file
existence alone is insufficient. Per-window metrics split upstream-commit wait
from local compute and durable commit so the overlap can be measured directly.

`volatile_backlog.py` owns the bounded process-local CPU backlog used by both
transient-outage adapters. Stage 0 continues its local backward/update while
stage 1 is unavailable. No checkpoint or boundary file is written in this mode.

`run_bpfree_volatile_outage.py` selects volatile window-streamed replay.
`run_exactbp_volatile_outage.py` uses the same geometry but disables Exact-BP
checkpoint writes during catch-up. These entries model a transient service or
link outage in which all worker processes and their RAM remain alive. They do
not model process loss. They provide the corresponding NCCL/GPU-P2P diagnostic.

The current CPU-transport entries are:

- `run_bpfree_cpu_volatile_outage.py`: the normal D2H path produces a pinned
  CPU hidden tensor. During the outage that transport-ready tensor is retained;
  rejoin posts it directly to Gloo without an H2D staging copy.
- `run_exactbp_cpu_volatile_outage.py`: the E4 fair 1F1B runtime sends forward
  hidden and backward hidden-gradient tensors through the same pinned CPU/Gloo
  byte budgets.

The CPU recovery point uses receive prepost depth zero so a receive posted
before the controlled outage cannot survive into the measured rejoin interval.

`run_exactbp_outage.py` is the paired Exact-BP runner. It uses the same model,
stage split, physical batch, microbatch count, outage backlog, and terminal
endpoint. During the outage it queues raw requests and records no partial stage
commit. After stage 1 rejoins, every queued window runs through the real
PyTorch `Schedule1F1B`; all three optimizer checkpoints and commit markers are
published before that global window counts as recovered.

`build_comparison.py` rejects mismatched protocol inputs and emits one JSON,
CSV, and Markdown table. The primary latency is method-neutral:
`stage_rejoined -> terminal_target_reached`.

`build_policy_comparison.py` combines repeated Exact-BP, drain-first BP-free,
and window-streamed BP-free reports. `plot_streamed_gantt.py` renders the
per-window wait/compute trace from a timestamped streamed run.

## Durable M=8 result (NCCL)

The paired point uses TinyLlama-1.1B on three L40 GPUs, one stage per GPU,
physical batch 1, eight microbatches per logical window, four prelude windows,
four outage windows, and four resumed windows. Recovery starts at the stage-1
rejoin event and ends when the same terminal stage-window target is durably
committed. Values are mean +/- sample standard deviation over three fresh
runs.

| Method | Commits at rejoin (s0/s1/s2) | Remaining stage-windows | Recovery (ms) | Durable state (MiB) |
|---|---:|---:|---:|---:|
| Exact BP / 1F1B | 0/0/0 | 12 | 1726.405 +/- 171.301 | 26.245 |
| BP-free / drain-first | 4/0/0 | 8 | 4112.404 +/- 60.502 | 58.358 |
| BP-free / window-streamed | 4/0/0 | 8 | 2212.006 +/- 240.405 | 58.358 |

Window streaming reduces BP-free recovery latency by 46.2% relative to
drain-first while preserving the same committed prefix, remaining work, and
durable state. It remains 28.1% slower than the current Exact-BP 1F1B baseline;
the result therefore establishes removal of the stage-major drain bubble, not
an end-to-end recovery-latency win over Exact BP.

## CPU transient volatile diagnostic

The volatile point uses the same M=8 geometry with three fresh process launches
at a fixed training seed. Neither method writes checkpoint or boundary files in
the measured interval. BP-free retains 32 hidden microbatches (16 MiB) in RAM
and has four stage-0 local windows completed at rejoin. Exact BP has no
outage-window update committed at rejoin and replays all three stages.

| Method | Progress at rejoin (s0/s1/s2) | Remaining stage-windows | Recovery (ms, mean +/- std) | RAM hidden (MiB) | Run send (MiB) |
|---|---:|---:|---:|---:|---:|
| Exact BP / CPU 1F1B volatile | 0/0/0 | 12 | 2381.215 +/- 66.579 | 0 | 192 |
| BP-free / CPU hidden replay | 4/0/0 | 8 | 2013.115 +/- 6.403 | 16 | 96 |

At this L40/Gloo CPU-transport point, BP-free reaches the terminal target 15.5%
sooner on average. It retains four stage-0 windows, leaves one-third fewer stage-windows
after rejoin, and sends no hidden-gradient messages. The byte ledger records
96 MiB sent over the complete BP-free run and 192 MiB for Exact BP.
For both methods the primary endpoint is the stage-2 local terminal completion;
the following global synchronization barrier is outside the measured interval.

## Remaining runner variants

- `exactbp_boundary_recompute`: saved boundary with upstream activation
  recomputation for an in-window failure.
- `exactbp_activation_journal`: saved boundary, activation stash, and weight
  version journal.

The single-window retry provides an additional correctness test.
