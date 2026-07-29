# E5 Independent Intermittent Dropout

This protocol evaluates accuracy when every device is independently unavailable
with probability `p` for each logical update window. A mask is sampled once per
paired seed and reused by all methods. With three stages, the probability that a
window contains at least one unavailable device is `1 - (1 - p)^3`.

## Window Semantics

- One logical window contains eight records (`b=1`, `m=8`, `B=8`).
- Availability is sampled independently for every `(window, stage)` pair.
- A selected stage rejects its first execution attempt for all eight records.
- Simultaneous and consecutive device outages are retained in the sampled mask.
- BP-free progress reaches stage `s` only when every prefix stage `0..s` is online.

## Accuracy Policies

- `bpfree_fault_free`: no injected outage.
- `bpfree_local_retain`: successful prefix stages commit; the failed request is not replayed.
- `bpfree_replay`: failed stage attempts are retried, so every stage eventually processes every record once.
- `bpfree_skip`: if any device is offline, the complete logical window is rejected before Stage 0.
- `exact_fault_free`: synchronous 1F1B without injected outage.
- `exact_skip`: synchronous 1F1B drops a logical window if any stage is offline.
- `exact_replay`: the failure occurs before execution; after rejoin, synchronous 1F1B executes the batch once.

The replay runs establish the quality semantics only. Recovery latency remains a
separate terminal-aligned E5 experiment because adding arbitrary sleep intervals
to an accuracy run would confound quality and timing.

## Formal Command

```bash
python experiments/e5_recovery/intermittent_dropout/run_formal.py \
  --output_root results/e5_recovery/quality/formal_v4_independent_dropout \
  --train_limit 10000 \
  --eval_limit 7600 \
  --probabilities 0.05,0.10 \
  --seeds 20260531,20260532,20260533
```

`protocol.json` locks runner and manifest hashes. `masks/` stores the sampled
availability matrices, and each completed run writes `normalized_result.json`.
The driver asserts every method's per-stage optimizer-step count before adding
the run to `results.csv` and `PROGRESS.md`.
