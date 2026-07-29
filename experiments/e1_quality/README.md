# E1 quality experiments

E1 is the AG News quality experiment used in the evaluation. This directory
defines the datasets, seeds, training budget, four methods, and output checks.
Training, scheduling, transport, and recovery live under
`src/sg_exe_trainer/runtime/`.

## Protocol

The E1 protocol consists of:

- `configs/agnews_quality_v1.json`: dataset, method, and training settings;
- `run_agnews_quality.py`: manifest verification, execution, result capture,
  and output checks.

The config reconstructs the protocol that produced the current paper table. It
pins the manifest hashes and record counts, three seeds, TinyLlama LoRA setup,
effective batch size 8, 1,250 optimizer boundaries, and validation every 125
boundaries.

| Method id | Paper label | Runtime semantics |
| --- | --- | --- |
| `full_bp_1gpu` | Full BP, 1 GPU | one exact-BP stage, batch 8 |
| `1f1b_3gpu` | Sync. 1F1B, 3 GPU | three exact-BP stages, eight microbatches |
| `bpfree_ce_3gpu` | BP-free CE, 3 GPU | three local optimizers, CE only, accumulation 8 |
| `bpfree_belief_3gpu` | BP-free belief, 3 GPU | same runtime with full belief transport, alpha 0.5 |

Run the full protocol from the repository root:

```bash
PYTHONPATH=src python experiments/e1_quality/run_agnews_quality.py \
  --data-root /path/to/artifact-data
```

`--data-root` may point at a separate data directory. The runner verifies its
manifest hashes against the configuration. `--seeds` and `--methods` select a
subset, and `--dry-run` prints the generated commands without launching them.

Each run records its command and protocol digest. The runner checks test and
validation cardinalities, validation boundaries, training cardinality, and
BP-free per-stage optimizer-step counts.

## Implementation

E1 calls these runtime entry points:

- `sg_exe_trainer.runtime.exactbp.distributed_runtime` for Full BP and 1F1B;
- `sg_exe_trainer.runtime.bpfree.orchestrated_runtime` for both BP-free arms.

The experiment driver supplies evaluation settings to the shared runtime. The
training, transport, scheduling, and checkpoint implementations remain under
`src/sg_exe_trainer/runtime/`.

## Additional diagnostics

The `diagnostics/` directory contains separate studies for:

- readout-adapter diagnostics;
- GPipe/PipeDream learning-curve comparisons;
- DroidCall and Mobile Actions evaluations.

These diagnostics use separate configurations and do not change the four E1
method definitions above.
