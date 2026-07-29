# Artifact code layout

The repository separates reusable training-system behavior from paper-specific
evaluation policy.

## Runtime (`src/sg_exe_trainer/runtime`)

Runtime modules implement behavior that is present when the system executes:

- `bpfree/gpu_runner.py`, `gpu_phase.py`, `gpu_stage.py`, and
  `gpu_transport.py`: the GPU/NCCL runtime path;
- `bpfree/model_runtime.py`: model partition and local optimizer construction
  shared by GPU, CPU, and recovery runners;
- `bpfree/cpu_phase.py` and `cpu_stage.py`: CPU/Gloo stage integration;
- `bpfree/schedule.py` and `schedule_runtime.py`: BPFree schedule semantics;
- `exactbp/`: exact-backpropagation runtime used by comparison systems;
- `transport/`: data-plane transport and link emulation;
- `recovery/`: checkpoints, durable boundaries, commit ledgers, window journals,
  catch-up, and recovery event recording.

Runtime code must not import from `experiments`. If an experiment needs a new
system mechanism, the mechanism belongs here and the experiment should expose
only its configuration.

## Experiments (`experiments`)

Experiment directories describe how a paper result is produced:

- E1 contains quality protocols, hyperparameter grids, and aggregation;
- E2 contains the fixed GPU-memory matrix, launcher, result audit, and report;
- E4 contains throughput configurations, launchers, and report/plot builders;
- E5 contains outage/fault schedules, recovery policies under evaluation,
  baseline adapters, launchers, and comparison builders.

An experiment may import the runtime. The runtime must never import an
experiment. Reusable comparison implementations that are not part of BP-free
remain under `experiments/shared/baselines/` and are named as baselines.

## Tools (`tools`)

Tools perform offline export, provenance checks, aggregation, and figure/table
generation. They are not imported by the runtime.

## Single implementation rule

Runtime APIs have one canonical module under `sg_exe_trainer.runtime`.
Experiment-local aliases and re-export compatibility modules are not allowed.
The maintained import tree contains one implementation of each runtime API.

## Dependency direction

```text
experiments  --->  sg_exe_trainer.runtime
     |                    |
     +----> tools/data    +----> common/tasks/metrics

sg_exe_trainer.runtime  -X->  experiments
```

Generated results, model weights, datasets, caches, and machine-specific
launch state remain outside the source tree.
