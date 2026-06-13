# Server Forward Pipeline Simulation

This note describes the stdlib-only simulator in:

- `tools/sim/simulate_forward_pipeline.py`

The simulator models the current forward-only stage runner, not a full-BP
PipeDream runtime. Each scheduled task is:

```text
stage0(request i) -> stage1(request i) -> ... -> terminal
```

Each stage task may do local loss, local backward, and local optimizer update
inside the worker. The scheduler still sees one atomic stage task and no
cross-stage backward edge.

## Why This Exists

For a single worker per stage and fixed-shape samples, bounded FIFO is already
the right baseline. The next useful server work is to quantify:

- the best possible throughput under measured stage times;
- queue depth and activation buffer pressure;
- which stage is the bottleneck;
- what happens under different buffer sizes;
- what-if stage replication or future placement choices could buy.

This should keep the project from inventing queue-order complexity where the
forward-only dependency graph does not need it.

## Quick Start

Run the included phone-derived example profile:

```bash
python tools/sim/simulate_forward_pipeline.py \
  --profile-json tools/sim/profiles/agnews_phone_stage_profile.example.json \
  --requests 512 \
  --buffer 3 \
  --replicas 1,1,1 \
  --output-json debug_runs/server_pipeline_sim/example_summary.json \
  --output-csv debug_runs/server_pipeline_sim/example_requests.csv
```

Compare against a no-overlap serial baseline:

```bash
python tools/sim/simulate_forward_pipeline.py \
  --profile-json tools/sim/profiles/agnews_phone_stage_profile.example.json \
  --requests 512 \
  --policy serial
```

Use real stage memory CSV output from phone or server runs:

```bash
python tools/sim/simulate_forward_pipeline.py \
  --stage-memory-csv debug_runs/stage-pipeline-eval-after256/results.stage_memory.csv \
  --requests 1024 \
  --duration-mode cycle \
  --buffer 3
```

## Runtime Semantics

The simulator intentionally mirrors `RunPreparedStagePipelineExperimentMain`:

- one bounded queue per stage;
- FIFO within each stage queue;
- stage workers block if the downstream queue is full;
- Q0 admission blocks if Q0 is full;
- terminal completion releases the request.

This means `--buffer 3` corresponds to the runner's `maxBufferedPerStage=3`.

## Important Outputs

- `throughput_per_s`: simulated completed requests per second.
- `stage_exec_utilization`: actual model/task execution occupancy.
- `stage_occupied_utilization`: execution plus blocked-on-downstream time.
- `max_queue_depth`: peak queued tasks per stage.
- `peak_buffer_bytes`: approximate queued plus blocked boundary tensor bytes.

If `stage_exec_utilization` is near 1.0 for one stage and lower for others, the
pipeline is bottlenecked by that stage. FIFO queue reordering will not fix that;
partitioning, placement, model export changes, or safe replication are the next
places to look.

## Next Server Extensions

The simulator is intentionally small. The next layer should be a planner that
uses measured per-layer or per-candidate-chunk profiles to choose:

- chunk boundaries;
- stage-to-device placement;
- buffer limits;
- optional stage replicas for what-if analysis.

The contribution should be framed as BP-free-aware partition and placement, with
bounded FIFO as the runtime baseline.
