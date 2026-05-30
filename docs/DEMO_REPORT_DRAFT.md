# Mobile BP-Free Training System Demo Draft

Last updated: 2026-05-30 Asia/Shanghai

## 1. One-Sentence Claim

This prototype demonstrates a mobile pipeline-training runtime for TinyLlama LoRA shards, where each phone performs local forward/backward/optimizer steps while inter-phone traffic remains forward-only hidden/belief information.

## 2. What Is Being Demonstrated

The current demo should be presented as a systems prototype, not as a final model-quality result.

Core demonstration points:

- Android workers can execute ExecuTorch `TrainingModule.executeForwardBackward()` and local `SGD.step()` on exported LoRA training PTEs.
- The coordinator manages a multi-stage mobile pipeline, including model shard assignment, next-hop routing, heartbeats, failure detection, and admin-visible scheduling state.
- The system has run real prepared Dolly SFT requests across phones, not only synthetic tensors.
- Three-phone full-model split has been smoke-tested with `tinyllama_lora_chunk_0/1/2`.
- Three-phone eval128 completed successfully on the full split; train512 exposed a heartbeat-stall failure after 84 successful training requests, which is useful control-plane evidence.
- The latest three-phone train512 rerun after the data-plane crash fix crossed the old failure point and completed a clean 128-request training window, then stopped at 193/194 because Android killed stage 2 for low memory.
- A bounded in-flight runner now demonstrates real cross-stage pipeline overlap instead of serial request submission.
- Fault-tolerance behavior has been measured on the real training path.

## 3. Algorithm Boundary

BP-free here does not mean no local backward pass.

The training flow is:

1. Stage 0 receives prepared embeddings, attention mask, position ids, and labels.
2. Each stage runs its local exported training graph.
3. Each stage applies local LoRA optimizer updates when `evalOnly=false`.
4. Stages forward hidden states and belief/log-prob signals to the next stage.
5. No cross-stage backward-gradient RPC is used.

This distinction is important for the report:

- Local computation: forward + local backward + local optimizer.
- Inter-stage communication: forward-only tensor/belief passing.

## 4. System Design Points

### Pipeline Design Contrast

Use these two figures as the design-comparison slide pair:

![Mobile BP-free pipeline mind map](figures/bpfree_pipeline_mindmap.png)

![Conventional BP pipeline mind map](figures/bp_pipeline_mindmap.png)

Suggested narration:

1. Conventional BP pipeline/1F1B is a strong baseline scheduling pattern, but it still couples stages through backward-gradient traffic. Activations flow forward; gradients flow back; a stage cannot finish its optimizer step until the downstream gradient dependency arrives.
2. This prototype keeps the pipeline partition, but changes the cross-phone contract. Every phone executes local forward/backward/LoRA optimizer on its own chunk, while the inter-phone data plane sends only hidden/belief information forward.
3. Because the algorithm removes cross-stage gradient RPC, the systems problem shifts toward mobile runtime concerns: shard placement, route readiness, heartbeat leases, retry behavior, worker telemetry, and persistent run evidence.

### Coordinator as Control Plane

The coordinator is responsible for:

- Registering Android workers as live nodes.
- Assigning nodes to model stages.
- Returning model artifact metadata and download URLs.
- Returning downstream next-hop routes.
- Tracking heartbeats and lease expiration.
- Recording request lifecycle events.
- Exposing `/api/v1/status` for system observability.

### Scheduler

The scheduler turns the original static mapping into a more robust mobile control plane:

- Preferred-device placement: known phones can be pinned to expected stages.
- Dynamic fill: unlisted or extra phones can fill empty stages.
- Late preferred-device correction: if a temporary worker occupies a stage, the preferred phone can take it back and the temporary worker can be relocated.
- Stage constraints: stages can declare `minMemoryGb`, `minComputeCapacity`, and `schedulingWeight`.
- Manual reconcile: `POST /api/v1/scheduler/reconcile` evicts expired leases and tries to restore preferred placement.
- Observability: admin status includes `assignmentReason` and recent `schedulerEvents`.

The systems angle is not that this is a complex cluster scheduler. The point is that mobile workers are unstable and orderless: phones change IPs, apps restart, devices arrive late, and stages must not blindly forward into dead downstreams.

### In-Flight Pipeline Runner

The original prepared-runner was intentionally conservative: submit one request, wait for the terminal stage, then submit the next request. That proves stability but does not prove pipeline overlap.

The new runner `:coordinator:runPreparedPipelineExperiment` keeps a bounded number of requests in flight. Android workers serialize only the local ExecuTorch/SGD section with a per-worker mutex; forwarding and downstream wait time are outside that mutex. This allows stage 0 to execute request `i+1` while stage 1 or stage 2 handles request `i`.

For timing figures, use coordinator-observed `event_epoch_ms` plus `total_measured_ms` from `stage-timings.csv`. Phone-local clocks are not a safe global timeline, and `local_ms` now includes local mutex queue wait.

### Failure Handling

The current implementation has:

- Heartbeat-based live-node detection.
- Offline-stage accounting.
- Worker eviction on lease expiration.
- Request-level failure recording.
- Admin retry/purge support.
- Checkpoint-file save attempt on Android.

Checkpoint restore is not yet a proven result. It should be presented as ongoing engineering, not as a validated guarantee.

## 5. Current Experimental Evidence

### Three-Phone Pipeline-Overlap Probe

Directory: `debug_runs/pipeline-overlap-20260530-114725`

Configuration:

- Stage 0: NX / `tinyllama_lora_chunk_0`
- Stage 1: Lenovo / `tinyllama_lora_chunk_1`
- Stage 2: Pixel / `tinyllama_lora_chunk_2`
- Runner: `runPreparedPipelineExperiment`, `maxInFlight=3`

| Probe | Success | Avg Loss | Token Accuracy | Overlap Evidence |
|---|---:|---:|---:|---|
| Eval-only 6 requests | 6/6 | 7.527053 | 0.373832 | 17 cross-stage local-execution overlaps; max 5994 ms |
| Train 6 requests | 6/6 | 7.478127 | 0.565957 | 19 cross-stage local-execution overlaps; max 6229 ms |
| Train512, window 3 | 512/512 | 7.4107 | 0.5579 | final run completed in 1h55m51s; throughput about 4.43 req/min; `maxTerminalInFlight=3` |

The current run's loss curve is mostly flat rather than monotonically improving: the 512 successful rows average `7.4107`, the first 50 successful requests average about `7.4113`, and the last 50 average about `7.5643`. That makes this a systems proof, not a convincing model-quality gain claim.

Important interpretation:

- This is the first direct evidence that the demo is not just serial request submission.
- The train probe uses the same Android `TrainingModule.executeForwardBackward()` + local optimizer path as the longer training runs.
- The probe is a scheduling/data-path proof, not a model-quality result.
- The temporary train-probe checkpoints were deleted afterward and workers were restarted, so future formal runs should start cleanly from PTE state.
- Request-order/FIFO is different from serial execution. Serial submission waits for the terminal stage before submitting the next request. The in-flight runner keeps up to three requests active, so stage 0 can execute a later request while stage 1 or stage 2 handles an earlier request. The full 512-request run completed with `512/512` final successes and `0` final failures.
- The current implementation proves bounded overlap with a local execution mutex. A stricter per-stage FIFO queue should be added before claiming deterministic request-order training quality.
- The 512 run included `5` transient failed attempts, all `Coordinator dispatch to stage 0 failed: Read timed out`, and all recovered by retry. This is a useful scheduler result: the window can be filled, but long-tail requests can occupy one slot and reduce effective concurrency. Present this as a next optimization target, not as a correctness failure.
- Worker telemetry was persisted by the coordinator during the run and exported after shutdown:
  - `worker-telemetry-final.csv`: 4135 heartbeat samples across the 117.62-minute run window.
  - `worker-telemetry-timeseries.png`: battery level, battery temperature, battery current, and app PSS over time.
  - NX / stage 0: 1380 samples, battery stayed `100%`, temp avg `36.0 C`, app PSS avg about `3256.0 MB`.
  - Lenovo / stage 1: 1379 samples, battery `99% -> 94%`, temp avg `35.4 C`, app PSS avg about `3259.4 MB`.
  - Pixel / stage 2: 1376 samples, battery `100% -> 95%`, temp avg `35.5 C`, app PSS avg about `3643.3 MB`.
  - Treat device-reported current as qualitative. It is useful for run telemetry, but a precise energy claim still needs Android Studio Power Profiler or an external power meter.

Report figure:

- `debug_runs/pipeline-overlap-20260530-114725/pipeline-train6-window3/figures/pipeline_overlap_gantt.png`
- `debug_runs/pipeline-overlap-20260530-114725/pipeline-train512-window3-20260530-121154/figures-128/pipeline_overlap_gantt.png`
- `debug_runs/pipeline-overlap-20260530-114725/pipeline-train512-window3-20260530-121154/figures-final/pipeline_overlap_gantt.png`
- `debug_runs/pipeline-overlap-20260530-114725/pipeline-train512-window3-20260530-121154/figures-final/worker-telemetry-timeseries.png`
- `debug_runs/pipeline-overlap-20260530-114725/pipeline-train512-window3-20260530-121154/figures-final/training_loss_curve.png`
- `debug_runs/pipeline-overlap-20260530-114725/pipeline-train512-window3-20260530-121154/figures-final/training_latency_curve.png`
- `debug_runs/pipeline-overlap-20260530-114725/pipeline-train512-window3-20260530-121154/figures-final/training_token_accuracy_curve.png`

### Two-Phone Formal Run

Directory: `debug_runs/formal-clean-20260526-215720`

This is currently the strongest complete training evidence.

| Phase | Success | Avg Latency | Avg Loss | Token Accuracy | Notes |
|---|---:|---:|---:|---:|---|
| Eval-before | 128/128 | 14187.05 ms | 5.754919 | 0.035949 | eval-only, no optimizer step |
| Train | 512/512 | 14172.34 ms | 5.647889 | 0.038854 | `evalOnly=false`, local optimizer steps |

Interpretation:

- This proves a repeatable two-phone mobile training path.
- Accuracy is low, but the run is useful as system evidence: real data, repeated requests, local optimizer, stage timing, and stable completion.
- Eval-after is not valid for this run because stage 0 native-crashed before the full eval-after completed, losing in-memory trained state.

### Fault-Tolerance Microbenchmark

Directory: `debug_runs/fault-tolerance-20260527-1516`

| Phase | Success | Avg Latency | Avg Loss | Token Accuracy |
|---|---:|---:|---:|---:|
| Pre-fault | 10/10 | 33363.30 ms | 5.717702 | 0.060274 |
| Post-fault | 10/10 | 30560.80 ms | 5.855204 | 0.039437 |
| Combined | 20/20 | 31962.05 ms | 5.786453 | 0.050000 |

Observed recovery:

- Stage 0 was killed manually.
- Coordinator observed the missing stage.
- Stage 0 was relaunched and re-registered as a new node.
- Post-fault training requests completed successfully.
- Approximate manual restart-to-live observation: 31.7 seconds.

Interpretation:

- This is good systems evidence: the pipeline can recover from a worker process crash and complete new training requests.
- It does not yet prove exact LoRA checkpoint restore correctness.

### Three-Phone Full-Split Run

Current directory: `debug_runs/three-phone-mainline-20260528-145049`

Configuration:

- Stage 0: `NX809J`, `tinyllama_lora_chunk_0`, `192.168.137.211:26052`
- Stage 1: `Lenovo_L71091`, `tinyllama_lora_chunk_1`, `192.168.137.124:26052`
- Stage 2: `23043RP34C`, `tinyllama_lora_chunk_2`, `192.168.137.174:26052`

Smoke result:

| Request | Success | Terminal Stage | Latency | Loss | Token Accuracy |
|---|---:|---:|---:|---:|---:|
| `three3-smoke-000001` | yes | stage 2 | 44233 ms | 8.2441435 | 0.333333 |

Completed eval128:

| Phase | Success | Avg Loss | Token Accuracy | Notes |
|---|---:|---:|---:|---|
| Eval128 | 128/128 | 7.405194 | 0.536203 | one wall-clock elapsed outlier should be flagged for latency |

Train512:

- Run id: `three-main-train512`
- Result: 84 successful training requests followed by 428 fast failures after stage 2 was evicted as `Stage 2 has no live worker`.
- Before failure, this run used `evalOnly=false`; coordinator stage timings show `optimizerStepApplied=true` on all stages.
- Diagnosis: stage 2 had normal battery, charging, temperature, and no app crash. The likely cause was a stuck heartbeat RPC/network stall combined with a short 15 s lease and a runner that kept submitting after transient route failure.
- Fix deployed on 2026-05-29: heartbeat/registration RPC deadlines, runner transient retry, and `heartbeatLeaseSeconds=45`.
- Worker heartbeat telemetry is now persisted by the coordinator, so battery/temperature/memory traces do not require ADB logcat or `monitor_perf.ps1`.

Data-plane crash fix and memory-stop rerun:

- A later clean run, `three-main-train512-cleanfix`, exposed a second bug: Lenovo and stage 2 crashed with `java.net.SocketException: Broken pipe` / `Software caused connection abort` in `HttpDataPlane.kt`.
- This was not an ExecuTorch or BP-free algorithm failure. It was a request transport bug: socket disconnect during large shard-response flush escaped the coroutine and killed the Android process.
- Raw crash logs are saved in `debug_runs/three-phone-mainline-20260528-145049/data-plane-crash-20260529-1224/`.
- Fix: data-plane per-client exceptions are now contained to the request, socket disconnects are logged as warnings, and the runner has a 420 s `SubmitRequest` deadline.
- Verification probe: `debug_runs/three-phone-mainline-20260528-145049/dataplane-fix-probe-20260529-1236/results.csv`, 3/3 success.
- Rerun: `three-main-train512-dataplanefix` in `debug_runs/three-phone-mainline-20260528-145049/train512-dataplanefix-20260529-1240`.
- Final result on 2026-05-29: 194 submitted, 193 successful training requests, 1 failed request; avg elapsed `42825.82 ms`, avg loss `7.4249`, token accuracy `0.5632`.
- This passed the previous request-84 heartbeat/offline failure point and completed a clean 128-request training window.
- Final failure: `three-main-train512-dataplanefix-000193`, message `Stage 2 has no live worker` after 19 attempts.
- Root cause: Android killed stage 2 process `pid=3797` with `reason=3 (LOW_MEMORY)` at `2026-05-29 14:58:14.673`, with PSS/RSS about `3.6GB`; no Java crash log was present. Stage 2 later restarted and re-registered, but the reloaded process cannot be treated as a valid continuation of the same in-memory LoRA training state.
- Memory interpretation: telemetry points to a high native/PTE/runtime memory plateau, not a step-by-step Java leak. Stage 2 rose from about `166 MB` PSS before model load to about `3.67 GB` after the training module became active, while Java heap stayed below about `98 MB`. The stage 2 artifact is the largest shard, and the current exporter duplicates `final_norm + lm_head` into every chunk for local CE/KL training.
- Post-diagnosis app optimization: checkpoint writes were throttled from every step to step 1 plus every 16 optimizer steps, and each local step now reports before/after PSS and Java heap in coordinator request events. Optional top-k belief transport was implemented but left default-disabled because simulation suggests belief may not improve over CE-only; it should not be treated as the main systems contribution.
- Final coordinator export and figures are saved under `debug_runs/three-phone-mainline-20260528-145049/coordinator-export-train512-dataplanefix-final193/`; raw failure evidence is saved under `debug_runs/three-phone-mainline-20260528-145049/train512-dataplanefix-20260529-1240/failure-evidence-20260529-1501/`.

Interpretation:

- The full TinyLlama LoRA split over three phones has moved beyond smoke: eval128 completed, the first train512 reached 84 real training updates before exposing a control-plane heartbeat failure, and the latest rerun reached 193 real training updates before exposing a stage-2 memory-pressure failure.
- The terminal stage was stage 2, so this is not the old two-of-four early-exit path.
- Token accuracy is still a systems sanity signal, not a final model-quality claim.

## 6. What Can Be Safely Claimed

Safe claims:

- The Android workers can run real ExecuTorch training modules.
- Local optimizer steps are applied for `evalOnly=false`.
- Inter-stage traffic remains forward-only.
- The coordinator can manage live stage routing and scheduler-visible worker assignment.
- The new in-flight runner demonstrates cross-stage overlap on both eval-only and training requests.
- The two-phone system completed a 512-request prepared Dolly SFT training sequence.
- The system recovered from a killed worker and completed post-fault training requests.
- A three-phone full-model split reached terminal stage 2 successfully.
- The mobile data plane now handles upstream socket disconnects without crashing the worker process.

Do not overclaim:

- Do not claim final model accuracy improvement on device yet.
- Do not claim checkpoint restore correctness yet.
- Do not use the older interrupted three-phone partial eval as the current eval128 result; the later full-split eval128 is the valid one, with its wall-clock latency outlier caveat.
- Do not compare two-phone two-of-four accuracy directly against three-phone full-split accuracy as a rigorous model-quality result; it is a useful demonstration contrast, not a controlled evaluation.

## 7. Suggested Demo Story

Recommended order for a live/report demo:

1. Motivation: mobile devices are abundant but unstable; training large models on them needs pipeline partitioning and fault-aware control.
2. Algorithm boundary: local BP-free style chunk training, forward-only inter-stage communication.
3. System architecture: coordinator plus Android workers.
4. Scheduler: preferred placement, dynamic fill, heartbeat leases, route readiness, event visibility.
5. Evidence 1: two-phone 512-request training run.
6. Evidence 2: three-phone pipeline-overlap Gantt from the in-flight runner.
7. Evidence 3: crash/restart fault-tolerance microbenchmark.
8. Evidence 4: three-phone full-model split eval128 and train512 memory-stop evidence.
9. Limitations: model quality and checkpoint restore still need controlled validation.
10. Next steps: reduce stage-2 memory pressure or validate checkpoint restore, then rerun train512/eval-after.

## 8. Report Figures

Generated figures live in `docs/figures/`.

Regenerate them from the current CSV/perf artifacts with:

```powershell
python tools\report\generate_demo_figures.py
```

For a new coordinator-recorded run, export the coordinator CSVs first:

```powershell
.\tools\report\export_coordinator_run.ps1 -RunId <runId> -OutDir debug_runs\<run>\coordinator-export -GenerateFigures
```

Recommended figure order:

1. `system_architecture.png`: control/data-plane picture. Use it to explain that each phone still performs local forward/backward/SGD, while cross-stage traffic is forward-only hidden/belief information.
2. `bpfree_pipeline_mindmap.png`: report-level design map for this prototype. Use it to show the coordinator, three Android stages, local forward/backward/optimizer, forward-only hidden/belief transfer, scheduling, failure handling, and persistent metrics.
3. `bp_pipeline_mindmap.png`: baseline mental model for conventional BP pipeline/1F1B. Use it to make the contrast explicit: forward activations go right, backward gradients come back left, and earlier stages remain coupled to downstream gradient availability.
4. `training_loss_curve.png`: 512-request two-phone training loss. This is the clearest "real training loop ran repeatedly" visual.
5. `training_latency_curve.png`: per-request end-to-end latency and rolling average for the same 512 training requests.
6. `training_token_accuracy_curve.png`: cumulative token accuracy during the 512-request run. Use cautiously; it is a systems sanity signal, not a final quality claim.
7. `experiment_summary_bars.png`: compact comparison across eval-before, train512, fault-tolerance, and three-phone partial runs.
8. `fault_tolerance_summary.png`: pre-fault vs post-fault latency/loss after a worker restart.
9. `fault_recovery_timeline.png`: coordinator live-node count during the recovery polling window.
10. `three_phone_partial_eval.png`: full-split three-phone partial eval, with the interrupted failure marked.
11. `device_memory_trace.png`: worker memory footprint from `monitor_perf.ps1`.
12. `battery_temperature_trace.png`: battery temperature during the monitored training window.
13. `battery_current_trace.png`: coarse device-reported current. Label this as qualitative unless using an external power meter or a controlled Android power-profile workflow.
14. `coordinator_run_metrics.png`: generated when coordinator `metrics.csv` is supplied.
15. `stage_local_timing_breakdown.png`: generated when coordinator `stage-timings.csv` contains local execution timing.
16. `stage_forward_timing_breakdown.png`: generated when coordinator `stage-timings.csv` contains forwarding timing.
17. `pipeline_overlap_gantt.png`: generated by `tools/report/plot_pipeline_overlap.py`; use it to prove stage-level overlap from coordinator-observed timings.
18. `worker-telemetry-timeseries.png`: generated by `tools/report/export_worker_telemetry.py`; use it to show battery level, battery temperature, device-reported current, and app PSS across the whole coordinator run window.

The generated `figure_metric_summary.csv` is a small audit table for the plotted averages.

Generate the pipeline-overlap figure with:

```powershell
python tools\report\plot_pipeline_overlap.py debug_runs\pipeline-overlap-20260530-114725\pipeline-train6-window3\stage-timings.csv --output_dir debug_runs\pipeline-overlap-20260530-114725\pipeline-train6-window3\figures
```

Export the coordinator-persisted worker telemetry time series with:

```powershell
python tools\report\export_worker_telemetry.py `
  --db coordinator\coordinator\data\coordinator.db `
  --results_csv debug_runs\pipeline-overlap-20260530-114725\pipeline-train512-window3-20260530-121154\results.csv `
  --output_dir debug_runs\pipeline-overlap-20260530-114725\pipeline-train512-window3-20260530-121154\figures-final
```

## 9. Long-Term Record Keeping

The current experiment record is good enough for a demo because `runPreparedExperiment` writes per-request CSVs and `monitor_perf.ps1` writes device samples. For a paper-quality system, the coordinator should own more of this record so the run is reproducible after logs scroll away.

Implemented after the initial report draft:

- Coordinator SQLite now persists `runs`, `request_metrics`, and `scheduler_events`.
- Requests submitted through the coordinator are automatically grouped into a run by request prefix, for example `train-000123` becomes run id `train`.
- Admin export endpoints:
  - `GET /api/v1/runs`
  - `GET /api/v1/runs/{runId}?metrics=1000`
  - `GET /api/v1/runs/{runId}/metrics.csv?limit=100000`
  - `GET /api/v1/runs/{runId}/stage-timings.csv?limit=100000`
- Scheduler events are persisted and restored after coordinator restart.
- Worker timing events are parsed into `stage_timing_metrics`, so future reports can plot per-stage `localMs`, `executeMs`, `optimizerStepMs`, `forwardMs`, and `totalStageMs`.

Recommended coordinator improvements:

1. Add a `runs` table keyed by `run_id`: phase, model shard names, pipeline config hash, git commit, dataset manifest path/hash, coordinator host, start/end timestamps, and notes.
2. Add a structured `request_metrics` table: request id, run id, record index, evalOnly, success, terminal stage, elapsed ms, loss, token correct/count, hidden/log-prob bytes, and failure message.
3. Extract per-stage timing into columns instead of leaving it only inside log strings: localMs, inputBuildMs, executeMs, gradientsMs, optimizerCreateMs, optimizerStepMs, outputConvertMs, forwardMs, totalStageMs.
4. Persist scheduler and fault events: assignment reason changes, lease expiration, manual reconcile, node lost/rejoined, route not ready, retry, and purge.
5. Let Android workers report resource snapshots in heartbeats or a side-channel: app PSS/RSS, battery level/status/current, thermal status, checkpoint save/restore status.
6. Add run lifecycle endpoints: `POST /api/v1/runs/start`, `POST /api/v1/runs/{id}/finish`, `GET /api/v1/runs/{id}/summary`, and `GET /api/v1/runs/{id}/metrics.csv`.
7. Add a checkpoint-restore proof event: checkpoint version/hash before crash, restore result after restart, and first post-restore request id.

Short-term rule for the next formal run:

- Create one directory per experiment under `debug_runs/<name>-<timestamp>/`.
- Store `results.csv`, `samples.csv`, `events.csv`, coordinator stdout/stderr, `/api/v1/status` snapshots, and figure outputs together.
- Copy or record the exact `pipeline.json`, model artifact hashes, dataset manifest hash, and git commit in a `run_manifest.json`.

## 10. Next Experiments To Finish Before Final Presentation

Highest priority:

1. Reduce stage-2 memory pressure or add a validated checkpoint/restore path before another full 512-request train.
2. Re-run three-phone `train512` after the memory fix, or deliberately report the 193-request run as a memory-failure systems result.
3. Keep all workers alive after train and run eval-after, or explicitly present why eval-after is blocked by in-memory state loss.
4. Preserve the coordinator heartbeat telemetry CSV for memory/thermal/current traces.
5. Add checkpoint-restore proof: train a few steps, record eval/loss, kill/restart one stage, confirm restore event/log, re-run same eval.

Optional but useful:

- Use a small router or stable hotspot with DHCP reservations.
- Reduce coordinator/Netty debug logging before final timed runs.
- Add a concise admin status screenshot showing three live stages and assignment reasons.
- Add a timing breakdown table from coordinator request events: stage local time and forward time.

## 11. Speaker Notes For Accuracy

If asked why token accuracy is low:

- The current mobile task is a systems validation task, not a fully tuned SFT run.
- The two-phone formal run used only two executed chunks from an earlier split, so it is not a full-model quality run.
- The three-phone split executes the full exported TinyLlama LoRA partition and already gives a more reasonable small-sample signal. Eval128 completed, and train512 passed the previous request-84 failure point, completed a clean 128-request training window, and then exposed a stage-2 low-memory limit at request 193.
- Dolly next-token accuracy at short sequence length is a harsh metric, especially with few LoRA steps and mobile constraints.

If asked what the novelty is:

- The novelty is the mobile distributed training runtime around a BP-free partitioned algorithm: forward-only inter-stage communication, Android ExecuTorch local training, scheduler-controlled model shard placement, heartbeat-based failure handling, and measurable fault recovery.

If asked what is still missing:

- Complete three-phone 512-request train/eval-after after memory or checkpoint work.
- Validated checkpoint restore.
- Cleaner energy measurement.
- A stronger controlled model-quality comparison.
- A controlled easy real-dataset run, preferably Rotten Tomatoes sentiment or AG News, to show a clearer loss trend without relying on a synthetic toy task.
