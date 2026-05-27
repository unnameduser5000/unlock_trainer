# Current Debug State

This file is the first file to read after any context reset.

Last updated: 2026-05-27 17:55 Asia/Shanghai

## Current Mainline Position

Do not restart from inference smoke or one-request smoke after a context reset. Those paths have already served their purpose.

Current valid system evidence:

- The real two-phone LoRA BP-free path has been demonstrated on Android workers with `TrainingModule.executeForwardBackward()` and local `SGD.step()` when `evalOnly=false`.
- Formal clean run directory: `debug_runs/formal-clean-20260526-215720`.
- `eval-before`: 128/128 success, average elapsed `14187.05 ms`, average local loss `5.754919`, token accuracy `0.035949`.
- Training: combined 512/512 success in `debug_runs/formal-clean-20260526-215720/train-combined/results.csv`, average elapsed `14172.34 ms`, average local loss `5.647889`, token accuracy `0.038854`.
- Train performance monitor: `debug_runs/formal-clean-20260526-215720/train/perf-monitor-20260526-225135`.
- `eval-after-full` is not valid: NX/stage 0 crashed natively before eval-after could complete, so the trained in-memory state was lost on stage 0.

Current engineering state:

- Windows hotspot routing is now recovered on `192.168.137.0/24`; coordinator currently sees both stages live.
- The Android APK installed on both phones includes the checkpoint attempt code in `TrainingCheckpointStore.kt` and the `NativeShardRunner.kt` calls around restore/save.
- Checkpoint support is not yet a proven paper result. It compiled and was deployed, but restore correctness has not been validated end-to-end because ExecuTorch Android exposes parameter tensors but no official Kotlin state-dict save/load API.
- A user-interrupted command on 2026-05-27 still completed one extra `evalOnly=false` request: `debug_runs/checkpoint-train-smoke-20260527/results.csv`, request `checkpoint-train-smoke-000001`, success, elapsed `27442 ms`. Treat this as an accidental checkpoint/debug probe, not part of formal metrics.

Scheduler state for three phones:

- Coordinator scheduling was extended on 2026-05-27. The scheduler now supports dynamic unlisted workers, configured-device preference, safe relocation of a temporary worker when the preferred phone later registers, stage-level minimum memory/compute requirements, a manual reconcile endpoint, and admin-visible scheduler events.
- New admin endpoint: `POST /api/v1/scheduler/reconcile`. Use it after editing config, after a worker restart sequence, or before a long run to evict expired leases and move live preferred devices back to their configured stages when a safe relocation exists.
- `/api/v1/status` now includes scheduler config flags, per-node `assignmentReason`, per-stage scheduling requirements, and recent `schedulerEvents`.
- Active `coordinator/config/pipeline.json` is still the two-stage LoRA pipeline. Do not switch it to three stages until `model/tinyllama_lora_chunk_2.pte` exists and the third phone IP/device id is known.
- Three-phone template added at `coordinator/config/pipeline_three_phone.template.json`. Stage 2 deliberately has blank `deviceId`; with `allowUnlistedDevices=true`, the first extra live phone can fill it.
- Server-side export command for a consistent 3-stage LoRA pipeline:

```bash
NUM_CHUNKS=3 CHUNK_IDX=-1 OUTPUT_DIR=model bash tools/export/export_lora_tinyllama.sh
```

- Regression check passed after the scheduler change:

```powershell
C:\Users\wentaodai\.gradle\wrapper\dists\gradle-8.7-bin\bhs2wmbdwecv87pi65oeuq5iu\gradle-8.7\bin\gradle.bat :coordinator:test --offline --no-daemon
```

Next non-smoke mainline choices:

- If the goal is a clean metric run, restart both Android workers to reload initial PTE state, then run a fresh `eval-before -> train512 -> eval-after` sequence.
- If the goal is fault tolerance, validate checkpoint save/restore explicitly by checking Android checkpoint files and proving restored parameters affect a post-restart eval/train request.
- If the goal is paper evidence, use the existing 128/512 formal run for throughput/loss and document the eval-after crash as the reason checkpoint/recovery became necessary.

## 2026-05-27 Fault Tolerance Microbenchmark

Experiment directory: `debug_runs/fault-tolerance-20260527-1516`.

Purpose: demonstrate worker-crash recovery for the real two-phone LoRA training path, not another inference or one-request smoke.

Setup:

- Backed up old Android checkpoint files into `files/shards/checkpoints/backup_20260527_1516` on both phones.
- Restarted both workers for a clean in-memory state.
- Initial recovered route before training:
  - stage 0 NX node `80` at `192.168.137.211:26052`
  - stage 1 Lenovo node `81` at `192.168.137.124:26052`
  - `liveNodeCount=2`
- Dataset: `data/sft_requests/tinyllama_dolly64_train512/requests.jsonl`.

Pre-fault run:

```powershell
C:\Users\wentaodai\.gradle\wrapper\dists\gradle-8.7-bin\bhs2wmbdwecv87pi65oeuq5iu\gradle-8.7\bin\gradle.bat :coordinator:runPreparedExperiment --offline --no-daemon --args="127.0.0.1 50051 data/sft_requests/tinyllama_dolly64_train512/requests.jsonl 0 10 debug_runs/fault-tolerance-20260527-1516/pre-fault/results.csv ft-prefault 1 0 false true"
```

- CSV: `debug_runs/fault-tolerance-20260527-1516/pre-fault/results.csv`.
- 10/10 requests succeeded.
- Average elapsed: `33363.3 ms`.
- Note: the command timed out at the Codex tool level after 300s, but the CSV and coordinator state show all 10 requests completed. Do not count that as a training failure.

Fault injection:

- Fault injected at `2026-05-27 15:46:26.488 +08:00`.
- Action: `adb -s 91260221021D shell am force-stop com.example.sid_trainer`.
- Stage 0 process disappeared. Coordinator status showed `liveNodeCount=1`, stage 0 unassigned, stage 1 still live as node `81`.
- Coordinator `offlineStageCount` remained `0` even while stage 0 had no assigned node. This is a useful fault-detection bug: the admin summary undercounts non-terminal offline stages.

Recovery:

- Restart command timestamp: `2026-05-27 15:48:24.935 +08:00`.
- Action: `adb -s 91260221021D shell monkey -p com.example.sid_trainer -c android.intent.category.LAUNCHER 1`.
- First observed recovered stage 0: `2026-05-27 15:48:56.633 +08:00`.
- End-to-end manual restart-to-live observation: about `31.7 s`.
- New stage 0 node: `82` at `192.168.137.211:26052`.
- Stage 1 stayed node `81`.

Post-fault run:

```powershell
C:\Users\wentaodai\.gradle\wrapper\dists\gradle-8.7-bin\bhs2wmbdwecv87pi65oeuq5iu\gradle-8.7\bin\gradle.bat :coordinator:runPreparedExperiment --offline --no-daemon --args="127.0.0.1 50051 data/sft_requests/tinyllama_dolly64_train512/requests.jsonl 10 10 debug_runs/fault-tolerance-20260527-1516/post-fault/results.csv ft-postfault 1 0 false true"
```

- CSV: `debug_runs/fault-tolerance-20260527-1516/post-fault/results.csv`.
- 10/10 requests succeeded after stage 0 restart.
- Average elapsed: `30560.8 ms`.
- Combined pre+post: 20/20 success, average elapsed `31962.05 ms`, average local loss `5.786453`, token accuracy `0.05`.

Evidence that the recovered stage continued real training:

- `ft-postfault-000010` was dispatched to stage 0 node `82`, then stage 1 node `81`, and completed successfully.
- Coordinator event for `ft-postfault-000010` stage 0: `runtime=TrainingModule`, `evalOnly=false`, `optimizerStepApplied=true`, `localMs=4467`, `loss=10.144974`.
- Coordinator event for `ft-postfault-000019` stage 0: `runtime=TrainingModule`, `evalOnly=false`, `optimizerStepApplied=true`, `localMs=3980`, `loss=10.7333555`.
- Stage 1 also continued applying optimizer steps on node `81`.
- Checkpoint files existed and were updated after the experiment:
  - NX: `files/shards/checkpoints/tinyllama_lora_chunk_0.latest.sidckpt`, about `501K`, timestamp `2026-05-27 15:56`.
  - Lenovo: `files/shards/checkpoints/tinyllama_lora_chunk_1.latest.sidckpt`, about `501K`, timestamp `2026-05-27 15:56`.

Interpretation boundary:

- This proves worker-crash recovery at the system level: after stage 0 was killed and relaunched, the pipeline accepted and completed new real training requests on a new node id.
- It also proves checkpoint files are being saved on both phones.
- It does not yet prove checkpoint restore numerically preserved the exact trained LoRA state. To claim that, run an explicit restore validation: record a post-training eval/loss, kill/restart the stage, confirm `Checkpoint restore status restored=true` in logs or add a coordinator-visible restore event, then repeat eval and compare outputs/loss.
- The `offlineStageCount=0` behavior during missing stage 0 was fixed on 2026-05-27 16:24. The bug was that summary used `!terminal && !routeReady`; `routeReady` describes downstream next-hop readiness, not whether this stage itself has a live worker. It now counts non-drained stages whose `assignedNode?.isLive != true`.
- Regression test added: `coordinator/src/test/kotlin/com/example/sid_coordinator/CoordinatorStateTest.kt`, covering the case where stage 1 is live but stage 0 is missing. `:coordinator:test --offline --no-daemon` passed.
- The currently running coordinator JVM must be restarted before `/api/v1/status` reflects this fix.

## 2026-05-27 Windows Hotspot Recovery

Current working network is the Windows mobile hotspot subnet `192.168.137.0/24`.

- Coordinator artifact/admin host: `192.168.137.1`
- Coordinator gRPC/admin ports: `50051` / `18080`
- Stage 0 `NX809J` / ADB `91260221021D`: `192.168.137.211:26052`
- Stage 1 `Lenovo_L71091` / ADB `ZY22G2HC5C`: `192.168.137.124:26052`
- `coordinator/config/pipeline.json` has been updated to these two `expectedHost` values and `artifactBaseUrl=http://192.168.137.1:18080`.
- `app/src/main/java/com/example/sid_trainer/MainActivity.kt` default coordinator host is `192.168.137.1`.
- Rebuilt with `:app:assembleDebug --offline --no-daemon`, installed to both phones, force-stopped, and relaunched both apps on 2026-05-27 14:46.

Coordinator status after relaunch:

- `liveNodeCount=2`
- `inactiveNodeCount=0`
- `offlineStageCount=0`
- Stage 0 assigned node `78` at `192.168.137.211:26052`
- Stage 1 assigned node `79` at `192.168.137.124:26052`
- Stage 0 next hop points to node `79` at `192.168.137.124:26052`

Real two-phone eval smoke also passed:

```powershell
C:\Users\wentaodai\.gradle\wrapper\dists\gradle-8.7-bin\bhs2wmbdwecv87pi65oeuq5iu\gradle-8.7\bin\gradle.bat :coordinator:runPreparedExperiment --offline --no-daemon --args="127.0.0.1 50051 data/sft_requests/tinyllama_dolly64_smoke/requests.jsonl 0 1 debug_runs/hotspot-smoke-20260527/results.csv hotspot-smoke 1 0 true true"
```

Result:

- Record index `0` skipped because `validLabels=0`.
- Submitted request `hotspot-smoke-000001` / record index `1`.
- `evalOnly=true`, so this did not apply optimizer steps.
- `success=true`, `terminal=true`, processed by stage 1.
- End-to-end elapsed time: `31303 ms`.
- Local loss: `6.8665123`.
- Token accuracy: `0.0` over `3` valid labels.
- CSV: `debug_runs/hotspot-smoke-20260527/results.csv`.

If the hotspot IPs change again, rerun `adb -s <serial> shell ip -f inet addr show wlan0`, update `coordinator/config/pipeline.json`, POST `/api/v1/routing/reload`, then relaunch both Android workers. Avoid parallel ADB commands when restarting the daemon.

## Mainline

The intended mainline experiment is BP-free / forward-only inter-stage training behavior on two Android phones.

Immediate target: restore the algorithm distinction before running more claims. BP-free means no cross-chunk backward-gradient traffic; it does not mean "no backward anywhere". A real BP-free training step still needs local backward/optimizer inside each chunk, while chunks exchange forward belief/log-prob signals.

The previously verified non-LoRA `tinyllama_chunk_0/1.pte` artifacts are training PTEs with `__et_training=3` and run through Android `TrainingModule.executeForwardBackward()` plus local `SGD.step()`. The current local runtime config was switched on 2026-05-25 17:19 to the LoRA artifacts `tinyllama_lora_chunk_0/1.pte` for the LoRA smoke test.

Do not confuse these three paths:

- `*_inf.pte`: relay / inference compatibility artifacts. This path was already tested earlier.
- `tinyllama_chunk_*.pte` without `_inf` exported by `sid_export_forward_mobile.py`: markerless `Module.execute()` forward graph carrying local CE/KD loss and belief/log-prob outputs. This is not full training.
- ExecuTorch `TrainingModule.executeForwardBackward()`: needed if the mobile experiment is meant to perform chunk-local backward from an exported joint graph. This still does not imply cross-chunk gradient RPC.

Real BP-free training target:

- Stage-to-stage traffic: forward-only hidden states and belief/log-prob signals.
- Inside each chunk: local loss, local backward, and local optimizer step.
- Android source now calls a chunk-local optimizer for training PTEs. In `NativeShardRunner.TrainingLoadedRuntime.execute`, the flow is `TrainingModule.executeForwardBackward("forward", *inputs)`, then `namedGradients("forward")`, lazy `SGD.create(namedParameters("forward"), 1e-5)`, then `SGD.step(gradients)`. This was verified by `:app:assembleDebug` on 2026-05-25 14:55. The currently deployed phone app still needs to be rebuilt/installed before this behavior exists on device.
- Maven `org.pytorch:executorch-android:1.2.0` exposes `SGD`, `namedParameters`, and `namedGradients`, but does not expose `TrainingModule.close()` to Kotlin. Do not re-add `module.close()` unless the dependency is upgraded and a build confirms it.

## 2026-05-26 Experiment/Measurement Scaffold

This is the current step-by-step path for turning the one-request smoke into a repeatable system experiment.

Current local status observed 2026-05-26 09:09 Asia/Shanghai:

- `adb devices` shows both phones online: `91260221021D` and `ZY22G2HC5C`.
- The Android app process was alive on both devices during the check (`NX809J` pid `22082`, `Lenovo_L71091` pid `25500`).
- Coordinator admin `http://127.0.0.1:18080` was not reachable in this Windows session. Start coordinator before submitting another experiment.
- Worktree was clean before this instrumentation pass.

New experiment support added on 2026-05-26:

- `coordinator/src/main/kotlin/com/example/sid_coordinator/PreparedRequestSupport.kt` centralizes prepared JSONL loading, tensor loading, and valid-label counting.
- `coordinator/src/main/kotlin/com/example/sid_coordinator/RunPreparedExperimentMain.kt` submits a range of prepared records and writes a CSV summary.
- `coordinator/build.gradle.kts` exposes `:coordinator:runPreparedExperiment`.
- `coordinator/src/main/kotlin/com/example/sid_coordinator/SubmitPreparedRequestMain.kt` now prints `validLabels` for a single-record run.
- `tools/data/prepare_lora_sft_requests.py` now writes `valid_label_count` into each manifest record and supports `--min_valid_labels`.
- `tools/android/collect_perf_snapshot.ps1` captures per-device battery, thermal, meminfo, top, and focused logcat snapshots under `debug_runs/`.
- `tools/android/monitor_perf.ps1` continuously samples both phones into `samples.csv` and extracts timing log events into `events.csv`.
- `tools/android/record_perfetto_trace.ps1` records simultaneous per-device Perfetto traces and pulls `.pftrace` files into `debug_runs/`.
- `ForwardChunkRequest` now has `eval_only = 9`. Default is false, preserving old training requests.
- `runPreparedExperiment` and `runSubmitPreparedRequest` can set eval-only mode. In eval-only mode Android still runs `TrainingModule.executeForwardBackward()` to get loss/log-probs, but skips `namedGradients()`, `SGD.create()`, and `SGD.step()`.
- `runPreparedExperiment` now computes shifted token accuracy from terminal `output_shift_log_p` and labels. CSV columns include `local_loss`, `token_correct`, `token_count`, and `token_accuracy`.
- Android worker logs now include local timing in `LOCAL_COMPLETED` request events: runtime, method, input count, local loss, `localMs`, `inputBuildMs`, `executeMs`, `gradientsMs`, `optimizerCreateMs`, `optimizerStepMs`, and `outputConvertMs`.
- Android worker forwarding completion events now include `forwardMs` and `totalStageMs`.
- Android trace sections were added for Perfetto / Android Studio Profiler: `sid_worker_local_execute`, `sid_build_training_inputs`, `sid_execute_forward_backward`, `sid_named_gradients`, `sid_create_sgd`, `sid_sgd_step`, `sid_worker_forward_next`, plus inference-side sections for controlled baselines.

Minimal repeatable experiment sequence:

```powershell
# 1. Build and reinstall the timed Android worker before measuring.
C:\Users\wentaodai\.gradle\wrapper\dists\gradle-8.7-bin\bhs2wmbdwecv87pi65oeuq5iu\gradle-8.7\bin\gradle.bat :app:assembleDebug --offline --no-daemon
adb -s 91260221021D install -r app\build\outputs\apk\debug\app-debug.apk
adb -s ZY22G2HC5C install -r app\build\outputs\apk\debug\app-debug.apk
adb -s 91260221021D shell am force-stop com.example.sid_trainer
adb -s ZY22G2HC5C shell am force-stop com.example.sid_trainer
adb -s 91260221021D shell monkey -p com.example.sid_trainer 1
adb -s ZY22G2HC5C shell monkey -p com.example.sid_trainer 1

# 2. Start coordinator in a dedicated terminal.
C:\Users\wentaodai\.gradle\wrapper\dists\gradle-8.7-bin\bhs2wmbdwecv87pi65oeuq5iu\gradle-8.7\bin\gradle.bat :coordinator:run --offline --no-daemon

# 3. Confirm both stages are live and active.
.\tools\android\check_status.ps1

# 4. Capture a pre-run device snapshot.
.\tools\android\collect_perf_snapshot.ps1 -Phase pre

# 4a. Optional but recommended: start continuous low-frequency monitoring in a second terminal.
# DurationSec=0 runs until Ctrl+C. Use a fixed duration when you want unattended runs.
.\tools\android\monitor_perf.ps1 -Phase dolly-lora-exp -IntervalSec 2 -MeminfoEverySamples 5 -DurationSec 180

# 4b. Optional heavier trace: start simultaneous Perfetto recording in another terminal.
# Open the resulting .pftrace with Android Studio Profiler or https://ui.perfetto.dev.
.\tools\android\record_perfetto_trace.ps1 -Phase dolly-lora-exp -DurationSec 180

# 5. Submit all eligible prepared requests. Argument order:
# host port manifest startIndex maxSubmitted outputCsv requestPrefix minValidLabels delayMs evalOnly
# maxSubmitted=0 means all eligible records.
C:\Users\wentaodai\.gradle\wrapper\dists\gradle-8.7-bin\bhs2wmbdwecv87pi65oeuq5iu\gradle-8.7\bin\gradle.bat :coordinator:runPreparedExperiment --offline --no-daemon --args="127.0.0.1 50051 data/sft_requests/tinyllama_dolly64_smoke/requests.jsonl 0 0 debug_runs/dolly-lora-exp/results.csv dolly-lora-exp 1 0 false"

# 6. Capture a post-run snapshot and full debug bundle.
.\tools\android\collect_perf_snapshot.ps1 -Phase post
.\tools\android\collect_debug_bundle.ps1
```

Performance evidence layers:

- `runPreparedExperiment` CSV is the canonical request-level result: one row per submitted record, with end-to-end submit latency and terminal success/failure.
- Coordinator request events are the canonical stage-level lifecycle log. They now contain Android-reported local compute timing and forwarding timing in event messages.
- `monitor_perf.ps1` CSV is the low-frequency device curve: battery level/status/temperature/voltage/current, thermal status, process RSS/VSZ, and lower-frequency app PSS/private dirty from `dumpsys meminfo`.
- `record_perfetto_trace.ps1` is the high-resolution trace path. Use it for timeline evidence around `sid_execute_forward_backward`, `sid_sgd_step`, and `sid_worker_forward_next`.
- Android Studio Power Profiler is still useful for visual exploration, but paper numbers should be backed by the CSV/event/trace artifacts above.

Train/eval effect-check sequence:

```powershell
# Eval-before: no optimizer step.
C:\Users\wentaodai\.gradle\wrapper\dists\gradle-8.7-bin\bhs2wmbdwecv87pi65oeuq5iu\gradle-8.7\bin\gradle.bat :coordinator:runPreparedExperiment --offline --no-daemon --args="127.0.0.1 50051 data/sft_requests/tinyllama_dolly64_eval128/requests.jsonl 0 0 debug_runs/eval-before/results.csv eval-before 1 0 true"

# Train: optimizer step enabled.
C:\Users\wentaodai\.gradle\wrapper\dists\gradle-8.7-bin\bhs2wmbdwecv87pi65oeuq5iu\gradle-8.7\bin\gradle.bat :coordinator:runPreparedExperiment --offline --no-daemon --args="127.0.0.1 50051 data/sft_requests/tinyllama_dolly64_train512/requests.jsonl 0 0 debug_runs/train/results.csv train 1 0 false"

# Eval-after: same app process / cached runtime, no optimizer step.
C:\Users\wentaodai\.gradle\wrapper\dists\gradle-8.7-bin\bhs2wmbdwecv87pi65oeuq5iu\gradle-8.7\bin\gradle.bat :coordinator:runPreparedExperiment --offline --no-daemon --args="127.0.0.1 50051 data/sft_requests/tinyllama_dolly64_eval128/requests.jsonl 0 0 debug_runs/eval-after/results.csv eval-after 1 0 true"
```

Important eval boundary:

- This measures eval loss and shifted token accuracy on the mobile pipeline's current in-memory weights.
- Keep both Android app processes alive between eval-before, train, and eval-after, or the cached runtime and in-memory optimizer/model updates will be lost.
- This is a practical mobile effect check, not a replacement for the larger server-side algorithm accuracy evaluation.

Monitoring verification on 2026-05-26:

- `monitor_perf.ps1 -Phase smoke4 -IntervalSec 1 -MeminfoEverySamples 2 -DurationSec 1` produced `debug_runs/perf-monitor-20260526-103323/samples.csv` with rows for both phones.
- `record_perfetto_trace.ps1 -Phase smoke3 -DurationSec 1` successfully recorded and pulled `.pftrace` files for both phones under `debug_runs/perfetto-20260526-103837/`.
- Perfetto configs must be placed under `/data/misc/perfetto-configs`; `/data/local/tmp` produced permission-denied errors on these phones.
- `:coordinator:compileKotlin --offline --no-daemon` and `:app:compileDebugKotlin --offline --no-daemon` passed after adding `eval_only` and token-accuracy calculation on 2026-05-26 11:09.
- `:app:assembleDebug --offline --no-daemon` passed and the rebuilt debug APK was installed on both phones on 2026-05-26 11:19.
- After reinstall, both Android app processes launched, but coordinator admin was still unavailable. Restart both workers after starting coordinator so registration and routing are fresh.

Prepared LoRA SFT train-smoke verification on 2026-05-26 13:35:

- Coordinator was already running before the smoke request. `netstat` showed PID `12880` listening on `0.0.0.0:50051` and `0.0.0.0:18080`.
- Initial coordinator status was incomplete: `liveNodeCount=1`; Lenovo/stage 1 was live as node `65`, but NX/stage 0 had no assigned live node. `check_status.ps1` showed the NX app process alive as pid `14443`, so the issue was stale/non-registered worker state rather than ADB offline.
- Recovery action was a minimal sequential NX worker restart only:
  - `adb -s 91260221021D shell am force-stop com.example.sid_trainer`
  - `adb -s 91260221021D shell monkey -p com.example.sid_trainer 1`
- After the restart, coordinator status showed routing epoch `306`, `liveNodeCount=2`, `inactiveNodeCount=0`, `offlineStageCount=0`, `drainedStageCount=0`.
- Recovered routing: stage 0 `NX809J` node `66` at `192.168.214.103:26052`; stage 1 `Lenovo_L71091` node `65` at `192.168.214.59:26052`; stage 0 next hop pointed to node `65`.
- Current Windows data availability at this point: `data/sft_requests/tinyllama_dolly64_smoke` exists; `data/sft_requests/tinyllama_dolly64_train512` and `data/sft_requests/tinyllama_dolly64_eval128` do not exist locally yet.
- `.\gradle.bat` is not a project file. `.\gradlew.bat` exists, but in this sandbox it attempted to download `gradle-8.7-bin.zip` and failed with network permission denial. The already-unpacked local Gradle binary worked only outside the sandbox because sandboxed Gradle failed with `Unable to establish loopback connection`.
- Smoke command actually used:

```powershell
C:\Users\wentaodai\.gradle\wrapper\dists\gradle-8.7-bin\bhs2wmbdwecv87pi65oeuq5iu\gradle-8.7\bin\gradle.bat :coordinator:runPreparedExperiment --offline --no-daemon --args="127.0.0.1 50051 data/sft_requests/tinyllama_dolly64_smoke/requests.jsonl 0 1 debug_runs/train-smoke/results.csv train-smoke 1 0 false"
```

- Argument meaning for that command: `startIndex=0`, `maxSubmitted=1`, `minValidLabels=1`, `delayMs=0`, `evalOnly=false`. This means "scan from record 0, skip records without training labels, submit one eligible training request, and allow Android optimizer steps."
- The run skipped record index `0` because `validLabels=0`, then submitted record index `1` because `validLabels=3`.
- CSV result path: `debug_runs/train-smoke/results.csv`.
- CSV row: `request_id=train-smoke-000001`, `record_index=1`, `dataset_index=1`, `valid_labels=3`, `eval_only=false`, `success=true`, `terminal=true`, `processed_stage_id=1`, `processed_chunk_idx=1`, `elapsed_ms=12359`, `output_hidden_bytes=262144`, `output_shift_log_p_bytes=4096000`, `local_loss=6.8664217`, `token_correct=0`, `token_count=3`, `token_accuracy=0.0`.
- Coordinator request detail for `train-smoke-000001` showed `lifecycleState=COMPLETED`, `terminal=true`, `storedPayload=true`, `submitAttempts=1`.
- Coordinator events prove both Android chunks ran the training path:
  - coordinator accepted and dispatched to stage 0 node `66` with `evalOnly=false`;
  - stage 0 node `66` received chunk 0;
  - stage 0 `LOCAL_COMPLETED`: `runtime=TrainingModule`, `method=forward`, `inputs=4`, `localMs=4446`, `loss=11.872195`, `optimizerStepApplied=true`, `executeMs=4390`;
  - stage 0 forwarded to `192.168.214.59:26052`;
  - stage 1 node `65` received chunk 1;
  - stage 1 `LOCAL_COMPLETED`: `runtime=TrainingModule`, `method=forward`, `inputs=5`, `localMs=5097`, `loss=6.8664217`, `optimizerStepApplied=true`, `executeMs=4927`;
  - stage 1 terminal completion reported `totalStageMs=5193`;
  - stage 0 completion reported downstream success with `forwardMs=6139` and `totalStageMs=10713`.
- This is a real two-phone prepared Dolly SFT LoRA training smoke. It is still not the full train/eval effect-check because the larger `train512` and `eval128` prepared request sets were not present on this Windows machine.

Eval-only and monitored train smoke verification on 2026-05-26 14:17-14:19:

- Pre-run coordinator status remained route-ready: routing epoch `322`, `liveNodeCount=2`, `inactiveNodeCount=0`, `offlineStageCount=0`, `drainedStageCount=0`. Stage 0 was NX node `69`; stage 1 was Lenovo node `68`; stage 0 next hop pointed to Lenovo at dynamic port `38395`.
- Local prepared data availability was unchanged: only `data/sft_requests/tinyllama_dolly64_smoke/requests.jsonl` existed; `tinyllama_dolly64_train512` and `tinyllama_dolly64_eval128` were still absent locally.
- Eval-only smoke command used the same smoke manifest with `evalOnly=true`:

```powershell
C:\Users\wentaodai\.gradle\wrapper\dists\gradle-8.7-bin\bhs2wmbdwecv87pi65oeuq5iu\gradle-8.7\bin\gradle.bat :coordinator:runPreparedExperiment --offline --no-daemon --args="127.0.0.1 50051 data/sft_requests/tinyllama_dolly64_smoke/requests.jsonl 0 1 debug_runs/eval-smoke/results.csv eval-smoke 1 0 true"
```

- Eval-only CSV result path: `debug_runs/eval-smoke/results.csv`.
- Eval-only CSV row: `request_id=eval-smoke-000001`, `record_index=1`, `dataset_index=1`, `valid_labels=3`, `eval_only=true`, `success=true`, `terminal=true`, `elapsed_ms=19551`, `output_hidden_bytes=262144`, `output_shift_log_p_bytes=4096000`, `local_loss=6.8660183`, `token_correct=0`, `token_count=3`, `token_accuracy=0.0`.
- Coordinator events for `eval-smoke-000001` prove eval-only behavior:
  - stage 0 `LOCAL_COMPLETED`: `runtime=TrainingModule`, `inputs=4`, `localMs=4071`, `loss=11.869716`, `evalOnly=true`, `optimizerStepApplied=false`, `executeMs=4066`;
  - stage 1 `LOCAL_COMPLETED`: `runtime=TrainingModule`, `inputs=5`, `localMs=5032`, `loss=6.8660183`, `evalOnly=true`, `optimizerStepApplied=false`, `executeMs=4866`;
  - stage 0 completion reported downstream success with `forwardMs=11191` and `totalStageMs=15808`.
- Synchronized train/perf-monitor smoke command shape:

```powershell
# Monitor window starts first; Gradle request starts 5 seconds later.
.\tools\android\monitor_perf.ps1 -Phase train-smoke-monitored -IntervalSec 1 -MeminfoEverySamples 2 -DurationSec 75
C:\Users\wentaodai\.gradle\wrapper\dists\gradle-8.7-bin\bhs2wmbdwecv87pi65oeuq5iu\gradle-8.7\bin\gradle.bat :coordinator:runPreparedExperiment --offline --no-daemon --args="127.0.0.1 50051 data/sft_requests/tinyllama_dolly64_smoke/requests.jsonl 0 1 debug_runs/train-smoke-monitored/results.csv train-smoke-monitored 1 0 false"
```

- Monitored train CSV result path: `debug_runs/train-smoke-monitored/results.csv`.
- Monitored train CSV row: `request_id=train-smoke-monitored-000001`, `record_index=1`, `dataset_index=1`, `valid_labels=3`, `eval_only=false`, `success=true`, `terminal=true`, `elapsed_ms=19794`, `output_hidden_bytes=262144`, `output_shift_log_p_bytes=4096000`, `local_loss=6.8660183`, `token_correct=0`, `token_count=3`, `token_accuracy=0.0`.
- Coordinator events for `train-smoke-monitored-000001` prove training behavior:
  - stage 0 `LOCAL_COMPLETED`: `runtime=TrainingModule`, `inputs=4`, `localMs=4037`, `loss=11.869716`, `evalOnly=false`, `optimizerStepApplied=true`, `executeMs=4031`;
  - stage 1 `LOCAL_COMPLETED`: `runtime=TrainingModule`, `inputs=5`, `localMs=5169`, `loss=6.8660183`, `evalOnly=false`, `optimizerStepApplied=true`, `executeMs=4994`;
  - stage 0 completion reported downstream success with `forwardMs=10944` and `totalStageMs=15206`.
- Perf-monitor output for the synchronized train smoke: `debug_runs/perf-monitor-20260526-141805/`.
  - `samples.csv`: 26 samples per phone from `2026-05-26T14:18:10.8857021+08:00` to `2026-05-26T14:19:23.0948696+08:00`.
  - `events.csv`: header only; focused timing logcat was not captured into this CSV during this run, so use coordinator request events for stage timing.
  - NX `91260221021D`: battery level stayed `100 -> 100`, battery temp `31.0C -> 31.0C`, `current_now` samples existed but varied from `-1518000` to `203000` uA while the device reported charging/full-style status. Do not treat this as paper-grade energy without a power setup decision.
  - Lenovo `ZY22G2HC5C`: battery level stayed `100 -> 100`, battery temp `31.0C -> 31.0C`, no numeric `current_now` samples were available from `/sys/class/power_supply/battery/current_now`.
- Throughput from the synchronized train smoke: one prepared request in `19.794s`, about `0.0505 requests/s`; valid-label throughput `3 / 19.794 = 0.152 label tokens/s`. This is a smoke number, not a steady-state throughput result.
- Current remaining blocker for full formal experiment: generate or copy `tinyllama_dolly64_eval128` and `tinyllama_dolly64_train512` into `data/sft_requests/`, then run eval-before, train, and eval-after without restarting either Android app process.

Server-side prepared-data command for a larger run:

```bash
python tools/data/prepare_lora_sft_requests.py \
  --model_name tinyllama \
  --dataset dolly \
  --seq_len 64 \
  --limit 64 \
  --attention_mask causal \
  --mask_prompt \
  --min_valid_labels 1 \
  --request_prefix dolly-lora-sft \
  --output_dir data/sft_requests/tinyllama_dolly64_train
```

Interpretation boundary:

- With the runner and timing logs, we can now run a multi-request, multi-step Android BP-free system experiment.
- Do not call it a full model-quality training experiment unless we also define dataset size, epochs, quality metric, and a way to persist/export updated mobile weights or adapter state.
- Current Android runtime can keep optimizer/model state in memory while the app process and cached runtime stay alive, but it still does not save updated weights back to disk.

The older `tinyllama-mainline-20260524-2349` request only proved a `Module.execute()` forward pipeline request at fixed `seqLen=64`; it did not prove the full training experiment. The newer `tinyllama-training-20260525-1542` request used training PTEs and completed through both phones.

## Current Devices

ADB serials:

- `91260221021D`: NX809J, stage 0, expected host `192.168.137.211`, expected port `26052`
- `ZY22G2HC5C`: Lenovo_L71091, stage 1, expected host `192.168.137.124`, expected port `26052`

Coordinator:

- gRPC: `127.0.0.1:50051`
- admin HTTP: `http://127.0.0.1:18080`
- config: `coordinator/config/pipeline.json`

For the current LoRA smoke path, local config points to:

- stage 0: `model/tinyllama_lora_chunk_0.pte`
- stage 1: `model/tinyllama_lora_chunk_1.pte`

The non-LoRA baseline config points to:

- stage 0: `model/tinyllama_chunk_0.pte`
- stage 1: `model/tinyllama_chunk_1.pte`

Verified 2026-05-25 00:39:

- Coordinator admin is reachable at `http://127.0.0.1:18080`.
- Routing epoch is `182`.
- `liveNodeCount=2`, `offlineStageCount=0`, `drainedStageCount=0`.
- Stage 0 is `NX809J`, node `47`, host `192.168.214.103:26052`, app pid `25557`.
- Stage 1 is `Lenovo_L71091`, node `46`, host `192.168.214.59:26052`, app pid `25922`.
- Stage 0 next hop is node `46` at `192.168.214.59:26052`.

Verified 2026-05-25 14:00:

- Coordinator admin is reachable at `http://127.0.0.1:18080`.
- Routing epoch is `200`.
- `liveNodeCount=2`, `offlineStageCount=0`, `drainedStageCount=0`.
- Stage 0 is `NX809J`, node `50`, host `192.168.214.103:26052`, app pid `25557`.
- Stage 1 is `Lenovo_L71091`, node `49`, host `192.168.214.59:26052`, app pid `25922`.
- Stage 0 next hop is node `49` at `192.168.214.59:26052`.
- Infrastructure is ready for a request. Earlier blocking condition "non-`_inf` equals `_inf`" is now known to be misleading because local `_inf` files are contaminated with the BP-free hash.

ADB serials verified 2026-05-25 00:39:

- `91260221021D`: product/model/device `NX809J`
- `ZY22G2HC5C`: product `halo`, model `Lenovo_L71091`, device `halo`

## Experiment Record Policy

Do not write a detailed "how it was run through" narrative from memory.

Use this split:

- Verified facts: only what is visible in code, config, coordinator request history, logcat, tombstones, or command output.
- Inference: explicitly label as inference and explain what evidence it comes from.
- Missing details: leave as `NEEDS_RECONSTRUCTION` instead of filling gaps from memory.

Current memory gap:

- The exact historical steps that first brought the two-phone chain up are not reliable in current context.
- Treat the two-phone chain as an older infrastructure milestone unless current coordinator logs / request history are collected again.
- For paper or experiment logs, reconstruct exact request id, command, device pids, artifact hashes, and coordinator events from logs or debug bundles before writing them as protocol details.

## Known Evidence

TinyLlama non-`_inf` files currently inspected:

- `model/tinyllama_chunk_0.pte`: size `1143135508`, SHA256 `C7A31652D9116B0848D579057478151DDDCC9FE0A8528C639FD905CA4BC31CBD`, no `__et_training`, no `aten::empty_permuted`
- `model/tinyllama_chunk_1.pte`: size `1143137700`, SHA256 `149D774EC8B2F4605DA85CD3F20909CBA3A4C7DB712FFA91A779101807CE338E`, no `__et_training`, no `aten::empty_permuted`

Critical artifact fact:

- `model/tinyllama_chunk_0.pte` is byte-identical to `model/tinyllama_chunk_0_inf.pte`.
- `model/tinyllama_chunk_1.pte` is byte-identical to `model/tinyllama_chunk_1_inf.pte`.
- Fresh server-side comparison from `compare_relay_vs_bpfree_20260525-141326` showed:
  - relay chunk 0 SHA256 `ffcbf6a119f350d06f34ee52eb53e03e1b407b9d757f75d8e56c6a1d7effd8fb`
  - BP-free chunk 0 SHA256 `c7a31652d9116b0848d579057478151dddcc9fe0a8528c639fd905ca4bc31cbd`
  - relay chunk 1 SHA256 `08ef26884b58fa9c4aaea4c4f4460c21a77a338efd118fcd7dbfe609539ac3db`
  - BP-free chunk 1 SHA256 `149d774ec8b2f4605da85cd3f20909cba3a4c7db712ffa91a779101807ce338e`
- Therefore the current local non-`_inf` hashes match freshly generated BP-free forward-only local CE/KD artifacts, and the local `_inf` files are not a clean relay baseline.

Phone artifact hashes verified 2026-05-25 00:39:

- NX809J stage 0 local file `files/shards/tinyllama_chunk_0.pte` SHA256 `c7a31652d9116b0848d579057478151dddcc9fe0a8528c639fd905ca4bc31cbd`, matching coordinator `model/tinyllama_chunk_0.pte`.
- Lenovo stage 1 local file `files/shards/tinyllama_chunk_1.pte` SHA256 `149d774ec8b2f4605da85cd3f20909cba3a4c7db712ffa91a779101807ce338e`, matching coordinator `model/tinyllama_chunk_1.pte`.
- Both phones contain stale extra shard files from earlier runs. Do not infer the active artifact from directory listing alone; use coordinator `modelShardId` plus matching filename/hash.

Phone artifact hashes rechecked 2026-05-25 14:00:

- NX809J stage 0 local file `files/shards/tinyllama_chunk_0.pte` SHA256 `c7a31652d9116b0848d579057478151dddcc9fe0a8528c639fd905ca4bc31cbd`, still matching the local `model/tinyllama_chunk_0.pte` and `_inf`.
- Lenovo stage 1 local file `files/shards/tinyllama_chunk_1.pte` SHA256 `149d774ec8b2f4605da85cd3f20909cba3a4c7db712ffa91a779101807ce338e`, still matching the local `model/tinyllama_chunk_1.pte` and `_inf`.
- These hashes now match the freshly generated forward local CE/KD/belief outputs from the server comparison above. It is OK to test the forward belief path with these hashes, but not OK to call it a full training run. Do not compare against local `_inf`; compare against freshly generated relay-only hashes if a baseline is needed.

Verified 2026-05-25 15:06:

- Coordinator admin is reachable at `http://127.0.0.1:18080`.
- Routing epoch is `200`.
- `liveNodeCount=2`, `offlineStageCount=0`, `drainedStageCount=0`.
- Stage 0 is `NX809J`, node `50`, host `192.168.214.103:26052`, app pid `25557`.
- Stage 1 is `Lenovo_L71091`, node `49`, host `192.168.214.59:26052`, app pid `25922`.
- Stage 0 next hop is node `49` at `192.168.214.59:26052`.
- Both Android processes are alive. No new NX tombstone beyond the old 2026-05-24 23:38 tombstone was listed. Lenovo tombstone listing is still permission-denied.
- Debug bundle: `debug_runs/android-20260525-150643` (ignored by git).

Verified 2026-05-25 15:15 after reinstalling the rebuilt debug APK:

- Coordinator admin is reachable at `http://127.0.0.1:18080`.
- Routing epoch is `210`.
- `liveNodeCount=2`, `inactiveNodeCount=0`, `offlineStageCount=0`, `drainedStageCount=0`.
- Stage 0 is `NX809J`, node `51`, host `192.168.214.103:26052`, app pid `16506`.
- Stage 1 is `Lenovo_L71091`, node `52`, host `192.168.214.59:26052`, app pid `14073`.
- Stage 0 next hop is node `52` at `192.168.214.59:26052`.
- Debug bundle: `debug_runs/android-20260525-151510` (ignored by git).

Fixed a worker-active self-lock bug on 2026-05-25:

- Coordinator returns `"PAUSED"` when a heartbeat itself reports `isActive=false`.
- Android previously treated `"PAUSED"` as a command to keep setting itself inactive.
- After reinstall, NX reproduced this as `isActive=false` while the process was alive and heartbeating.
- Fix: Android now treats `"PAUSED"` as informational; only `"DRAIN"` actively disables local scheduling. This restored both stages to active.

Verified 2026-05-25 15:44 after replacing `model/tinyllama_chunk_0.pte` and `model/tinyllama_chunk_1.pte` with training PTEs:

- Local PTE inspection:
  - `model/tinyllama_chunk_0.pte`: size `1143289984`, SHA256 `d936d5cd8fd24dd027ee52d638723e4bd66267a4bfd68e8c701f19b2f093b6ab`, `__et_training=3`, `aten::empty_permuted=0`
  - `model/tinyllama_chunk_1.pte`: size `1143294592`, SHA256 `ea0b4a9705df241ac72292f0a336163afa07eaba029209c67fb6f5f6f2ae33c7`, `__et_training=3`, `aten::empty_permuted=0`
  - `_inf` files remain markerless forward artifacts: chunk 0 SHA256 `c7a31652d9116b0848d579057478151dddcc9fe0a8528c639fd905ca4bc31cbd`; chunk 1 SHA256 `149d774ec8b2f4605da85cd3f20909cba3a4c7db712ffa91a779101807ce338e`.
- Ran `POST /api/v1/routing/reload`, restarted both Android workers, and both phones redownloaded the new training PTEs.
- Phone hashes:
  - NX stage 0 `files/shards/tinyllama_chunk_0.pte`: `d936d5cd8fd24dd027ee52d638723e4bd66267a4bfd68e8c701f19b2f093b6ab`
  - Lenovo stage 1 `files/shards/tinyllama_chunk_1.pte`: `ea0b4a9705df241ac72292f0a336163afa07eaba029209c67fb6f5f6f2ae33c7`
- Coordinator status after deployment: routing epoch `227`, `liveNodeCount=2`, `inactiveNodeCount=0`, both stages on port `26052`.
- Submitted true training-PTE request with:
  - `requestId=tinyllama-training-20260525-1542`
  - `modelPreset=tinyllama`, `seqLen=64`, `batchSize=1`, `chunkIdx=0`
- Stored payload decode for `tinyllama-training-20260525-1542`: `hidden_states` float32 `[1,64,2048]`, `attention_mask` float32 `[1,1,64,64]`, `position_ids` int64 `[1,64]`, `labels` int64 `[1,64]`, empty `shift_log_p_prev`.
- Result: success. `message=Stage 1 finished request tinyllama-training-20260525-1542`, `processedStageId=1`, `processedChunkIdx=1`, `terminal=true`, `outputHiddenBytes=262144`.
- Coordinator events show stage 0 node `51` received chunk 0, completed local shard, forwarded to `192.168.214.59:26052`; stage 1 node `52` received chunk 1, completed local shard, and returned terminal success.
- Lenovo logcat explicitly confirms training runtime and optimizer: `Creating ExecuTorch SGD optimizer ... parameters=47 lr=1.0E-5`, then `TrainingModule.executeForwardBackward() and SGD.step() succeeded ... with 5 inputs gradients=47`.
- NX focused logcat still returns no app-tag lines on this device, but stage 0 evidence is: phone hash is a training PTE with `__et_training=3`; `NativeShardRunner` can only load such an artifact through `TrainingModule`; coordinator recorded stage 0 `LOCAL_COMPLETED` and `FORWARDING`.
- No new NX tombstone beyond the old 2026-05-24 23:38 tombstone was listed. Lenovo tombstone listing is still permission-denied.
- Debug bundle: `debug_runs/android-20260525-154400` (ignored by git).

Important observed requests:

- `tinyllama-mainline-20260524-2349`: stored payload proves `batch=1`, `seqLen=64`, hidden shape `[1,64,2048]`, labels shape `[1,64]`, empty `shift_log_p_prev`, default `chunkIdx=0`; completed through both phones.
- `tinyllama-mainline-20260524-2337`: stored payload proves `batch=1`, `seqLen=8`, hidden shape `[1,8,2048]`, labels shape `[1,8]`, empty `shift_log_p_prev`, default `chunkIdx=0`; failed with coordinator dispatch EOF after stage 0 received it.

Event chain for `tinyllama-mainline-20260524-2349`:

- 2026-05-24 23:48:29 +08:00 coordinator accepted request and dispatched to stage 0 node 47.
- Stage 0 node 47 received chunk 0, completed local shard, and forwarded to `192.168.214.59:26052`.
- Stage 1 node 46 received chunk 1, completed local shard, and returned terminal success.
- Final message: `Stage 1 finished request tinyllama-mainline-20260524-2349`.

Event chain for `tinyllama-mainline-20260524-2337`:

- 2026-05-24 23:38:21 +08:00 coordinator accepted request and dispatched to stage 0 node 45.
- Stage 0 node 45 received chunk 0.
- Coordinator recorded failure: `Coordinator dispatch to stage 0 failed: Unexpected end of file from server`.

NEEDS_RECONSTRUCTION:

- exact command history and process lifecycle for the older chain-up success
- exact artifact hashes on each phone at that time
- exact coordinator request event dump for any claim stronger than "a forward pipeline request completed"

Interpretation so far:

- The `seqLen=8` crash is more consistent with fixed-shape export / input-shape mismatch than with "TinyLlama cannot run at all".
- The `seqLen=64` forward success is not a training-success claim.
- The export scripts are now validated enough for the forward local CE/KD/belief path: fresh relay-only and fresh local-loss/belief exports differ. The local `_inf` baseline is contaminated, not proof that export is broken.
- The actual training path remains incomplete until chunk-local backward plus optimizer update is wired and verified.

Important tombstones previously seen:

- `tombstone_04`, NX809J, 2026-05-24 23:31:07 +08:00, `SIGABRT`, abort message `ptr`, stack includes `org.pytorch.executorch.training.TrainingModule.executeForwardBackward` and `NativeShardRunner$TrainingLoadedRuntime.execute`.
- `tombstone_05`, NX809J, 2026-05-24 23:38:23 +08:00, `SIGABRT`, abort message `ptr`, stack includes `org.pytorch.executorch.Module.execute` and `NativeShardRunner$InferenceLoadedRuntime.execute`.
- Lenovo tombstone listing via normal `adb shell ls /data/tombstones` returns `Permission denied`.

Do not present either tombstone alone as the root cause. Use the newest logs and request details before claiming a cause.

Debug bundle:

- Latest local evidence bundle: `debug_runs/android-20260525-154400` (ignored by git).

## Current Controversial Commit

HEAD as of this note:

- `44a9081 Prevent TrainingModule fallback for inference PTEs`

What it actually did:

- It did not remove the explicit training path for PTEs that contain `__et_training`.
- It removed fallback from markerless PTEs to `TrainingModule` after `Module.load()` failure.

Why this is controversial:

- The commit name says "inference PTEs", which is misleading for BP-free non-`_inf` artifacts.
- It can make future agents think the mainline is inference/link testing again.

Before changing this again, decide explicitly whether the current artifact should run with:

- `Module.execute()` because it is a markerless forward graph carrying local loss outputs, or
- `TrainingModule.executeForwardBackward()` because it was exported by the joint forward/backward path and has `__et_training` metadata.

Do not randomly switch to LoRA/PEFT or rewrite the algorithm.

## LoRA / SFT Task Scaffold

Status as of 2026-05-25 16:18:

- LoRA is optional and does not change the default full-parameter training export. `tools/export/sid_export_mobile.py` keeps `--lora_rank 0` by default.
- When LoRA is enabled, the exporter wraps selected `nn.Linear` children with local LoRA adapters, freezes the base chunk weights, and leaves only adapter tensors trainable for Android `TrainingModule.executeForwardBackward()` plus local `SGD.step()`.
- Default LoRA wrapper: `tools/export/export_lora_tinyllama.sh`, with `CHUNK_IDX=0,1`, `SEQ_LEN=64`, `TRANSPORT_DTYPE=float16`, `ARTIFACT_PREFIX=tinyllama_lora`, `LORA_RANK=4`, `LORA_ALPHA=16`, `LORA_TARGETS=q_proj,v_proj`.
- This still preserves the BP-free system distinction: stage-to-stage traffic is forward-only hidden states and belief/log-prob signals. LoRA only reduces which local chunk parameters are updated.
- Real text data scaffold: `tools/data/prepare_lora_sft_requests.py`.
  - Dataset presets: `dolly` -> `databricks/databricks-dolly-15k`, `alpaca` -> `tatsu-lab/alpaca`, `gsm8k` -> `openai/gsm8k` with `main/train`.
  - Current mobile contract starts stage 0 from `hidden_states`, not token ids. Therefore the script tokenizes on the server, runs the model input embedding, and writes prepared tensor files plus `requests.jsonl`.
  - Default attention mask is now `causal` for SFT. Use `--attention_mask zero` only for reproducing the older synthetic demo shape.
  - Padding label handling was fixed so `pad_token=eos_token` does not mask a real EOS token; only positions added as padding are set to `-100`.
- Coordinator submit scaffold: `coordinator/src/main/kotlin/com/example/sid_coordinator/SubmitPreparedRequestMain.kt`, exposed through Gradle task `:coordinator:runSubmitPreparedRequest`.
- Generated request data under `data/sft_requests/` is ignored by git.

Validation performed locally:

- `python -m py_compile tools\export\sid_export_mobile.py tools\data\prepare_lora_sft_requests.py`: passed.
- `:coordinator:compileKotlin --offline --no-daemon`: passed when run outside the sandbox because sandboxed Gradle failed with a JVM loopback error.
- `python tools\data\prepare_lora_sft_requests.py --help` and `python tools\export\sid_export_mobile.py --help` did not run in this Windows shell because this local Python has no `torch`; this is an environment limitation, not a syntax result. Run these on the server export environment.
- Export dependency fix: `requirements-export.txt` now pins `torch==2.11.*` because `executorch==1.2.0` declares `torch>=2.11.0`; old `torch==2.9.0` caused pip resolver conflicts and `torchao` compatibility warnings.

Do not claim LoRA mobile training has completed yet. The completed phone run at request `tinyllama-training-20260525-1542` was the non-LoRA training-PTE path.

Verified 2026-05-25 17:41 after copying newly generated LoRA PTEs into `model/`:

- Local PTE inspection:
  - `model/tinyllama_lora_chunk_0.pte`: size `1143652096`, SHA256 `640cf3f4a92a89ad011453eea1bd1d6b430e40a39c351e2ae616b05692a7b5a1`, `__et_training=3`, `aten::empty_permuted=0`
  - `model/tinyllama_lora_chunk_1.pte`: size `1143656576`, SHA256 `6b2f7eca349b2b2df374b91573a96428251cba6153b02a6d5f50c2d84a96bc11`, `__et_training=3`, `aten::empty_permuted=0`
- `coordinator/config/pipeline.json` was switched locally to `tinyllama_lora_chunk_0/1`, then `POST /api/v1/routing/reload` succeeded.
- Direct ADB caching was used because coordinator admin HTTP became unresponsive while serving a large shard download:
  - `adb push model\tinyllama_lora_chunk_0.pte /data/local/tmp/tinyllama_lora_chunk_0.pte`, then `run-as com.example.sid_trainer cp ... files/shards/tinyllama_lora_chunk_0.pte`
  - same flow for `tinyllama_lora_chunk_1.pte`
- Phone cache hashes matched the local hashes above.
- Restarted both Android workers. Coordinator status showed `liveNodeCount=2`, `inactiveNodeCount=0`, node `56` for NX stage 0 and node `57` for Lenovo stage 1. During the successful request, stage 0 forwarded to Lenovo at `192.168.214.59:39899`; this dynamic Lenovo port was route-ready and is not itself a failure.
- Submitted LoRA synthetic smoke request:
  - `requestId=tinyllama-lora-smoke-20260525-1730`
  - `modelPreset=tinyllama`, `seqLen=64`, `batchSize=1`, `chunkIdx=0`
- Result: success. `message=Stage 1 finished request tinyllama-lora-smoke-20260525-1730`, `processedStageId=1`, `processedChunkIdx=1`, `terminal=true`, `outputHiddenBytes=262144`.
- Coordinator events show stage 0 node `56` received chunk 0, completed local shard, forwarded to `192.168.214.59:39899`, stage 1 node `57` received chunk 1, completed local shard, and returned terminal success.
- Lenovo logcat explicitly confirms LoRA training runtime and optimizer: `Creating ExecuTorch SGD optimizer ... tinyllama_lora_chunk_1.pte parameters=20 lr=1.0E-5`, then `TrainingModule.executeForwardBackward() and SGD.step() succeeded ... with 5 inputs gradients=20`.
- NX focused logcat still returns no app-tag lines on this device, but stage 0 evidence is: phone hash matches LoRA training PTE with `__et_training=3`; `NativeShardRunner` can only load such an artifact through `TrainingModule`; coordinator recorded stage 0 `LOCAL_COMPLETED` and `FORWARDING`.
- Debug bundle: `debug_runs/android-20260525-174128` (ignored by git).

LoRA real SFT request is still not completed because `data/sft_requests/` is not present on the Windows coordinator machine. Generate or copy a prepared request set before running `:coordinator:runSubmitPreparedRequest`.

Verified 2026-05-25 19:24 for real prepared Dolly SFT requests:

- Prepared request set is present at `data/sft_requests/tinyllama_dolly64_smoke`.
- `metadata.json` shows `dataset=databricks/databricks-dolly-15k`, `seq_len=64`, `limit=2`, `attention_mask=causal`, `mask_prompt=true`.
- Label audit:
  - record index `0`, `request_id=dolly-lora-smoke-000000`: `prompt_token_count=64`, `valid_labels=0`; do not use this record as a training-signal proof because the prompt fills the fixed 64-token window.
  - record index `1`, `request_id=dolly-lora-smoke-000001`: `prompt_token_count=22`, `valid_labels=3`; this is the valid smoke record.
- First attempt with relative manifest path failed because Gradle `JavaExec` ran from the `coordinator/` module directory and could not find `data/sft_requests/...`.
- Workaround run with absolute manifest path succeeded:
  - `requestId=dolly-lora-prepared-20260525-1909`
  - manifest: absolute path to `data\sft_requests\tinyllama_dolly64_smoke\requests.jsonl`
  - record index `1`
  - result: `success=true`, `message=Stage 1 finished request dolly-lora-prepared-20260525-1909`, `processedStageId=1`, `processedChunkIdx=1`, `terminal=true`, `outputHiddenBytes=262144`
  - decoded stored payload: `hidden_states` float32 `[1,64,2048]`, `attention_mask` float32 `[1,1,64,64]`, `position_ids` int64 `[1,64]`, `labels` int64 `[1,64]`, empty `shift_log_p_prev`
  - coordinator events: stage 0 node `60` received chunk 0, `LOCAL_COMPLETED`, forwarded to `192.168.214.59:26052`; stage 1 node `58` received chunk 1, `LOCAL_COMPLETED`, terminal `COMPLETED`
  - Lenovo logcat confirms LoRA training runtime: `TrainingModule.executeForwardBackward() and SGD.step() succeeded ... tinyllama_lora_chunk_1.pte with 5 inputs gradients=20`
- Fix added after that run: `coordinator/build.gradle.kts` now sets `workingDir = rootProject.projectDir` for `runSubmitPreparedRequest`.
- Relative-path verification run succeeded after the fix:
  - command shape: `:coordinator:runSubmitPreparedRequest --args="127.0.0.1 50051 data/sft_requests/tinyllama_dolly64_smoke/requests.jsonl 1 dolly-lora-prepared-relpath-20260525-1919"`
  - result: `success=true`, `message=Stage 1 finished request dolly-lora-prepared-relpath-20260525-1919`
  - coordinator events again show stage 0 node `60` local completion and forwarding, then stage 1 node `58` local completion and terminal success.
  - Lenovo logcat again confirms `TrainingModule.executeForwardBackward() and SGD.step() succeeded ... gradients=20`.
- Debug bundle after the prepared SFT run: `debug_runs/android-20260525-192457` (ignored by git).

Verified 2026-05-26 after a temporary network disconnect:

- Continue the active run instead of restarting if the Android worker processes are still alive. The LoRA optimizer state is in the app processes, so restarting workers during train invalidates the before/after comparison.
- Current valid formal run pointer: `debug_runs/CURRENT_FORMAL_RUN.txt` -> `debug_runs/formal-restart-20260526-185723`.
- Previous formal run `debug_runs/formal-20260526-171445` is invalid/aborted: `eval-before` submitted 128 requests, but Wi-Fi/IP changed mid-run and 53 requests failed with `Stage 0 has no live worker.` Do not use it for paper metrics.
- Current network route uses:
  - coordinator/admin host: `192.168.56.35`
  - stage 0 NX: `192.168.56.103:26052`, model `tinyllama_lora_chunk_0`
  - stage 1 Lenovo: `192.168.56.59:26052`, model `tinyllama_lora_chunk_1`
- The current valid run has completed `eval-before` successfully:
  - `debug_runs/formal-restart-20260526-185723/eval-before/results.csv`
  - 128/128 successful requests, 129 CSV lines including the header.
- The current `train` stage is still running and must not be interrupted:
  - command shape: `:coordinator:runPreparedExperiment --offline --no-daemon --args="127.0.0.1 50051 data/sft_requests/tinyllama_dolly64_train512/requests.jsonl 0 512 debug_runs/formal-restart-20260526-185723/train/results.csv train 1 0 false"`
  - progress checked after reconnect: 194/512 completed, 0 failure/build-error lines, both phones live.
  - `results.csv` may not exist until the Java process exits; while running, use `train/stdout.log` for progress.
- This run later became invalid because the network/worker dropped again:
  - `train/results.csv`: 512 submitted, 399 success, 113 failure.
  - Last successful request: `train-000398`.
  - First failed request: `train-000399`, message `Coordinator dispatch to stage 0 failed: Read timed out`, elapsed about 300 seconds.
  - Remaining failed requests reported `Stage 0 has no live worker.`
  - NX/stage 0 app process elapsed time was only about 28 minutes after recovery, while Lenovo/stage 1 app had been alive for about 2 hours 51 minutes. Therefore stage 0 was restarted and lost in-memory LoRA optimizer state. Do not tail-resume this run and do not use its `eval-after` for accuracy.
- Recovery rule after this failure: restart both Android workers to reset both chunks to the same initial LoRA state, start a new formal run, and keep `stopOnFailure=true`.
- `RunPreparedExperimentMain.kt` now has an optional 11th argument, `stopOnFailure`, defaulting to `true`. Formal commands should leave this default on or pass `true`; pass `false` only for deliberate robustness sweeps where collecting all failures matters.
- After a clean train reaches 512/512 success, immediately run `eval-after` without restarting Android workers:
  - `:coordinator:runPreparedExperiment --offline --no-daemon --args="127.0.0.1 50051 data/sft_requests/tinyllama_dolly64_eval128/requests.jsonl 0 128 <runDir>/eval-after/results.csv eval-after 1 0 true true"`

Formal run update 2026-05-27:

- Current valid run: `debug_runs/formal-clean-20260526-215720`.
- Android worker processes stayed alive through training:
  - stage 0 NX PID `9232`
  - stage 1 Lenovo PID `31464`
- `eval-before` completed cleanly:
  - `debug_runs/formal-clean-20260526-215720/eval-before/results.csv`
  - 128/128 success
  - average elapsed `14187.05 ms`
  - average local loss `5.754919`
  - token accuracy `0.035949` over 3950 valid shifted tokens
- Training completed as a clean 512-request sequence, but with one fail-fast coordinator heartbeat interruption:
  - front run `train/results.csv`: indices 0..340 succeeded, index 341 failed fast with `Stage 0 has no live worker.`
  - both Android worker PIDs were unchanged, so the LoRA optimizer state was still in memory.
  - resume run `train-resume-341/results.csv`: indices 341..511 completed successfully.
  - combined official training CSV: `debug_runs/formal-clean-20260526-215720/train-combined/results.csv`
  - combined train: 512/512 success, no missing indices
  - average elapsed `14172.34 ms`
  - average local loss `5.647889`
  - token accuracy `0.038854` over 16369 valid shifted tokens
- Train performance monitor:
  - `debug_runs/formal-clean-20260526-215720/train/perf-monitor-20260526-225135`
  - 2188 battery/memory/thermal samples and 420 timing events at train stop.
  - Energy numbers are coarse only because both phones reported 100% battery / charging-full states; use timing, memory, thermal, and relative current traces cautiously.
- A partial `eval-after` was started and interrupted by user request after one successful eval-only request. It should not be used for final metrics.
- Full official `eval-after` is running in:
  - `debug_runs/formal-clean-20260526-215720/eval-after-full`
  - command shape: `:coordinator:runPreparedExperiment --offline --no-daemon --args="127.0.0.1 50051 data/sft_requests/tinyllama_dolly64_eval128/requests.jsonl 0 128 debug_runs/formal-clean-20260526-215720/eval-after-full/results.csv eval-after-full 1 0 true true"`
  - Do not restart Android workers until this finishes.
- `eval-after-full` did not complete:
  - It failed on request `eval-after-full-000000`, record index 0.
  - `results.csv`: 1 submitted, 0 success, 1 failure.
  - failure message: `Coordinator dispatch to stage 0 failed: Unexpected end of file from server`
  - NX/stage 0 app process `9232` disappeared immediately after the failed request.
  - Android `dumpsys activity exit-info com.example.sid_trainer` reports `2026-05-27 00:57:20.340`, pid `9232`, reason `5 (APP CRASH(NATIVE))`, status `6`, RSS about `2.7GB`.
  - Lenovo/stage 1 process `31464` remained alive, but the trained state is no longer consistent because stage 0 was lost.
  - Debug bundle: `debug_runs/formal-clean-20260526-215720/android-20260527-010943`.
- Do not restart only stage 0 and call that a valid eval-after. Without a persisted LoRA/optimizer checkpoint, restarting stage 0 would reload the initial PTE while stage 1 still has trained in-memory LoRA state.
- Current conclusion for this run:
  - `eval-before` and 512-request training throughput/loss metrics are valid.
  - before/after eval accuracy is not valid because stage 0 native-crashed before eval-after could run.
  - The next fix should add a way to persist trained LoRA state/checkpoints before eval, or reduce/clear memory pressure before eval in a way that preserves trained weights.

## Commands That Should Be Used

Do not use `rg` in this Windows workspace. It repeatedly fails with Access Denied here. Use PowerShell `Get-ChildItem` and `Select-String`.

Export true BP-free training PTEs on the Linux server:

```bash
CHUNK_IDX=0,1 OUTPUT_DIR=model bash tools/export/export_bpfree_tinyllama.sh
```

Equivalent explicit command:

```bash
python tools/export/sid_export_mobile.py \
  --model_name tinyllama \
  --num_chunks 4 \
  --chunk_idx 0,1 \
  --seq_len 64 \
  --batch_size 1 \
  --transport_dtype float16 \
  --artifact_prefix tinyllama \
  --output_dir model
```

Forward belief/local-loss only, without local backward/SGD:

```bash
CHUNK_IDX=0,1 OUTPUT_DIR=model bash tools/export/export_forward_belief_tinyllama.sh
```

Inspect PTE markers:

```powershell
python tools\export\inspect_pte.py model\tinyllama_chunk_0.pte model\tinyllama_chunk_1.pte
```

Export LoRA TinyLlama training PTEs on the Linux server:

```bash
CHUNK_IDX=0,1 OUTPUT_DIR=model bash tools/export/export_lora_tinyllama.sh
```

Prepare a small Dolly SFT smoke set on the Linux server:

```bash
python tools/data/prepare_lora_sft_requests.py \
  --model_name tinyllama \
  --dataset dolly \
  --seq_len 64 \
  --limit 2 \
  --attention_mask causal \
  --request_prefix dolly-lora-smoke \
  --output_dir data/sft_requests/tinyllama_dolly64_smoke
```

Submit one prepared SFT request from Windows/coordinator:

```powershell
.\gradlew :coordinator:runSubmitPreparedRequest --args="127.0.0.1 50051 data/sft_requests/tinyllama_dolly64_smoke/requests.jsonl 0"
```

Run a prepared experiment with fail-fast enabled:

```powershell
C:\Users\wentaodai\.gradle\wrapper\dists\gradle-8.7-bin\bhs2wmbdwecv87pi65oeuq5iu\gradle-8.7\bin\gradle.bat :coordinator:runPreparedExperiment --offline --no-daemon --args="127.0.0.1 50051 data/sft_requests/tinyllama_dolly64_train512/requests.jsonl 0 512 debug_runs/<run>/train/results.csv train 1 0 false true"
```

Check coordinator:

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:18080/api/v1/status | Select-Object -ExpandProperty Content
```

Submit the current target request:

```powershell
.\tools\android\submit_tinyllama_request.ps1 -SeqLen 64 -BatchSize 1
```

Check phones and recent logs:

```powershell
.\tools\android\check_status.ps1
```

Check LAN topology before a long run:

```powershell
.\tools\android\check_network_topology.ps1
```

This verifies the three network directions required by the current direct pipeline:

- Android workers can reach the coordinator on `50051` and artifact/admin HTTP on `18080`.
- The coordinator PC can reach each Android worker server port.
- The upstream stage phone can reach the next-stage phone server port.

Windows Mobile Hotspot can replace a phone hotspot and is often more stable if the PC is kept awake, but it must be validated with this script. If the hotspot isolates Wi-Fi clients, the phones may both reach the PC while stage-to-stage forwarding still fails. A dedicated small router is the most stable option for long experiments because it supports DHCP reservations/static IPs and avoids laptop/phone hotspot power-management behavior.

Collect a debug bundle:

```powershell
.\tools\android\collect_debug_bundle.ps1
```

Decode stored request payload shapes:

```powershell
python tools\android\decode_request_payload.py tinyllama-mainline-20260524-2349
```

## Debugging Discipline

- First read this file, then inspect current `git status`.
- Treat `_inf` success as old compatibility evidence, not the current research target.
- Treat two-phone link success as older infrastructure evidence, not the mainline result.
- Do not run `_inf` artifacts again unless explicitly asked or using them as a controlled baseline.
- Keep `executorch/` untracked and do not push it to the project GitHub.
- If a phone exits, preserve the request id, coordinator request detail, `logcat`, process status, and tombstone evidence before changing code.
- Any fix must explain why the crash happens on the observed stage and shape.
