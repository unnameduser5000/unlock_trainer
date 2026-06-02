# Current Debug State

This file is the first file to read after any context reset.

Last updated: 2026-06-02 Asia/Shanghai

## 2026-06-02 Mainline: Scheduler Work

Do not go back to inference smoke as the default next step. The current mainline is scheduler improvement for real multi-phone training.

Implemented code path:

- `app/src/main/proto/sid.proto`
  - `ForwardChunkRequest.stop_after_local_stage`
  - `ForwardChunkRequest.belief_transport_mode`
  - `StageForwardChunkRequest`
  - `CoordinatingService.SubmitStageRequest`
- `app/src/main/java/com/example/sid_trainer/MainActivity.kt`
  - Android worker can now execute only its local stage and return without recursively forwarding to the downstream phone.
  - Android worker supports `belief_transport_mode=full|terminal|none`.
- `coordinator/src/main/kotlin/com/example/sid_coordinator/CoordinatorRequestOrchestrator.kt`
  - coordinator can dispatch a request to a specific stage worker.
- `coordinator/src/main/kotlin/com/example/sid_coordinator/RunPreparedStagePipelineExperimentMain.kt`
  - new experimental runner with explicit `Q0 -> stage0 -> Q1 -> stage1 -> Q2 -> stage2 -> Done` queues.
- `tools/export/sid_export_mobile.py`
  - export supports `--belief_transport_mode full|terminal|none`.
  - `full`: old belief/KL path, nonzero chunks consume previous full-vocab log-probs and every chunk returns full log-probs.
  - `terminal`: CE-only transport path, intermediate chunks do not consume or return full-vocab log-probs; terminal chunk returns full log-probs for accuracy metrics.
  - `none`: CE-only no-logprob-return path; no chunk returns full-vocab log-probs.
- `coordinator/build.gradle.kts`
  - new task `:coordinator:runPreparedStagePipelineExperiment`.

Verified locally:

```powershell
.\gradlew.bat :coordinator:compileKotlin --offline --no-daemon
.\gradlew.bat :app:compileDebugKotlin --offline --no-daemon
python -m py_compile tools\export\sid_export_mobile.py
```

All pass on 2026-06-02.

Important distinction:

- Old runner: `:coordinator:runPreparedPipelineExperiment`
  - bounded in-flight full-chain requests, `coordinator -> stage0 -> stage1 -> stage2`.
- New runner: `:coordinator:runPreparedStagePipelineExperiment`
  - coordinator-owned per-stage queues, each worker returns after local execute.

Suggested first phone validation after reinstalling workers with the new proto:

```powershell
.\gradlew.bat :coordinator:runPreparedStagePipelineExperiment --offline --no-daemon --args="127.0.0.1 50051 data/sft_requests/tinyllama_agnews128_label_train512_seed20260531/requests.jsonl 0 6 debug_runs/stage-pipeline-smoke-YYYYMMDD-HHMMSS/results.csv stage-pipeline-smoke 1 0 false true 0 10000 420000 3 3"
```

Argument order after `requestPrefix` is:

```text
minValidLabels delayMs evalOnly stopOnFailure transientRetryCount transientRetryDelayMs submitRpcDeadlineMs maxBufferedPerStage stageCount beliefTransportMode
```

For the current CE-only AG News mainline, use `beliefTransportMode=terminal` after exporting matching PTEs:

```powershell
.\gradlew.bat :coordinator:runPreparedStagePipelineExperiment --offline --no-daemon --args="127.0.0.1 50051 data/sft_requests/tinyllama_agnews128_label_train512_seed20260531/requests.jsonl 0 6 debug_runs/stage-pipeline-smoke-YYYYMMDD-HHMMSS/results.csv stage-pipeline-smoke 1 0 false true 0 10000 420000 3 3 terminal"
```

Server export command for matching LoRA PTEs:

```bash
cd ~/sg-exe-trainer
BELIEF_TRANSPORT_MODE=terminal \
NUM_CHUNKS=3 \
CHUNK_IDX=0,1,2 \
SEQ_LEN=128 \
TRANSPORT_DTYPE=float16 \
ALPHA=1.0 \
LABEL_SMOOTHING=0.0 \
ARTIFACT_PREFIX=tinyllama_lora \
ARTIFACT_SUFFIX=_seq128_terminal \
OUTPUT_DIR=model \
bash tools/export/export_lora_tinyllama.sh
```

Current limitation:

- Old PTEs exported before `--belief_transport_mode` still behave as `full`. Runtime `terminal`/`none` is only valid with newly exported matching PTEs.
- CE-only still computes vocab logits inside each chunk to form local CE loss. The optional full-vocab switch removes cross-stage log-prob input/output and KL/belief transport; it is not a sampled-CE or label-head replacement.
- The stage-pipeline runtime still needs phone validation and comparison against the old `window3` path.

Related algorithm-history note:

- Read `docs/RESULTS_REPORT_2026_5_NOTES.md` before rewriting the paper story from scratch. It summarizes the older ViT/LLM belief report and explains why the current mobile prototype should be framed as systems-first BP-free chunk-local LoRA, with belief as optional rather than the main quality claim.

## Current Mainline Position

Do not restart from inference smoke or one-request smoke after a context reset. Those paths have already served their purpose.

### 2026-05-31 Server-Side Label Quality Sweep

Why this stage exists:

- The Android phone demo has already proved the real distributed ExecuTorch training path can run, but the label-quality result is still fragile.
- The previous `train64` choice was a phone-runtime demo budget, not an algorithmic limit. It was intentionally small so eval-before/train/eval-after could finish on phones.
- Before spending more phone time, use the server simulation to sweep learning rate, training steps, and epochs on the same prepared request tensors.

Current server sweep result supplied from `data/debug_runs/server_label_sweeps/20260531-110827`:

- `lr=1e-5`: eval-after label-choice acc `0.6055`, total loss `10.5164`.
- `lr=3e-5`: eval-after label-choice acc `0.6172`, total loss `10.2685`.
- `lr=1e-4`: eval-after label-choice acc `0.6250`, total loss `7.9067`.
- `lr=3e-4`: eval-after label-choice acc `0.6367`, total loss `1.8972`, but check class balance before trusting it because the Android `3e-4` run collapsed toward positive.

Important interpretation:

- The simulator summary `avg_loss` is the BP-free local objective, not pure binary label loss.
- Report quality with `label_choice_accuracy`, constrained `choice_loss`, per-class accuracy, and prediction bias. Do not claim success only from a large total-loss drop.
- The target is a quality-preserving mobile-training point: eval-after accuracy does not collapse, constrained choice loss improves, per-class accuracy remains balanced, and the number of phone steps is feasible.

Implemented server sweep improvement:

- `tools/sim/run_bpfree_lora_label_experiment.py` supports `--train_epochs`.
- `tools/sim/run_bpfree_lora_label_sweep.sh` supports `TRAIN_EPOCHS=...`.
- The summary now records `train_epochs`, `unique_train_records`, and `train_steps`.

Recommended next server sweeps:

```bash
cd ~/sg-exe-trainer
git pull --ff-only origin main

DEVICE=cuda DTYPE=float32 \
TRAIN_LIMIT=64 \
TRAIN_EPOCHS=1 \
LRS="1e-5 3e-5 1e-4 3e-4" \
bash tools/sim/run_bpfree_lora_label_sweep.sh
```

Then test whether the issue is simply too few train steps:

```bash
DEVICE=cuda DTYPE=float32 \
TRAIN_LIMIT=64 \
TRAIN_EPOCHS=2 \
LRS="1e-5 3e-5 1e-4" \
OUTPUT_ROOT=debug_runs/server_label_sweeps/train64_epochs2 \
bash tools/sim/run_bpfree_lora_label_sweep.sh

DEVICE=cuda DTYPE=float32 \
TRAIN_LIMIT=64 \
TRAIN_EPOCHS=4 \
LRS="1e-5 3e-5 1e-4" \
OUTPUT_ROOT=debug_runs/server_label_sweeps/train64_epochs4 \
bash tools/sim/run_bpfree_lora_label_sweep.sh
```

If the multi-epoch `train64` sweep overfits or becomes class-biased, generate a larger balanced train set and run one epoch:

```bash
python tools/data/prepare_lora_sft_requests.py \
  --model_name tinyllama \
  --dataset rotten_tomatoes \
  --split train \
  --seq_len 128 \
  --limit 512 \
  --attention_mask causal \
  --mask_prompt \
  --response_style label \
  --no_append_eos \
  --max_prompt_tokens 24 \
  --min_valid_labels 1 \
  --learning_rate 0.0001 \
  --shuffle_seed 20260531 \
  --balance_labels \
  --request_prefix rt-label-train512-balanced \
  --output_dir data/sft_requests/tinyllama_rotten_tomatoes128_label_train512_prompt24_lr1e4_balanced

TRAIN_MANIFEST=data/sft_requests/tinyllama_rotten_tomatoes128_label_train512_prompt24_lr1e4_balanced/requests.jsonl \
TRAIN_LIMIT=512 \
TRAIN_EPOCHS=1 \
LRS="1e-5 3e-5 1e-4" \
DEVICE=cuda DTYPE=float32 \
OUTPUT_ROOT=debug_runs/server_label_sweeps/train512 \
bash tools/sim/run_bpfree_lora_label_sweep.sh
```

Decision rule before returning to Android:

- Prefer `lr=3e-5` or `lr=1e-4` unless `3e-4` shows stable per-class validation accuracy.
- Pick the smallest training budget that improves constrained choice loss without obvious class collapse.
- Once chosen, export/prep the same request manifest and run the phone eval-before/train/eval-after protocol.

### 2026-05-31 AG News Label Control Result

User synced server outputs locally under:

- `data/debug_runs/server_label_controls/agnews128_train512`

Task:

- TinyLlama AG News 4-class label-only classification.
- Labels are `World`, `Sports`, `Business`, `Science`.
- Eval set has 256 rows, balanced 64/class.
- Train budget: 512 steps, one epoch.
- Metric: constrained label-choice accuracy plus BP-free local objective loss.

Result summary:

| control | lr | eval-before acc | eval-after acc | delta acc | eval-before loss | eval-after loss | delta loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| full LoRA upper bound (`NUM_CHUNKS=1`) | `1e-4` | 0.3359 | 0.8359 | +0.5000 | 7.4003 | 2.2775 | -5.1228 |
| BP-free CE-only (`NUM_CHUNKS=3`, all chunks) | `1e-4` | 0.3359 | 0.8320 | +0.4961 | 7.4003 | 2.0444 | -5.3559 |
| BP-free CE-only (`NUM_CHUNKS=3`, all chunks) | `3e-4` | 0.3359 | 0.8359 | +0.5000 | 7.4003 | 2.0286 | -5.3717 |
| terminal chunk only (`NUM_CHUNKS=3`, chunk 2) | `3e-4` | 0.3359 | 0.8047 | +0.4688 | 7.4003 | 2.0343 | -5.3660 |

Per-class for the best BP-free CE-only `3e-4` run:

- Business: `11/64 -> 49/64`, `0.172 -> 0.766`.
- Science: `9/64 -> 55/64`, `0.141 -> 0.859`.
- Sports: `55/64 -> 62/64`, `0.859 -> 0.969`.
- World: `11/64 -> 48/64`, `0.172 -> 0.750`.

Interpretation:

- This is a successful quality-control result: the task is learnable by TinyLlama LoRA and the metric is visible.
- BP-free CE-only is not "obviously broken" on this task. It nearly matches the full-LoRA `NUM_CHUNKS=1` upper-bound accuracy and has slightly lower reported local objective loss.
- Terminal-only also improves, but all-chunk BP-free CE-only is better, so the deployable all-stage training path is worth taking to phones.
- Do not use Dolly short-run quality as the main demonstration target; Dolly remains useful for SFT/system realism, while AG News is better for visible accuracy/loss movement.
- Next phone demo target should be AG News label-only, preferably BP-free CE-only/all chunks with `lr=1e-4` first for stability. `3e-4` has the best/lower loss in server control, but `1e-4` reaches nearly identical accuracy and may be safer on Android.

### 2026-05-31 AG News Phone Mainline Run

Current run pointer:

```text
debug_runs/CURRENT_AGNEWS_RUN.txt -> debug_runs\agnews-phone-mainline-20260531-141409
```

Active phone mapping:

- stage 0: `NX809J`, shard `tinyllama_lora_chunk_0_seq128`
- stage 1: `Lenovo_L71091`, shard `tinyllama_lora_chunk_1_seq128`
- stage 2: `Pixel_10_Pro_XL`, shard `tinyllama_lora_chunk_2_seq128`

Prepared data:

- train: `data/sft_requests/tinyllama_agnews128_label_train512_seed20260531/requests.jsonl`
- eval: `data/sft_requests/tinyllama_agnews128_label_eval256_seed20260531/requests.jsonl`

Final status as of `2026-05-31 23:58 Asia/Shanghai`:

| phase | rows | failures | label-choice acc | avg local loss | notes |
|---|---:|---:|---:|---:|---|
| `eval-before256-window3` | 256 | 0 | 86/256 = 0.3359 | 7.6778 | three-phone seq128 baseline before mobile training |
| stitched `train512` mobile training | 512 covered | 9 transient failed rows recovered | train rows are not the held-out metric | see recovery notes | completed through documented continuations, not one clean uninterrupted run |
| `eval-after256-window3` | 256 | 0 | 80/256 = 0.3125 | 5.6342 | completed after mobile training; Gradle runner reported `BUILD SUCCESSFUL in 2h 5m 43s` |

Quality interpretation:

- Held-out avg local loss improved by `-2.0436` (`7.6778 -> 5.6342`).
- Held-out label-choice accuracy did not improve: `0.3359 -> 0.3125`, delta `-0.0234`.
- Use this as a successful real-device distributed training/system run with visible loss movement, but do not claim AG News accuracy improvement from this phone run.
- The `check-ag-news-run` heartbeat is no longer needed after this completion.

Failure/recovery note, `2026-05-31 16:1x Asia/Shanghai`:

- `train512-window3` stopped early with `191` rows: `188` terminal successes and `3` failures.
- The three failed requests are `000188`, `000189`, and `000190`; all report `Stage 2 has no live worker.`
- Coordinator status after failure: `liveNodeCount=2`, `offlineStageCount=1`; stage 2 / `Pixel_10_Pro_XL` was evicted with `reason=lease-expired`.
- ADB still sees Pixel and `pidof com.example.sid_trainer` returned a live process, so this is not an obvious Android process death.
- Pixel battery/power state was healthy: 100%, powered, `mWakefulness=Awake`, `mStayOn=true`.
- Pixel logcat root-cause evidence: repeated `GrpcManager` heartbeat failures to coordinator `192.168.137.1:50051` with `ENETUNREACH (Network is unreachable)`.
- Pixel Wi-Fi status at diagnosis: `Wifi is not connected`; `wlan0` had no IPv4 address and no route. The hotspot SSID `dwellerLAPTOP 5740` was visible in scan results, but CLI reconnect requires the hotspot passphrase.
- Do not treat this as a clean 512-row quality run. After reconnecting Pixel to the hotspot, relaunch/re-register stage 2 and either run a documented continuation from record index `191` or restart a fresh clean train512, depending on how strict the demo needs to be.

Recovery action:

- Pixel was manually reconnected to `dwellerLAPTOP 5740` and received `192.168.137.81`.
- Pixel worker was force-stopped/relaunched; coordinator returned to `liveNodeCount=3`, `offlineStageCount=0`.
- New stage 2 node: `129`, `Pixel_10_Pro_XL`, `192.168.137.81:26052`.
- Started documented continuation:

```text
debug_runs\agnews-phone-mainline-20260531-141409\train512-cont-from191-window3
```

- Continuation args:

```text
127.0.0.1 50051 data/sft_requests/tinyllama_agnews128_label_train512_seed20260531/requests.jsonl 191 321 debug_runs\agnews-phone-mainline-20260531-141409\train512-cont-from191-window3/results.csv agnews-train512-cont191-20260531 1 0 false true 18 10000 420000 3
```

- Initial continuation check: first `3/321` requests succeeded with `0` failures.
- Heartbeat automation `check-ag-news-run` now monitors `train512-cont-from191-window3`; when it reaches `321` terminal successes and `0` failures, it should launch `eval-after256-window3`.

Second interruption:

- `train512-cont-from191-window3` also stopped early with `111` rows: `108` terminal successes and `3` failures.
- Failed records are `299`, `300`, and `301`; all report `Stage 2 has no live worker.`
- Coordinator again showed `liveNodeCount=2`, `offlineStageCount=1`.
- Pixel app process was still alive, but Pixel Wi-Fi was disconnected again; `wlan0` had no IPv4 address and no route.
- Pixel logcat again showed `GrpcManager` heartbeat failures to `192.168.137.1:50051` with `ENETUNREACH`.
- Applied Pixel-side settings to reduce network-quality-triggered Wi-Fi drops:
  - `cmd wifi set-ipreach-disconnect disabled`
  - `settings put global captive_portal_mode 0`
  - `settings put global captive_portal_detection_enabled 0`
  - `settings put global wifi_watchdog_poor_network_test_enabled 0`
- ADB Wi-Fi toggle did not automatically reconnect to `dwellerLAPTOP 5740`; manual reconnection or the hotspot passphrase is needed.
- Recommendation after this repeated network loss: use the interrupted runs as fault-tolerance evidence, but for a clean quality result, stabilize the Wi-Fi first and then run a fresh train512. Further continuations are possible but become increasingly messy as a quality demo.

If `train512` still looks weak, do not continue blind LR tuning. Run controls that isolate the failure mode:

1. Full-LoRA upper bound on the same prepared data:

```bash
TRAIN_MANIFEST=data/sft_requests/tinyllama_rotten_tomatoes128_label_train512_prompt24_lr1e4_balanced/requests.jsonl \
EVAL_MANIFEST=data/sft_requests/tinyllama_rotten_tomatoes128_label_val256_prompt24_balanced/requests.jsonl \
NUM_CHUNKS=1 \
TRAIN_LIMIT=512 \
TRAIN_EPOCHS=1 \
LRS="3e-5 1e-4 3e-4" \
ALPHA=1.0 \
DEVICE=cuda DTYPE=float32 \
OUTPUT_ROOT=debug_runs/server_label_sweeps/full_lora_train512 \
bash tools/sim/run_bpfree_lora_label_sweep.sh
```

`NUM_CHUNKS=1` is a server-only full-backprop LoRA control because there is no cross-chunk detach boundary.

2. Terminal-chunk-only control:

```bash
TRAIN_MANIFEST=data/sft_requests/tinyllama_rotten_tomatoes128_label_train512_prompt24_lr1e4_balanced/requests.jsonl \
EVAL_MANIFEST=data/sft_requests/tinyllama_rotten_tomatoes128_label_val256_prompt24_balanced/requests.jsonl \
NUM_CHUNKS=3 \
TRAIN_CHUNKS=2 \
TRAIN_LIMIT=512 \
TRAIN_EPOCHS=1 \
LRS="3e-5 1e-4 3e-4" \
ALPHA=1.0 \
DEVICE=cuda DTYPE=float32 \
OUTPUT_ROOT=debug_runs/server_label_sweeps/terminal_chunk_ce_train512 \
bash tools/sim/run_bpfree_lora_label_sweep.sh
```

3. BP-free CE-only control:

```bash
TRAIN_MANIFEST=data/sft_requests/tinyllama_rotten_tomatoes128_label_train512_prompt24_lr1e4_balanced/requests.jsonl \
EVAL_MANIFEST=data/sft_requests/tinyllama_rotten_tomatoes128_label_val256_prompt24_balanced/requests.jsonl \
NUM_CHUNKS=3 \
TRAIN_CHUNKS=all \
TRAIN_LIMIT=512 \
TRAIN_EPOCHS=1 \
LRS="3e-5 1e-4 3e-4" \
ALPHA=1.0 \
DEVICE=cuda DTYPE=float32 \
OUTPUT_ROOT=debug_runs/server_label_sweeps/bpfree_ce_only_train512 \
bash tools/sim/run_bpfree_lora_label_sweep.sh
```

Interpretation:

- If `NUM_CHUNKS=1` is also weak, the task/prompt/data/LoRA capacity is the problem.
- If `NUM_CHUNKS=1` works but terminal-only is weak, the trainable location is the problem.
- If CE-only works but `ALPHA=0.5` is weak, the belief/KL term is hurting this label task.
- If server works but Android is weak, suspect PTE/export/runtime or optimizer/checkpoint mismatch.

### 2026-05-30 Label-Only Quality Demo Pivot

Why this pivot exists:

- The seq128 Rotten Tomatoes natural-language template run completed as a systems proof, but the loss did not show a meaningful downward trend.
- Completed run: `debug_runs/seq128-rotten-tomatoes-20260530-213247/train128-window3`.
- Result: `128/128` success, `0` failures, average terminal loss about `6.7117`, first 32 loss about `6.6797`, last 32 loss about `6.7000`.
- This is not a useful quality demo because the 10-token response template mostly measures predictable words such as "The movie review expresses..." rather than the sentiment decision.

Implemented fix:

- `tools/data/prepare_lora_sft_requests.py` now supports `--response_style label`.
  - For `rotten_tomatoes` and `sst2`, the response becomes only ` positive` or ` negative`.
  - Use `--no_append_eos` with this mode so the only supervised token is the class label.
- Prepared request records can now include optional `learning_rate`.
- `ForwardChunkRequest` now has optional `learning_rate = 10`.
- `PreparedRequestSupport.kt` copies manifest `learning_rate` into the protobuf request.
- Android `NativeShardRunner` uses request learning rate when it creates ExecuTorch `SGD`; old manifests still default to `1e-5`.

Recommended quality-demo dataset generation on the Linux export/server environment:

First run a cheap server-side probe before spending Android time:

```bash
python tools/data/probe_label_task.py \
  --model_name tinyllama \
  --dataset rotten_tomatoes \
  --limit 256 \
  --max_prompt_tokens 96 \
  --output_json debug_runs/label_probe_rt128_base.json
```

Interpretation:

- `choice_accuracy` and `avg_choice_loss` answer the binary question "positive vs negative" only.
- `full_vocab_accuracy` answers whether the full model's top token is exactly the target label token.
- If base `choice_accuracy` is already very high, the task is too easy for a learning demo.
- If base is near chance and server-side LoRA can overfit but Android cannot, debug the mobile BP-free path.
- If base is near chance and server-side LoRA also cannot overfit, change prompt/labels/LR before using phones.

Probe results so far:

- Train split with `--max_prompt_tokens 96` was too easy: `choice_accuracy=1.0`, `avg_choice_loss=0.0355`, while `full_vocab_accuracy=0.0`.
- Validation split with `--max_prompt_tokens 24` is a better demo target: `choice_accuracy=0.71875`, `avg_choice_loss=0.7612`, while `full_vocab_accuracy=0.0`.
- Therefore, do not judge this label task by full-vocab token accuracy. Use constrained `label_choice_accuracy` over the label tokens ` positive` and ` negative`.
- Coordinator prepared-request runners now emit `label_choice_correct`, `label_choice_count`, and `label_choice_accuracy` when the JSONL records contain `label_choices`.

```bash
python tools/data/prepare_lora_sft_requests.py \
  --model_name tinyllama \
  --dataset rotten_tomatoes \
  --split train \
  --seq_len 128 \
  --limit 64 \
  --attention_mask causal \
  --mask_prompt \
  --response_style label \
  --no_append_eos \
  --max_prompt_tokens 24 \
  --min_valid_labels 1 \
  --learning_rate 3e-4 \
  --request_prefix rt-label-train \
  --output_dir data/sft_requests/tinyllama_rotten_tomatoes128_label_train64_prompt24_lr3e4
```

Optional validation set:

```bash
python tools/data/prepare_lora_sft_requests.py \
  --model_name tinyllama \
  --dataset rotten_tomatoes \
  --split validation \
  --seq_len 128 \
  --limit 256 \
  --attention_mask causal \
  --mask_prompt \
  --response_style label \
  --no_append_eos \
  --max_prompt_tokens 24 \
  --min_valid_labels 1 \
  --request_prefix rt-label-val \
  --output_dir data/sft_requests/tinyllama_rotten_tomatoes128_label_val256_prompt24
```

Recommended demo protocol:

- Clean `_seq128.latest.sidckpt` checkpoints and restart all three workers.
- Run `eval-before` on the label-only manifest with `evalOnly=true`.
- Run one or more short train epochs over the same 64 real examples with unique run/request prefixes.
- Run `eval-after` on the same manifest with `evalOnly=true`.
- Report this honestly as a real-data overfit/learning-signal demo, not as a generalization claim.
- Use the existing `train128-window3` and pipeline-overlap artifacts for systems performance/stability, not for model-quality improvement.

Local status after this pivot:

- Windows local Python cannot generate prepared requests because it lacks `torch`; generate on the server environment and sync `data/sft_requests/...` back.
- New APK with request-level learning-rate support was built and installed on NX, Lenovo, and Pixel.
- Coordinator was restarted from the new code.
- Workers were relaunched and route is live: NX stage 0 -> Lenovo stage 1 -> Pixel stage 2.

### 2026-05-30 Pipeline-Overlap Scheduler Proof

Current pipeline-overlap evidence directory:

- `debug_runs/pipeline-overlap-20260530-114725`

Why this stage exists:

- The successful stay-awake `train512` proved the three-stage Android BP-free training chain is stable, but `runPreparedExperiment` was still request-level serial.
- The new target is to prove bounded in-flight scheduling: while one request is on downstream stages, upstream stages can execute later requests.

Code changes added for this stage:

- Android worker:
  - `MainActivity.kt` now wraps only `NativeShardRunner.execute(modelPath, request)` in a coroutine `Mutex`.
  - The downstream `sendDataToNextNode` call remains outside the local execution mutex, so one stage can start the next local request while an earlier request waits for downstream completion.
  - This protects ExecuTorch `TrainingModule` / local SGD from concurrent same-phone calls while still permitting inter-stage overlap.
- Coordinator:
  - New Gradle task: `:coordinator:runPreparedPipelineExperiment`.
  - New runner: `coordinator/src/main/kotlin/com/example/sid_coordinator/RunPreparedPipelineExperimentMain.kt`.
  - Arguments mirror `runPreparedExperiment`, plus final `maxInFlight` argument.
  - `CoordinatorState.reportRequestEvent()` now stores coordinator-observed request-event time in `event_epoch_ms`; raw worker phone time is appended as `workerEventEpochMs=...` in the message. Use this new data for Gantt plots because phone clocks are not guaranteed synchronized.
- Reporting:
  - New script: `tools/report/plot_pipeline_overlap.py`.
  - It reads `stage-timings.csv`, uses `LOCAL_COMPLETED` rows and `total_measured_ms` to estimate actual local ExecuTorch execution intervals, writes `pipeline_overlap_intervals.csv`, `pipeline_overlap_summary.txt`, and `pipeline_overlap_gantt.png`.
  - Do not use `local_ms` alone for Gantt intervals after the mutex change; it includes time waiting for the local execution mutex. Use `total_measured_ms` or `execute_ms` for the actual local work interval.

Build and deployment status:

- `.\gradlew.bat :coordinator:compileKotlin --offline --no-daemon`: passed.
- `.\gradlew.bat :app:assembleDebug --offline --no-daemon`: passed.
- New APK installed on:
  - NX ADB `91260221021D`
  - Lenovo ADB `ZY22G2HC5C`
  - Pixel ADB `58151FDCQ006A8`
- Coordinator was restarted from the new build and workers were relaunched.

Current clean post-probe device state:

- stage 0: NX / node `115` / `192.168.137.211:26052` / `tinyllama_lora_chunk_0`
- stage 1: Lenovo / node `108` / `192.168.137.124:26052` / `tinyllama_lora_chunk_1`
- stage 2: Pixel / node `116` / `192.168.137.139:26052` / `tinyllama_lora_chunk_2`
- Status after cleanup: `liveNodeCount=3`, `offlineStageCount=0`, route chain NX -> Lenovo -> Pixel.
- The small train pipeline probe saved `*.latest.sidckpt` files on all three phones. These were deleted after evidence export:
  - NX: `files/shards/checkpoints/tinyllama_lora_chunk_0.latest.sidckpt`
  - Lenovo: `files/shards/checkpoints/tinyllama_lora_chunk_1.latest.sidckpt`
  - Pixel: `files/shards/checkpoints/tinyllama_lora_chunk_2.latest.sidckpt`
- Workers were restarted after deleting latest checkpoints, so the next formal run should start from the PTE initial state. Old `backup_20260527_1516` checkpoint directories on NX/Lenovo were left untouched.

Pipeline eval-only probe:

- Directory: `debug_runs/pipeline-overlap-20260530-114725/pipeline-eval6-window3`
- Run id: `pipeline-eval6-window3-20260530-1150`
- Command: recorded in `command.txt`.
- Arguments: `evalOnly=true`, `maxSubmitted=6`, `maxInFlight=3`, `stopOnFailure=true`, transient retry count `0`.
- Result: 6/6 success.
- Summary: avg local loss `7.527053`, token accuracy `0.373832` over `107` tokens.
- Artifacts:
  - `results.csv`
  - `metrics.csv`
  - `stage-timings.csv`
  - `coordinator-run-summary.json`
  - `figures/pipeline_overlap_gantt.png`
  - `figures/pipeline_overlap_summary.txt`
  - `figures/pipeline_overlap_intervals.csv`
- Overlap evidence using `total_measured_ms` intervals:
  - `local_completed_intervals=18`
  - `overlap_pair_count=17`
  - `max_overlap_ms=5994`
  - Example: request `000005` stage 0 overlapped request `000004` stage 1 by `5994 ms`.

Pipeline train probe:

- Directory: `debug_runs/pipeline-overlap-20260530-114725/pipeline-train6-window3`
- Run id: `pipeline-train6-window3-20260530-1155`
- Command: recorded in `command.txt`.
- Arguments: `evalOnly=false`, `maxSubmitted=6`, `maxInFlight=3`, `delayMs=1000`, `stopOnFailure=true`, transient retry count `0`.
- Result: 6/6 success.
- Summary: avg local loss `7.478127`, token accuracy `0.565957` over `235` tokens.
- Artifacts:
  - `results.csv`
  - `metrics.csv`
  - `stage-timings.csv`
  - `coordinator-run-summary.json`
  - `figures/pipeline_overlap_gantt.png`
  - `figures/pipeline_overlap_summary.txt`
  - `figures/pipeline_overlap_intervals.csv`
- Overlap evidence using `total_measured_ms` intervals:
  - `local_completed_intervals=18`
  - `overlap_pair_count=19`
  - `max_overlap_ms=6229`
  - Examples:
    - request `000005` stage 0 overlapped request `000004` stage 1 by `6229 ms`
    - request `000004` stage 0 overlapped request `000003` stage 1 by `6011 ms`
    - request `000000` stage 2 overlapped request `000002` stage 0 by `5689 ms`

Pipeline train512 window-3 run:

- Directory: `debug_runs/pipeline-overlap-20260530-114725/pipeline-train512-window3-20260530-121154`
- Run id: `pipeline-train512-window3-20260530-121154`
- Command args are recorded in `args.txt`.
- Arguments: `maxSubmitted=512`, `maxInFlight=3`, `delayMs=0`, `evalOnly=false`, `stopOnFailure=true`, retry count `18`, retry delay `10000 ms`, submit deadline `420000 ms`.
- Status at the 128-request snapshot:
  - `128/512` terminal requests completed.
  - `128` success, `0` failed.
  - `maxTerminalInFlight=3`, so the window is actually being filled.
  - Throughput about `5.57 req/min`.
  - Average terminal elapsed about `32136 ms`.
  - Token accuracy snapshot about `0.5611`.
  - Battery snapshot around the same period: NX `100%`, Lenovo `97%`, Pixel `99%`; phone temperatures about `36-37 C`.
- Partial artifacts fixed at the 128-request snapshot:
  - `results.csv` keeps growing while the 512 run continues.
  - `coordinator-run-summary-128.json`
  - `metrics-128.csv`
  - `stage-timings-128.csv`
  - `figures-128/pipeline_overlap_gantt.png`
  - `figures-128/pipeline_overlap_summary.txt`
  - `figures-128/pipeline_overlap_intervals.csv`
- 128-snapshot overlap evidence using `total_measured_ms` intervals:
  - `local_completed_intervals=389`
  - `overlap_pair_count=483`
  - `max_overlap_ms=8317`
  - Examples:
    - request `000063` stage 0 overlapped request `000061` stage 2 by `8317 ms`
    - request `000064` stage 2 overlapped request `000066` stage 0 by `7962 ms`
    - request `000116` stage 0 overlapped request `000115` stage 1 by `7883 ms`
- Final status:
  - Completed `512/512` terminal requests with `0` final failures and no missing record indices through `511`.
  - Gradle stdout ended with `BUILD SUCCESSFUL in 1h 55m 51s`.
  - Wall-clock duration from runner timestamps: about `115.62 min`, throughput about `4.43 req/min`.
  - Runner-observed terminal latency: average about `40.6 s`, p50 about `33.9 s`, p95 about `54.8 s`, p99 about `174.3 s`, max about `463.9 s`.
  - Coordinator summary: average local loss `7.410730381496251`, token accuracy `0.557944895839697` over `16369` tokens.
  - `metrics-final.csv` contains `5` transient failed attempts, all recovered by retry:
    - request indices `294`, `299`, `303`, `306`, and `325`
    - message: `Coordinator dispatch to stage 0 failed: Read timed out`
  - This is useful systems evidence: bounded in-flight scheduling works and transient retry can recover requests, but stragglers/read-timeouts can occupy window slots and reduce effective concurrency.
  - Worker telemetry was exported from coordinator SQLite after the run, not just sampled as a final snapshot:
    - `worker-telemetry-final.csv`: 4135 heartbeat samples across the run window.
    - `worker-telemetry-summary.txt`: per-device battery, current, temperature, and memory summary.
    - `worker-telemetry-timeseries.png`: battery level, battery temperature, device-reported current, and app PSS over time.
    - NX / stage 0: 1380 samples, battery stayed `100%`, temp avg `36.0 C`, app PSS avg about `3256.0 MB`.
    - Lenovo / stage 1: 1379 samples, battery `99% -> 94%`, temp avg `35.4 C`, app PSS avg about `3259.4 MB`.
    - Pixel / stage 2: 1376 samples, battery `100% -> 95%`, temp avg `35.5 C`, app PSS avg about `3643.3 MB`.
    - `battery-final.txt` is only the final status snapshot before coordinator shutdown.
    - Treat device-reported current as qualitative unless backed by Android Studio Power Profiler or an external power meter.
  - Coordinator status before shutdown was saved in `status-final-before-shutdown.json`.
  - Coordinator process listening on `50051/18080` was stopped after final artifacts were exported.
- Final artifacts:
  - `coordinator-run-summary-final.json`
  - `metrics-final.csv`
  - `stage-timings-final.csv`
  - `figures-final/pipeline_overlap_gantt.png`
  - `figures-final/pipeline_overlap_summary.txt`
  - `figures-final/pipeline_overlap_intervals.csv`
  - `figures-final/worker-telemetry-final.csv`
  - `figures-final/worker-telemetry-summary.txt`
  - `figures-final/worker-telemetry-timeseries.png`
  - `figures-final/training_loss_curve.png`
  - `figures-final/training_latency_curve.png`
  - `figures-final/training_token_accuracy_curve.png`
  - `figures-final/run_metrics_summary.txt`
  - Training loss is roughly flat: average `7.4107`, first 50 successful requests about `7.4113`, last 50 about `7.5643`.

Interpretation:

- This is the first direct evidence that the mobile pipeline runner is not merely serial request submission.
- The probe demonstrates bounded in-flight request submission and cross-stage local execution overlap on the real Android ExecuTorch training path.
- The local execution mutex is a correctness guard, not a serialization of the whole request: upstream local execution can proceed while an earlier request is forwarding or waiting downstream.
- The train probe is a scheduler/data-path proof, not a model-quality result. For any next formal train/eval, start from the clean post-probe worker state or explicitly re-clear checkpoints.
- Request-order/FIFO and serial are not the same. Serial means submit one request and wait for the terminal stage before submitting the next. Request-ordered pipeline means each stage should apply local work in request order, while different requests can occupy different stages at the same time. The current implementation proves bounded in-flight overlap with a local execution mutex; strict per-stage FIFO ordering should still be added before making a model-quality/order-determinism claim.

Useful commands:

```powershell
.\gradlew.bat :coordinator:runPreparedPipelineExperiment --offline --no-daemon --args="127.0.0.1 50051 data/sft_requests/tinyllama_dolly64_train512/requests.jsonl 0 6 debug_runs/<run>/pipeline-train6-window3/results.csv pipeline-train6-window3-<ts> 1 1000 false true 0 10000 420000 3"

python tools\report\plot_pipeline_overlap.py debug_runs\<run>\pipeline-train6-window3\stage-timings.csv --output_dir debug_runs\<run>\pipeline-train6-window3\figures

python tools\report\export_worker_telemetry.py `
  --db coordinator\coordinator\data\coordinator.db `
  --results_csv debug_runs\pipeline-overlap-20260530-114725\pipeline-train512-window3-20260530-121154\results.csv `
  --output_dir debug_runs\pipeline-overlap-20260530-114725\pipeline-train512-window3-20260530-121154\figures-final

python tools\report\plot_run_metrics.py `
  --csv debug_runs\pipeline-overlap-20260530-114725\pipeline-train512-window3-20260530-121154\metrics-final.csv `
  --output_dir debug_runs\pipeline-overlap-20260530-114725\pipeline-train512-window3-20260530-121154\figures-final `
  --title "Train512 window=3"
```

### 2026-05-30 Four-Phone Scheduler Probe

Current four-phone scheduler evidence directory:

- `debug_runs/four-phone-scheduler-20260530-0005`

Connected ADB devices during the probe:

- Pixel / ADB `58151FDCQ006A8` / deviceId `Pixel_10_Pro_XL` / `192.168.137.139` / battery 99% / has cached `tinyllama_lora_chunk_2.pte`
- NX / ADB `91260221021D` / deviceId `NX809J` / `192.168.137.211` / battery 100% / has cached `tinyllama_lora_chunk_0.pte`
- Huawei / ADB `APFQUT2C19005486` / deviceId `AGT-AN00` / `192.168.137.243` / battery 67% / no shard cache
- Lenovo / ADB `ZY22G2HC5C` / deviceId `Lenovo_L71091` / `192.168.137.124` / battery 100% / has cached `tinyllama_lora_chunk_1.pte`

Important: the active model is still a three-stage LoRA TinyLlama pipeline. Four phones do not automatically mean four model stages. In this run the fourth phone was used to test scheduler spare/rejection behavior.

Observed scheduler result after relaunching workers in stage order:

- stage 0: NX node `111`, `assignmentReason=preferred-device-match`, shard `tinyllama_lora_chunk_0`
- stage 1: Lenovo node `108`, `assignmentReason=preferred-device-match`, shard `tinyllama_lora_chunk_1`
- stage 2: Pixel node `112`, `assignmentReason=dynamic-fill-offline-stage`, shard `tinyllama_lora_chunk_2`
- Huawei registered as `AGT-AN00` but was rejected with `reason=no-schedulable-stage` because all three stages were already live.

Status after the scheduler probe:

- `liveNodeCount=3`
- `offlineStageCount=0`
- route chain: NX `192.168.137.211:26052` -> Lenovo `192.168.137.124:26052` -> Pixel `192.168.137.139:26052`
- saved snapshots:
  - `debug_runs/four-phone-scheduler-20260530-0005/status-after-relaunch.json`
  - `debug_runs/four-phone-scheduler-20260530-0005/status-after-probe.json`

One prepared request was submitted as a scheduler/data-path probe, not as a new mainline smoke reset:

- Run id: `four-phone-scheduler-probe`
- CSV: `debug_runs/four-phone-scheduler-20260530-0005/scheduler-probe/results.csv`
- Result: 1/1 success, `evalOnly=true`, terminal stage 2 on Pixel, elapsed `44916 ms`, loss `8.239365`, token accuracy `1/3`.
- Coordinator export:
  - `debug_runs/four-phone-scheduler-20260530-0005/scheduler-probe/coordinator-run-summary.json`
  - `debug_runs/four-phone-scheduler-20260530-0005/scheduler-probe/stage-timings.csv`

Stage timing evidence confirms all three stages executed `TrainingModule` forward:

- stage 0 NX: `executeMs=5152`, local loss `12.037177`
- stage 1 Lenovo: `executeMs=7798`, local loss `5.682226`
- stage 2 Pixel: `executeMs=6777`, local loss `8.239365`

Caveat: Pixel's device-local `eventEpochMs` is not synchronized with the coordinator clock. For timeline figures, prefer coordinator request metrics or event ids over raw worker-local epoch ordering.

If a true four-stage model is needed, export new artifacts on the Linux server instead of trying to use the current three-stage PTEs:

```bash
NUM_CHUNKS=4 CHUNK_IDX=-1 OUTPUT_DIR=model bash tools/export/export_lora_tinyllama.sh
```

With joint graph diagnostics:

```bash
DUMP_JOINT_GRAPH=1 NUM_CHUNKS=4 CHUNK_IDX=-1 OUTPUT_DIR=model bash tools/export/export_lora_tinyllama.sh
```

Then update `coordinator/config/pipeline.json` to four stages with `tinyllama_lora_chunk_0/1/2/3`, pre-cache each shard on its intended phone, call `POST /api/v1/routing/reload`, and relaunch workers. Current exporter still duplicates `final_norm + lm_head` into each chunk, so four-stage memory will not shrink linearly.

512-row training retry after the scheduler probe:

- Run directory: `debug_runs/train512-pixel-stage2-20260530-0030`
- Run id: `train512-pixel-stage2-20260530`
- Command is recorded in `debug_runs/train512-pixel-stage2-20260530-0030/command-attempt2.txt`.
- Background runner PID is in `debug_runs/train512-pixel-stage2-20260530-0030/pid-attempt2.txt`; the process has exited normally after fail-fast.
- ADB perf monitor PID is in `debug_runs/train512-pixel-stage2-20260530-0030/monitor.pid.txt`; the monitor was stopped after failure evidence was saved.
- Worker mapping at start: NX stage 0, Lenovo stage 1, Pixel stage 2. Huawei was force-stopped to avoid spare-node noise.
- Arguments: train512 prepared Dolly SFT manifest, `startIndex=0`, `maxSubmitted=512`, `evalOnly=false`, `stopOnFailure=true`, explicit transient retry count `18`, retry delay `10000 ms`, submit deadline `420000 ms`.
- Result: 28 submitted, 27 successful training requests, 1 failed request.
- Final failed request: `train512-pixel-stage2-20260530-000027`, after 19 attempts including 18 transient retries, message `Stage 2 has no live worker.`
- Runner stdout/stderr and final CSV:
  - `debug_runs/train512-pixel-stage2-20260530-0030/gradle-combined.log`
  - `debug_runs/train512-pixel-stage2-20260530-0030/results.csv`
- Failure evidence bundle:
  - `debug_runs/train512-pixel-stage2-20260530-0030/failure-evidence-stage2-lease-expired/`
- Coordinator status after failure: `liveNodeCount=2`, `offlineStageCount=1`; stage 2 was evicted with `reason=lease-expired`.
- Pixel process did not disappear after the failure. `adb pidof com.example.sid_trainer` returned pid `20682`; Pixel battery was 100%, AC powered, about `29.3 C`.
- Pixel `dumpsys activity exit-info` showed only the earlier manual force-stop, not a new LOW_MEMORY/LMK or native crash record.
- Pixel logcat tail did not show a new app `AndroidRuntime` fatal or `OutOfMemory` line. It mainly showed repeated `BestClock: No network time available` noise and unrelated Google service `ManagedChannel allocation site` messages.
- Coordinator worker telemetry before eviction showed Pixel stage 2 active, thermal status `NONE`, app PSS about `3.79 GB`, private dirty about `3.78 GB`, Java/runtime memory about `76 MB`.
- Stage timing evidence shows stage 2 completed request `000026` successfully with `TrainingModule`, `evalOnly=false`, `optimizerStepApplied=true`, `executeMs=6900`, PSS about `3.77 GB`.
- Interpretation: replacing the old stage-2 phone with Pixel did not complete 512. This run did not reproduce the previous clear Android LOW_MEMORY kill; instead stage 2 remained as a process but stopped satisfying the coordinator lease, so the immediate failure mode is heartbeat/RPC stall or worker responsiveness collapse under the high native memory plateau.
- Caveat: the ADB `monitor_perf.ps1` invocation only sampled NX in this run, so use coordinator `worker-telemetry.csv` for Pixel memory/thermal evidence.

Stay-awake rerun started after noticing the previous failure may have coincided with screen-off behavior:

- Run directory pointer: `debug_runs/CURRENT_TRAIN512_RUN.txt`
- Active run directory at start: `debug_runs/train512-pixel-stage2-stayon-20260530-012301`
- Run id: `train512-pixel-stayon-20260530-012301`
- Before relaunching workers, NX, Lenovo, and Pixel were configured with:
  - `adb shell input keyevent KEYCODE_WAKEUP`
  - `adb shell svc power stayon true`
  - `adb shell settings put global stay_on_while_plugged_in 7`
- Verified on all three: `mWakefulness=Awake`, `mStayOn=true`, `mHoldingDisplaySuspendBlocker=true`, `mStayOnWhilePluggedInSetting=7`.
- Workers were force-stopped and relaunched cleanly: NX stage 0, Lenovo stage 1, Pixel stage 2 node `114`.
- Huawei was force-stopped as a spare to avoid repeated no-stage registration noise.
- Background runner PID: `debug_runs/train512-pixel-stage2-stayon-20260530-012301/pid.txt`.
- Monitor PID: `debug_runs/train512-pixel-stage2-stayon-20260530-012301/monitor.pid.txt`.
- Monitor wrapper fixed the previous serial passing issue; initial samples include all three serials: NX, Lenovo, and Pixel.
- Final result: 512 submitted, 512 succeeded, 0 failed.
- Runner stdout reports `BUILD SUCCESSFUL in 4h 23m 53s`.
- Local CSV summary from `results.csv`: avg elapsed `30851.48 ms`, avg loss `7.407001`, token accuracy `0.558067` over `16369` tokens.
- Coordinator summary: avg elapsed `30804.94 ms`, avg loss `7.40700065`, token accuracy `0.558067`.
- Last request `train512-pixel-stayon-20260530-012301-000511` succeeded on terminal stage 2 with elapsed `30757 ms`, loss `7.328057`, token accuracy `31/50`.
- Coordinator export was saved to `debug_runs/train512-pixel-stage2-stayon-20260530-012301/coordinator-export/`:
  - `run-summary.json`
  - `metrics.csv`
  - `stage-timings.csv`
- Monitor samples were saved under `debug_runs/train512-pixel-stage2-stayon-20260530-012301/monitor/perf-monitor-20260530-012352/`.
- Monitor captured all three active serials: NX 4027 rows, Lenovo 4027 rows, Pixel 4027 rows.
- Pixel stayed alive through the end: pid `21785`, battery 100%, about `28.8-29.0 C`, app PSS about `3.73 GB`, private dirty about `3.71 GB`.
- Interpretation: the previous 27-success failure was consistent with screen-off / lease-expiry behavior. Keeping the phones awake and plugged in allowed the same three-stage Pixel-as-stage2 pipeline to complete the full 512-row training run despite the high stage-2 memory plateau.
- The heartbeat follow-up `check-stay-awake-train512` was deleted after completion.

Eval-after started immediately after the successful 512-row train, without restarting workers:

- Eval directory: `debug_runs/train512-pixel-stage2-stayon-20260530-012301/eval-after`
- Run id: `train512-pixel-stayon-20260530-eval-after`
- Manifest: `data/sft_requests/tinyllama_dolly64_eval128/requests.jsonl`
- Arguments: `startIndex=0`, `maxSubmitted=128`, `evalOnly=true`, `stopOnFailure=true`, retry count `18`, retry delay `10000 ms`, submit deadline `420000 ms`.
- Background PID: `debug_runs/train512-pixel-stage2-stayon-20260530-012301/eval-after/pid.txt`; the process has exited normally.
- Final result: 128 submitted, 128 succeeded, 0 failed.
- Runner stdout reports `BUILD SUCCESSFUL in 1h 5m 19s`.
- Local CSV summary from `results.csv`: avg elapsed `30476.67 ms`, avg loss `7.389941`, token accuracy `0.536709` over `3950` tokens.
- Last request `train512-pixel-stayon-20260530-eval-after-000127` succeeded on terminal stage 2 with elapsed `25380 ms`, loss `6.7106795`, token accuracy `18/31`.
- Coordinator status after eval-after: `liveNodeCount=3`, `offlineStageCount=0`.
- Coordinator export was saved to `debug_runs/train512-pixel-stage2-stayon-20260530-012301/eval-after/coordinator-export/`:
  - `run-summary.json`
  - `metrics.csv`
  - `stage-timings.csv`
- Compared with the earlier three-phone eval128 (`avg loss 7.405194`, token accuracy `0.536203`), eval-after is a tiny improvement in loss and essentially unchanged token accuracy. Treat this as continuity/model-sanity evidence, not a strong accuracy claim.
- The heartbeat follow-up `check-eval-after-run` was deleted after completion.

### 2026-05-29 Active Three-Phone Mainline

Current active run directory:

- `debug_runs/three-phone-mainline-20260528-145049`

Current devices and stage mapping:

- stage 0: NX / ADB `91260221021D` / deviceId `NX809J` / `192.168.137.211:26052` / `tinyllama_lora_chunk_0`
- stage 1: Lenovo / ADB `ZY22G2HC5C` / deviceId `Lenovo_L71091` / `192.168.137.124:26052` / `tinyllama_lora_chunk_1`
- stage 2: third phone / ADB `267d1faa` / deviceId `23043RP34C` / `192.168.137.174:26052` / `tinyllama_lora_chunk_2`

Latest train512 rerun result:

- Run id: `three-main-train512-dataplanefix`
- Run directory: `debug_runs/three-phone-mainline-20260528-145049/train512-dataplanefix-20260529-1240`
- Started after reinstalling the APK and force-stopping/relaunching all workers so the LoRA state reloaded from the PTE files.
- Command args are recorded in `debug_runs/three-phone-mainline-20260528-145049/train512-dataplanefix-20260529-1240/command.txt`.
- PID file: `debug_runs/three-phone-mainline-20260528-145049/train512-dataplanefix-20260529-1240/pid.txt`.
- The run has finished; use the final `results.csv` plus coordinator export for analysis:

```powershell
Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:18080/api/v1/runs/three-main-train512-dataplanefix?metrics=10" |
  Select-Object -ExpandProperty Content
```

- Final result on `2026-05-29 15:01 Asia/Shanghai`: 194 submitted rows, 193 successful training requests, 1 failed request. Average elapsed about `42825.82 ms`, average loss `7.4249`, token accuracy `0.5632` over `6198` tokens.
- The run completed a clean 128-request training window and passed the old request-84 heartbeat/offline failure point, but did not complete the full 512-request train.
- First/final failure: `three-main-train512-dataplanefix-000193`, after 19 attempts, message `Stage 2 has no live worker.`
- Root cause evidence: stage 2 Android process `pid=3797` exited at `2026-05-29 14:58:14.673` with `reason=3 (LOW_MEMORY)`, PSS/RSS about `3.6GB`; no Java crash log was present. The worker later restarted as `pid=11750` and re-registered as node `110`, but that reloads stage 2 state and is not a valid continuation of the same in-memory LoRA training run.
- OOM interpretation: this does not look like a classic per-request Kotlin/Java leak. Worker telemetry shows stage 2 jumps from about `166 MB` PSS before model load to about `3.67 GB` after the training PTE/runtime is active, then stays near that high plateau until Android LMK kills it; Java heap remains under about `98 MB`. Stage 2 is also the largest artifact (`tinyllama_lora_chunk_2.pte` about `1.56 GiB` versus `1.39 GiB` for chunks 0/1). The exported BP-free LoRA chunk currently includes `final_norm` and a full `lm_head` in every chunk, so adding more stages reduces transformer-layer burden but does not remove the duplicated full-vocabulary head unless the exporter/algorithm is changed.
- ExecuTorch Android API note: the actual Maven dependency is `org.pytorch:executorch-android:1.2.0`. `javap` on the Gradle-cached AAR shows `TrainingModule` exposes `load`, `executeForwardBackward`, `namedParameters`, and `namedGradients`, but no `close/destroy`; `Tensor`/`EValue` also expose no explicit close. The local official `executorch/` clone may contain newer Java source with `TrainingModule.close()`, but that is not the API currently compiled into this app.
- Immediate low-risk app optimizations after the OOM diagnosis:
  - `NativeShardRunner.kt` no longer writes the LoRA checkpoint after every training step. It now saves step 1 and every 16 optimizer steps, reducing per-step parameter-copy and flash-write churn. The request event message includes `checkpointSaved` and `checkpointIntervalSteps`.
  - `MainActivity.kt` now records per-local-step memory telemetry in the `LOCAL_COMPLETED` request event message: `pssBeforeKb`, `pssAfterKb`, `pssDeltaKb`, `privateDirtyBeforeKb`, `privateDirtyAfterKb`, `javaHeapBeforeKb`, `javaHeapAfterKb`, and `javaHeapDeltaKb`. This gives ADB-free evidence for whether PSS grows request by request.
  - `BeliefTopKCodec.kt` adds optional top-k belief transport without changing the protobuf schema or re-exporting PTEs. It is currently default-disabled in `MainActivity.kt` because simulation suggests belief may not be a useful training signal compared with CE-only. If re-enabled, non-terminal workers encode dense `outputShiftLogP` as `topk_log_probs:k=16` in `TensorData`; the downstream worker decodes it back to dense float32 immediately before `TrainingModule.executeForwardBackward()`.
  - Verified with `.\gradlew.bat :app:assembleDebug --offline --no-daemon`.
- Operator-level ExecuTorch exploration is now recorded in `docs/BPFREE_EXECUTORCH_OPERATOR_NOTES.md`.
  - Do not claim that a redundant cross-stage hidden-gradient op has been found. Current export likely already avoids hidden/input gradient outputs because only LoRA params require grad and `dummy_hidden` does not.
  - The concrete mismatch points are dense CE/KL/log-prob branches, duplicated full `lm_head`, generic TrainingModule gradient/parameter outputs, Java/native `namedGradients -> SGD.step(map)`, and Android TrainingModule's `FileDataLoader` path.
  - The exporter now supports `--dump_joint_graph`, exposed through `DUMP_JOINT_GRAPH=1` in `tools/export/export_lora_tinyllama.sh`, to print/save joint graph output specs and operator counts on the Linux export server.
- Do not tail-resume this train512 as a valid continuous training run. Either restart all workers for a new clean run after reducing memory pressure, or first prove checkpoint restore correctness and resume from a restored state.
- A lightweight background monitor ran from `debug_runs/three-phone-mainline-20260528-145049/train512-dataplanefix-20260529-1240/monitor_run.ps1`; it stopped automatically after the failure and wrote snapshots to `monitor_snapshots.csv`.
- Live coordinator export at the 132-success point:
  - `debug_runs/three-phone-mainline-20260528-145049/coordinator-export-train512-dataplanefix-live132/run-summary.json`
  - `debug_runs/three-phone-mainline-20260528-145049/coordinator-export-train512-dataplanefix-live132/metrics.csv`
  - `debug_runs/three-phone-mainline-20260528-145049/coordinator-export-train512-dataplanefix-live132/stage-timings.csv`
  - generated figures in `debug_runs/three-phone-mainline-20260528-145049/coordinator-export-train512-dataplanefix-live132/figures/`
- Live worker telemetry snapshot at the same point: `debug_runs/three-phone-mainline-20260528-145049/worker-telemetry-train512-dataplanefix-live129.csv`.
- Final failure evidence bundle:
  - `debug_runs/three-phone-mainline-20260528-145049/train512-dataplanefix-20260529-1240/failure-evidence-20260529-1501/`
- Final coordinator export and figures:
  - `debug_runs/three-phone-mainline-20260528-145049/coordinator-export-train512-dataplanefix-final193/`

Completed in this run:

- Three-phone smoke: `debug_runs/three-phone-mainline-20260528-145049/smoke/results.csv`, 1/1 success.
- Three-phone eval128: `debug_runs/three-phone-mainline-20260528-145049/eval128/results.csv`, 128/128 success, avg loss `7.4051939`, token accuracy `0.5362025`.
- Eval128 timing caveat: Windows wall clock jumped from `2026-05-28 16:27` to `2026-05-29 00:27`, producing one bad `elapsed_ms` outlier. Keep raw data, but compute throughput/latency with that outlier flagged or removed.
- Data-plane fix probe: `debug_runs/three-phone-mainline-20260528-145049/dataplane-fix-probe-20260529-1236/results.csv`, 3/3 success. This specifically verified the previous second-request broken-pipe crash path, not a generic smoke rerun.

Fixes added before train512:

- Coordinator request metrics and `runPreparedExperiment` now use monotonic `System.nanoTime()` for `elapsed_ms`; epoch timestamps are still kept only for log readability.
- Android workers now report heartbeat telemetry to coordinator, reducing dependence on ADB/logcat:
  - battery level/status/current/voltage
  - charging source
  - battery temperature
  - thermal status
  - app PSS/private dirty
  - runtime used memory
  - worker state
- New coordinator endpoints:
  - `GET /api/v1/worker-telemetry?limit=1000`
  - `GET /api/v1/worker-telemetry.csv?limit=100000`
- Verified telemetry after reinstalling the APK on all three phones. Latest rows include real values such as NX battery `57%`, `powerSource=AC`, temperature about `30-33 C`, and per-worker PSS around `3.3-3.8 GB` during training.

Latest three-phone train512 result and control-plane fix:

- `three-main-train512` is no longer running.
- Local CSV: `debug_runs/three-phone-mainline-20260528-145049/train512/results.csv`.
- Result: 512 rows submitted, 84 succeeded, 428 failed.
- First failure: `three-main-train512-000084`, message `Stage 2 has no live worker.`
- Root-cause evidence: stage 2 battery was 96-97%, charging over USB, about 33 C, `thermal_status=NONE`, and the Android process did not crash. Logcat shows the same pid completed request `000083` with `TrainingModule.executeForwardBackward()` at `2026-05-29 11:17:41`, then coordinator evicted old node 99 after missing heartbeat lease, and the worker later re-registered. This points to a heartbeat RPC/network stall, not battery, thermal, or native model crash.
- Fix committed locally on `2026-05-29`:
  - `GrpcManager.kt` heartbeat and registration RPCs now have deadlines/timeouts so one stuck heartbeat cannot block the heartbeat loop for minutes.
  - `RunPreparedExperimentMain.kt` now retries transient route failures such as `Stage X has no live worker`, `connection reset`, `deadline exceeded`, and downstream route failures. With `stopOnFailure=false`, default retry is 18 attempts with 10 s delay.
  - `coordinator/config/pipeline.json` and the three-phone template now use `heartbeatLeaseSeconds=45` instead of 15.
- Verification/deployment after fix:
  - `:coordinator:test --offline --no-daemon` passed.
  - `:app:assembleDebug --offline --no-daemon` passed.
  - New APK installed on all three phones.
  - Coordinator restarted in `debug_runs/coordinator-restart-20260529-control-fix-2`.
  - Current `/api/v1/status`: `liveNodeCount=3`, `offlineStageCount=0`, `leaseDurationSeconds=45`.
- Coordinator run summaries now aggregate the latest metric attempt for each `requestId`; detailed metrics CSV still keeps all attempts. This avoids transient retry failures polluting the run-level success/failure counts after a later successful retry.
- To resume the partial training without replaying the first 84 successful rows, use start index 84 and max submitted 428 with a new output directory/request prefix.

Latest data-plane crash fix on 2026-05-29:

- The failed clean train run `three-main-train512-cleanfix` reached request 1, then Lenovo and stage 2 crashed with `java.net.SocketException: Broken pipe` / `Software caused connection abort` in `HttpDataPlane.kt:136`.
- Raw crash evidence was saved in `debug_runs/three-phone-mainline-20260528-145049/data-plane-crash-20260529-1224/`.
- Root cause: `HttpForwardChunkServer.handleClient()` wrote an error response and then unconditionally flushed the socket in `finally`; if the upstream client disconnected while a large shard response was being written, the flush exception escaped the coroutine and became an AndroidRuntime fatal exception.
- Fix: `HttpDataPlane.kt` now catches per-client data-plane exceptions, treats socket disconnects as request-level warnings, and uses quiet error-response/flush helpers. The accept loop also logs and retries instead of killing the data-plane server.
- Runner fix: `RunPreparedExperimentMain.kt` now applies a default `SubmitRequest` RPC deadline of `420000 ms`, configurable as optional arg 13, so a stuck request becomes a retryable deadline failure instead of hanging forever.
- Verification:
  - `:coordinator:test --offline --no-daemon` passed.
  - `:app:assembleDebug --offline --no-daemon` passed.
  - APK installed on all three phones.
  - `dataplane-fix-probe` completed 3/3 after relaunch.
  - The active `three-main-train512-dataplanefix` run reached 129/129 with no recorded failures, crossing the old `three-main-train512` failure point at request `000084` and completing a clean 128-request training window.

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
- Active `coordinator/config/pipeline.json` is the three-stage LoRA pipeline with `tinyllama_lora_chunk_0/1/2` on NX, Lenovo, and `23043RP34C`.
- Three-phone template added at `coordinator/config/pipeline_three_phone.template.json`. Stage 2 deliberately has blank `deviceId`; with `allowUnlistedDevices=true`, the first extra live phone can fill it.
- Server-side export command for a consistent 3-stage LoRA pipeline:

```bash
NUM_CHUNKS=3 CHUNK_IDX=-1 OUTPUT_DIR=model bash tools/export/export_lora_tinyllama.sh
```

- Regression check passed after the scheduler change:

```powershell
C:\Users\wentaodai\.gradle\wrapper\dists\gradle-8.7-bin\bhs2wmbdwecv87pi65oeuq5iu\gradle-8.7\bin\gradle.bat :coordinator:test --offline --no-daemon
```

Three-phone validation update on 2026-05-28:

- Three fresh LoRA full-split artifacts are present in `model/`:
  - `tinyllama_lora_chunk_0.pte`: 1496232320 bytes
  - `tinyllama_lora_chunk_1.pte`: 1496236800 bytes
  - `tinyllama_lora_chunk_2.pte`: 1672526848 bytes
- `coordinator/config/pipeline.json` was switched to a 3-stage pipeline:
  - stage 0: `NX809J`, `192.168.137.211:26052`
  - stage 1: `Lenovo_L71091`, `192.168.137.124:26052`
  - stage 2: `23043RP34C`, `192.168.137.77:26052`
- The three corresponding PTE files were pre-pushed into each Android app's `files/shards/` cache to avoid large coordinator HTTP downloads during the run.
- Three-stage coordinator status reached `liveNodeCount=3`, `offlineStageCount=0`, and all stages had `assignmentReason=preferred-device-match`.
- Completed three-phone eval-only smoke:
  - CSV: `debug_runs/three-phone-20260528/smoke/results.csv`
  - request `three3-smoke-000001`
  - terminal stage 2, processed chunk 2
  - success=true, elapsed `44233 ms`, local loss `8.2441435`, token accuracy `0.333333` over 3 valid labels
- A three-phone eval128 run was started but is partial because the user intentionally interrupted after one phone was borrowed for another experiment:
  - CSV: `debug_runs/three-phone-20260528/eval128/results.csv`
  - attempted 10 rows, 9 success, first failure `three3-eval128-000009`
  - failure message: `Coordinator dispatch to stage 0 failed: Connection reset`
  - successful subset average elapsed `31586.67 ms`, average loss `7.322979`, token accuracy `0.468468` over 222 valid labels
- Do not report this as completed eval128. It is safe to report as a full-model three-phone smoke plus partial eval evidence.
- A report draft was created at `docs/DEMO_REPORT_DRAFT.md`. Continue polishing the report from there instead of restarting low-level smoke debugging.
- Report figures were generated on 2026-05-28 and updated on 2026-05-29 into `docs/figures/` with:

```powershell
python tools\report\generate_demo_figures.py
```

- The figure script reads the existing formal/fault/three-phone CSVs and perf monitor samples, then emits architecture, BP-free/BP pipeline mind maps, loss, latency, token accuracy, fault-tolerance, partial three-phone, memory, temperature, and coarse current plots. Use these for presentation visuals before starting another phone run.
- For the report design section, use `docs/figures/bpfree_pipeline_mindmap.png` to explain this prototype and `docs/figures/bp_pipeline_mindmap.png` as the conventional 1F1B/BP pipeline contrast. The intended claim is: both can pipeline stages, but conventional BP sends backward gradients across workers while the current BP-free mobile path only sends forward hidden/belief information and lets every phone run local backward/optimizer.
- The figure script also accepts coordinator exports:

```powershell
python tools\report\generate_demo_figures.py --coordinator_run_dir debug_runs\<run>\coordinator-export
```

  Put `metrics.csv` and `stage-timings.csv` in that export directory. When present, the script also emits `coordinator_run_metrics.png`, `stage_local_timing_breakdown.png`, and `stage_forward_timing_breakdown.png`.
- Convenience export command for a live coordinator:

```powershell
.\tools\report\export_coordinator_run.ps1 -RunId <runId> -OutDir debug_runs\<run>\coordinator-export -GenerateFigures
```

- Coordinator experiment-record persistence was extended after the figures work:
  - New SQLite tables: `runs`, `request_metrics`, and `scheduler_events`.
  - Every coordinator-submitted request now writes a structured request metric row, grouped by `runId` inferred from request ids like `train-000123 -> train`.
  - Admin endpoints:
    - `GET /api/v1/runs`
    - `GET /api/v1/runs/{runId}?metrics=1000`
    - `GET /api/v1/runs/{runId}/metrics.csv?limit=100000`
    - `GET /api/v1/runs/{runId}/stage-timings.csv?limit=100000`
  - Scheduler events are now persisted and restored after coordinator restart, not only kept in memory.
  - Worker `RequestEvent` messages are parsed into `stage_timing_metrics` when they contain timing keys such as `localMs`, `executeMs`, `optimizerStepMs`, `forwardMs`, or `totalStageMs`. This enables per-stage timing breakdown plots from coordinator data.
  - Regression command passed:

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

## Easy Real-Dataset Training-Signal Run

Use this when you want a cleaner loss curve than Dolly without falling back to a fully synthetic toy dataset.

Recommended first choice: `rotten_tomatoes`.

- Real movie-review sentiment data.
- About 8.53k train rows, so the full dataset is smaller than Dolly.
- Binary labels, so the response entropy is much lower than Dolly/SFT.
- The formatted response is still text: `The movie review expresses a positive/negative sentiment.`

```bash
python tools/data/prepare_lora_sft_requests.py \
  --model_name tinyllama \
  --dataset rotten_tomatoes \
  --seq_len 64 \
  --limit 512 \
  --attention_mask causal \
  --mask_prompt \
  --max_prompt_tokens 48 \
  --min_valid_labels 4 \
  --request_prefix rt-lora-train \
  --output_dir data/sft_requests/tinyllama_rotten_tomatoes64_train512

# Then run the same prepared-experiment flow against the Rotten Tomatoes manifest.
```

Example train command:

```powershell
C:\Users\wentaodai\.gradle\wrapper\dists\gradle-8.7-bin\bhs2wmbdwecv87pi65oeuq5iu\gradle-8.7\bin\gradle.bat :coordinator:runPreparedExperiment --offline --no-daemon --args="127.0.0.1 50051 data/sft_requests/tinyllama_rotten_tomatoes64_train512/requests.jsonl 0 512 debug_runs/rotten-tomatoes-easy/train/results.csv rt-train 1 0 false true"
```

Second choice: `sst2`.

- Same binary sentiment framing, but the full train split is larger than Dolly.
- Still useful if you want a well-known benchmark-style comparison.

Third choice: `ag_news`.

- Real news-topic classification.
- Four labels: World, Sports, Business, Science and Technology.
- Slightly harder than SST-2, but still much more controlled than Dolly.

Optional QA-style choice: `sciq`.

- Real science multiple-choice questions.
- Useful if you want a task that feels closer to QA, but it may be noisier than SST-2/AG News.

The old `toy` preset still exists only as an internal overfit sanity check. Do not use it as the main report dataset.

## Debugging Discipline

- First read this file, then inspect current `git status`.
- Treat `_inf` success as old compatibility evidence, not the current research target.
- Treat two-phone link success as older infrastructure evidence, not the mainline result.
- Do not run `_inf` artifacts again unless explicitly asked or using them as a controlled baseline.
- Keep `executorch/` untracked and do not push it to the project GitHub.
- If a phone exits, preserve the request id, coordinator request detail, `logcat`, process status, and tombstone evidence before changing code.
- Any fix must explain why the crash happens on the observed stage and shape.

## Current Label-Only Rotten Tomatoes Mobile Run

Updated: 2026-05-31 00:20 Asia/Shanghai.

This is the current real-device quality run. Do not reset to old Dolly or smoke tests unless the user explicitly asks.

Run pointer:

```powershell
Get-Content debug_runs\CURRENT_LABEL_ONLY_RUN.txt
# debug_runs\label-only-demo-20260531-0010
```

Coordinator:

- gRPC: `127.0.0.1:50051`
- admin HTTP: `127.0.0.1:18080`
- stdout: `debug_runs\label-only-demo-20260531-0010\coordinator.stdout.log`
- stderr: `debug_runs\label-only-demo-20260531-0010\coordinator.stderr.log`

Active route:

- stage 0: `NX809J`, `192.168.137.211:26052`, shard `tinyllama_lora_chunk_0_seq128`
- stage 1: `Lenovo_L71091`, `192.168.137.124:26062`, shard `tinyllama_lora_chunk_1_seq128`
- stage 2: `Pixel_10_Pro_XL`, `192.168.137.139:26052`, shard `tinyllama_lora_chunk_2_seq128`

Prepared data:

- train: `data\sft_requests\tinyllama_rotten_tomatoes128_label_train64_prompt24_lr3e4\requests.jsonl`
- validation: `data\sft_requests\tinyllama_rotten_tomatoes128_label_val256_prompt24\requests.jsonl`
- Both manifests must contain `label_choices` with exactly:
  - `" positive"` token id `6374`
  - `" negative"` token id `8178`

Current phase:

- The first `eval-before` was intentionally stopped because the prepared Rotten Tomatoes manifests were label-skewed.
- Result CSV: `debug_runs\label-only-demo-20260531-0010\eval-before-val256-window3\results.csv`
- At 2026-05-31 00:20 it had reached `16/256` terminal successes, `0` failures, label-choice accuracy `12/16 = 75%`.
- At stop time it had reached about `78` terminal successes with `0` failures, but both existing manifests were all-positive:
  - train64: `64/64` positive
  - val256: `256/256` positive
- Do not use this aborted all-positive run as the quality result. It only shows that the three-phone seq128 route can keep processing requests.
- `token_accuracy` is expected to remain mostly `0.0` in this label-only setup because it checks the full-vocabulary target token. Use `label_choice_accuracy` as the meaningful binary-classification metric.
- `local_loss` is still recorded and should be summarized as mean/std for before/after comparison.

Regenerate balanced data on the server/export environment after pulling the current script update:

```bash
python tools/data/prepare_lora_sft_requests.py \
  --model_name tinyllama \
  --dataset rotten_tomatoes \
  --split train \
  --seq_len 128 \
  --limit 64 \
  --attention_mask causal \
  --mask_prompt \
  --response_style label \
  --no_append_eos \
  --max_prompt_tokens 24 \
  --min_valid_labels 1 \
  --learning_rate 0.0003 \
  --shuffle_seed 20260531 \
  --balance_labels \
  --request_prefix rt-label-train-balanced \
  --output_dir data/sft_requests/tinyllama_rotten_tomatoes128_label_train64_prompt24_lr3e4_balanced

python tools/data/prepare_lora_sft_requests.py \
  --model_name tinyllama \
  --dataset rotten_tomatoes \
  --split validation \
  --seq_len 128 \
  --limit 256 \
  --attention_mask causal \
  --mask_prompt \
  --response_style label \
  --no_append_eos \
  --max_prompt_tokens 24 \
  --min_valid_labels 1 \
  --shuffle_seed 20260531 \
  --balance_labels \
  --request_prefix rt-label-val-balanced \
  --output_dir data/sft_requests/tinyllama_rotten_tomatoes128_label_val256_prompt24_balanced
```

Before running, verify both manifests are balanced:

```powershell
foreach ($dir in @(
  'tinyllama_rotten_tomatoes128_label_train64_prompt24_lr3e4_balanced',
  'tinyllama_rotten_tomatoes128_label_val256_prompt24_balanced'
)) {
  $path = "data\sft_requests\$dir\requests.jsonl"
  $items = Get-Content $path | ForEach-Object {
    $o = $_ | ConvertFrom-Json
    $o.text.response.Trim()
  }
  $items | Group-Object
}
```

Eval-before command:

```powershell
.\gradlew.bat :coordinator:runPreparedPipelineExperiment --offline --no-daemon --args="127.0.0.1 50051 data/sft_requests/tinyllama_rotten_tomatoes128_label_val256_prompt24/requests.jsonl 0 256 debug_runs\label-only-demo-20260531-0010\eval-before-val256-window3\results.csv rt-label-eval-before-20260531-0014 1 0 true true 18 10000 420000 3"
```

After eval-before completes, run train64:

```powershell
.\gradlew.bat :coordinator:runPreparedPipelineExperiment --offline --no-daemon --args="127.0.0.1 50051 data/sft_requests/tinyllama_rotten_tomatoes128_label_train64_prompt24_lr3e4/requests.jsonl 0 64 debug_runs\label-only-demo-20260531-0010\train64-window3\results.csv rt-label-train64-window3-20260531 1 0 false true 18 10000 420000 3"
```

Then rerun the same validation manifest as `eval-after`:

```powershell
.\gradlew.bat :coordinator:runPreparedPipelineExperiment --offline --no-daemon --args="127.0.0.1 50051 data/sft_requests/tinyllama_rotten_tomatoes128_label_val256_prompt24/requests.jsonl 0 256 debug_runs\label-only-demo-20260531-0010\eval-after-val256-window3\results.csv rt-label-eval-after-20260531 1 0 true true 18 10000 420000 3"
```

Useful live checks:

```powershell
$run = Get-Content debug_runs\CURRENT_LABEL_ONLY_RUN.txt
$evalDir = Join-Path $run 'eval-before-val256-window3'
$rows = Import-Csv (Join-Path $evalDir 'results.csv')
$correct = ($rows | Measure-Object -Property label_choice_correct -Sum).Sum
$count = ($rows | Measure-Object -Property label_choice_count -Sum).Sum
"rows=$($rows.Count) success=$(($rows | ? { $_.success -eq 'true' }).Count) failed=$(($rows | ? { $_.success -ne 'true' }).Count) label_choice=$correct/$count"
Invoke-RestMethod http://127.0.0.1:18080/api/v1/status
Invoke-RestMethod 'http://127.0.0.1:18080/api/v1/worker-telemetry.csv?limit=20'
```

## Latest Balanced Label-Only Result

Updated: 2026-05-31 03:50 Asia/Shanghai.

Run:

```text
debug_runs\label-balanced-demo-20260531-0102
```

Summary file:

```text
debug_runs\label-balanced-demo-20260531-0102\SUMMARY.md
```

Data:

- train: `data/sft_requests/tinyllama_rotten_tomatoes128_label_train64_prompt24_lr3e4_balanced/requests.jsonl`
- validation: `data/sft_requests/tinyllama_rotten_tomatoes128_label_val256_prompt24_balanced/requests.jsonl`
- validation label split: `128 negative / 128 positive`

Completed phases:

| phase | rows | failures | label-choice acc | avg local loss | notes |
|---|---:|---:|---:|---:|---|
| eval-before | 256 | 0 | 153/256 = 0.5977 | 10.5673 | baseline before mobile training |
| train64 | 64 | 0 | 34/64 = 0.5312 | 10.2236 | `eval_only=false`; mobile training path exercised |
| eval-after | 256 | 0 | 123/256 = 0.4805 | 7.7905 | after train64 |

Per-class validation:

| phase | negative acc | positive acc |
|---|---:|---:|
| eval-before | 63/128 = 0.4922 | 90/128 = 0.7031 |
| eval-after | 2/128 = 0.0156 | 121/128 = 0.9453 |

Interpretation:

- System result is good: three-phone seq128 eval-before, train64, and eval-after all completed with `0` terminal failures.
- Training path is real: train64 used `eval_only=false`, and worker events showed optimizer steps.
- Quality result is not good for this hyperparameter setting: label-choice accuracy dropped by `0.1172`, while avg local loss dropped by `2.7768`.
- The after model became strongly positive-biased. This means the current `train64/lr=3e-4` setup is useful as an end-to-end system demo, but not yet a good quality-preserving training recipe.
- Useful fault-tolerance evidence: request `rt-label-balanced-train64-20260531-0214-000057` had a coordinator dispatch read timeout/retry and still ended as CSV success with `elapsed_ms=413879`.

## AG News Mainline Recovery Checkpoint

Updated: 2026-05-31 19:47 Asia/Shanghai.

Current run pointer:

```text
debug_runs\agnews-phone-mainline-20260531-141409
```

Three-phone AG News run status:

- `eval-before256-window3`: completed, `256/256` success, `0` failures.
- `train512-window3`: stopped early at record `190`; records `188..190` failed after Pixel/stage 2 lost Wi-Fi and coordinator evicted the stage.
- `train512-cont-from191-window3`: stopped early at record `301`; records `299..301` failed for the same Pixel/stage 2 Wi-Fi / route loss.
- `train512-cont-from299-window3`: active recovery continuation from record index `299`, limit `213`.

Important detail: `train512-cont-from299-window3` intentionally overlaps records `299..301` because those rows failed in `train512-cont-from191-window3`. As of this checkpoint it has written `11` terminal rows, all successful, through record `309`.

Live network status at this checkpoint:

- Coordinator `/api/v1/status`: `3` live nodes, `0` offline stages.
- Stage 0: `NX809J`, `192.168.137.211:26052`.
- Stage 1: `Lenovo_L71091`, `192.168.137.124:26052`.
- Stage 2: `Pixel_10_Pro_XL`, `192.168.137.139:26052`.
- Lenovo -> Pixel data-plane probe to `192.168.137.139:26052` succeeds and returns worker HTTP `404 Unsupported path`, which is expected for a raw HTTP probe against the worker port.

Current active command:

```powershell
.\gradlew.bat :coordinator:runPreparedPipelineExperiment --offline --no-daemon --args="127.0.0.1 50051 data/sft_requests/tinyllama_agnews128_label_train512_seed20260531/requests.jsonl 299 213 debug_runs\agnews-phone-mainline-20260531-141409\train512-cont-from299-window3\results.csv agnews-train512-cont299-20260531 1 0 false true 18 10000 420000 3"
```

If this continuation completes cleanly, it should reach `213` successful rows and last record index `511`. Then run `eval-after256-window3` on:

```text
data/sft_requests/tinyllama_agnews128_label_eval256_seed20260531/requests.jsonl
```

with offset `0`, limit `256`, `evalOnly=true`, `includeLabels=true`, retry count `18`, retry delay `10000 ms`, submit deadline `420000 ms`, and window `3`.

Update: 2026-05-31 20:51 Asia/Shanghai.

`train512-cont-from299-window3` stopped early after Pixel/stage 2 dropped and re-registered with a new IP:

- rows: `51`
- successes: `48`
- failures: `3`
- failed original record indices: `340`, `348`, `349`
- final summary: `selected=213 submitted=51 skipped=0 succeeded=48 failed=3`
- failure message: `Stage 2 has no live worker.`
- Pixel reappeared as `192.168.137.180`; coordinator again reported `3` live nodes and `0` offline stages.
- Lenovo -> Pixel `192.168.137.180:26052` probe succeeded with expected worker HTTP `404 Unsupported path`.

Because record `340` failed while later in-flight records `341..347` succeeded, a gap-only recovery manifest was generated to avoid duplicate training:

```text
data\sft_requests\tinyllama_agnews128_label_train512_seed20260531\requests_recovery_340_348_511.jsonl
```

This manifest contains exactly `165` rows: original record `340` followed by original records `348..511`.

Active recovery phase:

```text
debug_runs\agnews-phone-mainline-20260531-141409\train512-recovery-340-348to511-window3
```

Start command:

```powershell
.\gradlew.bat :coordinator:runPreparedPipelineExperiment --offline --no-daemon --args="127.0.0.1 50051 data/sft_requests/tinyllama_agnews128_label_train512_seed20260531/requests_recovery_340_348_511.jsonl 0 165 debug_runs\agnews-phone-mainline-20260531-141409\train512-recovery-340-348to511-window3\results.csv agnews-train512-recovery340-20260531 1 0 false true 18 10000 420000 3"
```

Note: in this recovery phase, CSV `record_index` is relative to the recovery manifest. Use `dataset_index` or the recovery manifest definition above to map back to original train512 records.

Update: 2026-05-31 20:57 Asia/Shanghai.

Immediately after starting `train512-recovery-340-348to511-window3`, Pixel/stage 2 again stopped heartbeating even though Android Wi-Fi still showed it connected to `dwellerLAPTOP 5740` with IP `192.168.137.180` and strong RSSI. Coordinator status dropped to `2` live nodes and `1` offline stage. The Pixel app process `com.example.sid_trainer` was still present, so the recovery action was to restart only the Pixel worker app:

```powershell
adb -s 58151FDCQ006A8 shell am force-stop com.example.sid_trainer
adb -s 58151FDCQ006A8 shell am start -n com.example.sid_trainer/.MainActivity
```

This restored coordinator status to `3` live nodes and `0` offline stages. The in-flight recovery runner was still within its retry window and successfully recovered:

- recovery rows: `3`
- successes: `3`
- failures: `0`
- covered original failed dataset indices: `38427`, `96958`, `94424`
- avg local loss over the first 3 recovery rows: `6.3168`

The first three rows correspond to original train512 record `340`, `348`, and `349`. Their high elapsed times (`~270-296 s`) are recovery/retry latency, not normal steady-state throughput.

## Pixel Stage-2 Drop Root-Cause Diagnosis

Updated: 2026-05-31 21:00 Asia/Shanghai.

The recurring Pixel/stage-2 failures are not model/OOM failures. The immediate coordinator-visible failure mode is:

```text
Stage 2 has no live worker.
```

Evidence:

- Coordinator repeatedly evicts Pixel/stage 2 by lease expiry, then later registers the same device again.
- Pixel often has good RSSI and link speed when checked, so this is not simple weak signal.
- Earlier Pixel Wi-Fi logs showed `CMD_IP_REACHABILITY_FAILURE`, `LOST_PROVISIONING`, `192.168.137.1 NUD_FAILED`, `CMD_IP_CONFIGURATION_LOST`, and association timeout/rejection events against the Windows hotspot.
- In the latest recovery attempt, Android Wi-Fi still showed connected with IP `192.168.137.180`, but coordinator had already expired stage 2. The app process still existed, but restarting only `com.example.sid_trainer` immediately re-registered Pixel and allowed the in-flight runner retries to recover.
- After restart, ExecuTorch training on Pixel succeeded again, including `TrainingModule.executeForwardBackward()` and optimizer steps, so the failure was outside the model execution path.

Interpretation:

1. Pixel is unusually sensitive to the Windows Mobile Hotspot path. Disabling captive portal, poor-network tests, and ipreach disconnect avoids some Android policy disconnects, but it does not fix Windows hotspot peer routing/ARP/NUD behavior, DHCP/IP churn, or driver-level association quirks.
2. The current Android worker is an Activity-driven app, not a foreground service with explicit wake/network recovery policy. When the network path changes or the process becomes cached/backgrounded, the gRPC heartbeat can stop long enough for the coordinator lease to expire.
3. Therefore the robust fix is not another one-off phone setting. The system should add worker-side auto recovery: foreground service, wake/Wi-Fi lock during runs, connectivity callback, heartbeat failure detection, channel rebuild, local server restart if needed, and explicit re-registration. Coordinator/runner retry logic already helps when the worker returns inside the retry window.

Current run status at this diagnosis checkpoint:

- `train512-recovery-340-348to511-window3`: `15/15` success, `0` failures.
- Coordinator: `3` live nodes, `0` offline stages.

Update: 2026-05-31 21:52 Asia/Shanghai.

The gap-only recovery phase finished successfully:

- phase: `train512-recovery-340-348to511-window3`
- rows: `165`
- successes: `165`
- failures: `0`
- avg local loss: `5.9178`
- label-choice: `57/165 = 0.3455`

Together with previous completed/partial phases, this covers the intended train512 set:

- `train512-window3`: original records `0..187` succeeded; `188..190` failed.
- `train512-cont-from191-window3`: original records `191..298` succeeded; `299..301` failed.
- `train512-cont-from299-window3`: original records `299..339` succeeded except `340`, and `341..347` succeeded; `348..349` failed; stopped before `350..511`.
- `train512-recovery-340-348to511-window3`: recovered original record `340` and original records `348..511`, all successful.

Started eval-after:

```text
debug_runs\agnews-phone-mainline-20260531-141409\eval-after256-window3
```

Command:

```powershell
.\gradlew.bat :coordinator:runPreparedPipelineExperiment --offline --no-daemon --args="127.0.0.1 50051 data/sft_requests/tinyllama_agnews128_label_eval256_seed20260531/requests.jsonl 0 256 debug_runs\agnews-phone-mainline-20260531-141409\eval-after256-window3\results.csv agnews-eval-after-20260531 1 0 true true 18 10000 420000 3"
```

At eval-after start, coordinator status was `3` live nodes and `0` offline stages.

Update: 2026-05-31 22:33 Asia/Shanghai.

`eval-after256-window3` is currently stalled at row `60` during transient retry, not because of model execution:

- rows written: `60`
- successes: `60`
- failures: `0`
- avg local loss so far: `5.5456`
- label-choice so far: `16/60 = 0.2667`
- runner is still alive and retrying request indices around `60..61`
- failure message in Gradle log: `Downstream forwarding failed: failed to connect to /192.168.137.139 (port 26052) from /192.168.137.124 ... after 15000ms`

Important network finding:

- Pixel/stage 2 Wi-Fi is connected to `dwellerLAPTOP 5740` with IP `192.168.137.139`, strong RSSI, and Pixel is listening on `*:26052`.
- Coordinator and the Windows host can connect to Pixel `192.168.137.139:26052`.
- NX and Lenovo can reach each other in both directions, including `26052`.
- Lenovo -> Pixel and Pixel -> Lenovo fail at the peer path: ping loses all packets and `nc` to `26052` times out.
- Restarting the Pixel app restored coordinator live status but did not restore the Lenovo <-> Pixel data path.

Interpretation: this checkpoint is a Windows hotspot / Pixel peer-to-peer data-plane failure. Coordinator heartbeats can be live while inter-stage forwarding is broken. This is separate from AG News label accuracy and separate from ExecuTorch OOM/model correctness.

Metric interpretation note:

- `label_choice_accuracy` in train CSV rows is useful only as a rough online sanity signal. It is computed after each training request on that same request, after the request may already have updated local LoRA weights, and the train stream distribution/order is not the held-out metric.
- The meaningful task-quality comparison is held-out `eval-before256-window3` versus completed `eval-after256-window3`.
- Server probe/control numbers are cleaner because they run in one process with a consistent model state and no phone network/retry/data-plane effects.

Update: 2026-05-31 22:41 Asia/Shanghai.

Recovery attempt details:

- Restarting Pixel app did not restore Lenovo -> Pixel reachability.
- Toggling Pixel Wi-Fi did not restore Lenovo -> Pixel reachability.
- Toggling Lenovo Wi-Fi restored Lenovo -> Pixel `26052`; the probe returned the expected `HTTP/1.1 404 Error` / `Unsupported path`.
- Coordinator then showed `3` live nodes and `0` offline stages again.
- The in-flight eval-after runner recovered within its retry window and wrote record index `61`, with elapsed time `760015 ms`; that high latency is retry recovery time, not steady-state runtime.

At this checkpoint:

- `eval-after256-window3`: rows `61`, successes `61`, failures `0`
- avg local loss so far: `5.5633`
- label-choice so far: `16/61 = 0.2623`

Note: after the Lenovo Wi-Fi refresh, Lenovo re-registered on gRPC port `26062` rather than `26052`. Coordinator routing updated to use `192.168.137.124:26062`.

Update: 2026-05-31 23:58 Asia/Shanghai.

`eval-after256-window3` completed successfully:

- rows: `256`
- successes: `256`
- failures: `0`
- avg local loss: `5.6342`
- label-choice: `80/256 = 0.3125`
- Gradle summary: `selected=256 submitted=256 skipped=0 succeeded=256 failed=0`
- runner duration: `BUILD SUCCESSFUL in 2h 5m 43s`

Final held-out comparison for this phone mainline:

| phase | success | failures | label-choice acc | avg local loss |
|---|---:|---:|---:|---:|
| `eval-before256-window3` | 256/256 | 0 | 86/256 = 0.3359 | 7.6778 |
| `eval-after256-window3` | 256/256 | 0 | 80/256 = 0.3125 | 5.6342 |

Conclusion:

- The system path is proven end-to-end on three phones with seq128 TinyLlama LoRA shards, bounded in-flight scheduling, transient retry, and documented recovery over Pixel/stage-2 network drops.
- The quality result is mixed: local loss drops clearly, but held-out constrained label-choice accuracy drops by `0.0234`. Do not present this as an accuracy-improving AG News run.
- The major systems issue observed during the run was Windows hotspot peer data-plane instability, especially Lenovo -> Pixel, not ExecuTorch OOM or model execution failure.

Update: 2026-06-01 00:38 Asia/Shanghai.

Started a new clean AG News phone run after Pixel/phones were moved to stable/device-MAC behavior and seq128 LoRA checkpoints were cleared.

- Run pointer: debug_runs\CURRENT_AGNEWS_RUN.txt -> debug_runs\agnews-clean-device-mac-20260601-0038
- Network at start: NX 192.168.137.98, Lenovo 192.168.137.251, Pixel 192.168.137.60 on dwellerLAPTOP 5740.
- Peer probes passed after worker restart: NX -> Lenovo, Lenovo -> Pixel, Pixel -> Lenovo, PC -> all worker ports.
- Deleted only tinyllama_lora_chunk_*_seq128.latest.sidckpt on the three devices so the run starts from clean seq128 adapters.
- Active phase: eval-before256-window3; early progress 16/16 successes, 0 failures.
- Monitor should start train512-window3 only after eval-before reaches 256 successes and 0 failures.

Update: 2026-06-01 00:53 Asia/Shanghai.

Clean run health check:

- Active phase: eval-before256-window3.
- Progress: 32 rows, 32 successes, 0 failures.
- Coordinator: 3 live nodes, 0 offline stages.
- Current route: NX 192.168.137.98:26052 -> Lenovo 192.168.137.251:26052 -> Pixel 192.168.137.5:26052.
- Peer probes passed with the current route: NX -> Lenovo and Lenovo -> Pixel returned worker 404 / Unsupported path.
- Pixel has already re-registered once from the start IP to 192.168.137.5, so future checks should read the live route from coordinator status rather than assuming the initial Pixel IP.
- monitor-ag-news-clean-run is ACTIVE every 10 minutes and should advance eval-before -> train512 -> eval-after without returning to smoke tests.

Update: 2026-06-01 01:10 Asia/Shanghai.

Clean run transient recovery during eval-before:

- `eval-before256-window3` reached 84 successes and 0 failures, then the runner entered transient retries for request indices around 84..87.
- Coordinator showed `liveNodeCount=2`, `offlineStageCount=1`; Gradle retry messages included `Stage 2 has no live worker` and `Downstream route is not ready`.
- Pixel still had ADB, `wlan0` IP 192.168.137.5, and the app process/server port existed, but coordinator had a stale/expired stage-2 route pointing at 192.168.137.139.
- Recovery action: force-stopped and relaunched only the Pixel worker app during eval-before. This is acceptable for eval-before because it is eval-only and no trained LoRA state exists yet.
- After relaunch, coordinator returned to 3 live nodes, 0 offline stages, route stage1 -> Pixel 192.168.137.5:26052, and Lenovo -> Pixel peer probe returned the expected worker 404 / Unsupported path.
- The in-flight runner recovered inside the retry window and advanced to 86 successes, 0 failures. Do not use this kind of single-stage relaunch during the train phase unless checkpoint/restore has been explicitly validated.

Update: 2026-06-01 02:09:48 Asia/Shanghai.

Clean run automation advanced phase: eval-before completed with 256 successes and 0 failures; started train512-window3 manually after PowerShell alias issue, pid 30860.


Update: 2026-06-01 02:11:26 Asia/Shanghai.

Clean run automation phase-start correction: the first train512 launch failed because Start-Process argument quoting split --args and Gradle treated 50051 as a task. Wrote train512-window3/start.ps1 with eval-before-style quoted args and relaunched train512-window3, pid 8016.


Update: 2026-06-01 04:41:17 Asia/Shanghai.

Clean run automation advanced phase: train512 completed with 512 successes and 0 failures; started eval-after256-window3 via start-eval-after.ps1, pid 34720.


Update: 2026-06-01 05:25:10 Asia/Shanghai.

Clean run eval-after recovery: eval-after256-window3 stopped at 115 successes and 3 failed in-flight rows (record_index 115..117) after Pixel DHCP changed from 192.168.137.5 to 192.168.137.135. Pixel worker was relaunched during eval-only recovery; coordinator route updated to stage1 -> 192.168.137.135:26052 and Lenovo -> Pixel probe passed. Started eval-after256-recovery-from115-window3 with offset 115, limit 141, pid 30316. This preserves the trained LoRA state only if checkpoints were restored on relaunch; label-quality comparison should note this recovery caveat.


Update: 2026-06-01 05:59:31 Asia/Shanghai.

Clean AG News device-MAC run final stitched result:

| phase | success | failures | label-choice acc | avg local loss |
|---|---:|---:|---:|---:|
| eval-before256-window3 | 256/256 | 0 | 86/256 = 0.3359 | 7.6778 |
| train512-window3 | 512/512 | 0 | 228/512 = 0.4453 | 6.3527 |
| eval-after stitched | 256/256 | 0 used rows, 3 interrupted rows ignored | 80/256 = 0.3125 | 5.6238 |

Held-out delta after train512: label-choice accuracy -0.0234, avg local loss -2.0540.

Recovery caveat: eval-after256-window3 stopped after 115 successful rows plus 3 failed in-flight rows when Pixel/stage 2 changed IP from 192.168.137.5 to 192.168.137.135. Pixel was relaunched and eval-after256-recovery-from115-window3 completed the remaining 141 rows. Treat the stitched quality result as valid only if checkpoint restore preserved the trained LoRA state across that Pixel relaunch; system-wise, this is a successful documented recovery from data-plane/IP churn.
