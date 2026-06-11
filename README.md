# Mobile BP-Free Modular Distillation Prototype

This repository is a prototype for deploying **BP-free / gradient-decoupled modular training** on phones with **ExecuTorch**.

The current goal is not "full distributed backpropagation on mobile". The goal is:

- split a causal LM into multiple `.pte` shards
- run each shard on a different phone
- keep **stage-to-stage communication forward-only**
- let each shard optimize with its own **local CE loss** plus **adjacent KD-style signals**
- provide a control plane that can register devices, distribute model artifacts, route requests, and keep the pipeline alive

This repository already contains a working control plane, Android workers, model artifact distribution, and an end-to-end request path. It is still a **prototype**, not a production scheduler.

## What This Repository Is

- An **Android worker app** that downloads a shard, runs it with ExecuTorch, and forwards tensors to the next worker.
- A **JVM coordinator** that handles registration, heartbeat, routing, request submission, artifact serving, and request lifecycle tracking.
- A **model export workflow** for turning chunked causal LM modules into ExecuTorch `.pte` shards that match the current mobile pipeline contract.

## What This Repository Is Not

- Not full cross-stage backpropagation.
- Not a final parallel scheduler.
- Not a production-ready mobile training platform.
- Not a general-purpose distributed training framework.

The current design is intentionally aligned with **module-decoupled / BP-free training**, where shard-to-shard traffic carries **representations and KD-related signals**, not true backward gradients.

## Current Status

What works today:

- Android worker registration and heartbeat
- dynamic routing from coordinator to workers
- worker-to-worker HTTP + protobuf forwarding
- automatic shard download from coordinator to phones
- request submission through coordinator
- request tracking, retry, drain, evict, reload, and admin status APIs
- ExecuTorch Android runtime integration in the worker

What is still incomplete or unstable:

- true scheduled parallel training across shards
- robust benchmarking
- memory-efficient execution for large shards on low-memory phones
- artifact/version lifecycle beyond the current prototype
- hardened production deployment, security, and observability

## Architecture

The system currently has three layers:

1. **Export layer**
   - A server-side script exports chunked modules into `.pte` shards.
   - Each shard keeps its own local training logic.

2. **Control plane**
   - The coordinator maps each device to a pipeline stage.
   - It serves shard files over HTTP.
   - It handles registration, heartbeat, request submission, routing state, and admin operations.

3. **Mobile data plane**
   - Each Android phone runs one worker.
   - A worker downloads its shard, starts a local HTTP data server, executes its chunk, and forwards results to the next worker.

Today the request path is still pipeline-shaped, but the intended research direction is to evolve from simple relay-style execution into a **scheduled, decoupled mobile training system**.

## Repository Layout

```text
.
|- app/                 Android worker
|- coordinator/         JVM coordinator and admin server
|- model/               Local shard artifacts (ignored by git)
|- tools/export/        Server-side model export scripts
|- CONTRIBUTING.md      Collaboration and handoff guide
|- environment.yml      Minimal conda environment for export
|- requirements-export.txt  Minimal pip dependencies for export
|- requirement.txt      Raw environment snapshot from a working machine
`- README.md
```

Key files:

- [app/src/main/java/com/example/sid_trainer/MainActivity.kt](app/src/main/java/com/example/sid_trainer/MainActivity.kt)
- [app/src/main/java/com/example/sid_trainer/GrpcManager.kt](app/src/main/java/com/example/sid_trainer/GrpcManager.kt)
- [app/src/main/java/com/example/sid_trainer/NativeShardRunner.kt](app/src/main/java/com/example/sid_trainer/NativeShardRunner.kt)
- [app/src/main/proto/sid.proto](app/src/main/proto/sid.proto)
- [coordinator/src/main/kotlin/com/example/sid_coordinator/CoordinatorMain.kt](coordinator/src/main/kotlin/com/example/sid_coordinator/CoordinatorMain.kt)
- [coordinator/config/pipeline.json](coordinator/config/pipeline.json)
- [coordinator/config/pipeline.example.json](coordinator/config/pipeline.example.json)
- [tools/export/sid_export_forward_mobile.py](tools/export/sid_export_forward_mobile.py)
- [tools/export/sid_export_mobile.py](tools/export/sid_export_mobile.py)
- [tools/export/export_bpfree_tinyllama.sh](tools/export/export_bpfree_tinyllama.sh)
- [tools/export/export_forward_belief_tinyllama.sh](tools/export/export_forward_belief_tinyllama.sh)
- [CONTRIBUTING.md](CONTRIBUTING.md)

## Data Contract Between Shards

The current shard contract is defined in [sid.proto](app/src/main/proto/sid.proto).

Each `ForwardChunkRequest` carries:

- `hidden_states`
- `attention_mask`
- `position_ids`
- `labels`
- `shift_log_p_prev`

Each `ForwardChunkResponse` returns:

- `local_loss`
- `output_hidden_states`
- `output_shift_log_p`

Important design constraint:

- **No stage-to-stage backward gradient RPC is used in this repository.**
- The coordinator and workers assume **forward-only inter-stage communication**.
- Any local optimization logic must live **inside each `.pte` shard**.

## Environment Setup

### 1. Android / coordinator workspace

Requirements:

- JDK 17
- Android Studio
- Android SDK 34
- `adb`

Build everything:

```bash
./gradlew :coordinator:build :app:assembleDebug
```

On Windows:

```powershell
./gradlew.bat :coordinator:build :app:assembleDebug
```

APK output:

- `app/build/outputs/apk/debug/app-debug.apk`

### 2. Export server environment

Use the minimal export environment first:

- [environment.yml](environment.yml)
- [requirements-export.txt](requirements-export.txt)

Create the export environment:

```bash
conda env create -f environment.yml
conda activate mobile-bpfree-export
```

This environment is intended for the **server-side export machine**, not for Android.

The file [requirement.txt](requirement.txt) is kept only as a **raw environment snapshot** from a working machine. It is not meant to be installed with `pip install -r requirement.txt`, and it is not a clean collaboration entrypoint.

Keep the Python `executorch` package version aligned with the Android `executorch-android` dependency when possible.

## Exporting Shards

The original export script snapshot is preserved at:

- [tools/export/sid_export_original_backup.py](tools/export/sid_export_original_backup.py)

The recommended first mobile system-test export script is:

- [tools/export/sid_export_forward_mobile.py](tools/export/sid_export_forward_mobile.py)

This script exports **forward-only** `.pte` shards for Android `Module.execute()`. It can export either:

- relay-only `_inf` shards for pure system validation
- forward belief/local-loss shards with local CE/KD loss outputs, but without local backward/optimizer

The joint forward/backward export script is:

- [tools/export/sid_export_mobile.py](tools/export/sid_export_mobile.py)

The joint script uses PyTorch/ExecuTorch forward-backward export and targets Android `TrainingModule.executeForwardBackward()`. This is the full BP-free training artifact path for the current prototype: each shard can run local loss, local backward, and a local optimizer step, while the system still does **not** add any cross-stage backward-gradient RPC. Use `sid_export_forward_mobile.py` only when validating the system routing or the forward belief signal path.

Why these mobile scripts exist:

- `chunk 0` can omit `prev_log_probs`
- stage-to-stage transport defaults to `float16`
- default `seq_len` is shorter
- XNNPACK lowering is opt-in for safer first runs
- it is tuned for the current worker/coordinator contract

### Recommended starting point

For the current phones, start with:

- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- `num_chunks=4`
- `seq_len=64`
- `transport_dtype=float16`

Two-phone forward-only compatibility export:

```bash
python tools/export/sid_export_forward_mobile.py \
  --model_name tinyllama \
  --num_chunks 4 \
  --chunk_idx 0,1 \
  --seq_len 64 \
  --transport_dtype float16 \
  --relay_only \
  --artifact_prefix tinyllama \
  --artifact_suffix _inf \
  --output_dir model
```

This writes:

- `model/tinyllama_chunk_0_inf.pte`
- `model/tinyllama_chunk_1_inf.pte`

Two-phone BP-free training export with local CE/KD loss/backward:

```bash
bash tools/export/export_bpfree_tinyllama.sh
```

Equivalent explicit command:

```bash
python tools/export/sid_export_mobile.py \
  --model_name tinyllama \
  --num_chunks 4 \
  --chunk_idx 0,1 \
  --seq_len 64 \
  --transport_dtype float16 \
  --artifact_prefix tinyllama \
  --output_dir model
```

Forward belief/local-loss export, for isolating the forward data plane without local backward:

```bash
bash tools/export/export_forward_belief_tinyllama.sh
```

For all four chunks:

```bash
python tools/export/sid_export_forward_mobile.py \
  --model_name tinyllama \
  --num_chunks 4 \
  --chunk_idx -1 \
  --seq_len 64 \
  --transport_dtype float16 \
  --relay_only \
  --artifact_prefix tinyllama \
  --artifact_suffix _inf \
  --output_dir model
```

Smaller sanity-check model:

```bash
python tools/export/sid_export_forward_mobile.py \
  --model_name smollm2_360m \
  --num_chunks 4 \
  --chunk_idx -1 \
  --seq_len 64 \
  --artifact_prefix smollm2_360m \
  --output_dir model
```

Single-chunk joint training-runtime export:

```bash
python tools/export/sid_export_mobile.py \
  --model_name tinyllama \
  --num_chunks 4 \
  --chunk_idx 0 \
  --seq_len 64 \
  --artifact_prefix tinyllama \
  --output_dir model
```

Two-phone LoRA training-runtime export:

```bash
CHUNK_IDX=0,1 OUTPUT_DIR=model bash tools/export/export_lora_tinyllama.sh
```

Equivalent explicit command:

```bash
python tools/export/sid_export_mobile.py \
  --model_name tinyllama \
  --num_chunks 4 \
  --chunk_idx 0,1 \
  --seq_len 64 \
  --transport_dtype float16 \
  --artifact_prefix tinyllama_lora \
  --output_dir model \
  --lora_rank 4 \
  --lora_alpha 16 \
  --lora_targets q_proj,v_proj
```

LoRA export freezes the base chunk weights and makes only the injected adapter tensors trainable inside `TrainingModule.executeForwardBackward()`. Cross-stage traffic is unchanged: hidden states and belief/log-prob signals go forward; no backward-gradient RPC is added.

Prepare real SFT samples as SID requests:

```bash
python tools/data/prepare_lora_sft_requests.py \
  --model_name tinyllama \
  --dataset dolly \
  --seq_len 64 \
  --limit 32 \
  --attention_mask causal \
  --request_prefix dolly-lora \
  --output_dir data/sft_requests/tinyllama_dolly64
```

Submit one prepared sample:

```bash
./gradlew :coordinator:runSubmitPreparedRequest \
  --args="127.0.0.1 50051 data/sft_requests/tinyllama_dolly64/requests.jsonl 0"
```

Important: the current mobile shard contract starts stage 0 from `hidden_states`, not token IDs. The data preparation script tokenizes text and computes input embeddings on the server, then stores those embedded tensors for coordinator submission. Use `--attention_mask zero` only when reproducing the older synthetic demo request shape; real SFT samples should use the default causal mask.

Notes:

- exported `.pte` files are **not committed** to git
- place the generated files under `model/` or another local artifact path referenced by the coordinator config
- after replacing coordinator-served artifacts, call `POST /api/v1/routing/reload` and restart the Android workers
- `--relay_only` is recommended only for the first end-to-end mobile system test. It preserves the stage-to-stage tensor relay while skipping local CE/KD loss inside the `.pte`, which avoids conflating coordinator/data-plane validation with model-loss runtime compatibility.
- Full BP-free training uses `sid_export_mobile.py` / `export_bpfree_tinyllama.sh`, not the markerless forward-only script.

## Coordinator Setup

The coordinator module is documented in more detail at [coordinator/README.md](coordinator/README.md).

The current default config file is:

- [coordinator/config/pipeline.json](coordinator/config/pipeline.json)

The repository also includes a sanitized example:

- [coordinator/config/pipeline.example.json](coordinator/config/pipeline.example.json)

Before running, update at least:

- `artifactBaseUrl`
- `stages[].deviceId`
- `stages[].modelArtifactPath`
- `stages[].expectedHost`
- `stages[].expectedPort`

The checked-in `pipeline.json` may contain local development values. For team setup, prefer copying `pipeline.example.json` into your own local config first.

Start the coordinator:

```bash
./gradlew :coordinator:run
```

On Windows:

```powershell
./gradlew.bat :coordinator:run
```

If needed, pass a custom config:

```bash
./gradlew :coordinator:run --args="--config /path/to/pipeline.json"
```

Admin endpoints:

- `GET /healthz`
- `GET /api/v1/status`
- `GET /api/v1/requests`
- `POST /api/v1/routing/reload`
- `POST /api/v1/stages/{stageId}/drain`
- `POST /api/v1/stages/{stageId}/resume`
- `POST /api/v1/nodes/{nodeId}/evict`

Example:

```powershell
Invoke-RestMethod http://127.0.0.1:18080/api/v1/status
```

## Android Worker Setup

The worker can be launched with adb intent extras, so routine phone startup does not require manually typing values into the UI.

PowerShell:

```powershell
.\tools\android\start_worker.ps1 `
  -Serial <serial> `
  -CoordinatorHost 192.168.1.10 `
  -DeviceId android_stage_0 `
  -Install
```

Bash:

```bash
tools/android/start_worker.sh \
  --serial <serial> \
  --coordinator-host 192.168.1.10 \
  --device-id android_stage_0 \
  --install
```

The script builds and installs the debug APK when `-Install` / `--install` is set, then starts `MainActivity` with:

- `sid.coordinator_host`
- `sid.coordinator_port`, default `50051`
- `sid.device_id`, which must match `coordinator/config/pipeline.json`
- `sid.local_port`, default `26052`
- `sid.auto_start=true`

Without those launch extras, the app opens with auto-start disabled. You can still edit the fields in the UI and press `Start Worker`.

When the worker starts:

1. it starts a local data server
2. it registers with the coordinator
3. it downloads its shard if needed
4. it becomes active only after the shard is ready

Logs are visible both in-app and in `logcat`.

```powershell
adb -s <serial> logcat -s SidWorkerUi ExecuTorchShardRunner GrpcManager AndroidRuntime
```

## End-to-End Demo

### 1. Start the coordinator

```bash
./gradlew :coordinator:run
```

### 2. Start the configured workers

For each configured stage, launch one phone with the matching `deviceId` from `coordinator/config/pipeline.json`:

```bash
tools/android/start_worker.sh \
  --serial <serial> \
  --coordinator-host 192.168.1.10 \
  --device-id android_stage_0 \
  --install
```

Make sure every worker device:

- is on the same LAN as the coordinator
- uses a `deviceId` that exists in the pipeline config
- can reach the coordinator host and HTTP artifact port

### 3. Check control-plane status

```powershell
Invoke-RestMethod http://127.0.0.1:18080/api/v1/status
```

You want to see:

- `liveNodeCount` equals the number of active stages
- `assignedNode` exists for each stage you are testing
- `routeReady=true` where appropriate

### 4. Submit a demo request

The coordinator module includes a small submit client:

```bash
./gradlew :coordinator:runSubmitDemo
```

On Windows:

```powershell
./gradlew.bat :coordinator:runSubmitDemo
```

### 5. Inspect the request lifecycle

```powershell
Invoke-RestMethod http://127.0.0.1:18080/api/v1/requests
Invoke-RestMethod http://127.0.0.1:18080/api/v1/requests/<request-id>?events=20
```

## Replacing Model Artifacts

If you replace a shard file on disk:

1. overwrite the local `.pte` file referenced by `modelArtifactPath`
2. reload the coordinator config and artifact metadata
3. restart the workers so they re-check the shard

Reload command:

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:18080/api/v1/routing/reload
```

This is required because the coordinator caches:

- `modelSha256`
- `modelBytes`
- artifact download URLs

## Collaboration Notes

For a teammate-facing workflow guide, see [CONTRIBUTING.md](CONTRIBUTING.md).

For teammates integrating against this repository, the most important contracts are:

1. **Do not introduce cross-stage backward RPC by accident.**
   - This repository is explicitly built around forward-only inter-stage communication.

2. **If you change `sid.proto`, rebuild both modules.**
   - `app` and `coordinator` share the same protobuf definitions.

3. **Treat `pipeline.json` as environment-specific.**
   - Device IDs, IPs, and artifact paths should not be assumed stable across machines.

4. **Do not commit `.pte` files or SQLite state.**
   - Those are intentionally ignored by `.gitignore`.

5. **Keep coordinator and worker logs readable.**
   - This project is still debug-heavy; logs are part of the interface during collaboration.

## Recommended Research Direction

The current repository is still pipeline-shaped, but the intended next step is **scheduled decoupled execution**, not naive relay.

In other words:

- today: request enters stage 0 and flows through the active shard sequence
- next: decoupled shards should be schedulable more independently, because the training method is BP-free across stages

That scheduler is **not finished yet**. The current system should be treated as the control-plane and deployment prototype that future scheduling work will build on.

## Known Limitations

- Large `phi-2` shards are likely too heavy for 4 GB phones.
- Memory pressure is currently the main blocker on weaker devices.
- The current coordinator is not yet the final parallel scheduling system.
- Security is minimal and oriented toward LAN testing.
- The export path is still evolving and model-dependent.

## Suggested First Tasks For New Collaborators

- export a smaller TinyLlama-based shard set
- validate end-to-end request flow on two phones
- profile memory use on stage 0
- define the next scheduling abstraction beyond relay-style forwarding
- formalize the training method name and benchmark protocol

## License / Release State

No release process or license text has been added yet in this repository snapshot. Add those before public release.
