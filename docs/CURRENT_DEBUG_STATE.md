# Current Debug State

This file is the first file to read after any context reset.

Last updated: 2026-05-25 17:41 Asia/Shanghai

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

The older `tinyllama-mainline-20260524-2349` request only proved a `Module.execute()` forward pipeline request at fixed `seqLen=64`; it did not prove the full training experiment. The newer `tinyllama-training-20260525-1542` request used training PTEs and completed through both phones.

## Current Devices

ADB serials:

- `91260221021D`: NX809J, stage 0, expected host `192.168.214.103`, expected port `26052`
- `ZY22G2HC5C`: Lenovo_L71091, stage 1, expected host `192.168.214.59`, expected port `26052`

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
