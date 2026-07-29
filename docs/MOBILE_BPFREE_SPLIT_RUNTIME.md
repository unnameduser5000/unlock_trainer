# Mobile BP-free split runtime

## Scope

This path keeps one ExecuTorch `Method` and one active request per phone. It
does not create a session pool or implement same-device 1F1B. A boundary-marked
joint PTE executes as:

1. local forward, local head, and local loss;
2. copy detached hidden/belief tensors at the boundary;
3. start the downstream request;
4. resume the same `Method` through local backward;
5. apply the existing per-request AdamW update.

The downstream transfer can therefore overlap the upstream phone's backward.
PTE files without the boundary marker continue to use the original atomic
`TrainingModule.executeForwardBackward()` path.

## Forward failure policies

The request protocol exposes two mutually exclusive recovery modes:

- `durable_pipeline=true` stores the detached boundary in the phone-local
  outbox, publishes it before resuming local backward, and retries it until
  the downstream stage acknowledges it. The downstream forward can therefore
  overlap the upstream backward without giving up durable replay.
- `drop_on_forward_failure=true` keeps the contiguous prefix-stage optimizer
  commits but drops the unavailable boundary without retry or catch-up. The
  response is a successful non-terminal partial commit with
  `forward_dropped=true` and `first_unprocessed_stage_id` identifying where
  execution stopped.

Durable replay is request-idempotent, not only optimizer-idempotent. Every
optimizer commit persists the request key with the checkpoint. A downstream
ACK replaces the pending boundary payload with a compact phone-local receipt.
When the same request arrives again, the worker restores/checks that state
before entering ExecuTorch: a pending boundary keeps its original outbox
entry, an acknowledged boundary returns its receipt, and a terminal commit
returns a terminal receipt. None of these committed replay paths reruns local
forward/backward, creates another boundary, forwards the request again, or
applies another optimizer step.

The best-effort mode applies only to downstream availability failures. Local
PTE execution, optimizer, checkpoint, and invalid-request failures remain hard
failures. The pipeline experiment runner accepts a dropped request only when
its stage metrics prove a contiguous committed prefix from Stage 0. It records
the policy and outcome in `drop_on_forward_failure`,
`dropped_prefix_commit_valid`, `forward_dropped`,
`first_unprocessed_stage_id`, and `failure_kind` CSV columns. This mode is for
sparse quality-tolerated omissions; it does not impose a long-outage drop
budget by itself.

Belief transport defaults to `terminal`: intermediate stages send no
full-vocabulary belief tensor, while the terminal stage may return log-probs
for evaluation. The PTE output contract is static. A request with
`beliefTransportMode=none` prevents transport and Java-side copies, but only a
PTE exported with `--belief_transport_mode none` removes the terminal
full-log-prob branch from the graph.

## Source layout

- `tools/export/bpfree_pipeline_boundary.py`: custom PyTorch operator and graph audit.
- `tools/export/sid_export_mobile.py`: opt-in `--enable_pipeline_boundary` export.
- `third_party/executorch_patches/executorch-v1.2.0-bpfree-boundary.patch`:
  ExecuTorch 1.2.0 C++/JNI/Java implementation.
- `app/src/main/java/com/example/sid_trainer/BPFreeShardRuntime.kt`: one-Method
  forward-boundary/backward lifecycle.
- `app/src/main/java/com/example/sid_trainer/MobileTrainingState.kt`: shared
  deduplication, checkpoint, and AdamW state used by atomic and split paths.
- `app/src/main/java/com/example/sid_trainer/ForwardChunkProcessor.kt`:
  split/atomic execution policy and downstream dispatch.
- `app/src/main/java/com/example/sid_trainer/DurableBoundaryOutbox.kt`:
  phone-local boundary replay and acknowledgement state.
- `app/src/main/java/com/example/sid_trainer/WorkerController.kt`: worker
  lifecycle and control-plane orchestration outside the Android activity.

The app uses `app/libs/executorch-android-bpfree-1.2.0.aar`. Its SHA-256 is:

```text
dfef4039fc757d93578521054d360a1be5b4b873bf580f494665ff38913ebf8b
```

`app/libs/executorch-android-bpfree-1.2.0.json` records the exact upstream
commit, native-library hash, ABI, and required API/operator markers. Validate
the checked-in artifact before building or installing the app:

```bash
python tools/android/verify_bpfree_aar.py
```

## Rebuild

The canonical runtime was built from ExecuTorch tag `v1.2.0`, commit
`0b0e2c5cdd67c8b4396a46ea1d1aa72ffb0128d7`. The repository build wrapper
creates a temporary worktree, applies the checked-in patch, enables optimized
kernels, builds only `arm64-v8a`, and structurally verifies the result. It does
not modify the supplied ExecuTorch checkout:

```bash
ANDROID_NDK=/path/to/android-ndk \
ANDROID_SDK=/path/to/android-sdk \
tools/android/build_bpfree_aar.sh /path/to/executorch
```

A rebuilt AAR may have a different full byte hash when the Android toolchain
differs. The wrapper still requires the BP-free Java/JNI API, boundary operator,
and optimized CPU kernel markers. Replace the canonical manifest hash only
after validating the new artifact on the physical-phone pipeline.

Export remains opt-in so existing phone artifacts are unchanged:

```bash
python -m tools.export.sid_export_mobile \
  --num_chunks 3 \
  --chunk_idx 0 \
  --belief_transport_mode none \
  --seed 20260715 \
  --enable_pipeline_boundary \
  --output_dir debug_runs/mobile_bpfree_split_runtime_v1/export
```

Every export writes a `.runtime_contract.json` containing its static belief
mode, shape, LoRA, and seed settings. A boundary export also writes a
`.boundary_audit.json` and fails if the marker is absent, duplicated, after the
backward seed, or after any trainable gradient.

Stage metrics split downstream HTTP work into request serialization/write,
response wait/read/parse, server request read/parse, handler time, and response
serialization. `rpc_response_wait_ms` still contains downstream queue and
compute; it is not a communication-only number. The client-send,
server-request-received, server-response-ready, and client-response-received
epoch fields allow calibrated-phone runs to estimate both wire intervals. For
clock-independent accounting, `forward_ms - rpc_server_handler_ms` is the
implementation-level RPC overhead, including serialization, socket setup,
request/response transfer, and parsing.

## Verification

The production TinyLlama chunk-0 PTE was exercised through the new C++ module,
using the same startup order as Android: named parameters, forward boundary,
backward resume, named gradients, then a second split cycle. A stock atomic
`TrainingModule` execution provided the gradient reference.

```json
{"parameters":28,"gradients":28,"hidden_bytes":32768,"belief_bytes":4,"gradient_max_abs_diff":0,"second_cycle":true,"passed":true}
```

`tests/mobile_runtime/bpfree_module_harness.cpp` contains the repeatable host
check, kept outside experiment launchers because it verifies a production
runtime contract. On 2026-07-29, the synchronized app passed
`:app:testDebugUnitTest`, `:coordinator:test`, and `:app:assembleDebug` with the
existing local JDK/Gradle/AAR environment.

## Three-phone verification

On 2026-07-13, the optimized release boundary path completed training on three
physical arm64 phones. Chunk 0 ran on an NX809J, chunk 1 on a Lenovo L71091,
and chunk 2 on a Pixel 10 Pro XL. The requests used batch size 1, sequence
length 8, `maxInFlight=3`, split execution, and terminal belief transport.

| Stage | Restart probe boundary | Restart probe backward | Steady execute mean | Optimizer mean |
| --- | ---: | ---: | ---: | ---: |
| 0 | 2.773 s | 1.724 s | 9.185 s | 0.152 s |
| 1 | 4.380 s | 2.525 s | 5.271 s | 0.285 s |
| 2 | 2.993 s | 2.030 s | 4.968 s | 0.072 s |

The restart probe includes checkpoint restore and cold-page effects. The
steady means come from the later eight-request group crossing optimizer step
32. Stage 0 becomes the sustained bottleneck as the NX809J warms up. The group
completed 8/8 requests in 82.477 seconds (0.0970 request/s); checkpoint writes
at step 32 took 9, 11, and 8 ms for stages 0, 1, and 2.

Stage 0 forwarded copied boundary tensors before local backward completed, and
stage 1 did the same. The selected three-window Gantt audit measured upstream
backward/downstream-local overlap at every boundary. Intermediate belief
payloads were all zero bytes.

Per-request memory sampling uses `Debug.getMemoryInfo()` for the current
process. `ActivityManager.getProcessMemoryInfo()` is intentionally not used:
Android may rate-limit that API and return a stale pre-model value during a
500-ms sampling loop.

The current calibration CSVs, request event snapshots, Gantt, release AAR, and
audit report are under
`debug_runs/mobile_bpfree_split_runtime_v1/atomic_split_calibration_20260713`.
The earlier portable-kernel phone capture provides an additional semantic
check. The optimized release capture above provides the performance values.

An earlier request intentionally remains as negative-path evidence. It sent
`beliefTransportMode=full` to a terminal-mode PTE, so JNI passed five tensors
to a four-input method. ExecuTorch reported the input-count error correctly,
but the current fbjni error translation then aborted the worker process. The
valid terminal path is unaffected; graceful JNI propagation for invalid input
is still an open robustness issue.
