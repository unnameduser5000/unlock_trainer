# UnLok: Resilient Collaborative Training

This repository implements the system described in **"Resilient Collaborative
On-Device Training System: Backpropagation-Beyond Approach."** It includes the
multi-stage training runtime, Android implementation, ExecuTorch extension,
comparison systems, and scripts for the five evaluation sections.

UnLok removes the end-to-end backward dependency between pipeline stages. Each
stage constructs a local objective, performs a local backward pass, and commits
its optimizer update independently. Stage boundaries carry detached forward
hidden states; they do not carry backward gradients or optimizer state.

In this repository, **BP-free** means *free of cross-stage backpropagation*.
Local backpropagation remains part of every stage update.

## Key Features

- **Early forwarding.** A downstream stage can start its forward pass while
  the upstream stage completes local backward and optimization.
- **Stage-local state.** Each executor owns its trainable LoRA parameters,
  optimizer state, checkpoints, and commit journal.
- **Forward-only boundaries.** Inter-stage traffic contains detached hidden
  states and request metadata.
- **Stage-granular recovery.** Committed upstream progress can be retained when
  a downstream stage is temporarily unavailable.
- **Two deployment paths.** The repository provides CPU/Gloo and GPU/NCCL
  server runtimes as well as an Android/ExecuTorch runtime for physical phones.

## Architecture

For each request at stage `s`, execution follows this order:

```text
receive hidden state
       |
       v
local forward + local objective
       |
       +---- detached boundary ----> stage s + 1
       |
       v
local backward
       |
       v
local optimizer update + commit
```

The mobile deployment separates the control and data planes:

- The **coordinator** registers executors, tracks heartbeat leases, assigns
  stages and model artifacts, admits requests, and maintains routing state.
- Each **executor** runs one model stage, sends its detached boundary directly
  to the next executor, updates local LoRA parameters, and persists its local
  training state.

Boundary entries are versioned by request, stage, update window, and
microbatch. Durable replay resends a committed boundary after rejoin without
repeating the producer's forward, backward, or optimizer update. An optional
prefix-retention policy records the stages that committed before an unavailable
boundary and omits the remaining suffix.

## Repository Structure

| Path | Description |
| --- | --- |
| `src/sg_exe_trainer/runtime/bpfree/` | BP-free CPU/Gloo and GPU/NCCL runtimes |
| `src/sg_exe_trainer/runtime/exactbp/` | GPipe and synchronous 1F1B runtimes |
| `src/sg_exe_trainer/runtime/recovery/` | Checkpoints, durable boundaries, journals, and catch-up |
| `src/sg_exe_trainer/runtime/transport/` | Pinned CPU transport and link pacing |
| `experiments/shared/baselines/` | PipeDream comparison implementation |
| `app/` | Android executor and on-device training state |
| `coordinator/` | JVM coordinator and request ingress |
| `tools/export/` | Model partitioning and ExecuTorch export |
| `experiments/` | E1, E2, E4, and E5 configurations and launchers |
| `tests/` | Runtime, protocol, and mobile artifact tests |
| `third_party/executorch_patches/` | ExecuTorch 1.2.0 extension and upstream license |

Model weights, generated PTE files, and datasets are downloaded or produced by
the commands below and are stored outside the source tree.

## Environment

### Server experiments

Server experiments require Linux, Python 3.10 or newer, PyTorch with
`torch.distributed`, and either CUDA/NCCL or Gloo. The complete package list
from the evaluated server environment is provided in
[`requirement.txt`](requirement.txt).

Install the repository package after preparing the PyTorch environment:

```bash
python -m pip install -e .
```

### Mobile export and Android build

The PTE export environment is defined separately:

```bash
conda env create -f environment.yml
conda activate mobile-bpfree-export
```

Building the Android application requires JDK 17, Android SDK 34, and `adb`.
The application targets arm64-v8a devices.

## Verification

Verify the supplied ExecuTorch Android runtime:

```bash
python tools/android/verify_bpfree_aar.py
```

Run the Python tests after installing `pytest` and the server dependencies:

```bash
PYTHONPATH=src python -m pytest tests
```

Build and test the coordinator and Android application:

```bash
./gradlew :coordinator:test :app:testDebugUnitTest :app:assembleDebug
```

On Windows, use `gradlew.bat`. The APK is written to
`app/build/outputs/apk/debug/app-debug.apk`.

## Experiments

| Evaluation | Contents | Documentation |
| --- | --- | --- |
| E1 | AG News quality and local-objective comparison | [`experiments/e1_quality/`](experiments/e1_quality/) |
| E2 | Per-stage GPU memory across batch geometries | [`experiments/e2_memory/`](experiments/e2_memory/) |
| E3 | Three-phone ExecuTorch execution | [`docs/MOBILE_BPFREE_SPLIT_RUNTIME.md`](docs/MOBILE_BPFREE_SPLIT_RUNTIME.md) |
| E4 | Stage scaling, scheduling, and link sensitivity | [`experiments/e4_throughput/`](experiments/e4_throughput/) |
| E5 | Middle-stage outage and recovery | [`experiments/e5_recovery/`](experiments/e5_recovery/) |

### E1: quality

E1 compares Full BP, synchronous 1F1B, BP-free with local cross entropy, and
BP-free with belief consistency. The configuration fixes TinyLlama LoRA,
effective batch size 8, 1,250 optimizer boundaries, and three paired seeds.

```bash
PYTHONPATH=src python experiments/e1_quality/run_agnews_quality.py \
  --data-root /path/to/artifact-data
```

The data root supplies the AG News manifests referenced by
`experiments/e1_quality/configs/agnews_quality_v1.json`.

### E2: GPU memory

E2 compares BP-free, GPipe, synchronous 1F1B, and PipeDream at effective batch
size 32 over `(b,m) = (1,32), (2,16), (4,8), (8,4)`.

```bash
PYTHONPATH=src python experiments/e2_memory/run_gpu_memory.py \
  --data-root /path/to/artifact-data \
  --output-root results/e2_memory

PYTHONPATH=src python experiments/e2_memory/build_report.py \
  --raw-root results/e2_memory \
  --output-dir results/e2_memory_report
```

### E4: throughput and communication

E4 evaluates two-to-four-stage scaling, fixed-effective-batch geometry,
low-batch scheduling, controlled link latency/bandwidth, and runtime overhead.
The launch queue runs the configured experiment set:

```bash
PYTHONPATH=src bash experiments/e4_throughput/run_formal_queue.sh
```

Individual configurations and report builders are listed in
[`experiments/e4_throughput/README.md`](experiments/e4_throughput/README.md).

### E5: recovery

E5 uses a three-stage pipeline with a controlled Stage-1 outage. BP-free and
synchronous 1F1B use the same model split, physical batch, update window, and
terminal progress target.

```bash
PYTHONPATH=src python experiments/e5_recovery/formal_v2/run_bpfree_outage.py \
  --run-id bpfree-run \
  --train-manifest /path/to/train_manifest.jsonl \
  --output-dir results/e5/bpfree

PYTHONPATH=src python experiments/e5_recovery/formal_v2/run_exactbp_outage.py \
  --run-id exactbp-run \
  --train-manifest /path/to/train_manifest.jsonl \
  --output-dir results/e5/exactbp

PYTHONPATH=src python experiments/e5_recovery/formal_v2/build_comparison.py \
  --bpfree-summary results/e5/bpfree/summary.json \
  --exactbp-summary results/e5/exactbp/summary.json \
  --output-dir results/e5/comparison
```

Additional E5 entry points cover streamed catch-up, volatile hidden replay,
balanced skipping, and independent intermittent dropout. They are described in
[`experiments/e5_recovery/formal_v2/README.md`](experiments/e5_recovery/formal_v2/README.md),
[`formal_v3_no_recovery_quality/`](experiments/e5_recovery/formal_v3_no_recovery_quality/),
and [`intermittent_dropout/`](experiments/e5_recovery/intermittent_dropout/).

## Three-Phone Runtime

The Android runtime uses a custom ExecuTorch boundary operator. A
boundary-enabled training method pauses after producing its detached output,
starts the downstream request, and then resumes the same method for local
backward. PTE files without the marker continue to use atomic execution.

Verify the checked-in AAR and export one boundary-enabled stage:

```bash
python tools/android/verify_bpfree_aar.py

python -m tools.export.sid_export_mobile \
  --num_chunks 3 \
  --chunk_idx 0 \
  --belief_transport_mode none \
  --enable_pipeline_boundary \
  --output_dir model
```

Configure one stage per device in `coordinator/config/pipeline.example.json`,
start the coordinator, and launch each worker with its configured device ID:

```bash
./gradlew :coordinator:run

tools/android/start_worker.sh \
  --serial <adb-serial> \
  --coordinator-host <host> \
  --device-id android_stage_0 \
  --install
```

The complete PTE contract, AAR build procedure, host harness, and phone
measurements are documented in
[`docs/MOBILE_BPFREE_SPLIT_RUNTIME.md`](docs/MOBILE_BPFREE_SPLIT_RUNTIME.md).
Coordinator configuration and endpoints are documented in
[`coordinator/README.md`](coordinator/README.md).

## ExecuTorch Extension

The mobile runtime is based on ExecuTorch `v1.2.0` at commit
`0b0e2c5cdd67c8b4396a46ea1d1aa72ffb0128d7`. The source patch adds the
boundary operator, split training module, JNI/Java bindings, native
registration, and optimized-kernel build switch.

```bash
ANDROID_NDK=/path/to/android-ndk \
ANDROID_SDK=/path/to/android-sdk \
tools/android/build_bpfree_aar.sh /path/to/executorch
```

The build script works in a temporary Git worktree and leaves the supplied
ExecuTorch checkout unchanged. Exact source and build information is provided
in [`third_party/executorch_patches/README.md`](third_party/executorch_patches/README.md)
and `app/libs/executorch-android-bpfree-1.2.0.json`.

## License

Project source is released under the BSD 3-Clause License; see
[`LICENSE`](LICENSE). Third-party components retain their original licenses.
