# Contributing

Contributions to the runtime, Android implementation, experiment drivers, and
documentation are welcome. Keep changes focused and include the validation
needed for the affected component.

## Development Setup

Install the Python package in an environment with the required PyTorch stack:

```bash
python -m pip install -e .
```

The Android and coordinator modules require JDK 17 and Android SDK 34:

```bash
./gradlew :coordinator:build :app:assembleDebug
```

The mobile export environment is defined by `environment.yml`:

```bash
conda env create -f environment.yml
conda activate mobile-bpfree-export
```

## Code Organization

- Reusable training, transport, and recovery behavior belongs under
  `src/sg_exe_trainer/runtime/`.
- Experiment directories contain configurations, launchers, and report
  builders; they call the shared runtime.
- Android executor logic belongs under `app/`; coordinator and routing logic
  belongs under `coordinator/`.
- Offline export, aggregation, and visualization utilities belong under
  `tools/`.

The cross-stage BP-free interface carries detached forward state. Changes that
add a backward-gradient RPC alter the system model and require a corresponding
protocol and evaluation update.

## Generated Files

Keep generated and machine-specific data outside commits, including:

- model weights and PTE files;
- datasets and prepared request manifests;
- checkpoints, SQLite state, logs, and result directories;
- local IP addresses, device IDs, and signing material.

Use `coordinator/config/pipeline.example.json` as the template for local
deployments.

## Validation

Run the Python tests for runtime or experiment changes:

```bash
PYTHONPATH=src python -m pytest tests
```

Verify the Android runtime artifact after changes to the AAR, manifest, build
wrapper, or ExecuTorch patch:

```bash
python tools/android/verify_bpfree_aar.py
```

Run the Gradle tests for coordinator or Android changes:

```bash
./gradlew :coordinator:test :app:testDebugUnitTest :app:assembleDebug
```

Changes to `app/src/main/proto/sid.proto` affect both the Android worker and the
coordinator and should be validated on both sides.

## Pull Requests

Describe:

- the behavior changed;
- the affected runtime or experiment contract;
- the commands used for validation;
- any required model export or configuration update.

Keep unrelated refactors, generated results, and protocol changes in separate
commits where practical.
