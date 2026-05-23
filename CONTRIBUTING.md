# Contributing

This repository is still a research prototype. The main goal of collaboration is to keep interfaces stable while we iterate on:

- mobile shard execution
- model export
- coordinator control-plane behavior
- future scheduling logic for BP-free modular training

## Before You Start

Read these files first:

- [README.md](README.md)
- [coordinator/README.md](coordinator/README.md)
- [app/src/main/proto/sid.proto](app/src/main/proto/sid.proto)
- [tools/export/sid_export_mobile.py](tools/export/sid_export_mobile.py)

## Ground Rules

1. Do not introduce stage-to-stage backward gradient RPC.
   This repository is intentionally built around forward-only inter-stage communication.

2. Do not commit model artifacts.
   `.pte`, downloaded shards, and generated exports must stay out of git.

3. Treat `pipeline.json` as local state.
   Use the example config as the starting point and keep machine-specific IPs and device IDs out of commits unless the change is intentional.

4. Keep logging readable.
   This project is still debugging-heavy. If a change affects execution flow, add or preserve logs that make failures diagnosable.

5. Keep protocol changes coordinated.
   Any protobuf change affects both Android workers and the coordinator.

## Recommended Workflow

1. Create a branch for your work.
2. Keep changes scoped to one area when possible:
   - export
   - Android worker
   - coordinator
   - docs
3. Rebuild the affected modules before sending changes.
4. Document any config or artifact assumptions in your PR or handoff notes.

## Local Development

### Android + coordinator

Build:

```bash
./gradlew :coordinator:build :app:assembleDebug
```

On Windows:

```powershell
./gradlew.bat :coordinator:build :app:assembleDebug
```

### Export server

The minimal export environment is described by:

- [environment.yml](environment.yml)
- [requirements-export.txt](requirements-export.txt)

```bash
conda env create -f environment.yml
conda activate mobile-bpfree-export
```

## Config Files

Do not edit the checked-in `coordinator/config/pipeline.example.json` with local machine values. Instead:

1. copy it to a working config
2. fill in your LAN IPs, device IDs, and model artifact paths
3. pass the working config to the coordinator

Example:

```bash
cp coordinator/config/pipeline.example.json coordinator/config/pipeline.local.json
```

Then run:

```bash
./gradlew :coordinator:run --args="--config coordinator/config/pipeline.local.json"
```

## If You Change The Protocol

If you edit [app/src/main/proto/sid.proto](app/src/main/proto/sid.proto):

1. rebuild `app`
2. rebuild `coordinator`
3. verify request submission still works
4. verify worker registration and heartbeat still work

Minimum command:

```bash
./gradlew :coordinator:build :app:assembleDebug
```

## If You Change Export Logic

If you modify [tools/export/sid_export_forward_mobile.py](tools/export/sid_export_forward_mobile.py) or [tools/export/sid_export_mobile.py](tools/export/sid_export_mobile.py):

1. keep a reproducible command example
2. note which model preset you tested
3. note `num_chunks`, `seq_len`, and `transport_dtype`
4. mention whether the artifact is forward-only or joint forward/backward
5. mention whether XNNPACK lowering was enabled

## If You Replace Model Artifacts

After replacing `.pte` files used by the coordinator:

1. overwrite the artifact on disk
2. reload the coordinator config
3. restart the workers

Reload:

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:18080/api/v1/routing/reload
```

## Suggested Commit Scope

Good commit boundaries:

- one commit for docs only
- one commit for worker runtime fixes
- one commit for coordinator control-plane changes
- one commit for export script changes

Avoid mixing:

- protocol changes
- export changes
- scheduler changes
- large unrelated formatting churn

## What To Include In A Handoff

When handing work to another teammate, include:

- what changed
- which commands you ran
- which phones or devices you tested on
- whether the change requires new `.pte` artifacts
- whether the coordinator config must be updated
- known failure modes or limitations
