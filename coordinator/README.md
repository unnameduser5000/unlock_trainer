# Coordinator

This module provides a standalone JVM control-plane service for the SID worker pipeline.

## What It Does

- Accepts worker registration over gRPC.
- Maps each `device_id` to a configured pipeline stage.
- Returns `stage_id`, `model_shard_id`, `terminal`, and `next_hop`.
- Tracks worker leases from heartbeat traffic.
- Publishes `routing_epoch` and `route_ready` on heartbeat so workers can reconnect live.
- Persists node identity and routing metadata in SQLite for restart recovery.
- Tracks request lifecycle and marks overdue in-flight requests as `STALLED` in the admin view.
- Accepts new request ingress over coordinator gRPC and forwards them into stage 0 only when the full pipeline is live.
- Evicts dead nodes when heartbeats expire.
- Purges old resolved request history automatically.
- Falls back to configured `expectedHost` and `expectedPort` for downstream routing.
- Exposes an HTTP admin interface for health, routing, and request maintenance.

## Config

Default config path:

```text
coordinator/config/pipeline.json
```

Default SQLite state path:

```text
coordinator/data/coordinator.db
```

Default admin HTTP port:

```text
18080
```

Each stage must define:

- `stageId`
- `deviceId`
- `modelShardId`
- `modelArtifactPath`
- `expectedHost`
- `expectedPort`

Example:

```json
{
  "bindPort": 50051,
  "adminBindPort": 18080,
  "heartbeatLeaseSeconds": 15,
  "cleanupIntervalSeconds": 5,
  "requestStallTimeoutSeconds": 30,
  "resolvedRequestRetentionSeconds": 86400,
  "artifactBaseUrl": "http://192.168.5.29:18080",
  "pipelineName": "demo-pipeline",
  "stateDbPath": "coordinator/data/coordinator.db",
  "stages": [
    {
      "stageId": 0,
      "deviceId": "android_stage_0",
      "modelShardId": "chunk_0",
      "modelArtifactPath": "../../model/phi2_sid_full_ft_chunk_0.pte",
      "expectedHost": "192.168.5.10",
      "expectedPort": 50052
    },
    {
      "stageId": 1,
      "deviceId": "android_stage_1",
      "modelShardId": "chunk_1",
      "modelArtifactPath": "../../model/phi2_sid_full_ft_chunk_1.pte",
      "expectedHost": "192.168.5.11",
      "expectedPort": 50052
    }
  ]
}
```

Restart behavior:

- Node registrations and routing epoch are restored from SQLite.
- Restored nodes start as `offline` until they heartbeat or register again.
- This preserves node identity without assuming an old socket address is still alive.

## Admin Interface

Endpoints:

- `GET /healthz`
- `GET /artifacts/stages/{stageId}/model`
- `GET /api/v1/status`
- `GET /api/v1/requests`
- `GET /api/v1/requests/{requestId}`
- `POST /api/v1/requests/{requestId}/retry`
- `POST /api/v1/requests/{requestId}/purge`
- `POST /api/v1/requests/purge-resolved`
- `POST /api/v1/stages/{stageId}/drain`
- `POST /api/v1/stages/{stageId}/resume`
- `POST /api/v1/nodes/{nodeId}/evict`
- `POST /api/v1/routing/reload`

Example:

```powershell
Invoke-RestMethod http://127.0.0.1:18080/api/v1/status
Invoke-RestMethod http://127.0.0.1:18080/api/v1/requests
Invoke-RestMethod http://127.0.0.1:18080/api/v1/requests?state=STALLED
Invoke-RestMethod http://127.0.0.1:18080/api/v1/requests/request-123?events=20
```

Mutation examples:

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:18080/api/v1/requests/request-123/purge
Invoke-RestMethod -Method Post http://127.0.0.1:18080/api/v1/requests/request-123/retry
Invoke-RestMethod -Method Post http://127.0.0.1:18080/api/v1/requests/purge-resolved?older_than_seconds=3600
Invoke-RestMethod -Method Post http://127.0.0.1:18080/api/v1/stages/1/drain
Invoke-RestMethod -Method Post http://127.0.0.1:18080/api/v1/stages/1/resume
Invoke-RestMethod -Method Post http://127.0.0.1:18080/api/v1/nodes/3/evict
Invoke-RestMethod -Method Post http://127.0.0.1:18080/api/v1/routing/reload
```

Runtime reload behavior:

- `pipelineName`, `heartbeatLeaseSeconds`, `cleanupIntervalSeconds`, and stage definitions can be reloaded.
- `requestStallTimeoutSeconds` and `resolvedRequestRetentionSeconds` can be reloaded.
- `bindPort`, `adminBindPort`, and `stateDbPath` are fixed after process start.
- Drained stages are persisted in SQLite and survive restart.
- `/api/v1/requests` accepts `state=IN_FLIGHT|STALLED|FAILED|COMPLETED`.
- Requests can only be admitted when every configured stage has a live worker and no stage is drained.

## gRPC Ingress

The coordinator gRPC service now exposes:

- `SubmitRequest(ForwardChunkRequest) -> ForwardChunkResponse`

The caller must provide a non-blank `request_id`. The coordinator persists the original request payload for later retry and forwards it to stage 0 only if the entire configured pipeline is currently live.

## Model Distribution

When `artifactBaseUrl` and each stage's `modelArtifactPath` are configured:

- the coordinator computes `modelSha256` and `modelBytes` at startup
- registration and heartbeat responses include `modelDownloadUrl`, `modelSha256`, and `modelBytes`
- Android workers download their shard into app-private storage automatically before they become schedulable

This removes the need to push `.pte` files over `adb` for routine startup.

## Run

```powershell
./gradlew.bat :coordinator:run
```

Custom config:

```powershell
./gradlew.bat :coordinator:run --args="--config C:\path\to\pipeline.json"
```
