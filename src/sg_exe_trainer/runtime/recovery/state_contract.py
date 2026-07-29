from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import torch

from .durable_io import atomic_write_json, fsync_directory, sha256_file


SCHEMA_VERSION = 1


class StateContractError(RuntimeError):
    """Base class for a violated E5 persistence or idempotency contract."""


class BoundaryConflictError(StateContractError):
    """The same logical boundary was written with different contents."""


class CommitConflictError(StateContractError):
    """The same stage/window commit was recorded with different contents."""


class CorruptStateError(StateContractError):
    """Durable metadata and its payload do not agree."""


class OutboxCapacityError(StateContractError):
    """The configured pending-window limit would be exceeded."""


def _safe_component(value: str, field: str) -> str:
    value = str(value).strip()
    if not value or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for ch in value):
        raise ValueError(f"{field} contains unsupported path characters: {value!r}")
    return value


def _tensor_nbytes(tensor: Optional[torch.Tensor]) -> int:
    if tensor is None:
        return 0
    return int(tensor.numel() * tensor.element_size())


def _cpu_snapshot(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    return tensor.detach().to(device="cpu").contiguous().clone()


@dataclass(frozen=True, order=True)
class BoundaryKey:
    run_id: str
    source_stage: int
    target_stage: int
    window_id: int
    microbatch_id: int

    def __post_init__(self) -> None:
        _safe_component(self.run_id, "run_id")
        if self.source_stage < 0 or self.target_stage < 0:
            raise ValueError("stage ids must be non-negative")
        if self.target_stage != self.source_stage + 1:
            raise ValueError("a boundary must connect adjacent stages")
        if self.window_id < 0 or self.microbatch_id < 0:
            raise ValueError("window_id and microbatch_id must be non-negative")

    @property
    def stem(self) -> str:
        return f"window-{self.window_id:08d}.mb-{self.microbatch_id:04d}"


@dataclass(frozen=True)
class BoundaryMetadata:
    schema_version: int
    key: BoundaryKey
    request_ids: tuple[str, ...]
    producer_version: int
    payload_file: str
    payload_sha256: str
    payload_file_bytes: int
    hidden_tensor_bytes: int
    belief_tensor_bytes: int
    created_ns: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["request_ids"] = list(self.request_ids)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BoundaryMetadata":
        key = BoundaryKey(**payload["key"])
        return cls(
            schema_version=int(payload["schema_version"]),
            key=key,
            request_ids=tuple(str(item) for item in payload["request_ids"]),
            producer_version=int(payload["producer_version"]),
            payload_file=str(payload["payload_file"]),
            payload_sha256=str(payload["payload_sha256"]),
            payload_file_bytes=int(payload["payload_file_bytes"]),
            hidden_tensor_bytes=int(payload["hidden_tensor_bytes"]),
            belief_tensor_bytes=int(payload["belief_tensor_bytes"]),
            created_ns=int(payload["created_ns"]),
        )

    def semantic_identity(self) -> tuple[Any, ...]:
        return (
            self.schema_version,
            self.key,
            self.request_ids,
            self.producer_version,
            self.payload_sha256,
            self.hidden_tensor_bytes,
            self.belief_tensor_bytes,
        )


@dataclass(frozen=True)
class LoadedBoundary:
    metadata: BoundaryMetadata
    hidden: torch.Tensor
    prev_log_probs: Optional[torch.Tensor]


class DurableBoundaryOutbox:
    """Filesystem-backed, versioned stage-boundary tensors.

    The tensor file is made durable before its JSON commit marker. Consumers
    only enumerate JSON markers, so a process failure cannot expose a partial
    tensor as a committed boundary.
    """

    def __init__(self, root: Path, run_id: str, *, max_pending_windows: Optional[int] = None) -> None:
        self.root = Path(root)
        self.run_id = _safe_component(run_id, "run_id")
        if max_pending_windows is not None and max_pending_windows <= 0:
            raise ValueError("max_pending_windows must be positive when set")
        self.max_pending_windows = max_pending_windows
        self.run_root = self.root / self.run_id

    def _boundary_dir(self, source_stage: int, target_stage: int) -> Path:
        return self.run_root / "boundaries" / f"stage-{source_stage}-to-{target_stage}"

    def _ack_path(self, key: BoundaryKey) -> Path:
        return (
            self.run_root
            / "acks"
            / f"stage-{key.source_stage}-to-{key.target_stage}"
            / f"{key.stem}.json"
        )

    def _paths(self, key: BoundaryKey) -> tuple[Path, Path]:
        if key.run_id != self.run_id:
            raise ValueError(f"boundary run_id={key.run_id!r} does not match outbox run_id={self.run_id!r}")
        directory = self._boundary_dir(key.source_stage, key.target_stage)
        return directory / f"{key.stem}.pt", directory / f"{key.stem}.json"

    def put(
        self,
        key: BoundaryKey,
        *,
        hidden: torch.Tensor,
        request_ids: Iterable[str],
        producer_version: int,
        prev_log_probs: Optional[torch.Tensor] = None,
    ) -> BoundaryMetadata:
        if producer_version < 0:
            raise ValueError("producer_version must be non-negative")
        request_ids_tuple = tuple(str(item) for item in request_ids)
        if not request_ids_tuple:
            raise ValueError("request_ids cannot be empty")

        self.ensure_window_capacity(key.source_stage, key.target_stage, key.window_id)
        payload_path, metadata_path = self._paths(key)
        payload_path.parent.mkdir(parents=True, exist_ok=True)

        hidden_cpu = _cpu_snapshot(hidden)
        belief_cpu = _cpu_snapshot(prev_log_probs)
        assert hidden_cpu is not None

        tmp_payload = payload_path.with_name(
            f".{payload_path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
        )
        try:
            torch.save(
                {"hidden": hidden_cpu, "prev_log_probs": belief_cpu},
                tmp_payload,
            )
            with tmp_payload.open("rb") as handle:
                os.fsync(handle.fileno())

            candidate = BoundaryMetadata(
                schema_version=SCHEMA_VERSION,
                key=key,
                request_ids=request_ids_tuple,
                producer_version=int(producer_version),
                payload_file=payload_path.name,
                payload_sha256=sha256_file(tmp_payload),
                payload_file_bytes=tmp_payload.stat().st_size,
                hidden_tensor_bytes=_tensor_nbytes(hidden_cpu),
                belief_tensor_bytes=_tensor_nbytes(belief_cpu),
                created_ns=time.time_ns(),
            )

            if metadata_path.exists():
                existing = self._read_metadata(metadata_path)
                if existing.semantic_identity() == candidate.semantic_identity():
                    return existing
                raise BoundaryConflictError(
                    f"boundary {key} already exists with different content or version"
                )

            os.replace(tmp_payload, payload_path)
            fsync_directory(payload_path.parent)
            atomic_write_json(metadata_path, candidate.to_dict())
            return candidate
        finally:
            tmp_payload.unlink(missing_ok=True)

    def load(self, key: BoundaryKey, *, verify_checksum: bool = True) -> LoadedBoundary:
        payload_path, metadata_path = self._paths(key)
        if not metadata_path.exists() or not payload_path.exists():
            raise FileNotFoundError(f"committed boundary not found: {key}")

        metadata = self._read_metadata(metadata_path)
        if metadata.key != key:
            raise CorruptStateError(f"boundary key mismatch in {metadata_path}")
        if metadata.payload_file != payload_path.name:
            raise CorruptStateError(f"payload filename mismatch in {metadata_path}")
        if verify_checksum and sha256_file(payload_path) != metadata.payload_sha256:
            raise CorruptStateError(f"payload checksum mismatch for {key}")

        try:
            payload = torch.load(payload_path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(payload_path, map_location="cpu")
        hidden = payload.get("hidden")
        belief = payload.get("prev_log_probs")
        if not torch.is_tensor(hidden):
            raise CorruptStateError(f"hidden tensor missing for {key}")
        if belief is not None and not torch.is_tensor(belief):
            raise CorruptStateError(f"prev_log_probs is not a tensor for {key}")
        return LoadedBoundary(metadata=metadata, hidden=hidden, prev_log_probs=belief)

    def acknowledge(self, key: BoundaryKey, *, consumer_version: int) -> None:
        if consumer_version < 0:
            raise ValueError("consumer_version must be non-negative")
        self.load(key, verify_checksum=False)
        path = self._ack_path(key)
        candidate = {
            "schema_version": SCHEMA_VERSION,
            "key": asdict(key),
            "consumer_version": int(consumer_version),
        }
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing == candidate:
                return
            raise BoundaryConflictError(f"boundary {key} has a conflicting acknowledgement")
        atomic_write_json(path, candidate)

    def list_committed(self, source_stage: int, target_stage: int) -> list[BoundaryMetadata]:
        directory = self._boundary_dir(source_stage, target_stage)
        if not directory.exists():
            return []
        records = [self._read_metadata(path) for path in directory.glob("window-*.json")]
        return sorted(records, key=lambda item: item.key)

    def list_pending(self, source_stage: int, target_stage: int) -> list[BoundaryMetadata]:
        return [
            record
            for record in self.list_committed(source_stage, target_stage)
            if not self._ack_path(record.key).exists()
        ]

    def pending_window_ids(self, source_stage: int, target_stage: int) -> list[int]:
        return sorted(
            {record.key.window_id for record in self.list_pending(source_stage, target_stage)}
        )

    def ensure_window_capacity(
        self,
        source_stage: int,
        target_stage: int,
        window_id: int,
    ) -> None:
        if self.max_pending_windows is None:
            return
        pending = set(self.pending_window_ids(source_stage, target_stage))
        if window_id in pending:
            return
        if len(pending) >= self.max_pending_windows:
            raise OutboxCapacityError(
                f"outbox stage-{source_stage}->stage-{target_stage} already has "
                f"{len(pending)} pending windows; limit={self.max_pending_windows}"
            )

    def is_acknowledged(self, key: BoundaryKey) -> bool:
        return self._ack_path(key).exists()

    @staticmethod
    def _read_metadata(path: Path) -> BoundaryMetadata:
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata = BoundaryMetadata.from_dict(payload)
        if metadata.schema_version != SCHEMA_VERSION:
            raise CorruptStateError(
                f"unsupported boundary schema {metadata.schema_version} in {path}"
            )
        return metadata


@dataclass(frozen=True)
class StageCommit:
    run_id: str
    stage_id: int
    window_id: int
    optimizer_step: int
    request_ids: tuple[str, ...]
    input_producer_versions: tuple[int, ...]
    checkpoint_id: str
    committed_ns: int = 0
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _safe_component(self.run_id, "run_id")
        if min(self.stage_id, self.window_id, self.optimizer_step) < 0:
            raise ValueError("stage_id, window_id, and optimizer_step must be non-negative")
        if not self.request_ids:
            raise ValueError("request_ids cannot be empty")
        if any(version < 0 for version in self.input_producer_versions):
            raise ValueError("input producer versions must be non-negative")

    def with_commit_time(self) -> "StageCommit":
        if self.committed_ns:
            return self
        return StageCommit(
            run_id=self.run_id,
            stage_id=self.stage_id,
            window_id=self.window_id,
            optimizer_step=self.optimizer_step,
            request_ids=self.request_ids,
            input_producer_versions=self.input_producer_versions,
            checkpoint_id=self.checkpoint_id,
            committed_ns=time.time_ns(),
            schema_version=self.schema_version,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["request_ids"] = list(self.request_ids)
        payload["input_producer_versions"] = list(self.input_producer_versions)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StageCommit":
        return cls(
            run_id=str(payload["run_id"]),
            stage_id=int(payload["stage_id"]),
            window_id=int(payload["window_id"]),
            optimizer_step=int(payload["optimizer_step"]),
            request_ids=tuple(str(item) for item in payload["request_ids"]),
            input_producer_versions=tuple(int(item) for item in payload["input_producer_versions"]),
            checkpoint_id=str(payload.get("checkpoint_id", "")),
            committed_ns=int(payload["committed_ns"]),
            schema_version=int(payload["schema_version"]),
        )

    def semantic_identity(self) -> tuple[Any, ...]:
        return (
            self.schema_version,
            self.run_id,
            self.stage_id,
            self.window_id,
            self.optimizer_step,
            self.request_ids,
            self.input_producer_versions,
            self.checkpoint_id,
        )


class StageCommitLedger:
    """One immutable commit marker per stage-local optimizer window."""

    def __init__(self, root: Path, run_id: str) -> None:
        self.root = Path(root)
        self.run_id = _safe_component(run_id, "run_id")
        self.run_root = self.root / self.run_id / "commits"

    def _path(self, stage_id: int, window_id: int) -> Path:
        return self.run_root / f"stage-{stage_id}" / f"window-{window_id:08d}.json"

    def record(self, commit: StageCommit) -> StageCommit:
        if commit.run_id != self.run_id:
            raise ValueError(f"commit run_id={commit.run_id!r} does not match ledger run_id={self.run_id!r}")
        candidate = commit.with_commit_time()
        path = self._path(candidate.stage_id, candidate.window_id)
        if path.exists():
            existing = StageCommit.from_dict(json.loads(path.read_text(encoding="utf-8")))
            if existing.semantic_identity() == candidate.semantic_identity():
                return existing
            raise CommitConflictError(
                f"stage {candidate.stage_id} window {candidate.window_id} already has a different commit"
            )
        atomic_write_json(path, candidate.to_dict())
        return candidate

    def get(self, stage_id: int, window_id: int) -> StageCommit:
        path = self._path(stage_id, window_id)
        if not path.exists():
            raise FileNotFoundError(path)
        return StageCommit.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list_stage(self, stage_id: int) -> list[StageCommit]:
        directory = self.run_root / f"stage-{stage_id}"
        if not directory.exists():
            return []
        commits = [
            StageCommit.from_dict(json.loads(path.read_text(encoding="utf-8")))
            for path in directory.glob("window-*.json")
        ]
        return sorted(commits, key=lambda item: item.window_id)
