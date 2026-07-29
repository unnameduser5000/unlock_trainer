from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import torch

from .durable_io import atomic_write_json, fsync_directory, sha256_file
from .state_contract import CorruptStateError, StateContractError


class CheckpointConflictError(StateContractError):
    """The same stage/version checkpoint was written with different state."""


def _cpu_clone_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().to(device="cpu").contiguous().clone()
    if isinstance(value, dict):
        return {key: _cpu_clone_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone_tree(item) for item in value)
    return value


def _tree_tensor_bytes(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(_tree_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tree_tensor_bytes(item) for item in value)
    return 0


def _update_tree_digest(digest: Any, value: Any) -> None:
    if torch.is_tensor(value):
        tensor = value.detach().to(device="cpu").contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        return
    if isinstance(value, dict):
        digest.update(b"dict\0")
        for key in sorted(value, key=lambda item: repr(item)):
            _update_tree_digest(digest, key)
            _update_tree_digest(digest, value[key])
        return
    if isinstance(value, (list, tuple)):
        digest.update(b"list\0" if isinstance(value, list) else b"tuple\0")
        for item in value:
            _update_tree_digest(digest, item)
        return
    digest.update(type(value).__name__.encode("utf-8"))
    digest.update(b"\0")
    digest.update(repr(value).encode("utf-8"))


def _tree_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    _update_tree_digest(digest, value)
    return digest.hexdigest()


def _move_optimizer_state_to_parameter_device(optimizer: torch.optim.Optimizer) -> None:
    parameter = next(
        (item for group in optimizer.param_groups for item in group["params"]),
        None,
    )
    if parameter is None:
        return
    device = parameter.device
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device=device)


@dataclass(frozen=True)
class StageCheckpointMetadata:
    schema_version: int
    run_id: str
    stage_id: int
    window_id: int
    optimizer_step: int
    checkpoint_id: str
    payload_file: str
    payload_sha256: str
    state_sha256: str
    payload_file_bytes: int
    trainable_parameter_bytes: int
    optimizer_state_bytes: int
    created_ns: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StageCheckpointMetadata":
        return cls(**{key: payload[key] for key in cls.__dataclass_fields__})

    def semantic_identity(self) -> tuple[Any, ...]:
        return (
            self.schema_version,
            self.run_id,
            self.stage_id,
            self.window_id,
            self.optimizer_step,
            self.checkpoint_id,
            self.state_sha256,
        )


class StageCheckpointStore:
    """Atomic trainable-parameter, optimizer, and RNG checkpoints."""

    SCHEMA_VERSION = 1

    def __init__(self, root: Path, run_id: str) -> None:
        self.root = Path(root)
        self.run_id = str(run_id)
        self.run_root = self.root / self.run_id / "checkpoints"

    def _paths(self, stage_id: int, optimizer_step: int) -> tuple[Path, Path]:
        directory = self.run_root / f"stage-{stage_id}"
        stem = f"step-{optimizer_step:08d}"
        return directory / f"{stem}.pt", directory / f"{stem}.json"

    def save(
        self,
        *,
        module: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        stage_id: int,
        window_id: int,
        optimizer_step: int,
        device: Optional[torch.device] = None,
    ) -> StageCheckpointMetadata:
        if min(stage_id, window_id, optimizer_step) < 0:
            raise ValueError("stage_id, window_id, and optimizer_step must be non-negative")
        params = {
            name: parameter.detach().to(device="cpu").contiguous().clone()
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        }
        if not params:
            raise ValueError("module has no trainable parameters")

        rng_state: dict[str, Any] = {"cpu": torch.get_rng_state().clone()}
        if device is not None and device.type == "cuda":
            rng_state["cuda"] = torch.cuda.get_rng_state(device).cpu().clone()
            rng_state["cuda_device_index"] = device.index

        payload = {
            "trainable_parameters": params,
            "optimizer_state": _cpu_clone_tree(optimizer.state_dict()),
            "rng_state": rng_state,
        }
        state_sha256 = _tree_sha256(payload)
        checkpoint_id = f"stage{stage_id}-step{optimizer_step}"
        payload_path, metadata_path = self._paths(stage_id, optimizer_step)
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_payload = payload_path.with_name(
            f".{payload_path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
        )
        try:
            torch.save(payload, tmp_payload)
            with tmp_payload.open("rb") as handle:
                os.fsync(handle.fileno())
            candidate = StageCheckpointMetadata(
                schema_version=self.SCHEMA_VERSION,
                run_id=self.run_id,
                stage_id=stage_id,
                window_id=window_id,
                optimizer_step=optimizer_step,
                checkpoint_id=checkpoint_id,
                payload_file=payload_path.name,
                payload_sha256=sha256_file(tmp_payload),
                state_sha256=state_sha256,
                payload_file_bytes=tmp_payload.stat().st_size,
                trainable_parameter_bytes=_tree_tensor_bytes(params),
                optimizer_state_bytes=_tree_tensor_bytes(payload["optimizer_state"]),
                created_ns=time.time_ns(),
            )

            if metadata_path.exists():
                existing = self._read_metadata(metadata_path)
                if existing.semantic_identity() == candidate.semantic_identity():
                    return existing
                raise CheckpointConflictError(
                    f"stage {stage_id} optimizer step {optimizer_step} already has different state"
                )

            os.replace(tmp_payload, payload_path)
            fsync_directory(payload_path.parent)
            atomic_write_json(metadata_path, candidate.to_dict())
            return candidate
        finally:
            tmp_payload.unlink(missing_ok=True)

    def restore(
        self,
        *,
        module: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        stage_id: int,
        optimizer_step: int,
        restore_rng: bool = True,
        device: Optional[torch.device] = None,
    ) -> StageCheckpointMetadata:
        payload_path, metadata_path = self._paths(stage_id, optimizer_step)
        if not payload_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"checkpoint stage={stage_id} step={optimizer_step} not found")
        metadata = self._read_metadata(metadata_path)
        if sha256_file(payload_path) != metadata.payload_sha256:
            raise CorruptStateError(f"checkpoint file checksum mismatch: {payload_path}")
        try:
            payload = torch.load(payload_path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(payload_path, map_location="cpu")
        if _tree_sha256(payload) != metadata.state_sha256:
            raise CorruptStateError(f"checkpoint state fingerprint mismatch: {payload_path}")

        parameter_map = dict(module.named_parameters())
        saved_params = payload["trainable_parameters"]
        expected_names = {name for name, parameter in parameter_map.items() if parameter.requires_grad}
        if set(saved_params) != expected_names:
            raise CorruptStateError(
                f"trainable parameter names differ: saved={sorted(saved_params)} expected={sorted(expected_names)}"
            )
        with torch.no_grad():
            for name, saved in saved_params.items():
                parameter = parameter_map[name]
                parameter.copy_(saved.to(device=parameter.device, dtype=parameter.dtype))

        optimizer.load_state_dict(payload["optimizer_state"])
        _move_optimizer_state_to_parameter_device(optimizer)
        optimizer.zero_grad(set_to_none=True)

        if restore_rng:
            rng_state = payload["rng_state"]
            torch.set_rng_state(rng_state["cpu"])
            if device is not None and device.type == "cuda" and "cuda" in rng_state:
                torch.cuda.set_rng_state(rng_state["cuda"], device=device)
        return metadata

    def list_stage(self, stage_id: int) -> list[StageCheckpointMetadata]:
        directory = self.run_root / f"stage-{stage_id}"
        if not directory.exists():
            return []
        records = [
            self._read_metadata(path)
            for path in directory.glob("step-*.json")
        ]
        return sorted(records, key=lambda item: item.optimizer_step)

    def _read_metadata(self, path: Path) -> StageCheckpointMetadata:
        metadata = StageCheckpointMetadata.from_dict(json.loads(path.read_text(encoding="utf-8")))
        if metadata.schema_version != self.SCHEMA_VERSION or metadata.run_id != self.run_id:
            raise CorruptStateError(f"checkpoint metadata mismatch in {path}")
        return metadata
