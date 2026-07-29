from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch

from .state_contract import (
    BoundaryKey,
    BoundaryMetadata,
    CorruptStateError,
    DurableBoundaryOutbox,
    LoadedBoundary,
    StageCommit,
    StageCommitLedger,
)


class WindowJournalError(RuntimeError):
    """Base class for an invalid BP-free local-window transition."""


class WindowAlreadyOpenError(WindowJournalError):
    pass


class NoOpenWindowError(WindowJournalError):
    pass


class IncompleteWindowError(WindowJournalError):
    pass


@dataclass(frozen=True)
class PendingBoundary:
    microbatch_id: int
    request_ids: tuple[str, ...]
    hidden: torch.Tensor
    prev_log_probs: Optional[torch.Tensor]


@dataclass(frozen=True)
class WindowCommitResult:
    commit: StageCommit
    boundaries: tuple[BoundaryMetadata, ...]

    @property
    def durable_payload_bytes(self) -> int:
        return sum(item.payload_file_bytes for item in self.boundaries)

    @property
    def hidden_tensor_bytes(self) -> int:
        return sum(item.hidden_tensor_bytes for item in self.boundaries)


class BPFreeWindowJournal:
    """Publishes a complete BP-free local-update window.

    Hidden tensors are snapshotted in process memory during forward. They are
    written to the durable outbox only after the local optimizer step succeeds.
    The stage commit marker is written last. A recovery consumer must use
    `CommittedBoundaryReader`, which rejects an outbox window without that final
    commit marker.
    """

    def __init__(
        self,
        *,
        run_id: str,
        stage_id: int,
        world_size: int,
        outbox: DurableBoundaryOutbox,
        ledger: StageCommitLedger,
    ) -> None:
        if stage_id < 0 or stage_id >= world_size:
            raise ValueError("stage_id must be in [0, world_size)")
        self.run_id = run_id
        self.stage_id = stage_id
        self.world_size = world_size
        self.outbox = outbox
        self.ledger = ledger
        self._window_id: Optional[int] = None
        self._producer_version: Optional[int] = None
        self._expected_microbatch_ids: tuple[int, ...] = ()
        self._input_producer_versions: tuple[int, ...] = ()
        self._boundaries: dict[int, PendingBoundary] = {}

    @property
    def has_open_window(self) -> bool:
        return self._window_id is not None

    def begin(
        self,
        *,
        window_id: int,
        microbatch_ids: Iterable[int],
        producer_version: int,
        input_producer_versions: Iterable[int] = (),
    ) -> None:
        if self.has_open_window:
            raise WindowAlreadyOpenError(f"stage {self.stage_id} already has an open window")
        if window_id < 0 or producer_version < 0:
            raise ValueError("window_id and producer_version must be non-negative")
        expected = tuple(int(item) for item in microbatch_ids)
        if not expected or len(set(expected)) != len(expected) or any(item < 0 for item in expected):
            raise ValueError("microbatch_ids must be unique non-negative values")
        input_versions = tuple(int(item) for item in input_producer_versions)
        if any(item < 0 for item in input_versions):
            raise ValueError("input producer versions must be non-negative")

        if self.stage_id < self.world_size - 1:
            self.outbox.ensure_window_capacity(self.stage_id, self.stage_id + 1, window_id)

        self._window_id = window_id
        self._producer_version = producer_version
        self._expected_microbatch_ids = expected
        self._input_producer_versions = input_versions
        self._boundaries.clear()

    def capture_output(
        self,
        *,
        microbatch_id: int,
        request_ids: Iterable[str],
        hidden: torch.Tensor,
        prev_log_probs: Optional[torch.Tensor] = None,
    ) -> None:
        self._require_open()
        if self.stage_id == self.world_size - 1:
            raise WindowJournalError("terminal stage has no downstream boundary")
        if microbatch_id not in self._expected_microbatch_ids:
            raise ValueError(f"unexpected microbatch_id={microbatch_id}")
        if microbatch_id in self._boundaries:
            raise WindowJournalError(f"microbatch_id={microbatch_id} was captured twice")
        ids = tuple(str(item) for item in request_ids)
        if not ids:
            raise ValueError("request_ids cannot be empty")
        self._boundaries[microbatch_id] = PendingBoundary(
            microbatch_id=microbatch_id,
            request_ids=ids,
            hidden=hidden.detach().to(device="cpu").contiguous().clone(),
            prev_log_probs=(
                prev_log_probs.detach().to(device="cpu").contiguous().clone()
                if prev_log_probs is not None
                else None
            ),
        )

    def commit_after_optimizer(
        self,
        *,
        optimizer_step: int,
        request_ids: Iterable[str],
        checkpoint_id: str,
    ) -> WindowCommitResult:
        self._require_open()
        assert self._window_id is not None
        assert self._producer_version is not None
        if optimizer_step != self._producer_version + 1:
            raise WindowJournalError(
                f"optimizer_step={optimizer_step} must follow producer_version={self._producer_version}"
            )

        if self.stage_id < self.world_size - 1:
            captured = tuple(sorted(self._boundaries))
            expected = tuple(sorted(self._expected_microbatch_ids))
            if captured != expected:
                raise IncompleteWindowError(
                    f"stage {self.stage_id} window {self._window_id} expected boundaries "
                    f"{expected}, captured {captured}"
                )

        boundary_records: list[BoundaryMetadata] = []
        for microbatch_id in self._expected_microbatch_ids:
            if self.stage_id == self.world_size - 1:
                break
            pending = self._boundaries[microbatch_id]
            key = BoundaryKey(
                run_id=self.run_id,
                source_stage=self.stage_id,
                target_stage=self.stage_id + 1,
                window_id=self._window_id,
                microbatch_id=microbatch_id,
            )
            boundary_records.append(
                self.outbox.put(
                    key,
                    hidden=pending.hidden,
                    request_ids=pending.request_ids,
                    producer_version=self._producer_version,
                    prev_log_probs=pending.prev_log_probs,
                )
            )

        commit = self.ledger.record(
            StageCommit(
                run_id=self.run_id,
                stage_id=self.stage_id,
                window_id=self._window_id,
                optimizer_step=optimizer_step,
                request_ids=tuple(str(item) for item in request_ids),
                input_producer_versions=self._input_producer_versions,
                checkpoint_id=str(checkpoint_id),
            )
        )
        result = WindowCommitResult(commit=commit, boundaries=tuple(boundary_records))
        self._reset()
        return result

    def abort_before_optimizer(self) -> None:
        self._require_open()
        self._reset()

    def _require_open(self) -> None:
        if not self.has_open_window:
            raise NoOpenWindowError(f"stage {self.stage_id} has no open window")

    def _reset(self) -> None:
        self._window_id = None
        self._producer_version = None
        self._expected_microbatch_ids = ()
        self._input_producer_versions = ()
        self._boundaries.clear()


class CommittedBoundaryReader:
    """Reads only complete windows whose producer commit marker exists."""

    def __init__(self, *, outbox: DurableBoundaryOutbox, ledger: StageCommitLedger) -> None:
        self.outbox = outbox
        self.ledger = ledger

    def load(self, key: BoundaryKey) -> LoadedBoundary:
        commit = self.ledger.get(key.source_stage, key.window_id)
        loaded = self.outbox.load(key)
        expected_forward_version = commit.optimizer_step - 1
        if loaded.metadata.producer_version != expected_forward_version:
            raise CorruptStateError(
                f"boundary {key} producer_version={loaded.metadata.producer_version} "
                f"does not match committed forward version={expected_forward_version}"
            )
        if not set(loaded.metadata.request_ids).issubset(set(commit.request_ids)):
            raise CorruptStateError(f"boundary {key} contains request ids outside its stage commit")
        return loaded
