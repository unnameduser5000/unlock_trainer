from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Iterable, Optional

import torch

from sg_exe_trainer.runtime.bpfree.schedule import BPFreeMicrobatch, BPFreeUpdateWindow
from sg_exe_trainer.runtime.bpfree.gpu_stage import (
    BPFreePipelineStageV0,
    BodyForwardOutput,
    OptimizerStepOutput,
    ForwardInput,
    SendOutput,
)

from .checkpoint_store import StageCheckpointStore
from .window_journal import BPFreeWindowJournal, WindowCommitResult


InputVersionProvider = Callable[[BPFreeUpdateWindow], Iterable[int]]
SkipP2PPolicy = Callable[[BPFreeMicrobatch], bool]
ReadOutboxPolicy = Callable[[BPFreeMicrobatch], bool]
InputAcknowledger = Callable[[BPFreeUpdateWindow, int], None]


@dataclass(frozen=True)
class JournalWindowSelection:
    start_window: int
    end_window_exclusive: int

    def __post_init__(self) -> None:
        if self.start_window < 0 or self.end_window_exclusive <= self.start_window:
            raise ValueError("journal window range must be non-empty and non-negative")

    def contains(self, window_id: int) -> bool:
        return self.start_window <= window_id < self.end_window_exclusive


def request_ids_for_microbatch(mb: BPFreeMicrobatch) -> tuple[str, ...]:
    ids: list[str] = []
    for offset, record in enumerate(mb.records):
        value = next(
            (
                record[key]
                for key in ("request_id", "record_id", "id", "seq_no")
                if key in record and record[key] not in (None, "")
            ),
            None,
        )
        ids.append(str(value) if value is not None else f"seq-{mb.seq_start + offset}")
    return tuple(ids)


def request_ids_for_window(window: BPFreeUpdateWindow) -> tuple[str, ...]:
    return tuple(
        request_id
        for microbatch in window.microbatches
        for request_id in request_ids_for_microbatch(microbatch)
    )


class BPFreeStageJournalObserver:
    """Turns selected local optimizer windows into durable recovery state."""

    def __init__(
        self,
        *,
        stage_id: int,
        journal: BPFreeWindowJournal,
        checkpoint_store: StageCheckpointStore,
        selection: JournalWindowSelection,
        optimizer_steps_start: int = 0,
        input_version_provider: Optional[InputVersionProvider] = None,
        input_acknowledger: Optional[InputAcknowledger] = None,
    ) -> None:
        if optimizer_steps_start < 0:
            raise ValueError("optimizer_steps_start must be non-negative")
        if stage_id > 0 and input_version_provider is None:
            raise ValueError("downstream stages require an explicit input_version_provider")
        self.stage_id = stage_id
        self.journal = journal
        self.checkpoint_store = checkpoint_store
        self.selection = selection
        self.optimizer_step = optimizer_steps_start
        self.input_version_provider = input_version_provider
        self.input_acknowledger = input_acknowledger
        self.active_window_id: Optional[int] = None
        self.active_request_ids: tuple[str, ...] = ()
        self.results: list[WindowCommitResult] = []

    def before_window(self, window: BPFreeUpdateWindow) -> bool:
        if not self.selection.contains(window.window_id):
            return False
        if self.active_window_id is not None:
            raise RuntimeError(f"journal window {self.active_window_id} is still active")
        input_versions = (
            tuple(self.input_version_provider(window))
            if self.input_version_provider is not None
            else ()
        )
        self.journal.begin(
            window_id=window.window_id,
            microbatch_ids=(mb.mb_id for mb in window.microbatches),
            producer_version=self.optimizer_step,
            input_producer_versions=input_versions,
        )
        self.active_window_id = window.window_id
        self.active_request_ids = request_ids_for_window(window)
        return True

    def capture_output(self, mb: BPFreeMicrobatch, body_output: BodyForwardOutput) -> None:
        if self.active_window_id != mb.window_id:
            return
        self.journal.capture_output(
            microbatch_id=mb.mb_id,
            request_ids=request_ids_for_microbatch(mb),
            hidden=body_output.next_hidden,
        )

    def after_optimizer(
        self,
        *,
        window: BPFreeUpdateWindow,
        module: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> Optional[WindowCommitResult]:
        next_step = self.optimizer_step + 1
        if self.active_window_id is None:
            self.optimizer_step = next_step
            return None
        if self.active_window_id != window.window_id:
            raise RuntimeError(
                f"optimizer completed window {window.window_id}, but journal window "
                f"{self.active_window_id} is active"
            )

        checkpoint = self.checkpoint_store.save(
            module=module,
            optimizer=optimizer,
            stage_id=self.stage_id,
            window_id=window.window_id,
            optimizer_step=next_step,
            device=device,
        )
        result = self.journal.commit_after_optimizer(
            optimizer_step=next_step,
            request_ids=self.active_request_ids,
            checkpoint_id=checkpoint.checkpoint_id,
        )
        if self.input_acknowledger is not None:
            self.input_acknowledger(window, next_step)
        self.optimizer_step = next_step
        self.active_window_id = None
        self.active_request_ids = ()
        self.results.append(result)
        return result

    def abort_before_optimizer(self) -> None:
        if self.active_window_id is None:
            return
        self.journal.abort_before_optimizer()
        self.active_window_id = None
        self.active_request_ids = ()


class JournaledBPFreePipelineStage(BPFreePipelineStageV0):
    """Thin E5-only adapter around the clean BP-free stage implementation."""

    def __init__(
        self,
        *,
        journal_observer: BPFreeStageJournalObserver,
        skip_p2p_policy: Optional[SkipP2PPolicy] = None,
        read_outbox_policy: Optional[ReadOutboxPolicy] = None,
        committed_boundary_reader=None,
        recovery_run_id: str = "",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.journal_observer = journal_observer
        self.skip_p2p_policy = skip_p2p_policy or (lambda _mb: False)
        self.read_outbox_policy = read_outbox_policy or (lambda _mb: False)
        self.committed_boundary_reader = committed_boundary_reader
        self.recovery_run_id = str(recovery_run_id)

    def load_or_recv_forward_input(self, mb: BPFreeMicrobatch) -> ForwardInput:
        if not self.read_outbox_policy(mb):
            return super().load_or_recv_forward_input(mb)
        if self.rank == 0 or self.committed_boundary_reader is None:
            raise RuntimeError("outbox input requires a downstream stage and committed reader")

        from .state_contract import BoundaryKey

        started = time.perf_counter()
        loaded = self.committed_boundary_reader.load(
            BoundaryKey(
                self.recovery_run_id,
                self.rank - 1,
                self.rank,
                mb.window_id,
                mb.mb_id,
            )
        )
        hidden = loaded.hidden.to(device=self.device, dtype=self.dtype)
        prev_log_probs = loaded.prev_log_probs
        if prev_log_probs is not None:
            prev_log_probs = prev_log_probs.to(device=self.device)
        self._sync_cuda(mb)
        return ForwardInput(
            hidden=hidden,
            prev_log_probs=prev_log_probs,
            load_hidden_ms=(time.perf_counter() - started) * 1000.0,
        )

    def begin_window(
        self,
        *,
        window: BPFreeUpdateWindow,
        learning_rate_override: Optional[float],
    ) -> None:
        self.journal_observer.before_window(window)
        try:
            super().begin_window(
                window=window,
                learning_rate_override=learning_rate_override,
            )
        except Exception:
            self.journal_observer.abort_before_optimizer()
            raise

    def post_body_forward_send(
        self,
        *,
        mb: BPFreeMicrobatch,
        body_output: BodyForwardOutput,
    ) -> SendOutput:
        if self.is_last:
            return super().post_body_forward_send(mb=mb, body_output=body_output)
        self.journal_observer.capture_output(mb, body_output)
        if self.skip_p2p_policy(mb):
            if self.journal_observer.active_window_id != mb.window_id:
                raise RuntimeError(
                    f"cannot skip P2P for unjournaled window {mb.window_id}"
                )
            return SendOutput(send_hidden_ms=0.0, send_log_probs_ms=0.0)
        return super().post_body_forward_send(mb=mb, body_output=body_output)

    def maybe_optimizer_step(
        self,
        *,
        mb: BPFreeMicrobatch,
        window: BPFreeUpdateWindow,
    ) -> OptimizerStepOutput:
        output = super().maybe_optimizer_step(mb=mb, window=window)
        if output.applied:
            if self.optimizer is None:
                raise RuntimeError("optimizer disappeared after a local optimizer step")
            self.journal_observer.after_optimizer(
                window=window,
                module=self.chunk,
                optimizer=self.optimizer,
                device=self.device,
            )
        return output
