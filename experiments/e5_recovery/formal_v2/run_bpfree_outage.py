#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoConfig, AutoModelForCausalLM

from sg_exe_trainer.common.trainable_modes import configure_model_trainable
from sg_exe_trainer.metrics.activation_memory import SavedTensorTracker
from sg_exe_trainer.runtime.bpfree.model_runtime import (
    build_optimizer,
    build_stage_chunk,
)
from sg_exe_trainer.runtime.bpfree.schedule import (
    BPFreeUpdateWindow,
    split_records_into_update_windows,
)
from sg_exe_trainer.runtime.bpfree.schedule_runtime import ScheduleBPFreeBodySendHeadV1
from sg_exe_trainer.runtime.bpfree.trace import ActionTracer, NoOpActionTracer
from sg_exe_trainer.tasks.label_experiment import read_manifest, resolve_dtype, resolve_model_name

from sg_exe_trainer.runtime.recovery.checkpoint_store import StageCheckpointStore
from sg_exe_trainer.runtime.recovery.catchup_stream import wait_for_stage_commit
from sg_exe_trainer.runtime.recovery.durable_io import atomic_write_json
from sg_exe_trainer.runtime.recovery.event_log import (
    RecoveryEventName,
    RecoveryEventRecorder,
    RecoveryTimeline,
)
from sg_exe_trainer.runtime.recovery.runtime_adapter import (
    BPFreeStageJournalObserver,
    JournaledBPFreePipelineStage,
    JournalWindowSelection,
)
from sg_exe_trainer.runtime.recovery.state_contract import (
    BoundaryKey,
    DurableBoundaryOutbox,
    StageCommitLedger,
)
from sg_exe_trainer.runtime.recovery.volatile_backlog import (
    VolatileBoundaryBuffer,
    VolatileCaptureBPFreePipelineStage,
)
from sg_exe_trainer.runtime.recovery.window_journal import (
    BPFreeWindowJournal,
    CommittedBoundaryReader,
)

from .protocol import E5OutageProtocol


def _parse_devices(raw: str, expected: int) -> list[str]:
    devices = [item.strip() for item in raw.split(",") if item.strip()]
    if len(devices) != expected:
        raise ValueError(f"expected {expected} stage devices, got {devices}")
    return devices


def _seed_everything(seed: int, rank: int) -> None:
    actual = int(seed) + int(rank)
    random.seed(actual)
    np.random.seed(actual % (2**32 - 1))
    torch.manual_seed(actual)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(actual)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _run_windows(
    *,
    phase: str,
    windows: list[BPFreeUpdateWindow],
    schedule: ScheduleBPFreeBodySendHeadV1,
    device: torch.device,
    rank_metrics: list[dict[str, Any]],
    event_recorder: RecoveryEventRecorder,
    record_prefix_commits: bool = False,
    before_window: Optional[Callable[[BPFreeUpdateWindow], dict[str, Any]]] = None,
) -> None:
    _sync(device)
    phase_started = time.monotonic_ns()
    for window in windows:
        ready_wait_started = time.monotonic_ns()
        extra_metrics = before_window(window) if before_window is not None else {}
        _sync(device)
        started = time.monotonic_ns()
        schedule.step_window(window)
        _sync(device)
        ended = time.monotonic_ns()
        elapsed_ms = (ended - started) / 1_000_000.0
        rank_metrics.append(
            {
                "phase": phase,
                "window_id": window.window_id,
                "records": window.num_records,
                "elapsed_ms": elapsed_ms,
                "optimizer_steps": schedule.optimizer_steps,
                "ready_wait_start_monotonic_ns": ready_wait_started,
                "compute_start_monotonic_ns": started,
                "compute_end_monotonic_ns": ended,
                **extra_metrics,
            }
        )
        if record_prefix_commits:
            event_recorder.record(
                RecoveryEventName.PREFIX_WINDOW_COMMIT,
                window_id=window.window_id,
                details={"elapsed_ms": elapsed_ms},
            )
    phase_ended = time.monotonic_ns()
    rank_metrics.append(
        {
            "phase": f"{phase}_total",
            "window_id": -1,
            "records": sum(window.num_records for window in windows),
            "elapsed_ms": (phase_ended - phase_started) / 1_000_000.0,
            "optimizer_steps": schedule.optimizer_steps,
            "phase_start_monotonic_ns": phase_started,
            "phase_end_monotonic_ns": phase_ended,
        }
    )


def _run_drain_first_catchup(
    *,
    rank: int,
    protocol: E5OutageProtocol,
    outage: list[BPFreeUpdateWindow],
    schedule: ScheduleBPFreeBodySendHeadV1,
    device: torch.device,
    rank_metrics: list[dict[str, Any]],
    event_recorder: RecoveryEventRecorder,
) -> None:
    if rank == 1:
        event_recorder.record(
            RecoveryEventName.STAGE_REJOINED,
            window_id=protocol.outage_end_window_exclusive,
        )
        event_recorder.record(
            RecoveryEventName.CATCHUP_STAGE1_START,
            window_id=protocol.outage_start_window,
        )
        _run_windows(
            phase="catchup_stage1",
            windows=outage,
            schedule=schedule,
            device=device,
            rank_metrics=rank_metrics,
            event_recorder=event_recorder,
        )
        event_recorder.record(
            RecoveryEventName.CATCHUP_STAGE1_DONE,
            window_id=protocol.outage_end_window_exclusive - 1,
        )
    dist.barrier()

    if rank == 2:
        event_recorder.record(
            RecoveryEventName.CATCHUP_STAGE2_START,
            window_id=protocol.outage_start_window,
        )
        _run_windows(
            phase="catchup_stage2",
            windows=outage,
            schedule=schedule,
            device=device,
            rank_metrics=rank_metrics,
            event_recorder=event_recorder,
        )
        event_recorder.record(
            RecoveryEventName.CATCHUP_STAGE2_DONE,
            window_id=protocol.outage_end_window_exclusive - 1,
        )
        event_recorder.record(
            RecoveryEventName.TERMINAL_TARGET_REACHED,
            window_id=protocol.outage_end_window_exclusive - 1,
        )
    dist.barrier()


def _run_window_streamed_catchup(
    *,
    rank: int,
    protocol: E5OutageProtocol,
    outage: list[BPFreeUpdateWindow],
    schedule: ScheduleBPFreeBodySendHeadV1,
    device: torch.device,
    rank_metrics: list[dict[str, Any]],
    event_recorder: RecoveryEventRecorder,
    ledger: StageCommitLedger,
    wait_timeout_s: float,
    poll_ms: float,
) -> None:
    if rank == 1:
        event_recorder.record(
            RecoveryEventName.STAGE_REJOINED,
            window_id=protocol.outage_end_window_exclusive,
        )
    # Stage 2 must not poll before the rejoin transition is visible to all ranks.
    dist.barrier()

    if rank == 1:
        event_recorder.record(
            RecoveryEventName.CATCHUP_STAGE1_START,
            window_id=protocol.outage_start_window,
        )
        _run_windows(
            phase="catchup_stage1_streamed",
            windows=outage,
            schedule=schedule,
            device=device,
            rank_metrics=rank_metrics,
            event_recorder=event_recorder,
        )
        event_recorder.record(
            RecoveryEventName.CATCHUP_STAGE1_DONE,
            window_id=protocol.outage_end_window_exclusive - 1,
        )
    elif rank == 2:
        event_recorder.record(
            RecoveryEventName.CATCHUP_STAGE2_START,
            window_id=protocol.outage_start_window,
        )

        def wait_for_upstream(window: BPFreeUpdateWindow) -> dict[str, Any]:
            result = wait_for_stage_commit(
                ledger=ledger,
                stage_id=1,
                window_id=window.window_id,
                timeout_s=wait_timeout_s,
                poll_ms=poll_ms,
            )
            return {
                "upstream_commit_wait_ms": result.wait_ms,
                "upstream_commit_polls": result.polls,
                "upstream_optimizer_step": result.commit.optimizer_step,
            }

        _run_windows(
            phase="catchup_stage2_streamed",
            windows=outage,
            schedule=schedule,
            device=device,
            rank_metrics=rank_metrics,
            event_recorder=event_recorder,
            before_window=wait_for_upstream,
        )
        event_recorder.record(
            RecoveryEventName.CATCHUP_STAGE2_DONE,
            window_id=protocol.outage_end_window_exclusive - 1,
        )
        event_recorder.record(
            RecoveryEventName.TERMINAL_TARGET_REACHED,
            window_id=protocol.outage_end_window_exclusive - 1,
        )
    dist.barrier()


def _run_volatile_window_streamed_catchup(
    *,
    rank: int,
    protocol: E5OutageProtocol,
    outage: list[BPFreeUpdateWindow],
    schedule: ScheduleBPFreeBodySendHeadV1,
    stage: VolatileCaptureBPFreePipelineStage,
    buffer: VolatileBoundaryBuffer,
    device: torch.device,
    rank_metrics: list[dict[str, Any]],
    event_recorder: RecoveryEventRecorder,
) -> None:
    if rank == protocol.failure_stage:
        event_recorder.record(
            RecoveryEventName.STAGE_REJOINED,
            window_id=protocol.outage_end_window_exclusive,
        )
    dist.barrier()

    if rank == 0:
        phase_started = time.monotonic_ns()
        staging = stage.prepare_buffered_replay(outage)
        staging_ended = time.monotonic_ns()
        rank_metrics.append(
            {
                "phase": "catchup_stage0_volatile_staging",
                "window_id": -1,
                "records": sum(window.num_records for window in outage),
                "elapsed_ms": float(staging["elapsed_ms"]),
                "optimizer_steps": schedule.optimizer_steps,
                "phase_start_monotonic_ns": phase_started,
                "phase_end_monotonic_ns": staging_ended,
                "microbatches": int(staging["microbatches"]),
                "tensor_bytes": int(staging["tensor_bytes"]),
            }
        )
        for window in outage:
            buffer.validate_window(window)
            started = time.monotonic_ns()
            replay_metrics = [
                stage.replay_buffered_hidden(mb) for mb in window.microbatches
            ]
            stage.drain_pending_sends()
            _sync(device)
            ended = time.monotonic_ns()
            rank_metrics.append(
                {
                    "phase": "catchup_stage0_volatile_replay",
                    "window_id": window.window_id,
                    "records": window.num_records,
                    "elapsed_ms": (ended - started) / 1_000_000.0,
                    "optimizer_steps": schedule.optimizer_steps,
                    "compute_start_monotonic_ns": started,
                    "compute_end_monotonic_ns": ended,
                    "h2d_ms": sum(float(item["h2d_ms"]) for item in replay_metrics),
                    "send_hidden_ms": sum(
                        float(item["send_hidden_ms"]) for item in replay_metrics
                    ),
                    "tensor_bytes": sum(
                        int(item["tensor_bytes"]) for item in replay_metrics
                    ),
                }
            )
        phase_ended = time.monotonic_ns()
        rank_metrics.append(
            {
                "phase": "catchup_stage0_volatile_replay_total",
                "window_id": -1,
                "records": sum(window.num_records for window in outage),
                "elapsed_ms": (phase_ended - phase_started) / 1_000_000.0,
                "optimizer_steps": schedule.optimizer_steps,
                "phase_start_monotonic_ns": phase_started,
                "phase_end_monotonic_ns": phase_ended,
            }
        )
        stage.clear_replay_cache()
    elif rank == 1:
        event_recorder.record(
            RecoveryEventName.CATCHUP_STAGE1_START,
            window_id=protocol.outage_start_window,
        )
        _run_windows(
            phase="catchup_stage1_volatile",
            windows=outage,
            schedule=schedule,
            device=device,
            rank_metrics=rank_metrics,
            event_recorder=event_recorder,
        )
        stage.drain_pending_sends()
        event_recorder.record(
            RecoveryEventName.CATCHUP_STAGE1_DONE,
            window_id=protocol.outage_end_window_exclusive - 1,
        )
    else:
        event_recorder.record(
            RecoveryEventName.CATCHUP_STAGE2_START,
            window_id=protocol.outage_start_window,
        )
        _run_windows(
            phase="catchup_stage2_volatile",
            windows=outage,
            schedule=schedule,
            device=device,
            rank_metrics=rank_metrics,
            event_recorder=event_recorder,
        )
        event_recorder.record(
            RecoveryEventName.CATCHUP_STAGE2_DONE,
            window_id=protocol.outage_end_window_exclusive - 1,
        )
        event_recorder.record(
            RecoveryEventName.TERMINAL_TARGET_REACHED,
            window_id=protocol.outage_end_window_exclusive - 1,
        )
    dist.barrier()


def _worker(
    rank: int,
    world_size: int,
    cfg: dict[str, Any],
    volatile_stage_cls: type[VolatileCaptureBPFreePipelineStage],
) -> None:
    os.environ["MASTER_ADDR"] = str(cfg["master_addr"])
    os.environ["MASTER_PORT"] = str(cfg["master_port"])
    device = torch.device(cfg["stage_devices"][rank])
    if device.type == "cuda":
        torch.cuda.set_device(device)
    process_group_kwargs: dict[str, Any] = {
        "backend": str(cfg["backend"]),
        "rank": rank,
        "world_size": world_size,
    }
    if str(cfg["backend"]) == "nccl":
        process_group_kwargs["device_id"] = device
    try:
        dist.init_process_group(**process_group_kwargs)
    except TypeError:
        process_group_kwargs.pop("device_id", None)
        dist.init_process_group(**process_group_kwargs)

    try:
        _seed_everything(int(cfg["seed"]), rank)
        output_dir = Path(cfg["output_dir"])
        protocol = E5OutageProtocol(**cfg["protocol_constructor"])
        recovery_state_mode = str(cfg["recovery_state_mode"])
        durable_recovery = recovery_state_mode == "durable"
        state_root = output_dir / (
            "durable_state" if durable_recovery else "runtime_state"
        )
        event_recorder = RecoveryEventRecorder(
            state_root,
            run_id=protocol.run_id,
            rank=rank,
            stage_id=rank,
        )

        dtype = resolve_dtype(str(cfg["dtype"]))
        model = AutoModelForCausalLM.from_pretrained(
            str(cfg["resolved_model"]),
            torch_dtype=dtype,
        )
        hidden_size = int(
            getattr(model.config, "hidden_size", 0)
            or getattr(model.config, "n_embd", 0)
            or 0
        )
        if hidden_size <= 0:
            raise ValueError("model hidden size is not positive")

        trainable_setup = configure_model_trainable(
            module=model,
            mode=str(cfg["trainable_mode"]),
            lora_targets=str(cfg["lora_targets"]),
            lora_rank=int(cfg["lora_rank"]),
            lora_alpha=float(cfg["lora_alpha"]),
            lora_init_std=float(cfg["lora_init_std"]),
            lora_init_seed=int(cfg["lora_init_seed"]),
        )
        chunk = build_stage_chunk(
            model=model,
            stage_id=rank,
            num_chunks=world_size,
            belief_transport_mode="terminal",
            alpha=float(cfg["alpha"]),
            label_smoothing=float(cfg["label_smoothing"]),
        )
        chunk.to(device)
        local_params = [parameter for parameter in chunk.parameters() if parameter.requires_grad]
        optimizer = build_optimizer(
            params=local_params,
            cfg=argparse.Namespace(
                learning_rate=float(cfg["learning_rate"]),
                optimizer=str(cfg["optimizer"]),
                sgd_momentum=0.0,
                sgd_dampening=0.0,
                sgd_weight_decay=0.0,
                sgd_nesterov=False,
            ),
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        train_manifest = Path(cfg["train_manifest"])
        required_records = protocol.total_logical_windows * protocol.effective_batch_size
        records = read_manifest(train_manifest, required_records)
        if len(records) < required_records:
            raise ValueError(f"need {required_records} records, found {len(records)}")
        windows = split_records_into_update_windows(
            records=records,
            effective_batch_size=protocol.effective_batch_size,
            n_microbatches=protocol.microbatches_per_window,
            drop_last=True,
        )
        prelude = windows[: protocol.outage_start_window]
        outage = windows[protocol.outage_start_window : protocol.outage_end_window_exclusive]
        resumed = windows[protocol.outage_end_window_exclusive : protocol.total_logical_windows]

        def is_outage_window(mb) -> bool:
            return protocol.outage_start_window <= mb.window_id < protocol.outage_end_window_exclusive

        tracer = (
            ActionTracer(
                path=output_dir / f"rank{rank}.actions.csv",
                phase="e5_formal_v2",
                rank=rank,
                stage_id=rank,
            )
            if bool(cfg["enable_action_trace"])
            else NoOpActionTracer()
        )
        stage_kwargs = {
            "rank": rank,
            "world_size": world_size,
            "phase": "train",
            "mode": "train",
            "request_prefix": str(cfg["request_prefix"]),
            "chunk": chunk,
            "optimizer": optimizer,
            "train_this_rank": True,
            "dtype": dtype,
            "device": device,
            "belief_transport_mode": "terminal",
            "grad_clip": float(cfg["grad_clip"]),
            "hidden_size": hidden_size,
            "vocab_size": int(cfg["vocab_size"]),
            "manifest_dir": train_manifest.parent,
            "tracer": tracer,
            "max_pending_sends": 2,
            "recv_inflight_depth": int(cfg["recv_inflight_depth"]),
            "sync_action_trace": bool(cfg["sync_action_trace"]),
            "perf_minimal_metrics": True,
        }

        outbox: Optional[DurableBoundaryOutbox] = None
        ledger: Optional[StageCommitLedger] = None
        checkpoint_store: Optional[StageCheckpointStore] = None
        observer: Optional[BPFreeStageJournalObserver] = None
        volatile_buffer: Optional[VolatileBoundaryBuffer] = None

        if durable_recovery:
            outbox = DurableBoundaryOutbox(
                state_root,
                protocol.run_id,
                max_pending_windows=protocol.max_pending_windows,
            )
            ledger = StageCommitLedger(state_root, protocol.run_id)
            reader = CommittedBoundaryReader(outbox=outbox, ledger=ledger)

            def input_versions(window: BPFreeUpdateWindow) -> tuple[int, ...]:
                return tuple(
                    reader.load(
                        BoundaryKey(
                            protocol.run_id,
                            rank - 1,
                            rank,
                            window.window_id,
                            mb.mb_id,
                        )
                    ).metadata.producer_version
                    for mb in window.microbatches
                )

            def acknowledge_inputs(
                window: BPFreeUpdateWindow,
                consumer_version: int,
            ) -> None:
                if rank == 0:
                    return
                for mb in window.microbatches:
                    outbox.acknowledge(
                        BoundaryKey(
                            protocol.run_id,
                            rank - 1,
                            rank,
                            window.window_id,
                            mb.mb_id,
                        ),
                        consumer_version=consumer_version,
                    )

            checkpoint_store = StageCheckpointStore(state_root, protocol.run_id)
            observer = BPFreeStageJournalObserver(
                stage_id=rank,
                journal=BPFreeWindowJournal(
                    run_id=protocol.run_id,
                    stage_id=rank,
                    world_size=world_size,
                    outbox=outbox,
                    ledger=ledger,
                ),
                checkpoint_store=checkpoint_store,
                selection=JournalWindowSelection(
                    protocol.outage_start_window,
                    protocol.outage_end_window_exclusive,
                ),
                optimizer_steps_start=0,
                input_version_provider=input_versions if rank > 0 else None,
                input_acknowledger=acknowledge_inputs if rank > 0 else None,
            )
            stage = JournaledBPFreePipelineStage(
                journal_observer=observer,
                skip_p2p_policy=is_outage_window,
                read_outbox_policy=(is_outage_window if rank > 0 else None),
                committed_boundary_reader=reader,
                recovery_run_id=protocol.run_id,
                **stage_kwargs,
            )
        else:
            volatile_buffer = VolatileBoundaryBuffer(
                max_pending_windows=protocol.max_pending_windows
            )
            stage = volatile_stage_cls(
                volatile_buffer=volatile_buffer,
                skip_p2p_policy=(is_outage_window if rank == 0 else lambda _mb: False),
                **stage_kwargs,
            )
        schedule = ScheduleBPFreeBodySendHeadV1(
            stage=stage,
            train_this_rank=True,
            track_activation_memory=False,
            activation_tracker=SavedTensorTracker(),
            vocab_size=int(cfg["vocab_size"]),
            learning_rate_override=float(cfg["learning_rate"]),
            optimizer_steps_start=0,
            perf_minimal_metrics=True,
            window_input_staging=False,
        )
        recv_windows = (
            prelude + resumed
            if durable_recovery
            else prelude + outage + resumed
        )
        stage.set_forward_recv_plan(
            [mb for window in recv_windows for mb in window.microbatches]
        )
        rank_metrics: list[dict[str, Any]] = []

        dist.barrier()
        _run_windows(
            phase="prelude",
            windows=prelude,
            schedule=schedule,
            device=device,
            rank_metrics=rank_metrics,
            event_recorder=event_recorder,
        )
        stage.drain_pending_sends()
        dist.barrier()

        if rank == protocol.failure_stage:
            event_recorder.record(
                RecoveryEventName.OUTAGE_INJECTED,
                window_id=protocol.outage_start_window,
            )
            event_recorder.record(
                RecoveryEventName.OUTAGE_DETECTED,
                window_id=protocol.outage_start_window,
            )
        dist.barrier()

        if rank == 0:
            _run_windows(
                phase="outage_prefix",
                windows=outage,
                schedule=schedule,
                device=device,
                rank_metrics=rank_metrics,
                event_recorder=event_recorder,
                record_prefix_commits=True,
            )
        dist.barrier()

        if durable_recovery:
            assert ledger is not None and outbox is not None
            stage_commit_counts_at_rejoin = {
                stage_id: len(ledger.list_stage(stage_id))
                for stage_id in range(world_size)
            }
            pending_outbox_windows_at_rejoin = {
                "stage0_to_stage1": outbox.pending_window_ids(0, 1),
                "stage1_to_stage2": outbox.pending_window_ids(1, 2),
            }
        else:
            stage_commit_counts_at_rejoin = {
                stage_id: (protocol.outage_windows if stage_id == 0 else 0)
                for stage_id in range(world_size)
            }
            pending_outbox_windows_at_rejoin = {
                "stage0_to_stage1": [],
                "stage1_to_stage2": [],
            }

        volatile_state_at_rejoin = {
            "window_ids": volatile_buffer.window_ids() if volatile_buffer is not None else [],
            "microbatches": (
                volatile_buffer.microbatch_count if volatile_buffer is not None else 0
            ),
            "hidden_tensor_bytes": (
                volatile_buffer.tensor_bytes if volatile_buffer is not None else 0
            ),
            "capture_ms": volatile_buffer.capture_ms if volatile_buffer is not None else 0.0,
        }

        if not durable_recovery:
            assert isinstance(stage, volatile_stage_cls)
            assert volatile_buffer is not None
            _run_volatile_window_streamed_catchup(
                rank=rank,
                protocol=protocol,
                outage=outage,
                schedule=schedule,
                stage=stage,
                buffer=volatile_buffer,
                device=device,
                rank_metrics=rank_metrics,
                event_recorder=event_recorder,
            )
            volatile_buffer.clear()
        elif protocol.catchup_policy == "window_streamed":
            assert ledger is not None
            _run_window_streamed_catchup(
                rank=rank,
                protocol=protocol,
                outage=outage,
                schedule=schedule,
                device=device,
                rank_metrics=rank_metrics,
                event_recorder=event_recorder,
                ledger=ledger,
                wait_timeout_s=float(cfg["catchup_wait_timeout_s"]),
                poll_ms=float(cfg["catchup_poll_ms"]),
            )
        else:
            _run_drain_first_catchup(
                rank=rank,
                protocol=protocol,
                outage=outage,
                schedule=schedule,
                device=device,
                rank_metrics=rank_metrics,
                event_recorder=event_recorder,
            )

        completed_outage_stage_windows = schedule.optimizer_steps - len(prelude)

        if rank == 0:
            event_recorder.record(
                RecoveryEventName.LIVE_P2P_RESUMED,
                window_id=protocol.outage_end_window_exclusive,
            )
        dist.barrier()
        _run_windows(
            phase="resumed",
            windows=resumed,
            schedule=schedule,
            device=device,
            rank_metrics=rank_metrics,
            event_recorder=event_recorder,
        )
        stage.drain_pending_sends()
        _sync(device)

        checkpoint_records = (
            checkpoint_store.list_stage(rank) if checkpoint_store is not None else []
        )
        observer_results = observer.results if observer is not None else []
        boundary_file_bytes = sum(
            result.durable_payload_bytes for result in observer_results
        )
        rank_payload = {
            "rank": rank,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
            ),
            "optimizer_steps": schedule.optimizer_steps,
            "completed_outage_stage_windows": completed_outage_stage_windows,
            "journaled_windows": [result.commit.window_id for result in observer_results],
            "checkpoint_ids": [result.commit.checkpoint_id for result in observer_results],
            "durable_checkpoint_file_bytes": sum(
                item.payload_file_bytes for item in checkpoint_records
            ),
            "durable_boundary_file_bytes": boundary_file_bytes,
            "durable_hidden_tensor_bytes": sum(
                result.hidden_tensor_bytes for result in observer_results
            ),
            "volatile_state_at_rejoin": volatile_state_at_rejoin,
            "transport_summary": (
                stage.transport_summary()
                if hasattr(stage, "transport_summary")
                else {}
            ),
            "peak_cuda_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            ),
            "trainable_mode": trainable_setup.mode,
            "metrics": rank_metrics,
        }
        atomic_write_json(output_dir / f"rank{rank}.summary.json", rank_payload)
        dist.barrier()

        if rank == 0:
            timeline = RecoveryTimeline.load(state_root, protocol.run_id)
            rank_summaries = [
                json.loads((output_dir / f"rank{stage_id}.summary.json").read_text(encoding="utf-8"))
                for stage_id in range(world_size)
            ]
            checkpoint_file_bytes = sum(
                int(item["durable_checkpoint_file_bytes"])
                for item in rank_summaries
            )
            boundary_file_bytes = sum(
                int(item["durable_boundary_file_bytes"])
                for item in rank_summaries
            )
            summary = {
                "runner": (
                    "e5_formal_v2_bpfree_volatile_outage"
                    if not durable_recovery
                    else (
                        "e5_formal_v2_bpfree_window_streamed_outage"
                        if protocol.catchup_policy == "window_streamed"
                        else "e5_formal_v2_bpfree_outage"
                    )
                ),
                "transport": cfg["transport_name"],
                "transport_details": cfg["transport_details"],
                "recovery_state_mode": recovery_state_mode,
                "protocol": protocol.to_dict(),
                "model_name": cfg["model_name"],
                "resolved_model": cfg["resolved_model"],
                "dtype": cfg["dtype"],
                "learning_rate": cfg["learning_rate"],
                "seed": cfg["seed"],
                "training_config": {
                    "optimizer": cfg["optimizer"],
                    "grad_clip": cfg["grad_clip"],
                    "trainable_mode": cfg["trainable_mode"],
                    "lora_targets": cfg["lora_targets"],
                    "lora_rank": cfg["lora_rank"],
                    "lora_alpha": cfg["lora_alpha"],
                    "lora_init_std": cfg["lora_init_std"],
                    "lora_init_seed": cfg["lora_init_seed"],
                    "label_smoothing": cfg["label_smoothing"],
                },
                "environment": {
                    "torch_version": torch.__version__,
                    "cuda_version": torch.version.cuda,
                    "gpu_names": [item["device_name"] for item in rank_summaries],
                },
                "recovery_timing": {
                    **timeline.recovery_summary(),
                    **timeline.common_recovery_summary(),
                },
                "stage_commit_counts": {
                    str(stage_id): (
                        len(ledger.list_stage(stage_id)) if ledger is not None else 0
                    )
                    for stage_id in range(world_size)
                },
                "completed_stage_windows": {
                    str(item["rank"]): int(item["completed_outage_stage_windows"])
                    for item in rank_summaries
                },
                "progress_at_rejoin": {
                    "local_optimizer_windows": {
                        str(key): value
                        for key, value in stage_commit_counts_at_rejoin.items()
                    }
                },
                "stage_commit_counts_at_rejoin": {
                    str(key): value
                    for key, value in stage_commit_counts_at_rejoin.items()
                },
                "pending_outbox_windows_at_rejoin": pending_outbox_windows_at_rejoin,
                "pending_outbox_windows": {
                    "stage0_to_stage1": (
                        outbox.pending_window_ids(0, 1) if outbox is not None else []
                    ),
                    "stage1_to_stage2": (
                        outbox.pending_window_ids(1, 2) if outbox is not None else []
                    ),
                },
                "volatile_state_at_rejoin": {
                    "hidden_tensor_bytes": sum(
                        int(item["volatile_state_at_rejoin"]["hidden_tensor_bytes"])
                        for item in rank_summaries
                    ),
                    "microbatches": sum(
                        int(item["volatile_state_at_rejoin"]["microbatches"])
                        for item in rank_summaries
                    ),
                    "capture_ms": sum(
                        float(item["volatile_state_at_rejoin"]["capture_ms"])
                        for item in rank_summaries
                    ),
                },
                "durable_state": {
                    "checkpoint_file_bytes": checkpoint_file_bytes,
                    "boundary_file_bytes": boundary_file_bytes,
                    "total_file_bytes": checkpoint_file_bytes + boundary_file_bytes,
                },
                "rank_summaries": rank_summaries,
            }
            atomic_write_json(output_dir / "summary.json", summary)
            print(json.dumps(summary, indent=2), flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def build_parser(
    *,
    default_catchup_policy: str = "drain_first",
    default_recovery_state_mode: str = "durable",
    default_backend: str = "nccl",
    default_recv_inflight_depth: int = 1,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="E5 formal-v2 BP-free stage-1 outage runner")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="tinyllama")
    parser.add_argument("--stage-devices", default="cuda:0,cuda:1,cuda:2")
    parser.add_argument("--backend", default=default_backend)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29751)
    parser.add_argument("--prelude-windows", type=int, default=4)
    parser.add_argument("--outage-windows", type=int, default=4)
    parser.add_argument("--resumed-windows", type=int, default=2)
    parser.add_argument("--physical-batch-size", type=int, default=1)
    parser.add_argument("--microbatches-per-window", type=int, default=8)
    parser.add_argument("--max-pending-windows", type=int, default=4)
    parser.add_argument(
        "--catchup-policy",
        choices=("drain_first", "window_streamed"),
        default=default_catchup_policy,
    )
    parser.add_argument(
        "--recovery-state-mode",
        choices=("durable", "volatile"),
        default=default_recovery_state_mode,
        help="volatile keeps outage hidden tensors in process-local RAM",
    )
    parser.add_argument("--catchup-wait-timeout-s", type=float, default=120.0)
    parser.add_argument("--catchup-poll-ms", type=float, default=1.0)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--optimizer", default="adamw")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--trainable-mode", default="lora")
    parser.add_argument("--lora-targets", default="q_proj,v_proj")
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-init-std", type=float, default=0.01)
    parser.add_argument("--lora-init-seed", type=int, default=20260531)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--request-prefix", default="e5")
    parser.add_argument(
        "--recv-inflight-depth",
        type=int,
        default=default_recv_inflight_depth,
    )
    parser.add_argument("--enable-action-trace", action="store_true")
    parser.add_argument("--sync-action-trace", action="store_true")
    return parser


def main(
    *,
    default_catchup_policy: str = "drain_first",
    default_recovery_state_mode: str = "durable",
    default_backend: str = "nccl",
    transport_name: str = "nccl-gpu-p2p",
    transport_details: str = "GPU-resident point-to-point transport",
    volatile_stage_cls: type[VolatileCaptureBPFreePipelineStage] = VolatileCaptureBPFreePipelineStage,
    default_recv_inflight_depth: int = 1,
) -> None:
    args = build_parser(
        default_catchup_policy=default_catchup_policy,
        default_recovery_state_mode=default_recovery_state_mode,
        default_backend=default_backend,
        default_recv_inflight_depth=default_recv_inflight_depth,
    ).parse_args()
    if args.catchup_wait_timeout_s <= 0 or args.catchup_poll_ms <= 0:
        raise ValueError("catch-up wait timeout and poll interval must be positive")
    if args.recovery_state_mode == "volatile" and args.catchup_policy != "window_streamed":
        raise ValueError("volatile recovery requires window_streamed catch-up")
    protocol = E5OutageProtocol(
        run_id=args.run_id,
        prelude_windows=args.prelude_windows,
        outage_windows=args.outage_windows,
        resumed_windows=args.resumed_windows,
        physical_batch_size=args.physical_batch_size,
        microbatches_per_window=args.microbatches_per_window,
        max_pending_windows=args.max_pending_windows,
        catchup_policy=args.catchup_policy,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    atomic_write_json(args.output_dir / "protocol.json", protocol.to_dict())

    resolved_model = resolve_model_name(args.model_name)
    model_config = AutoConfig.from_pretrained(resolved_model)
    cfg = {
        "protocol_constructor": {
            key: value
            for key, value in protocol.to_dict().items()
            if key in E5OutageProtocol.__dataclass_fields__
        },
        "output_dir": str(args.output_dir),
        "train_manifest": str(args.train_manifest),
        "model_name": args.model_name,
        "resolved_model": resolved_model,
        "vocab_size": int(model_config.vocab_size),
        "stage_devices": _parse_devices(args.stage_devices, protocol.num_stages),
        "backend": args.backend,
        "transport_name": transport_name,
        "transport_details": transport_details,
        "master_addr": args.master_addr,
        "master_port": args.master_port,
        "dtype": args.dtype,
        "learning_rate": args.learning_rate,
        "optimizer": args.optimizer,
        "grad_clip": args.grad_clip,
        "trainable_mode": args.trainable_mode,
        "lora_targets": args.lora_targets,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_init_std": args.lora_init_std,
        "lora_init_seed": args.lora_init_seed,
        "alpha": args.alpha,
        "label_smoothing": args.label_smoothing,
        "seed": args.seed,
        "request_prefix": args.request_prefix,
        "recv_inflight_depth": args.recv_inflight_depth,
        "enable_action_trace": args.enable_action_trace,
        "sync_action_trace": args.sync_action_trace,
        "catchup_wait_timeout_s": args.catchup_wait_timeout_s,
        "catchup_poll_ms": args.catchup_poll_ms,
        "recovery_state_mode": args.recovery_state_mode,
    }
    mp.spawn(
        _worker,
        args=(protocol.num_stages, cfg, volatile_stage_cls),
        nprocs=protocol.num_stages,
        join=True,
    )


if __name__ == "__main__":
    main()
