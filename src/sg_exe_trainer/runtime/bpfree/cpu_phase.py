from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torch.distributed as dist

from sg_exe_trainer.common.trainable_modes import optimizer_state_nbytes
from sg_exe_trainer.metrics.activation_memory import SavedTensorTracker
from sg_exe_trainer.runtime.bpfree import pipeline_support
from sg_exe_trainer.runtime.bpfree.schedule import split_records_into_update_windows
from sg_exe_trainer.runtime.bpfree.schedule_runtime import ScheduleBPFreeBodySendHeadV1
from sg_exe_trainer.runtime.bpfree.cpu_stage import BPFreePipelineStageV0
from sg_exe_trainer.runtime.transport.cpu import sync_max_wall_ms
from sg_exe_trainer.runtime.bpfree.trace import ActionTracer, NoOpActionTracer


def _sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _nbytes(tensor: Optional[torch.Tensor]) -> int:
    if tensor is None:
        return 0
    return int(tensor.numel() * tensor.element_size())


def run_phase_schedule_cpu(
    *,
    rank: int,
    world_size: int,
    phase: str,
    records: list[dict[str, Any]],
    manifest_dir: Path,
    mode: str,
    request_prefix: str,
    chunk: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    train_chunks: set[int],
    dtype: torch.dtype,
    device: torch.device,
    belief_transport_mode: str,
    grad_clip: float,
    learning_rate_override: Optional[float],
    vocab_size: int,
    hidden_size: int,
    input_embedding: Optional[torch.nn.Module] = None,
    output_dir: Path,
    progress_interval: int,
    physical_batch_size: int,
    track_activation_memory: bool,
    gradient_accumulation_steps: int,
    enable_action_trace: bool,
    sync_action_trace: bool,
    action_trace_start_window: int,
    action_trace_end_window: int | None,
    perf_minimal_metrics: bool,
    local_param_stats: Any,
    memory_ledger: dict[str, int],
    recv_prepost_depth: int = 0,
    max_pending_send_bytes: int = 67_108_864,
    max_posted_recv_bytes: int = 67_108_864,
) -> Optional[dict[str, Any]]:
    train_this_rank = mode == "train" and rank in train_chunks
    grad_accum_steps = max(1, int(gradient_accumulation_steps))

    if mode == "train":
        effective_batch_size = physical_batch_size * grad_accum_steps
        n_microbatches = grad_accum_steps
    else:
        effective_batch_size = physical_batch_size
        n_microbatches = 1

    update_windows = split_records_into_update_windows(
        records=records,
        effective_batch_size=effective_batch_size,
        n_microbatches=n_microbatches,
        drop_last=True,
    )

    scheduled_records = sum(window.num_records for window in update_windows)
    records = records[:scheduled_records]

    result_rows: list[dict[str, Any]] = []
    terminal_metric_rows: list[dict[str, Any]] = []
    activation_tracker = SavedTensorTracker()

    metrics_path = output_dir / f"{phase}.stage{rank}.metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    action_tracer = (
        ActionTracer(
            path=output_dir / f"{phase}.stage{rank}.actions.csv",
            phase=phase,
            rank=rank,
            stage_id=rank,
            min_window_id=action_trace_start_window,
            max_window_id=action_trace_end_window,
        )
        if enable_action_trace
        else NoOpActionTracer()
    )

    stage = BPFreePipelineStageV0(
        rank=rank,
        world_size=world_size,
        phase=phase,
        mode=mode,
        request_prefix=request_prefix,
        chunk=chunk,
        optimizer=optimizer,
        train_this_rank=train_this_rank,
        dtype=dtype,
        device=device,
        belief_transport_mode=belief_transport_mode,
        grad_clip=grad_clip,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        input_embedding=input_embedding,
        manifest_dir=manifest_dir,
        tracer=action_tracer,
        recv_prepost_depth=recv_prepost_depth,
        max_pending_send_bytes=max_pending_send_bytes,
        max_posted_recv_bytes=max_posted_recv_bytes,
        sync_action_trace=sync_action_trace,
        perf_minimal_metrics=perf_minimal_metrics,
    )

    schedule = ScheduleBPFreeBodySendHeadV1(
        stage=stage,
        train_this_rank=train_this_rank,
        track_activation_memory=track_activation_memory,
        activation_tracker=activation_tracker,
        vocab_size=vocab_size,
        learning_rate_override=learning_rate_override,
        optimizer_steps_start=0,
        perf_minimal_metrics=perf_minimal_metrics,
    )

    result_path = output_dir / f"{phase}.csv"
    result_handle = None
    result_writer = None

    if rank == world_size - 1 and not perf_minimal_metrics:
        result_handle = result_path.open("w", newline="", encoding="utf-8")
        result_writer = csv.DictWriter(result_handle, fieldnames=pipeline_support.result_fieldnames())
        result_writer.writeheader()

    _sync_cuda(device)
    dist.barrier()
    _sync_cuda(device)
    phase_started = time.perf_counter()

    if hasattr(stage, "set_forward_recv_plan"):
        stage.set_forward_recv_plan([mb for window in update_windows for mb in window.microbatches])

    try:
        with metrics_path.open("w", newline="", encoding="utf-8") as metrics_handle:
            metric_writer = csv.DictWriter(metrics_handle, fieldnames=pipeline_support.metric_fieldnames())
            metric_writer.writeheader()



            for window in update_windows:
                first_mb = window.microbatches[0]
                with stage._span(first_mb, "PHASE_STEP_WINDOW"):
                    runs = schedule.step_window(window)

                for run in runs:
                    mb = run.mb
                    batch_seq = mb.global_batch_seq
                    batch_records = mb.records
                    batch_start_seq = mb.seq_start
                    request_id = pipeline_support.request_id_for(request_prefix, phase, batch_start_seq)

                    if perf_minimal_metrics:
                        if rank == world_size - 1 and progress_interval > 0:
                            completed = batch_start_seq + len(batch_records)
                            if completed % progress_interval == 0 or completed == len(records):
                                print(
                                    f"[rank {rank}] {phase}: {completed}/{len(records)}",
                                    flush=True,
                                )
                        continue

                    stage_started = time.perf_counter()
                    stage_started_epoch_ms = time.time() * 1000.0

                    # This timing is no longer exact stage wall time because schedule already ran
                    # the actions. For v2, we keep metric fields populated from action timings.
                    execute_ms = (
                        run.fwd_output.forward_ms
                        + run.backward_output.backward_ms
                        + run.opt_output.optimizer_ms
                    )

                    opt_state_bytes = optimizer_state_nbytes(optimizer)

                    if rank == world_size - 1:
                        assert result_writer is not None

                        # Use the measured action sum as a stable debug elapsed proxy.
                        elapsed_per_request_ms = execute_ms / max(1, len(batch_records))

                        labels_cpu = run.common.labels.detach().cpu()
                        log_probs_cpu = (
                            run.fwd_output.next_log_probs.detach().float().cpu()
                            if run.fwd_output.next_log_probs is not None
                            else None
                        )

                        for offset, record in enumerate(batch_records):
                            seq = batch_start_seq + offset
                            row_log_probs = (
                                log_probs_cpu[offset : offset + 1]
                                if log_probs_cpu is not None
                                else None
                            )
                            row_labels = labels_cpu[offset : offset + 1]
                            result = pipeline_support.make_terminal_result(
                                phase=phase,
                                seq=seq,
                                request_id=pipeline_support.request_id_for(request_prefix, phase, seq),
                                record=record,
                                mode=mode,
                                loss_value=run.loss_value,
                                final_log_probs=row_log_probs,
                                labels=row_labels,
                                elapsed_ms=elapsed_per_request_ms,
                            )
                            pipeline_support.write_result_row(result_writer, result)
                            result_rows.append(result)

                    if (
                        sync_action_trace
                        and action_tracer.is_enabled_for_window(mb.window_id)
                    ):
                        _sync_cuda(device)
                    stage_ended_epoch_ms = time.time() * 1000.0

                    nccl_blocking_ms = (
                        run.fwd_input.recv_hidden_ms
                        + run.fwd_input.recv_log_probs_ms
                        + run.send_output.send_hidden_ms
                        + run.send_output.send_log_probs_ms
                    )

                    known_ms = (
                        run.fwd_input.recv_hidden_ms
                        + run.fwd_input.recv_log_probs_ms
                        + run.fwd_input.load_hidden_ms
                        + run.common.load_input_ms
                        + run.fwd_output.forward_ms
                        + run.backward_output.backward_ms
                        + run.opt_output.optimizer_ms
                        + run.send_output.send_hidden_ms
                        + run.send_output.send_log_probs_ms
                    )

                    cuda_peak_alloc = (
                        torch.cuda.max_memory_allocated(device)
                        if device.type == "cuda"
                        else 0
                    )
                    cuda_peak_reserved = (
                        torch.cuda.max_memory_reserved(device)
                        if device.type == "cuda"
                        else 0
                    )

                    metric = {
                        "phase": phase,
                        "seq": batch_start_seq,
                        "request_id": request_id,
                        "batch_seq": batch_seq,
                        "physical_batch_size": physical_batch_size,
                        "records": len(batch_records),
                        "stage_id": rank,
                        "device": str(device),
                        "mode": mode,
                        "train": bool(train_this_rank),
                        "local_loss": run.loss_value,
                        "optimizer_step": run.optimizer_step_index_after,
                        "gradient_accumulation_steps": grad_accum_steps,
                        "optimizer_step_applied": run.opt_output.applied,
                        "recv_hidden_ms": run.fwd_input.recv_hidden_ms,
                        "recv_log_probs_ms": run.fwd_input.recv_log_probs_ms,
                        "load_hidden_ms": run.fwd_input.load_hidden_ms,
                        "load_input_ms": run.common.load_input_ms,
                        "execute_ms": execute_ms,
                        "forward_ms": run.fwd_output.forward_ms,
                        "backward_ms": run.backward_output.backward_ms,
                        "optimizer_ms": run.opt_output.optimizer_ms,
                        "send_hidden_ms": run.send_output.send_hidden_ms,
                        "send_log_probs_ms": run.send_output.send_log_probs_ms,
                        "nccl_blocking_ms": nccl_blocking_ms,
                        "unattributed_ms": 0.0,
                        "stage_total_ms": known_ms,
                        "stage_start_epoch_ms": stage_started_epoch_ms,
                        "stage_end_epoch_ms": stage_ended_epoch_ms,
                        "output_hidden_bytes": _nbytes(run.fwd_output.next_hidden),
                        "output_log_probs_bytes": _nbytes(run.fwd_output.next_log_probs),
                        "local_params": getattr(local_param_stats, "params", ""),
                        "local_trainable_params": getattr(local_param_stats, "trainable_params", ""),
                        "local_param_bytes": getattr(local_param_stats, "bytes", ""),
                        "local_trainable_param_bytes": getattr(local_param_stats, "trainable_bytes", ""),
                        **memory_ledger,
                        "gradient_storage_bytes": run.backward_output.gradient_storage_bytes,
                        "optimizer_state_bytes": opt_state_bytes,
                        **run.activation_stats,
                        "cuda_peak_memory_allocated": cuda_peak_alloc,
                        "cuda_peak_memory_reserved": cuda_peak_reserved,
                        "identified_allocated_bytes": "",
                        "runtime_residual_bytes": "",
                    }

                    pipeline_support.write_metric_row(metric_writer, metric)

                    if rank == world_size - 1:
                        terminal_metric_rows.append(metric)

                    if rank == world_size - 1 and progress_interval > 0:
                        completed = batch_start_seq + len(batch_records)
                        if completed % progress_interval == 0 or completed == len(records):
                            if mode == "train":
                                print(
                                    f"[rank {rank}] {phase}: {completed}/{len(records)} "
                                    f"loss={run.loss_value:.4f} train={train_this_rank}",
                                    flush=True,
                                )
                            else:
                                correct = sum(int(row["choice_correct"]) for row in result_rows)
                                count = sum(int(row["choice_count"]) for row in result_rows)
                                acc = correct / count if count else 0.0
                                print(
                                    f"[rank {rank}] {phase}: {completed}/{len(records)} acc={acc:.4f}",
                                    flush=True,
                                )

            schedule.drain()

    finally:
        action_tracer.flush()
        if result_handle is not None:
            result_handle.close()

    _sync_cuda(device)
    local_wall_ms = (time.perf_counter() - phase_started) * 1000.0
    wall_ms = sync_max_wall_ms(local_wall_ms)

    local_transport_budget = stage.transport_summary()
    transport_budget_by_rank: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(transport_budget_by_rank, local_transport_budget)

    if rank == world_size - 1:
        completed_records = len(records)

        if perf_minimal_metrics and mode == "train":
            return {
                "phase": phase,
                "rows": 0,
                "csv": "",
                "mode": mode,
                "completed_records": completed_records,
                "optimizer_steps": schedule.optimizer_steps,
                "batches": sum(window.num_microbatches for window in update_windows),
                "wall_ms": wall_ms,
                "throughput_per_s": completed_records / (wall_ms / 1000.0) if wall_ms > 0 else 0.0,
                "perf_minimal_metrics": True,
                "transport_budget_by_rank": transport_budget_by_rank,
            }

        base_summary = pipeline_support.summarize_phase(phase, result_rows, result_path)

        if mode == "train":
            base_summary.update(
                {
                    "mode": mode,
                    "completed_records": completed_records,
                    "optimizer_steps": schedule.optimizer_steps,
                    "batches": sum(window.num_microbatches for window in update_windows),
                    "wall_ms": wall_ms,
                    "throughput_per_s": completed_records / (wall_ms / 1000.0) if wall_ms > 0 else 0.0,
                    "transport_budget_by_rank": transport_budget_by_rank,
                    "pipeline_phase_metrics": pipeline_support.summarize_bpfree_phase_metrics(
                        terminal_metric_rows,
                        completed_records=completed_records,
                        wall_ms=wall_ms,
                    ),
                }
            )
        else:
            correct = sum(int(row["choice_correct"]) for row in result_rows)
            count = sum(int(row["choice_count"]) for row in result_rows)
            base_summary.update(
                {
                    "mode": mode,
                    "completed_records": completed_records,
                    "optimizer_steps": 0,
                    "batches": sum(window.num_microbatches for window in update_windows),
                    "choice_correct": correct,
                    "choice_count": count,
                    "choice_accuracy": correct / count if count else 0.0,
                    "wall_ms": wall_ms,
                    "throughput_per_s": completed_records / (wall_ms / 1000.0) if wall_ms > 0 else 0.0,
                    "transport_budget_by_rank": transport_budget_by_rank,
                }
            )

        return base_summary

    return None
