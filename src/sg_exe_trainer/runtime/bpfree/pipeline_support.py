"""Shared tensor loading, memory accounting, and result records for BPFree."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Optional

import torch

from sg_exe_trainer.metrics.activation_memory import SavedTensorTracker
from sg_exe_trainer.tasks.label_experiment import (
    label_choice_metrics,
    load_tensor,
    one_token_choice_ids,
)
from sg_exe_trainer.common.trainable_modes import module_param_stats


def parse_devices(raw: str, expected: int) -> list[str]:
    devices = [item.strip() for item in raw.split(",") if item.strip()]
    if len(devices) != expected:
        raise ValueError(f"--stage_devices must contain {expected} devices, got {devices}.")
    return devices


def tensor_nbytes(tensor: Optional[torch.Tensor]) -> int:
    if tensor is None:
        return 0
    return int(tensor.numel() * tensor.element_size())


def request_id_for(prefix: str, phase: str, seq: int) -> str:
    return f"{prefix}-{phase}-{seq:06d}"


def _module_stats_or_zero(module: Optional[torch.nn.Module]) -> tuple[int, int]:
    if module is None:
        return 0, 0
    stats = module_param_stats(module)
    return stats.bytes, stats.trainable_bytes


def stage_memory_ledger(
    chunk: torch.nn.Module,
    input_embedding: Optional[torch.nn.Module] = None,
) -> dict[str, int]:
    layer_stats = module_param_stats(chunk.layers)
    final_norm_bytes, final_norm_trainable_bytes = _module_stats_or_zero(chunk.final_norm)
    lm_head_bytes, lm_head_trainable_bytes = _module_stats_or_zero(chunk.lm_head)
    local_readout_adapter_bytes, local_readout_adapter_trainable_bytes = _module_stats_or_zero(
        getattr(chunk, "local_readout_adapter", None)
    )
    input_embedding_bytes, input_embedding_trainable_bytes = _module_stats_or_zero(input_embedding)

    local_readout_param_bytes = final_norm_bytes + lm_head_bytes
    local_readout_trainable_param_bytes = final_norm_trainable_bytes + lm_head_trainable_bytes
    resident_model_param_bytes = (
        layer_stats.bytes
        + local_readout_param_bytes
        + local_readout_adapter_bytes
        + input_embedding_bytes
    )
    resident_trainable_param_bytes = (
        layer_stats.trainable_bytes
        + local_readout_trainable_param_bytes
        + local_readout_adapter_trainable_bytes
        + input_embedding_trainable_bytes
    )
    return {
        "resident_model_param_bytes": resident_model_param_bytes,
        "resident_frozen_param_bytes": resident_model_param_bytes - resident_trainable_param_bytes,
        "base_shard_param_bytes": layer_stats.bytes,
        "base_shard_trainable_param_bytes": layer_stats.trainable_bytes,
        "local_readout_param_bytes": local_readout_param_bytes,
        "local_readout_trainable_param_bytes": local_readout_trainable_param_bytes,
        "local_readout_adapter_param_bytes": local_readout_adapter_bytes,
        "local_readout_adapter_trainable_param_bytes": local_readout_adapter_trainable_bytes,
        "input_embedding_param_bytes": input_embedding_bytes,
        "input_embedding_trainable_param_bytes": input_embedding_trainable_bytes,
    }


def batched(records: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    if batch_size <= 0:
        raise ValueError("physical batch size must be positive.")
    if len(records) % batch_size != 0:
        raise ValueError(
            f"Record count ({len(records)}) must divide --physical_batch_size ({batch_size})."
        )
    return [records[index : index + batch_size] for index in range(0, len(records), batch_size)]


def batched_shape(record: dict[str, Any], tensor_name: str, batch_size: int) -> list[int]:
    shape = list(record["tensors"][tensor_name]["shape"])
    if not shape or shape[0] != 1:
        raise ValueError(
            f"Expected manifest tensor {tensor_name!r} to have a singleton batch axis, got {shape}."
        )
    return [batch_size, *shape[1:]]


def load_common_tensors(
    *,
    records: list[dict[str, Any]],
    manifest_dir: Path,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    attention_mask = torch.cat(
        [load_tensor(manifest_dir, record["tensors"]["attention_mask"]) for record in records], dim=0
    ).to(device)
    position_ids = torch.cat(
        [load_tensor(manifest_dir, record["tensors"]["position_ids"]) for record in records], dim=0
    ).to(device)
    labels = torch.cat(
        [load_tensor(manifest_dir, record["tensors"]["labels"]) for record in records], dim=0
    ).to(device)
    return attention_mask, position_ids, labels


def load_stage0_hidden(
    *,
    records: list[dict[str, Any]],
    manifest_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    input_embedding: Optional[torch.nn.Module] = None,
) -> torch.Tensor:
    tensors = records[0].get("tensors") or {}

    if "hidden_states" in tensors:
        hidden = torch.cat(
            [load_tensor(manifest_dir, record["tensors"]["hidden_states"]) for record in records],
            dim=0,
        ).to(device)
        return hidden.to(dtype=dtype)

    if "input_ids" in tensors:
        if input_embedding is None:
            raise ValueError("input_embedding is required for input_ids-only manifests.")
        input_ids = torch.cat(
            [load_tensor(manifest_dir, record["tensors"]["input_ids"]) for record in records],
            dim=0,
        ).to(device=device, dtype=torch.long)
        with torch.no_grad():
            hidden = input_embedding(input_ids)
        return hidden.to(dtype=dtype)

    raise KeyError("Manifest record must contain either tensors.hidden_states or tensors.input_ids.")


def hidden_shape(record: dict[str, Any], batch_size: int, hidden_size: int) -> list[int]:
    tensors = record.get("tensors") or {}

    if "hidden_states" in tensors:
        return batched_shape(record, "hidden_states", batch_size)

    if "input_ids" in tensors:
        input_shape = batched_shape(record, "input_ids", batch_size)
        if len(input_shape) != 2:
            raise ValueError(f"Expected input_ids shape [batch, seq], got {input_shape}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive to infer hidden shape, got {hidden_size}")
        return [input_shape[0], input_shape[1], hidden_size]

    raise KeyError("Manifest record must contain either tensors.hidden_states or tensors.input_ids.")


def log_probs_shape(record: dict[str, Any], batch_size: int, vocab_size: int) -> list[int]:
    label_shape = batched_shape(record, "labels", batch_size)
    if len(label_shape) != 2:
        raise ValueError(f"Expected labels shape [batch, seq], got {label_shape}")
    return [label_shape[0], label_shape[1], vocab_size]


def summarize_bpfree_phase_metrics(
    metric_rows: list[dict[str, Any]],
    *,
    completed_records: int,
    wall_ms: float,
) -> dict[str, Any]:
    complete_windows: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for row in metric_rows:
        current.append(row)
        if bool(row.get("optimizer_step_applied")):
            complete_windows.append(current)
            current = []

    def window_records(window: list[dict[str, Any]]) -> int:
        return sum(int(row.get("records", 0)) for row in window)

    def window_duration_ms(window: list[dict[str, Any]]) -> float:
        started = float(window[0].get("stage_start_epoch_ms", 0.0))
        ended = float(window[-1].get("stage_end_epoch_ms", 0.0))
        return max(0.0, ended - started)

    payload: dict[str, Any] = {
        "full_run_throughput_per_s": completed_records / (wall_ms / 1000.0) if wall_ms > 0 else 0.0,
        "steady_state_throughput_per_s": "",
        "warmup_or_fill_ms": "",
        "drain_ms": "",
        "fill_drain_overhead_ms": "",
        "status": "insufficient_complete_update_windows",
        "source": "summary.pipeline_phase_metrics",
        "phase_semantics": "bpfree_update_window_start_steady_body_update_tail",
        "phase_alignment_note": (
            "Shared fill/drain report fields are only semantically aligned. "
            "For BP-free, warmup_or_fill_ms and drain_ms refer to the first and last "
            "logical update windows; the tail side is update-tail cost, not exact-BP pipeline drain."
        ),
        "trim_policy": "drop_first_last_complete_logical_update_windows",
        "logical_update_windows": len(complete_windows),
        "steady_state_windows": 0,
    }
    if len(complete_windows) < 3:
        return payload

    first = complete_windows[0]
    last = complete_windows[-1]
    interior = complete_windows[1:-1]
    interior_records = sum(window_records(window) for window in interior)
    interior_ms = sum(window_duration_ms(window) for window in interior)
    warmup_or_fill_ms = window_duration_ms(first)
    drain_ms = window_duration_ms(last)
    payload.update(
        {
            "steady_state_throughput_per_s": (
                interior_records / (interior_ms / 1000.0) if interior_ms > 0 else ""
            ),
            "warmup_or_fill_ms": warmup_or_fill_ms,
            "drain_ms": drain_ms,
            "fill_drain_overhead_ms": warmup_or_fill_ms + drain_ms,
            "status": "explicit_update_window_trim",
            "steady_state_windows": len(interior),
            "steady_state_records": interior_records,
        }
    )
    return payload


def result_fieldnames() -> list[str]:
    return [
        "seq",
        "request_id",
        "dataset_index",
        "response",
        "mode",
        "loss",
        "choice_correct",
        "choice_count",
        "choice_accuracy",
        "choice_loss",
        "elapsed_ms",
    ]


def metric_fieldnames() -> list[str]:
    return [
        "phase",
        "seq",
        "request_id",
        "batch_seq",
        "physical_batch_size",
        "records",
        "stage_id",
        "device",
        "mode",
        "train",
        "local_loss",
        "optimizer_step",
        "gradient_accumulation_steps",
        "optimizer_step_applied",
        "recv_hidden_ms",
        "recv_log_probs_ms",
        "load_hidden_ms",
        "load_input_ms",
        "execute_ms",
        "forward_ms",
        "backward_ms",
        "optimizer_ms",
        "send_hidden_ms",
        "send_log_probs_ms",
        "nccl_blocking_ms",
        "unattributed_ms",
        "stage_total_ms",
        "stage_start_epoch_ms",
        "stage_end_epoch_ms",
        "output_hidden_bytes",
        "output_log_probs_bytes",
        "local_params",
        "local_trainable_params",
        "local_param_bytes",
        "local_trainable_param_bytes",
        "resident_model_param_bytes",
        "resident_frozen_param_bytes",
        "base_shard_param_bytes",
        "base_shard_trainable_param_bytes",
        "local_readout_param_bytes",
        "local_readout_trainable_param_bytes",
        "local_readout_adapter_param_bytes",
        "local_readout_adapter_trainable_param_bytes",
        "input_embedding_param_bytes",
        "input_embedding_trainable_param_bytes",
        "gradient_storage_bytes",
        "optimizer_state_bytes",
        *list(SavedTensorTracker().snapshot()),
        "cuda_peak_memory_allocated",
        "cuda_peak_memory_reserved",
        "identified_allocated_bytes",
        "runtime_residual_bytes",
    ]


def write_result_row(writer: csv.DictWriter, result: dict[str, Any]) -> None:
    writer.writerow({name: result.get(name, "") for name in result_fieldnames()})


def write_metric_row(writer: csv.DictWriter, metric: dict[str, Any]) -> None:
    writer.writerow({name: metric.get(name, "") for name in metric_fieldnames()})


def make_terminal_result(
    *,
    phase: str,
    seq: int,
    request_id: str,
    record: dict[str, Any],
    mode: str,
    loss_value: float,
    final_log_probs: Optional[torch.Tensor],
    labels: torch.Tensor,
    elapsed_ms: float,
    choice_metrics: Optional[tuple[int, int, float]] = None,
) -> dict[str, Any]:
    if choice_metrics is not None:
        choice_correct, choice_count, choice_loss = choice_metrics
    elif final_log_probs is None or not (record.get("label_choices") or []):
        choice_correct, choice_count, choice_loss = 0, 0, 0.0
    else:
        choice_ids = one_token_choice_ids(record)
        choice_correct, choice_count, choice_loss = label_choice_metrics(
            final_log_probs.detach().float().cpu(),
            labels.detach().cpu(),
            choice_ids,
        )
    response = (record.get("text") or {}).get("response", "").strip()
    return {
        "phase": phase,
        "seq": seq,
        "request_id": request_id,
        "dataset_index": int(record.get("dataset_index", -1)),
        "response": response,
        "mode": mode,
        "loss": loss_value,
        "choice_correct": choice_correct,
        "choice_count": choice_count,
        "choice_accuracy": (choice_correct / choice_count) if choice_count else 0.0,
        "choice_loss": choice_loss,
        "elapsed_ms": elapsed_ms,
    }


def summarize_phase(phase: str, rows: list[dict[str, Any]], output_csv: Path) -> dict[str, Any]:
    correct = sum(int(row["choice_correct"]) for row in rows)
    count = sum(int(row["choice_count"]) for row in rows)
    losses = [float(row["loss"]) for row in rows]
    return {
        "phase": phase,
        "rows": len(rows),
        "choice_correct": correct,
        "choice_count": count,
        "choice_accuracy": (correct / count) if count else 0.0,
        "avg_loss": sum(losses) / len(losses) if losses else 0.0,
        "avg_elapsed_ms": (
            sum(float(row["elapsed_ms"]) for row in rows) / len(rows) if rows else 0.0
        ),
        "csv": str(output_csv),
    }
