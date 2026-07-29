#!/usr/bin/env python3
"""Run a real full-backward LoRA pipeline baseline.

Launch with torchrun, one rank per pipeline stage:

    torchrun --standalone --nproc_per_node=3 src/sg_exe_trainer/runtime/exactbp/distributed_runtime.py ...

This runner is intentionally separate from the BP-free scheduler lab. It keeps
the same stage partition, LoRA injection, request manifests, dtype, and label
metrics, while supporting either Schedule1F1B or ScheduleGPipe so the terminal
loss backpropagates through all stages.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.pipelining import PipelineStage, Schedule1F1B, ScheduleGPipe
from torch.distributed.pipelining.schedules import _ScheduleForwardOnly
from transformers import AutoModelForCausalLM

from sg_exe_trainer.metrics.activation_memory import SavedTensorTracker
from sg_exe_trainer.artifacts.lora_export import save_lora_state
from sg_exe_trainer.common.experiment_common import (
    get_model_parts,
    infer_module_compute_dtype,
    label_choice_details,
    load_tensor,
    read_manifest,
    resolve_dtype,
    resolve_model_name,
    stage0_tensor_name,
)
from sg_exe_trainer.common.lora_layers import lora_parameter_fingerprint
from sg_exe_trainer.common.trainable_modes import (
    configure_model_trainable,
    gradient_storage_nbytes,
    module_param_stats,
    optimizer_state_nbytes,
)


@dataclass
class RankConfig:
    stage_id: int
    rank: int
    world_size: int
    device: torch.device
    device_name: str
    dtype: torch.dtype
    label_smoothing: float
    hidden_size: int | None = None
    vocab_size: int | None = None
    trainable_mode: str = "lora"
    local_params: int = 0
    local_trainable_params: int = 0
    local_param_bytes: int = 0
    local_trainable_param_bytes: int = 0
    resident_model_param_bytes: int = 0
    resident_frozen_param_bytes: int = 0
    base_shard_param_bytes: int = 0
    base_shard_trainable_param_bytes: int = 0
    local_readout_param_bytes: int = 0
    local_readout_trainable_param_bytes: int = 0
    input_embedding_param_bytes: int = 0
    input_embedding_trainable_param_bytes: int = 0


@dataclass
class RecoveryRuntimeState:
    policy: str
    failure_stage: int | None
    failure_batch_seq: int | None
    failure_microbatch_index: int | None
    checkpoint_interval_batches: int
    worker_rejoin_delay_ms: float
    event_triggered: bool = False
    event_batch_seq: int | None = None
    failure_detected_ms: float | None = None
    recovery_unit_done_ms: float | None = None
    replay_start_ms: float | None = None
    replay_done_ms: float | None = None
    recovery_scope: str = ""
    recovery_records: int = 0
    initial_checkpoint_bytes: int = 0
    initial_checkpoint_ms: float = 0.0
    checkpoint_captures: int = 0
    checkpoint_capture_bytes_total: int = 0
    checkpoint_capture_ms_total: float = 0.0
    checkpoint_restores: int = 0
    checkpoint_restore_bytes_total: int = 0
    checkpoint_restore_ms_total: float = 0.0
    recovery_wait_events: int = 0
    recovery_wait_ms_total: float = 0.0
    replayed_batches: int = 0
    replayed_records: int = 0
    interrupted_batches: int = 0
    latest_checkpoint_batch_seq: int = -1
    latest_checkpoint_version: int = 0
    latest_checkpoint_bytes: int = 0
    latest_checkpoint: Optional[dict[str, Any]] = None
    replay_plan: list[int] = field(default_factory=list)
    pending_batch_overhead: dict[str, Any] = field(default_factory=dict)
    failure_event: str = "batch_boundary"
    global_window_committed: bool = False

    def to_rank_summary(self, *, stage_id: int, rank: int) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "failure_stage": self.failure_stage,
            "failure_batch_seq": self.failure_batch_seq,
            "failure_microbatch_index": self.failure_microbatch_index,
            "failure_detected_ms": self.failure_detected_ms,
            "recovery_unit_done_ms": self.recovery_unit_done_ms,
            "replay_start_ms": self.replay_start_ms,
            "replay_done_ms": self.replay_done_ms,
            "recovery_scope": self.recovery_scope,
            "recovery_records": self.recovery_records,
            "checkpoint_interval_batches": self.checkpoint_interval_batches,
            "worker_rejoin_delay_ms": self.worker_rejoin_delay_ms,
            "event_triggered": self.event_triggered,
            "event_batch_seq": self.event_batch_seq,
            "stage_id": stage_id,
            "rank": rank,
            "initial_checkpoint_bytes": self.initial_checkpoint_bytes,
            "initial_checkpoint_ms": self.initial_checkpoint_ms,
            "checkpoint_captures": self.checkpoint_captures,
            "checkpoint_capture_bytes_total": self.checkpoint_capture_bytes_total,
            "checkpoint_capture_ms_total": self.checkpoint_capture_ms_total,
            "checkpoint_restores": self.checkpoint_restores,
            "checkpoint_restore_bytes_total": self.checkpoint_restore_bytes_total,
            "checkpoint_restore_ms_total": self.checkpoint_restore_ms_total,
            "recovery_wait_events": self.recovery_wait_events,
            "recovery_wait_ms_total": self.recovery_wait_ms_total,
            "replayed_batches": self.replayed_batches,
            "replayed_records": self.replayed_records,
            "interrupted_batches": self.interrupted_batches,
            "latest_checkpoint_batch_seq": self.latest_checkpoint_batch_seq,
            "latest_checkpoint_version": self.latest_checkpoint_version,
            "latest_checkpoint_bytes": self.latest_checkpoint_bytes,
            "replay_plan": list(self.replay_plan),
            "failure_event": self.failure_event,
            "global_window_committed": self.global_window_committed,
        }


class StageEventRecorder:
    def __init__(
        self,
        *,
        stage_id: int,
        rank: int,
        device_name: str,
        enabled: bool,
        failure_target_batch_seq: int | None = None,
        failure_target_event: str | None = None,
        failure_target_microbatch_index: int | None = None,
    ) -> None:
        self.stage_id = stage_id
        self.rank = rank
        self.device_name = device_name
        self.enabled = enabled
        self.phase = ""
        self.batch_seq = -1
        self.forward_index = 0
        self.backward_index = 0
        self.skip_forward_events = 0
        self._backward_started: Optional[tuple[int, float, float, Any]] = None
        self.rows: list[dict[str, Any]] = []
        self._cuda_event_pairs: list[tuple[dict[str, Any], Any, Any]] = []
        self.failure_target_batch_seq = failure_target_batch_seq
        self.failure_target_event = failure_target_event
        self.failure_target_microbatch_index = failure_target_microbatch_index
        self._cooperative_failure_epoch_ms: float | None = None

    def start_batch(self, *, phase: str, batch_seq: int) -> None:
        self.phase = phase
        self.batch_seq = batch_seq
        self.forward_index = 0
        self.backward_index = 0
        self._backward_started = None
        self._cooperative_failure_epoch_ms = None

    def start_cuda_event(self) -> Any:
        if not self.enabled or not self.device_name.startswith("cuda") or not torch.cuda.is_available():
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def record(
        self,
        *,
        event: str,
        microbatch_index: int,
        start_epoch_ms: float,
        start_perf: float,
        start_cuda_event: Any,
    ) -> None:
        if not self.enabled:
            return
        end_epoch_ms = time.time() * 1000.0
        row = {
            "phase": self.phase,
            "batch_seq": self.batch_seq,
            "stage_id": self.stage_id,
            "rank": self.rank,
            "device": self.device_name,
            "event": event,
            "microbatch_index": microbatch_index,
            "start_epoch_ms": start_epoch_ms,
            "end_epoch_ms": end_epoch_ms,
            "duration_ms": (time.perf_counter() - start_perf) * 1000.0,
        }
        self.rows.append(row)
        if (
            self.failure_target_batch_seq is not None
            and self.failure_target_event is not None
            and self.failure_target_microbatch_index is not None
            and self.phase.startswith("train")
            and self.batch_seq == self.failure_target_batch_seq
            and event == self.failure_target_event
            and microbatch_index == self.failure_target_microbatch_index
            and self._cooperative_failure_epoch_ms is None
        ):
            self._cooperative_failure_epoch_ms = end_epoch_ms
        if start_cuda_event is not None:
            end_cuda_event = torch.cuda.Event(enable_timing=True)
            end_cuda_event.record()
            self._cuda_event_pairs.append((row, start_cuda_event, end_cuda_event))

    def finalize_cuda_durations(self) -> None:
        """Replace host launch durations with GPU stream durations after one step sync."""
        for row, start_event, end_event in self._cuda_event_pairs:
            row["duration_ms"] = float(start_event.elapsed_time(end_event))
        self._cuda_event_pairs.clear()

    def next_forward_index(self) -> int:
        index = self.forward_index
        self.forward_index += 1
        return index

    def skip_next_forward_event(self) -> None:
        if self.enabled:
            self.skip_forward_events += 1

    def consume_forward_skip(self) -> bool:
        if self.skip_forward_events <= 0:
            return False
        self.skip_forward_events -= 1
        return True

    def backward_pre_hook(self, _module: nn.Module, _grad_output: tuple[torch.Tensor, ...]) -> None:
        if not self.enabled:
            return
        index = self.backward_index
        self._backward_started = (
            index,
            time.time() * 1000.0,
            time.perf_counter(),
            self.start_cuda_event(),
        )

    def backward_hook(
        self,
        _module: nn.Module,
        _grad_input: tuple[torch.Tensor, ...],
        _grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        if not self.enabled or self._backward_started is None:
            return
        index, start_epoch_ms, start_perf, start_cuda_event = self._backward_started
        self.backward_index += 1
        self._backward_started = None
        self.record(
            event="backward",
            microbatch_index=index,
            start_epoch_ms=start_epoch_ms,
            start_perf=start_perf,
            start_cuda_event=start_cuda_event,
        )

    def consume_cooperative_failure_epoch_ms(self) -> float:
        value = float(self._cooperative_failure_epoch_ms or 0.0)
        self._cooperative_failure_epoch_ms = None
        return value


def summarize_batch_timing(
    events: list[dict[str, Any]],
    *,
    step_started_epoch_ms: float,
    step_ended_epoch_ms: float,
    optimizer_ms: float,
) -> dict[str, float | str]:
    """Split observed stage time without calling the residual pure communication."""
    if not events:
        return {
            "forward_ms": "",
            "backward_ms": "",
            "pipeline_fill_wait_ms": "",
            "pipeline_interior_wait_ms": "",
            "pipeline_tail_wait_ms": "",
            "pipeline_unattributed_ms": "",
        }

    forward_ms = sum(float(event["duration_ms"]) for event in events if event["event"] == "forward")
    backward_ms = sum(float(event["duration_ms"]) for event in events if event["event"] == "backward")
    first_compute_ms = min(float(event["start_epoch_ms"]) for event in events)
    last_compute_ms = max(float(event["end_epoch_ms"]) for event in events)
    compute_span_ms = max(0.0, last_compute_ms - first_compute_ms)
    fill_wait_ms = max(0.0, first_compute_ms - step_started_epoch_ms)
    interior_wait_ms = max(0.0, compute_span_ms - forward_ms - backward_ms)
    tail_wait_ms = max(0.0, step_ended_epoch_ms - last_compute_ms - optimizer_ms)
    step_ms = max(0.0, step_ended_epoch_ms - step_started_epoch_ms)
    unattributed_ms = max(0.0, step_ms - forward_ms - backward_ms - optimizer_ms)
    return {
        "forward_ms": forward_ms,
        "backward_ms": backward_ms,
        "pipeline_fill_wait_ms": fill_wait_ms,
        "pipeline_interior_wait_ms": interior_wait_ms,
        "pipeline_tail_wait_ms": tail_wait_ms,
        # Schedule1F1B does not expose per-send/recv spans. This includes NCCL
        # rendezvous, pipeline scheduling waits, synchronization, and clipping.
        "pipeline_unattributed_ms": unattributed_ms,
    }


class FullBackwardStageChunk(nn.Module):
    def __init__(
        self,
        *,
        stage_id: int,
        layer_start: int,
        layer_end: int,
        layers: list[nn.Module],
        final_norm: Optional[nn.Module],
        lm_head: Optional[nn.Module],
        rotary_emb: Optional[nn.Module],
        recorder: StageEventRecorder,
    ) -> None:
        super().__init__()
        self.stage_id = stage_id
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.layers = nn.ModuleList(layers)
        self.final_norm = final_norm
        self.lm_head = lm_head
        self.rotary_emb = rotary_emb
        self.recorder = recorder
        self.register_full_backward_pre_hook(self.recorder.backward_pre_hook)
        self.register_full_backward_hook(self.recorder.backward_hook)

    @property
    def is_last_stage(self) -> bool:
        return self.final_norm is not None and self.lm_head is not None

    def compute_dtype(self) -> torch.dtype:
        return infer_module_compute_dtype(self)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        skip_event = self.recorder.consume_forward_skip()
        microbatch_index = -1 if skip_event else self.recorder.next_forward_index()
        started_epoch_ms = time.time() * 1000.0
        started_perf = time.perf_counter()
        started_cuda_event = self.recorder.start_cuda_event()

        dtype = self.compute_dtype()
        hidden_states = hidden_states.to(dtype=dtype)
        attention_mask = attention_mask.to(device=hidden_states.device, dtype=dtype)
        position_ids = position_ids.to(device=hidden_states.device, dtype=torch.long)

        position_embeddings = None
        if self.rotary_emb is not None:
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        curr_hidden = hidden_states
        for layer in self.layers:
            layer_out = layer(
                curr_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )
            curr_hidden = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        if self.is_last_stage:
            assert self.final_norm is not None
            assert self.lm_head is not None
            output = self.lm_head(self.final_norm(curr_hidden))
        else:
            output = curr_hidden

        if not skip_event:
            self.recorder.record(
                event="forward",
                microbatch_index=microbatch_index,
                start_epoch_ms=started_epoch_ms,
                start_perf=started_perf,
                start_cuda_event=started_cuda_event,
            )
        return output


def parse_devices(raw: str, expected: int) -> list[str]:
    devices = [item.strip() for item in raw.split(",") if item.strip()]
    if len(devices) != expected:
        raise ValueError(f"--stage_devices must contain {expected} devices, got {devices}")
    return devices


def stage_layer_range(stage_id: int, total_layers: int, num_chunks: int) -> tuple[int, int]:
    chunk_size = total_layers // num_chunks
    start = stage_id * chunk_size
    end = (stage_id + 1) * chunk_size if stage_id < num_chunks - 1 else total_layers
    return start, end


def build_stage_module(
    *,
    model: nn.Module,
    stage_id: int,
    num_chunks: int,
    recorder: StageEventRecorder,
) -> FullBackwardStageChunk:
    layers, final_norm, lm_head, _vocab_size, rotary_emb = get_model_parts(model)
    start, end = stage_layer_range(stage_id, len(layers), num_chunks)
    print(f"[rank {stage_id}] 1F1B stage layers=[{start}, {end - 1}]", flush=True)
    is_last = stage_id == num_chunks - 1
    return FullBackwardStageChunk(
        stage_id=stage_id,
        layer_start=start,
        layer_end=end,
        layers=[layers[index] for index in range(start, end)],
        final_norm=final_norm if is_last else None,
        lm_head=lm_head if is_last else None,
        rotary_emb=rotary_emb,
        recorder=recorder,
    )


def _module_stats_or_zero(module: Optional[nn.Module]) -> tuple[int, int]:
    if module is None:
        return 0, 0
    stats = module_param_stats(module)
    return stats.bytes, stats.trainable_bytes


def stage_memory_ledger(
    *,
    module: FullBackwardStageChunk,
    input_embedding: Optional[nn.Module],
) -> dict[str, int]:
    layer_stats = module_param_stats(module.layers)
    final_norm_bytes, final_norm_trainable_bytes = _module_stats_or_zero(module.final_norm)
    lm_head_bytes, lm_head_trainable_bytes = _module_stats_or_zero(module.lm_head)
    input_embedding_bytes, input_embedding_trainable_bytes = _module_stats_or_zero(input_embedding)

    local_readout_param_bytes = final_norm_bytes + lm_head_bytes
    local_readout_trainable_param_bytes = final_norm_trainable_bytes + lm_head_trainable_bytes
    base_shard_param_bytes = layer_stats.bytes + input_embedding_bytes
    base_shard_trainable_param_bytes = layer_stats.trainable_bytes + input_embedding_trainable_bytes
    resident_model_param_bytes = base_shard_param_bytes + local_readout_param_bytes
    resident_trainable_param_bytes = base_shard_trainable_param_bytes + local_readout_trainable_param_bytes
    return {
        "resident_model_param_bytes": resident_model_param_bytes,
        "resident_frozen_param_bytes": resident_model_param_bytes - resident_trainable_param_bytes,
        "base_shard_param_bytes": base_shard_param_bytes,
        "base_shard_trainable_param_bytes": base_shard_trainable_param_bytes,
        "local_readout_param_bytes": local_readout_param_bytes,
        "local_readout_trainable_param_bytes": local_readout_trainable_param_bytes,
        "input_embedding_param_bytes": input_embedding_bytes,
        "input_embedding_trainable_param_bytes": input_embedding_trainable_bytes,
    }


def build_optimizer(
    *,
    params: list[nn.Parameter],
    optimizer_name: str,
    learning_rate: float,
    sgd_momentum: float,
    sgd_dampening: float,
    sgd_weight_decay: float,
    sgd_nesterov: bool,
) -> torch.optim.Optimizer:
    if optimizer_name == "adamw":
        return torch.optim.AdamW(params, lr=learning_rate)
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            params,
            lr=learning_rate,
            momentum=sgd_momentum,
            dampening=sgd_dampening,
            weight_decay=sgd_weight_decay,
            nesterov=sgd_nesterov,
        )
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def clone_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: clone_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_to_cpu(item) for item in value)
    return value


def payload_nbytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(payload_nbytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(payload_nbytes(item) for item in value)
    return 0


def capture_trainable_checkpoint(
    chunk: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
) -> tuple[dict[str, Any], int]:
    checkpoint = {
        "trainable_parameters": {
            name: parameter.detach().cpu().clone()
            for name, parameter in chunk.named_parameters()
            if parameter.requires_grad
        },
        "optimizer_state": clone_to_cpu(optimizer.state_dict()) if optimizer is not None else None,
    }
    return checkpoint, payload_nbytes(checkpoint)


def restore_trainable_checkpoint(
    *,
    chunk: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    checkpoint: dict[str, Any],
    device: torch.device,
) -> None:
    expected = {name: parameter for name, parameter in chunk.named_parameters() if parameter.requires_grad}
    incoming = checkpoint.get("trainable_parameters", {})
    if set(expected) != set(incoming):
        missing = sorted(set(expected) - set(incoming))
        unexpected = sorted(set(incoming) - set(expected))
        raise RuntimeError(
            "Checkpoint trainable-parameter mismatch: "
            f"missing={missing[:3]} unexpected={unexpected[:3]}"
        )
    with torch.no_grad():
        for name, parameter in expected.items():
            parameter.copy_(incoming[name].to(device=parameter.device, dtype=parameter.dtype))
    optimizer_state = checkpoint.get("optimizer_state")
    if optimizer is None or optimizer_state is None:
        return
    optimizer.load_state_dict(optimizer_state)
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=device)


def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor, *, label_smoothing: float) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].float()
    shift_labels = labels[..., 1:].long()
    valid_mask = (shift_labels != -100).float()
    valid_count = valid_mask.sum().clamp_min(1.0)
    safe_labels = torch.where(shift_labels != -100, shift_labels, torch.zeros_like(shift_labels))
    loss_unmasked = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        safe_labels.reshape(-1),
        reduction="none",
        label_smoothing=label_smoothing,
    ).reshape_as(shift_labels)
    return (loss_unmasked * valid_mask).sum() / valid_count


def batched(records: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    if len(records) % batch_size != 0:
        raise ValueError(
            f"Record count {len(records)} must be divisible by --batch_size {batch_size}. "
            "Use train/eval limits that divide evenly so PipelineStage shapes stay fixed."
        )
    return [records[index : index + batch_size] for index in range(0, len(records), batch_size)]


def load_batch_tensors(
    *,
    records: list[dict[str, Any]],
    manifest_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    load_hidden: bool,
    load_labels: bool,
    input_embedding: Optional[nn.Module] = None,
) -> dict[str, Optional[torch.Tensor]]:
    hidden_parts = []
    input_id_parts = []
    attention_parts = []
    position_parts = []
    label_parts = []
    stage0_kind = None
    for record in records:
        tensors = record["tensors"]
        if load_hidden:
            record_stage0_kind = stage0_tensor_name(record)
            if stage0_kind is None:
                stage0_kind = record_stage0_kind
            elif stage0_kind != record_stage0_kind:
                raise ValueError("Mixed stage0 input kinds in one batch are not supported.")
            if record_stage0_kind == "hidden_states":
                hidden_parts.append(load_tensor(manifest_dir, tensors["hidden_states"]))
            else:
                input_id_parts.append(load_tensor(manifest_dir, tensors["input_ids"]))
        attention_parts.append(load_tensor(manifest_dir, tensors["attention_mask"]))
        position_parts.append(load_tensor(manifest_dir, tensors["position_ids"]))
        if load_labels:
            label_parts.append(load_tensor(manifest_dir, tensors["labels"]))

    batch: dict[str, Optional[torch.Tensor]] = {
        "hidden": None,
        "attention_mask": torch.cat(attention_parts, dim=0).to(device=device, dtype=dtype),
        "position_ids": torch.cat(position_parts, dim=0).to(device=device, dtype=torch.long),
        "labels": None,
    }
    if load_hidden:
        if stage0_kind == "hidden_states":
            batch["hidden"] = torch.cat(hidden_parts, dim=0).to(device=device, dtype=dtype)
        elif stage0_kind == "input_ids":
            if input_embedding is None:
                raise ValueError("input_embedding is required for input_ids-based stage0 manifests.")
            input_ids = torch.cat(input_id_parts, dim=0).to(device=device, dtype=torch.long)
            with torch.no_grad():
                batch["hidden"] = input_embedding(input_ids).detach().to(dtype=dtype)
        else:
            raise ValueError("Unable to resolve stage0 input kind for batch.")
    if load_labels:
        batch["labels"] = torch.cat(label_parts, dim=0).to(device=device, dtype=torch.long)
    return batch


def metric_fieldnames() -> list[str]:
    return [
        "phase",
        "batch_seq",
        "stage_id",
        "rank",
        "device",
        "mode",
        "records",
        "microbatches",
        "h2d_ms",
        "step_ms",
        "optimizer_ms",
        "forward_ms",
        "backward_ms",
        "pipeline_fill_wait_ms",
        "pipeline_interior_wait_ms",
        "pipeline_tail_wait_ms",
        "pipeline_unattributed_ms",
        "avg_loss",
        "loss_count",
        "start_epoch_ms",
        "end_epoch_ms",
        "trainable_mode",
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
        "input_embedding_param_bytes",
        "input_embedding_trainable_param_bytes",
        "gradient_storage_bytes",
        "optimizer_state_bytes",
        "autograd_saved_tensors",
        "autograd_saved_bytes_total",
        "autograd_saved_cuda_bytes_total",
        "autograd_saved_cuda_nonleaf_bytes_total",
        "autograd_saved_cuda_leaf_bytes_total",
        "autograd_saved_bytes_peak",
        "autograd_saved_cuda_bytes_peak",
        "autograd_saved_cuda_nonleaf_bytes_peak",
        "autograd_saved_cuda_leaf_bytes_peak",
        "autograd_saved_cuda_unique_bytes_peak",
        "autograd_saved_cuda_nonleaf_unique_bytes_peak",
        "autograd_saved_cuda_leaf_unique_bytes_peak",
        "autograd_saved_cuda_nonleaf_unique_hidden_bytes_peak",
        "autograd_saved_cuda_nonleaf_unique_vocab_bytes_peak",
        "autograd_saved_cuda_nonleaf_unique_attention_bytes_peak",
        "autograd_saved_cuda_nonleaf_unique_other_bytes_peak",
        "autograd_saved_cuda_bytes_live_final",
        "autograd_saved_cuda_nonleaf_bytes_live_final",
        "autograd_saved_cuda_unique_bytes_live_final",
        "autograd_saved_cuda_nonleaf_unique_bytes_live_final",
        "cuda_peak_memory_allocated",
        "cuda_peak_memory_reserved",
        "identified_allocated_bytes",
        "runtime_residual_bytes",
        "offline_skipped",
        "recovery_event",
        "recovery_policy",
        "recovery_action",
    ]


def timeline_fieldnames() -> list[str]:
    return [
        "phase",
        "batch_seq",
        "stage_id",
        "rank",
        "device",
        "event",
        "microbatch_index",
        "start_epoch_ms",
        "end_epoch_ms",
        "duration_ms",
    ]


def eval_result_fieldnames() -> list[str]:
    return [
        "phase",
        "seq",
        "dataset_index",
        "response",
        "predicted_response",
        "predicted_token_id",
        "target_token_id",
        "choice_correct",
        "choice_count",
        "choice_accuracy",
        "choice_loss",
    ]


def train_batch_fieldnames() -> list[str]:
    return [
        "phase",
        "batch_seq",
        "records",
        "status",
        "offline_skipped",
        "avg_loss",
        "loss_count",
        "step_ms",
        "throughput_per_s",
    ]


def load_transient_dropout_mask(
    path: Optional[Path],
    *,
    num_stages: int,
    window_size: int,
) -> dict[int, frozenset[int]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError(f"Unsupported transient dropout mask schema in {path}")
    if int(payload.get("num_stages", -1)) != num_stages:
        raise ValueError("Transient dropout mask num_stages does not match --num_chunks")
    if int(payload.get("window_size", -1)) != window_size:
        raise ValueError("Transient dropout mask window_size must match --batch_size")
    raw_by_stage = payload.get("offline_windows_by_stage")
    if not isinstance(raw_by_stage, dict):
        raise ValueError("Transient dropout mask requires offline_windows_by_stage")
    result: dict[int, frozenset[int]] = {}
    for stage_id in range(num_stages):
        raw_windows = raw_by_stage.get(str(stage_id), [])
        if not isinstance(raw_windows, list):
            raise ValueError(f"Mask windows for stage {stage_id} must be a list")
        windows = [int(value) for value in raw_windows]
        if any(value < 0 for value in windows):
            raise ValueError("Transient dropout window ids must be non-negative")
        if len(set(windows)) != len(windows):
            raise ValueError(f"Duplicate transient dropout window for stage {stage_id}")
        result[stage_id] = frozenset(windows)
    return result


def transient_offline_stages(*, batch_seq: int, mode: str, args: argparse.Namespace) -> tuple[int, ...]:
    if mode != "train":
        return ()
    return tuple(
        stage_id
        for stage_id, windows in args.transient_offline_windows.items()
        if batch_seq in windows
    )


def batch_overlaps_offline_window(
    *,
    batch_seq: int,
    batch_size: int,
    mode: str,
    args: argparse.Namespace,
) -> bool:
    if mode != "train":
        return False
    contiguous_outage = False
    if args.offline_stage is not None:
        assert args.offline_start_seq is not None
        assert args.offline_end_seq is not None
        start = batch_seq * batch_size
        end = start + batch_size
        contiguous_outage = start < args.offline_end_seq and args.offline_start_seq < end
    transient_skip = (
        args.transient_dropout_policy == "skip"
        and bool(transient_offline_stages(batch_seq=batch_seq, mode=mode, args=args))
    )
    return contiguous_outage or transient_skip


def batch_matches_failure_event(
    *,
    batch_seq: int,
    mode: str,
    cfg: RankConfig,
    recovery_state: Optional[RecoveryRuntimeState],
) -> bool:
    if mode != "train" or recovery_state is None:
        return False
    if recovery_state.event_triggered:
        return False
    if recovery_state.failure_stage is None or recovery_state.failure_batch_seq is None:
        return False
    if recovery_state.failure_microbatch_index is not None:
        return False
    return batch_seq == recovery_state.failure_batch_seq


def prepare_restart_from_last_commit(
    *,
    recovery_state: RecoveryRuntimeState,
    module: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: RankConfig,
    batch_seq: int,
    batch_size: int,
) -> tuple[float, int, int]:
    recovery_state.recovery_wait_events += 1
    recovery_state.recovery_wait_ms_total += recovery_state.worker_rejoin_delay_ms
    restore_started = time.perf_counter()
    checkpoint = recovery_state.latest_checkpoint
    restore_ms = 0.0
    restore_bytes = 0
    if checkpoint is not None:
        restore_trainable_checkpoint(
            chunk=module,
            optimizer=optimizer,
            checkpoint=checkpoint,
            device=cfg.device,
        )
        if cfg.device.type == "cuda":
            torch.cuda.synchronize(cfg.device)
        restore_ms = (time.perf_counter() - restore_started) * 1000.0
        restore_bytes = recovery_state.latest_checkpoint_bytes
        recovery_state.checkpoint_restores += 1
        recovery_state.checkpoint_restore_ms_total += restore_ms
        recovery_state.checkpoint_restore_bytes_total += restore_bytes
    checkpoint_batch_seq = recovery_state.latest_checkpoint_batch_seq
    replay_start = checkpoint_batch_seq + 1
    replay_plan = list(range(replay_start, batch_seq + 1))
    replayed_batches_this_event = max(0, batch_seq - replay_start + 1)
    recovery_state.replay_plan = replay_plan
    recovery_state.replayed_batches += replayed_batches_this_event
    recovery_state.replayed_records += replayed_batches_this_event * batch_size
    recovery_state.replay_start_ms = None
    recovery_state.replay_done_ms = None
    batch_start_seq = batch_seq * batch_size
    batch_end_seq = batch_start_seq + batch_size - 1
    if recovery_state.failure_microbatch_index is None:
        recovery_state.recovery_scope = (
            f"batch{recovery_state.failure_batch_seq} recovery unit; "
            f"replay batch{replay_start}..batch{batch_seq}"
        )
    else:
        recovery_state.recovery_scope = (
            f"global window records {batch_start_seq}..{batch_end_seq}; "
            f"failure at backward microbatch {recovery_state.failure_microbatch_index}; "
            f"replay batch{replay_start}..batch{batch_seq}"
        )
    recovery_state.recovery_records = replayed_batches_this_event * batch_size
    recovery_state.global_window_committed = False
    return restore_ms, restore_bytes, replay_start


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def validation_curve_fieldnames() -> list[str]:
    return [
        "optimizer_step",
        "train_samples_seen",
        "phase",
        "validation_records",
        "choice_correct",
        "choice_count",
        "choice_accuracy",
        "avg_loss",
        "wall_ms",
    ]


def aggregate_train_segments(segments: list[dict[str, Any]]) -> dict[str, Any]:
    if not segments:
        return {}
    records = sum(int(segment.get("rows", 0)) for segment in segments)
    completed = sum(int(segment.get("completed_records", 0)) for segment in segments)
    wall_ms = sum(float(segment.get("wall_ms", 0.0)) for segment in segments)
    loss_weighted_sum = sum(
        float(segment.get("avg_loss", 0.0)) * int(segment.get("completed_records", 0))
        for segment in segments
    )
    return {
        "phase": "train",
        "mode": "train",
        "rows": records,
        "completed_records": completed,
        "skipped_records": sum(int(segment.get("skipped_records", 0)) for segment in segments),
        "skipped_batches": sum(int(segment.get("skipped_batches", 0)) for segment in segments),
        "optimizer_steps": sum(int(segment.get("optimizer_steps", 0)) for segment in segments),
        "batches": sum(int(segment.get("batches", 0)) for segment in segments),
        "choice_correct": 0,
        "choice_count": 0,
        "choice_accuracy": 0.0,
        "avg_loss": loss_weighted_sum / completed if completed else 0.0,
        "wall_ms": wall_ms,
        "throughput_per_s": completed / (wall_ms / 1000.0) if wall_ms > 0 else 0.0,
        "segment_phases": [str(segment.get("phase", "")) for segment in segments],
    }


def summarize_1f1b_phase_metrics(
    phase_metrics: list[dict[str, Any]],
    *,
    completed_records: int,
    wall_ms: float,
) -> dict[str, Any]:
    usable = [row for row in phase_metrics if not bool(row.get("offline_skipped"))]
    payload: dict[str, Any] = {
        "full_run_throughput_per_s": completed_records / (wall_ms / 1000.0) if wall_ms > 0 else 0.0,
        "steady_state_throughput_per_s": "",
        "warmup_or_fill_ms": "",
        "drain_ms": "",
        "fill_drain_overhead_ms": "",
        "status": "insufficient_batches_for_trim",
        "source": "summary.pipeline_phase_metrics",
        "phase_semantics": "1f1b_batch_transaction_fill_steady_drain",
        "phase_alignment_note": (
            "Shared fill/drain report fields follow native 1F1B batch-transaction "
            "pipeline fill/steady/drain semantics."
        ),
        "trim_policy": "drop_first_last_completed_batches",
        "completed_batches": len(usable),
        "steady_state_batches": 0,
    }
    if len(usable) < 3:
        return payload

    first = usable[0]
    last = usable[-1]
    interior = usable[1:-1]
    interior_records = sum(int(row.get("records", 0)) for row in interior)
    interior_ms = sum(float(row.get("step_ms", 0.0)) for row in interior)
    warmup_or_fill_ms = float(first.get("step_ms", 0.0))
    drain_ms = float(last.get("step_ms", 0.0))
    payload.update(
        {
            "steady_state_throughput_per_s": (
                interior_records / (interior_ms / 1000.0) if interior_ms > 0 else ""
            ),
            "warmup_or_fill_ms": warmup_or_fill_ms,
            "drain_ms": drain_ms,
            "fill_drain_overhead_ms": warmup_or_fill_ms + drain_ms,
            "status": "explicit_batch_trim",
            "steady_state_batches": len(interior),
            "steady_state_records": interior_records,
        }
    )
    return payload


def evaluate_logits(
    *,
    phase: str,
    batch_start_seq: int,
    records: list[dict[str, Any]],
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[list[dict[str, Any]], int, int, float, int]:
    log_probs = F.log_softmax(logits.detach().float(), dim=-1)
    rows: list[dict[str, Any]] = []
    correct_total = 0
    count_total = 0
    loss_sum = 0.0
    loss_count = 0
    for offset, record in enumerate(records):
        if record.get("label_choices"):
            choice_info = label_choice_details(
                record,
                log_probs[offset : offset + 1],
                labels[offset : offset + 1],
            )
            correct = int(choice_info["choice_correct"])
            count = int(choice_info["choice_count"])
            choice_loss = float(choice_info["choice_loss"])
            loss_sum += choice_loss * count
            loss_count += count
            predicted_response = str(choice_info["predicted_response"])
            predicted_token_id = choice_info["predicted_token_id"]
            target_token_id = choice_info["target_token_id"]
        else:
            correct, count = 0, 0
            choice_loss = float(
                causal_lm_loss(
                    logits[offset : offset + 1],
                    labels[offset : offset + 1],
                    label_smoothing=0.0,
                ).item()
            )
            loss_sum += choice_loss
            loss_count += 1
            predicted_response = ""
            predicted_token_id = ""
            target_token_id = ""
        correct_total += correct
        count_total += count
        response = (record.get("text") or {}).get("response", "").strip()
        rows.append(
            {
                "phase": phase,
                "seq": batch_start_seq + offset,
                "dataset_index": int(record.get("dataset_index", -1)),
                "response": response,
                "predicted_response": predicted_response,
                "predicted_token_id": predicted_token_id,
                "target_token_id": target_token_id,
                "choice_correct": correct,
                "choice_count": count,
                "choice_accuracy": (correct / count) if count else 0.0,
                "choice_loss": choice_loss,
            }
        )
    return rows, correct_total, count_total, loss_sum, loss_count


def sync_max_ms(local_ms: float, device: torch.device) -> float:
    tensor = torch.tensor([local_ms], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def sync_sum_value(local_value: float, device: torch.device) -> float:
    tensor = torch.tensor([local_value], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def set_optimizer_lr(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(learning_rate)


def run_phase(
    *,
    name: str,
    mode: str,
    records: list[dict[str, Any]],
    manifest_dir: Path,
    cfg: RankConfig,
    stage: PipelineStage,
    module: FullBackwardStageChunk,
    train_schedule: Schedule1F1B,
    eval_schedule: _ScheduleForwardOnly,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    batch_size: int,
    output_dir: Path,
    stage0_input_embedding: Optional[nn.Module],
    recovery_state: Optional[RecoveryRuntimeState] = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    is_first = cfg.rank == 0
    is_last = cfg.rank == cfg.world_size - 1
    batches = batched(records, batch_size)
    phase_metrics: list[dict[str, Any]] = []
    phase_timeline_start = len(module.recorder.rows)
    eval_rows: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    loss_weighted_sum = 0.0
    loss_count = 0
    choice_correct = 0
    choice_count = 0
    choice_loss_sum = 0.0
    eval_loss_count = 0
    skipped_records = 0
    skipped_batches = 0
    transient_replayed_batches = 0
    activation_tracker = SavedTensorTracker(hidden_size=cfg.hidden_size, vocab_size=cfg.vocab_size)

    module.train(mode == "train")
    dist.barrier()
    phase_started = time.perf_counter()

    batch_seq = 0
    while batch_seq < len(batches):
        batch_records = batches[batch_seq]
        module.recorder.start_batch(phase=name, batch_seq=batch_seq)
        batch_timeline_start = len(module.recorder.rows)
        masked_stages = transient_offline_stages(
            batch_seq=batch_seq,
            mode=mode,
            args=args,
        )
        if masked_stages and args.transient_dropout_policy == "replay":
            # Accuracy protocol: the selected stage fails before any work for
            # this batch, rejoins on retry, and the global batch executes once.
            transient_replayed_batches += 1
        if batch_overlaps_offline_window(
            batch_seq=batch_seq,
            batch_size=batch_size,
            mode=mode,
            args=args,
        ):
            # A standard backward pipeline cannot commit a partial batch when
            # one stage is unavailable. Drop it before Schedule1F1B begins so
            # every rank observes the same strict no-recovery semantics.
            skipped_records += len(batch_records)
            skipped_batches += 1
            started_epoch_ms = time.time() * 1000.0
            phase_metrics.append(
                {
                    "phase": name,
                    "batch_seq": batch_seq,
                    "stage_id": cfg.stage_id,
                    "rank": cfg.rank,
                    "device": cfg.device_name,
                    "mode": mode,
                    "records": len(batch_records),
                    "microbatches": args.microbatches,
                    "h2d_ms": 0.0,
                    "step_ms": 0.0,
                    "optimizer_ms": 0.0,
                    "avg_loss": "",
                    "loss_count": 0,
                    "start_epoch_ms": started_epoch_ms,
                    "end_epoch_ms": time.time() * 1000.0,
                    "trainable_mode": cfg.trainable_mode,
                    "local_params": cfg.local_params,
                    "local_trainable_params": cfg.local_trainable_params,
                    "local_param_bytes": cfg.local_param_bytes,
                    "local_trainable_param_bytes": cfg.local_trainable_param_bytes,
                    "resident_model_param_bytes": cfg.resident_model_param_bytes,
                    "resident_frozen_param_bytes": cfg.resident_frozen_param_bytes,
                    "base_shard_param_bytes": cfg.base_shard_param_bytes,
                    "base_shard_trainable_param_bytes": cfg.base_shard_trainable_param_bytes,
                    "local_readout_param_bytes": cfg.local_readout_param_bytes,
                    "local_readout_trainable_param_bytes": cfg.local_readout_trainable_param_bytes,
                    "input_embedding_param_bytes": cfg.input_embedding_param_bytes,
                    "input_embedding_trainable_param_bytes": cfg.input_embedding_trainable_param_bytes,
                    "gradient_storage_bytes": 0,
                    "optimizer_state_bytes": optimizer_state_nbytes(optimizer),
                    "offline_skipped": True,
                }
            )
            if is_last:
                train_rows.append(
                    {
                        "phase": name,
                        "batch_seq": batch_seq,
                        "records": len(batch_records),
                        "status": "offline_skipped",
                        "offline_skipped": True,
                        "avg_loss": "",
                        "loss_count": 0,
                        "step_ms": 0.0,
                        "throughput_per_s": 0.0,
                    }
                )
            dist.barrier()
            batch_seq += 1
            continue
        if batch_matches_failure_event(
            batch_seq=batch_seq,
            mode=mode,
            cfg=cfg,
            recovery_state=recovery_state,
        ):
            assert recovery_state is not None
            recovery_state.event_triggered = True
            recovery_state.event_batch_seq = batch_seq
            recovery_state.interrupted_batches += 1
            started_epoch_ms = time.time() * 1000.0
            if (
                recovery_state.failure_detected_ms is None
                and cfg.stage_id == recovery_state.failure_stage
            ):
                recovery_state.failure_detected_ms = started_epoch_ms
            overhead: dict[str, Any] = {
                "phase": name,
                "batch_seq": batch_seq,
                "stage_id": cfg.stage_id,
                "rank": cfg.rank,
                "device": cfg.device_name,
                "mode": mode,
                "records": len(batch_records),
                "microbatches": args.microbatches,
                "h2d_ms": 0.0,
                "step_ms": 0.0,
                "optimizer_ms": 0.0,
                "avg_loss": "",
                "loss_count": 0,
                "start_epoch_ms": started_epoch_ms,
                "end_epoch_ms": started_epoch_ms,
                "trainable_mode": cfg.trainable_mode,
                "local_params": cfg.local_params,
                "local_trainable_params": cfg.local_trainable_params,
                "local_param_bytes": cfg.local_param_bytes,
                "local_trainable_param_bytes": cfg.local_trainable_param_bytes,
                "resident_model_param_bytes": cfg.resident_model_param_bytes,
                "resident_frozen_param_bytes": cfg.resident_frozen_param_bytes,
                "base_shard_param_bytes": cfg.base_shard_param_bytes,
                "base_shard_trainable_param_bytes": cfg.base_shard_trainable_param_bytes,
                "local_readout_param_bytes": cfg.local_readout_param_bytes,
                "local_readout_trainable_param_bytes": cfg.local_readout_trainable_param_bytes,
                "input_embedding_param_bytes": cfg.input_embedding_param_bytes,
                "input_embedding_trainable_param_bytes": cfg.input_embedding_trainable_param_bytes,
                "gradient_storage_bytes": 0,
                "optimizer_state_bytes": optimizer_state_nbytes(optimizer),
                "offline_skipped": False,
                "recovery_event": True,
                "recovery_policy": recovery_state.policy,
                "recovery_action": "interrupted_batch",
            }
            if recovery_state.policy == "strict_skip":
                skipped_records += len(batch_records)
                skipped_batches += 1
                overhead["recovery_action"] = "strict_skip"
                phase_metrics.append(overhead)
                if is_last:
                    train_rows.append(
                        {
                            "phase": name,
                            "batch_seq": batch_seq,
                            "records": len(batch_records),
                            "status": "recovery_dropped",
                            "offline_skipped": False,
                            "avg_loss": "",
                            "loss_count": 0,
                            "step_ms": 0.0,
                            "throughput_per_s": 0.0,
                        }
                    )
                dist.barrier()
                batch_seq += 1
                continue
            if recovery_state.policy == "wait_for_rejoin_batch_boundary":
                recovery_state.recovery_wait_events += 1
                recovery_state.recovery_wait_ms_total += recovery_state.worker_rejoin_delay_ms
                overhead["step_ms"] = recovery_state.worker_rejoin_delay_ms
                overhead["end_epoch_ms"] = started_epoch_ms + recovery_state.worker_rejoin_delay_ms
                overhead["recovery_action"] = "wait_then_resume"
                phase_metrics.append(overhead)
                dist.barrier()
                batch_seq += 1
                continue
            if recovery_state.policy == "restart_from_last_commit":
                restore_ms, restore_bytes, replay_start = prepare_restart_from_last_commit(
                    recovery_state=recovery_state,
                    module=module,
                    optimizer=optimizer,
                    cfg=cfg,
                    batch_seq=batch_seq,
                    batch_size=batch_size,
                )
                skipped_records += len(batch_records)
                skipped_batches += 1
                total_overhead_ms = recovery_state.worker_rejoin_delay_ms + restore_ms
                overhead["step_ms"] = total_overhead_ms
                overhead["optimizer_ms"] = restore_ms
                overhead["end_epoch_ms"] = started_epoch_ms + total_overhead_ms
                overhead["recovery_action"] = "restore_checkpoint_and_replay"
                overhead["checkpoint_restore_bytes"] = restore_bytes
                overhead["checkpoint_restore_ms"] = restore_ms
                overhead["recovery_replayed_batches"] = max(0, batch_seq - replay_start + 1)
                phase_metrics.append(overhead)
                if is_last:
                    train_rows.append(
                        {
                            "phase": name,
                            "batch_seq": batch_seq,
                            "records": len(batch_records),
                            "status": "recovery_dropped",
                            "offline_skipped": False,
                            "avg_loss": "",
                            "loss_count": 0,
                            "step_ms": total_overhead_ms,
                            "throughput_per_s": 0.0,
                        }
                    )
                dist.barrier()
                batch_seq = replay_start
                continue
            raise ValueError(f"Unsupported recovery policy: {recovery_state.policy}")
        if cfg.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(cfg.device)

        h2d_started = time.perf_counter()
        loaded = load_batch_tensors(
            records=batch_records,
            manifest_dir=manifest_dir,
            device=cfg.device,
            dtype=cfg.dtype,
            load_hidden=is_first,
            load_labels=is_last,
            input_embedding=stage0_input_embedding if is_first else None,
        )
        if cfg.device.type == "cuda":
            torch.cuda.synchronize(cfg.device)
        h2d_ms = (time.perf_counter() - h2d_started) * 1000.0

        kwargs = {
            "attention_mask": loaded["attention_mask"],
            "position_ids": loaded["position_ids"],
        }
        target = loaded["labels"] if is_last else None
        if mode == "train":
            learning_rate = args.learning_rate
            if learning_rate is None:
                learning_rate = batch_records[0].get("learning_rate")
            if learning_rate is not None:
                set_optimizer_lr(optimizer, float(learning_rate))
            optimizer.zero_grad(set_to_none=True)

        step_started_epoch_ms = time.time() * 1000.0
        if (
            recovery_state is not None
            and batch_seq in recovery_state.replay_plan
            and recovery_state.replay_start_ms is None
        ):
            recovery_state.replay_start_ms = step_started_epoch_ms
        step_started = time.perf_counter()
        losses: list[torch.Tensor] = []
        cooperative_failure_ms = 0.0
        if args.track_activation_memory:
            activation_tracker.reset()
        if stage.inputs_meta is None:
            module.recorder.skip_next_forward_event()
        if mode == "train":
            hook_context = (
                torch.autograd.graph.saved_tensors_hooks(
                    activation_tracker.pack,
                    activation_tracker.unpack,
                )
                if args.track_activation_memory
                else nullcontext()
            )
            with torch.enable_grad(), hook_context:
                if is_first:
                    assert loaded["hidden"] is not None
                    output = train_schedule.step(
                        loaded["hidden"],
                        target=target,
                        losses=losses if is_last else None,
                        return_outputs=False,
                        **kwargs,
                    )
                else:
                    output = train_schedule.step(
                        target=target,
                        losses=losses if is_last else None,
                        return_outputs=False,
                        **kwargs,
                    )
                del output
                if (
                    recovery_state is not None
                    and not recovery_state.event_triggered
                    and recovery_state.failure_microbatch_index is not None
                ):
                    cooperative_failure_ms = sync_max_ms(
                        module.recorder.consume_cooperative_failure_epoch_ms(),
                        cfg.device,
                    )
                if cooperative_failure_ms <= 0:
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(module.parameters(), args.grad_clip)
                    optimizer_started = time.perf_counter()
                    optimizer.step()
                    if cfg.device.type == "cuda":
                        torch.cuda.synchronize(cfg.device)
                    optimizer_ms = (time.perf_counter() - optimizer_started) * 1000.0
                else:
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_ms = 0.0
        else:
            optimizer_ms = 0.0
            with torch.no_grad():
                if is_first:
                    assert loaded["hidden"] is not None
                    output = eval_schedule.step(loaded["hidden"], return_outputs=True, **kwargs)
                else:
                    output = eval_schedule.step(return_outputs=True, **kwargs)
            if is_last:
                assert isinstance(output, torch.Tensor)
                assert loaded["labels"] is not None
                rows, correct, count, eval_loss_sum, current_eval_loss_count = evaluate_logits(
                    phase=name,
                    batch_start_seq=batch_seq * batch_size,
                    records=batch_records,
                    logits=output,
                    labels=loaded["labels"],
                )
                eval_rows.extend(rows)
                choice_correct += correct
                choice_count += count
                choice_loss_sum += eval_loss_sum
                eval_loss_count += current_eval_loss_count

        if mode == "train" and cooperative_failure_ms > 0:
            assert recovery_state is not None
            recovery_state.event_triggered = True
            recovery_state.event_batch_seq = batch_seq
            recovery_state.interrupted_batches += 1
            recovery_state.failure_event = "backward"
            if recovery_state.failure_detected_ms is None:
                recovery_state.failure_detected_ms = cooperative_failure_ms
            restore_ms, restore_bytes, replay_start = prepare_restart_from_last_commit(
                recovery_state=recovery_state,
                module=module,
                optimizer=optimizer,
                cfg=cfg,
                batch_seq=batch_seq,
                batch_size=batch_size,
            )
            skipped_records += len(batch_records)
            skipped_batches += 1
            total_overhead_ms = recovery_state.worker_rejoin_delay_ms + restore_ms
            phase_metrics.append(
                {
                    "phase": name,
                    "batch_seq": batch_seq,
                    "stage_id": cfg.stage_id,
                    "rank": cfg.rank,
                    "device": cfg.device_name,
                    "mode": mode,
                    "records": len(batch_records),
                    "microbatches": args.microbatches,
                    "h2d_ms": h2d_ms,
                    "step_ms": total_overhead_ms,
                    "optimizer_ms": restore_ms,
                    "avg_loss": "",
                    "loss_count": 0,
                    "start_epoch_ms": cooperative_failure_ms,
                    "end_epoch_ms": cooperative_failure_ms + total_overhead_ms,
                    "trainable_mode": cfg.trainable_mode,
                    "local_params": cfg.local_params,
                    "local_trainable_params": cfg.local_trainable_params,
                    "local_param_bytes": cfg.local_param_bytes,
                    "local_trainable_param_bytes": cfg.local_trainable_param_bytes,
                    "resident_model_param_bytes": cfg.resident_model_param_bytes,
                    "resident_frozen_param_bytes": cfg.resident_frozen_param_bytes,
                    "base_shard_param_bytes": cfg.base_shard_param_bytes,
                    "base_shard_trainable_param_bytes": cfg.base_shard_trainable_param_bytes,
                    "local_readout_param_bytes": cfg.local_readout_param_bytes,
                    "local_readout_trainable_param_bytes": cfg.local_readout_trainable_param_bytes,
                    "input_embedding_param_bytes": cfg.input_embedding_param_bytes,
                    "input_embedding_trainable_param_bytes": cfg.input_embedding_trainable_param_bytes,
                    "gradient_storage_bytes": 0,
                    "optimizer_state_bytes": optimizer_state_nbytes(optimizer),
                    "offline_skipped": False,
                    "recovery_event": True,
                    "recovery_policy": recovery_state.policy,
                    "recovery_action": "restore_checkpoint_and_replay",
                    "checkpoint_restore_bytes": restore_bytes,
                    "checkpoint_restore_ms": restore_ms,
                    "recovery_replayed_batches": max(0, batch_seq - replay_start + 1),
                }
            )
            if is_last:
                train_rows.append(
                    {
                        "phase": name,
                        "batch_seq": batch_seq,
                        "records": len(batch_records),
                        "status": "recovery_dropped",
                        "offline_skipped": False,
                        "avg_loss": "",
                        "loss_count": 0,
                        "step_ms": total_overhead_ms,
                        "throughput_per_s": 0.0,
                    }
                )
            dist.barrier()
            batch_seq = replay_start
            continue

        activation_stats = activation_tracker.snapshot() if args.track_activation_memory else {}
        opt_state_bytes = optimizer_state_nbytes(optimizer)
        gradient_storage_bytes = gradient_storage_nbytes(module)
        if cfg.device.type == "cuda":
            torch.cuda.synchronize(cfg.device)
        module.recorder.finalize_cuda_durations()
        step_ms = (time.perf_counter() - step_started) * 1000.0
        step_end_epoch_ms = time.time() * 1000.0
        peak_allocated = int(torch.cuda.max_memory_allocated(cfg.device)) if cfg.device.type == "cuda" else 0
        peak_reserved = int(torch.cuda.max_memory_reserved(cfg.device)) if cfg.device.type == "cuda" else 0
        activation_nonleaf_peak = int(activation_stats.get("autograd_saved_cuda_nonleaf_unique_bytes_peak", 0) or 0)
        identified_allocated_bytes = (
            cfg.resident_model_param_bytes
            + gradient_storage_bytes
            + opt_state_bytes
            + activation_nonleaf_peak
        )
        runtime_residual_bytes = max(0, peak_allocated - identified_allocated_bytes)
        timing_breakdown = summarize_batch_timing(
            module.recorder.rows[batch_timeline_start:],
            step_started_epoch_ms=step_started_epoch_ms,
            step_ended_epoch_ms=step_end_epoch_ms,
            optimizer_ms=optimizer_ms,
        )
        avg_loss = ""
        current_loss_count = 0
        if is_last and mode == "train":
            current_loss_count = len(losses)
            if losses:
                loss_values = [float(loss.detach().cpu().item()) for loss in losses]
                avg_loss_value = sum(loss_values) / len(loss_values)
                avg_loss = avg_loss_value
                loss_weighted_sum += sum(loss_values)
                loss_count += len(loss_values)
                train_rows.append(
                    {
                        "phase": name,
                        "batch_seq": batch_seq,
                        "records": len(batch_records),
                        "status": "completed",
                        "offline_skipped": False,
                        "avg_loss": avg_loss_value,
                        "loss_count": len(loss_values),
                        "step_ms": step_ms,
                        "throughput_per_s": len(batch_records) / (step_ms / 1000.0) if step_ms > 0 else 0.0,
                    }
                )

        phase_metrics.append(
            {
                "phase": name,
                "batch_seq": batch_seq,
                "stage_id": cfg.stage_id,
                "rank": cfg.rank,
                "device": cfg.device_name,
                "mode": mode,
                "records": len(batch_records),
                "microbatches": args.microbatches,
                "h2d_ms": h2d_ms,
                "step_ms": step_ms,
                "optimizer_ms": optimizer_ms,
                **timing_breakdown,
                "avg_loss": avg_loss,
                "loss_count": current_loss_count,
                "start_epoch_ms": step_started_epoch_ms,
                "end_epoch_ms": step_end_epoch_ms,
                "trainable_mode": cfg.trainable_mode,
                "local_params": cfg.local_params,
                "local_trainable_params": cfg.local_trainable_params,
                "local_param_bytes": cfg.local_param_bytes,
                "local_trainable_param_bytes": cfg.local_trainable_param_bytes,
                "resident_model_param_bytes": cfg.resident_model_param_bytes,
                "resident_frozen_param_bytes": cfg.resident_frozen_param_bytes,
                "base_shard_param_bytes": cfg.base_shard_param_bytes,
                "base_shard_trainable_param_bytes": cfg.base_shard_trainable_param_bytes,
                "local_readout_param_bytes": cfg.local_readout_param_bytes,
                "local_readout_trainable_param_bytes": cfg.local_readout_trainable_param_bytes,
                "input_embedding_param_bytes": cfg.input_embedding_param_bytes,
                "input_embedding_trainable_param_bytes": cfg.input_embedding_trainable_param_bytes,
                "gradient_storage_bytes": gradient_storage_bytes,
                "optimizer_state_bytes": opt_state_bytes,
                **activation_stats,
                "cuda_peak_memory_allocated": peak_allocated,
                "cuda_peak_memory_reserved": peak_reserved,
                "identified_allocated_bytes": identified_allocated_bytes,
                "runtime_residual_bytes": runtime_residual_bytes,
                "offline_skipped": False,
                "recovery_event": False,
                "recovery_policy": recovery_state.policy if recovery_state is not None else "",
                "recovery_action": (
                    "transient_replay"
                    if masked_stages and args.transient_dropout_policy == "replay"
                    else (
                        "replay_batch"
                        if recovery_state is not None and batch_seq in recovery_state.replay_plan
                        else "normal"
                    )
                ),
            }
        )

        if recovery_state is not None and batch_seq in recovery_state.replay_plan:
            recovery_state.replay_plan.remove(batch_seq)
            recovery_state.replay_done_ms = step_end_epoch_ms

        if (
            mode == "train"
            and recovery_state is not None
            and recovery_state.event_triggered
            and recovery_state.failure_batch_seq is not None
            and batch_seq == recovery_state.failure_batch_seq
            and recovery_state.recovery_unit_done_ms is None
            and not recovery_state.replay_plan
            and cfg.stage_id == cfg.world_size - 1
        ):
            recovery_state.recovery_unit_done_ms = step_end_epoch_ms
            recovery_state.global_window_committed = True

        if (
            mode == "train"
            and recovery_state is not None
            and recovery_state.policy == "restart_from_last_commit"
            and ((batch_seq + 1) % recovery_state.checkpoint_interval_batches == 0)
        ):
            capture_started = time.perf_counter()
            checkpoint, checkpoint_bytes = capture_trainable_checkpoint(module, optimizer)
            if cfg.device.type == "cuda":
                torch.cuda.synchronize(cfg.device)
            capture_ms = (time.perf_counter() - capture_started) * 1000.0
            recovery_state.latest_checkpoint = checkpoint
            recovery_state.latest_checkpoint_batch_seq = batch_seq
            recovery_state.latest_checkpoint_version += 1
            recovery_state.latest_checkpoint_bytes = checkpoint_bytes
            recovery_state.checkpoint_captures += 1
            recovery_state.checkpoint_capture_bytes_total += checkpoint_bytes
            recovery_state.checkpoint_capture_ms_total += capture_ms

        if cfg.rank == cfg.world_size - 1 and args.progress_interval > 0:
            completed = (batch_seq + 1) * batch_size
            if completed % args.progress_interval == 0 or completed == len(records):
                if mode == "train":
                    printable_loss = float(avg_loss) if avg_loss != "" else 0.0
                    print(f"{name}: {completed}/{len(records)} loss={printable_loss:.4f}", flush=True)
                else:
                    acc = choice_correct / choice_count if choice_count else 0.0
                    print(f"{name}: {completed}/{len(records)} acc={acc:.4f}", flush=True)

        del loaded
        if args.gc_interval_batches > 0 and (batch_seq + 1) % args.gc_interval_batches == 0:
            gc.collect()
        batch_seq += 1

    dist.barrier()
    wall_ms = sync_max_ms((time.perf_counter() - phase_started) * 1000.0, cfg.device)
    phase_timeline = module.recorder.rows[phase_timeline_start:]

    rank_recovery = recovery_state.to_rank_summary(stage_id=cfg.stage_id, rank=cfg.rank) if recovery_state is not None else {}
    recovery_rank_summaries: list[dict[str, Any]] = []
    if recovery_state is not None:
        recovery_rank_summaries = [{} for _ in range(cfg.world_size)]
        dist.all_gather_object(recovery_rank_summaries, rank_recovery)

    local_summary: Optional[dict[str, Any]]
    if is_last:
        if mode == "train":
            avg_loss_final = loss_weighted_sum / loss_count if loss_count else 0.0
            output_csv = output_dir / f"{name}_batches.csv"
            write_csv(output_csv, train_batch_fieldnames(), train_rows)
            final_batch_rows: dict[int, dict[str, Any]] = {}
            for row in train_rows:
                final_batch_rows[int(row.get("batch_seq", -1))] = row
            completed_records_final = sum(
                int(row.get("records", 0))
                for row in final_batch_rows.values()
                if row.get("status") == "completed"
            )
            skipped_records_final = sum(
                int(row.get("records", 0))
                for row in final_batch_rows.values()
                if row.get("status") != "completed"
            )
            completed_batches_final = sum(
                1 for row in final_batch_rows.values() if row.get("status") == "completed"
            )
            skipped_batches_final = sum(
                1 for row in final_batch_rows.values() if row.get("status") != "completed"
            )
            local_summary = {
                "phase": name,
                "mode": mode,
                "rows": len(records),
                "completed_records": completed_records_final,
                "skipped_records": skipped_records_final,
                "skipped_batches": skipped_batches_final,
                "transient_replayed_batches": transient_replayed_batches,
                "optimizer_steps": completed_batches_final,
                "batches": len(batches),
                "choice_correct": 0,
                "choice_count": 0,
                "choice_accuracy": 0.0,
                "avg_loss": avg_loss_final,
                "wall_ms": wall_ms,
                "throughput_per_s": (
                    (completed_records_final / (wall_ms / 1000.0)) if wall_ms > 0 else 0.0
                ),
                "csv": str(output_csv),
                "pipeline_phase_metrics": summarize_1f1b_phase_metrics(
                    phase_metrics,
                    completed_records=completed_records_final,
                    wall_ms=wall_ms,
                ),
                "recovery": rank_recovery,
                "recovery_rank_summaries": recovery_rank_summaries,
            }
        else:
            output_csv = output_dir / f"{name}.csv"
            write_csv(output_csv, eval_result_fieldnames(), eval_rows)
            avg_choice_loss = choice_loss_sum / eval_loss_count if eval_loss_count else 0.0
            local_summary = {
                "phase": name,
                "mode": mode,
                "rows": len(records),
                "completed_records": len(records),
                "skipped_records": 0,
                "skipped_batches": 0,
                "optimizer_steps": 0,
                "batches": len(batches),
                "choice_correct": choice_correct,
                "choice_count": choice_count,
                "choice_accuracy": (choice_correct / choice_count) if choice_count else 0.0,
                "avg_loss": avg_choice_loss,
                "loss_count": eval_loss_count,
                "wall_ms": wall_ms,
                "throughput_per_s": len(records) / (wall_ms / 1000.0) if wall_ms > 0 else 0.0,
                "csv": str(output_csv),
            }
    else:
        local_summary = None

    summary_box: list[Optional[dict[str, Any]]] = [local_summary]
    dist.broadcast_object_list(summary_box, src=cfg.world_size - 1)
    return summary_box[0] or {}, phase_metrics, phase_timeline


def merge_rank_csvs(output_dir: Path, *, prefix: str, fieldnames: list[str], world_size: int, merged_name: str) -> None:
    rows: list[dict[str, Any]] = []
    for rank in range(world_size):
        path = output_dir / f"rank{rank}_{prefix}.csv"
        if not path.is_file():
            continue
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    rows.sort(
        key=lambda row: (
            row.get("phase", ""),
            int(float(row.get("batch_seq") or 0)),
            int(float(row.get("stage_id") or 0)),
            row.get("event", ""),
            int(float(row.get("microbatch_index") or 0)),
        )
    )
    write_csv(output_dir / merged_name, fieldnames, rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-backward LoRA pipeline baseline.")
    parser.add_argument("--model_name", default="tinyllama")
    parser.add_argument("--train_manifest", type=Path, required=True)
    parser.add_argument("--eval_manifest", type=Path, required=True)
    parser.add_argument(
        "--validation_manifest",
        type=Path,
        default=None,
        help="Held-out development manifest evaluated at fixed optimizer-step checkpoints.",
    )
    parser.add_argument("--validation_limit", type=int, default=None)
    parser.add_argument(
        "--validation_interval_steps",
        type=int,
        default=0,
        help="Evaluate validation_manifest every N optimizer steps; 0 disables intermediate validation.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_chunks", type=int, default=3)
    parser.add_argument("--stage_devices", required=True)
    parser.add_argument("--train_limit", type=int, default=None)
    parser.add_argument("--train_epochs", type=int, default=1)
    parser.add_argument("--eval_limit", type=int, default=None)
    parser.add_argument("--microbatches", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument(
        "--pipeline_schedule",
        default="1f1b",
        choices=["auto", "1f1b", "gpipe"],
        help=(
            "Exact-BP pipeline schedule. auto selects 1f1b when microbatches "
            ">= num_chunks and gpipe otherwise."
        ),
    )
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--sgd_momentum", type=float, default=0.0)
    parser.add_argument("--sgd_dampening", type=float, default=0.0)
    parser.add_argument("--sgd_weight_decay", type=float, default=0.0)
    parser.add_argument("--sgd_nesterov", action="store_true")
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--trainable_mode", default="lora", choices=["lora", "full", "full_layers"])
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_targets", default="q_proj,v_proj")
    parser.add_argument("--lora_init_std", type=float, default=0.01)
    parser.add_argument(
        "--lora_init_seed",
        type=int,
        default=None,
        help="Process-independent LoRA initialization seed. Formal comparisons should set this explicitly.",
    )
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--progress_interval", type=int, default=128)
    parser.add_argument(
        "--gc_interval_batches",
        type=int,
        default=0,
        help="Run Python gc.collect() every N completed batches; 0 disables per-batch collection.",
    )
    parser.add_argument(
        "--offline_stage",
        type=int,
        default=None,
        help=(
            "Audit-only unavailable stage for strict no-recovery experiments. "
            "Any training batch overlapping the sequence window is dropped before 1F1B scheduling."
        ),
    )
    parser.add_argument("--offline_start_seq", type=int, default=None)
    parser.add_argument("--offline_end_seq", type=int, default=None)
    parser.add_argument(
        "--transient_dropout_mask",
        type=Path,
        default=None,
        help="JSON mask of independently unavailable stages per logical batch.",
    )
    parser.add_argument(
        "--transient_dropout_policy",
        choices=["skip", "replay"],
        default="skip",
        help=(
            "skip drops any batch with an unavailable stage; replay models failure "
            "before execution followed by rejoin and executes the batch once."
        ),
    )
    parser.add_argument(
        "--recovery_policy",
        default="strict_skip",
        choices=["strict_skip", "wait_for_rejoin_batch_boundary", "restart_from_last_commit"],
        help=(
            "Exact-BP outage semantics. strict_skip is the historical no-recovery control; "
            "wait_for_rejoin_batch_boundary drops the interrupted batch and resumes after a fixed wait; "
            "restart_from_last_commit restores the latest committed batch-boundary checkpoint and replays "
            "later committed batches before continuing."
        ),
    )
    parser.add_argument(
        "--failure_stage",
        type=int,
        default=None,
        help="For recovery-aware baselines: logical stage id that becomes unavailable at the injected batch boundary.",
    )
    parser.add_argument(
        "--failure_batch_seq",
        type=int,
        default=None,
        help="Inject a single batch-boundary failure at the given train batch index.",
    )
    parser.add_argument(
        "--failure_microbatch_index",
        type=int,
        default=None,
        help=(
            "Optional cooperative exact-BP fault point inside the injected train batch. "
            "When set, the target batch is interrupted after the specified backward microbatch "
            "finishes on --failure_stage, and the whole global BP window is replayed before commit."
        ),
    )
    parser.add_argument(
        "--checkpoint_interval_batches",
        type=int,
        default=1,
        help="For restart_from_last_commit: capture trainable state every K committed train batches.",
    )
    parser.add_argument(
        "--worker_rejoin_delay_ms",
        type=float,
        default=2000.0,
        help="Synthetic wait before recovery resumes after the injected batch-boundary outage.",
    )
    parser.add_argument("--skip_eval_before", action="store_true")
    parser.add_argument("--skip_eval_after", action="store_true")
    parser.add_argument(
        "--track_activation_memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record autograd saved-tensor storage; disable only for pure-throughput runs.",
    )
    parser.add_argument("--record_timeline", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_chunks <= 0:
        raise ValueError("--num_chunks must be positive")
    if args.microbatches <= 0:
        raise ValueError("--microbatches must be positive")
    resolved_pipeline_schedule = args.pipeline_schedule
    if resolved_pipeline_schedule == "auto":
        resolved_pipeline_schedule = (
            "1f1b" if args.microbatches >= args.num_chunks else "gpipe"
        )
    if resolved_pipeline_schedule == "1f1b" and args.microbatches < args.num_chunks:
        raise ValueError("--microbatches must be >= --num_chunks for Schedule1F1B")
    if resolved_pipeline_schedule == "gpipe" and any(
        value is not None
        for value in (
            args.offline_stage,
            args.offline_start_seq,
            args.offline_end_seq,
            args.transient_dropout_mask,
            args.failure_stage,
            args.failure_batch_seq,
            args.failure_microbatch_index,
        )
    ):
        raise ValueError(
            "ScheduleGPipe currently supports throughput runs without failure injection"
        )
    batch_size = args.batch_size or args.microbatches
    if batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if batch_size % args.microbatches != 0:
        raise ValueError("--batch_size must be divisible by --microbatches")
    if args.train_epochs <= 0:
        raise ValueError("--train_epochs must be positive")
    if args.gc_interval_batches < 0:
        raise ValueError("--gc_interval_batches must be non-negative")
    if args.checkpoint_interval_batches <= 0:
        raise ValueError("--checkpoint_interval_batches must be positive")
    if args.worker_rejoin_delay_ms < 0:
        raise ValueError("--worker_rejoin_delay_ms must be non-negative")
    offline_values = (args.offline_stage, args.offline_start_seq, args.offline_end_seq)
    if args.transient_dropout_mask is not None and any(
        value is not None for value in offline_values
    ):
        raise ValueError("Use either a contiguous offline window or --transient_dropout_mask")
    if any(value is not None for value in offline_values):
        if any(value is None for value in offline_values):
            raise ValueError(
                "--offline_stage, --offline_start_seq, and --offline_end_seq must be supplied together"
            )
        assert args.offline_stage is not None
        assert args.offline_start_seq is not None
        assert args.offline_end_seq is not None
        if args.offline_stage < 0 or args.offline_stage >= args.num_chunks:
            raise ValueError("--offline_stage must name a pipeline stage")
        if args.offline_start_seq < 0 or args.offline_end_seq <= args.offline_start_seq:
            raise ValueError("offline window must satisfy 0 <= start < end")
    if args.failure_batch_seq is not None and args.failure_batch_seq < 0:
        raise ValueError("--failure_batch_seq must be non-negative")
    if args.failure_microbatch_index is not None and args.failure_microbatch_index < 0:
        raise ValueError("--failure_microbatch_index must be non-negative")
    if args.failure_stage is not None and (args.failure_stage < 0 or args.failure_stage >= args.num_chunks):
        raise ValueError("--failure_stage must name a pipeline stage")
    if args.failure_batch_seq is None and args.recovery_policy != "strict_skip":
        raise ValueError("Recovery-aware 1F1B policies require --failure_batch_seq")
    if args.failure_batch_seq is not None and args.failure_stage is None:
        raise ValueError("Injected recovery baselines require --failure_stage")
    if args.failure_microbatch_index is not None:
        if args.failure_batch_seq is None or args.failure_stage is None:
            raise ValueError("--failure_microbatch_index requires --failure_batch_seq and --failure_stage")
        if args.recovery_policy != "restart_from_last_commit":
            raise ValueError(
                "--failure_microbatch_index currently requires --recovery_policy restart_from_last_commit."
            )
        if args.failure_microbatch_index >= args.microbatches:
            raise ValueError("--failure_microbatch_index must be smaller than --microbatches")
    if args.failure_batch_seq is not None and any(value is not None for value in offline_values):
        raise ValueError("Use either offline-window strict degradation or batch-boundary recovery injection, not both")
    if args.transient_dropout_mask is not None and args.failure_batch_seq is not None:
        raise ValueError("Use either intermittent dropout or a single recovery injection")
    if args.transient_dropout_mask is not None and args.validation_interval_steps > 0:
        raise ValueError("Intermittent dropout currently requires one uninterrupted training phase")

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    devices = parse_devices(args.stage_devices, args.num_chunks)
    if local_rank < 0 or local_rank >= len(devices):
        raise ValueError(f"LOCAL_RANK={local_rank} is outside parsed stage devices {devices}")
    device = torch.device(devices[local_rank])
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != args.num_chunks:
        raise ValueError(f"torchrun world_size={world_size} must equal --num_chunks={args.num_chunks}")
    dtype = resolve_dtype(args.dtype)
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    resolved_model = resolve_model_name(args.model_name)
    if rank == 0:
        print(
            f"Starting Exact-BP {resolved_pipeline_schedule} baseline "
            f"model={resolved_model} chunks={args.num_chunks} "
            f"devices={devices} microbatches={args.microbatches} batch_size={batch_size}",
            flush=True,
        )

    recorder = StageEventRecorder(
        stage_id=rank,
        rank=rank,
        device_name=devices[rank],
        enabled=args.record_timeline,
        failure_target_batch_seq=(
            args.failure_batch_seq
            if args.failure_microbatch_index is not None and rank == args.failure_stage
            else None
        ),
        failure_target_event=(
            "backward"
            if args.failure_microbatch_index is not None and rank == args.failure_stage
            else None
        ),
        failure_target_microbatch_index=(
            args.failure_microbatch_index
            if args.failure_microbatch_index is not None and rank == args.failure_stage
            else None
        ),
    )
    print(f"[rank {rank}] loading model={resolved_model} dtype={dtype} device={device}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(resolved_model, torch_dtype=dtype)
    hidden_size = int(getattr(model.config, "hidden_size", 0) or getattr(model.config, "n_embd", 0) or 0)
    vocab_size = int(getattr(model.config, "vocab_size", 0) or 0)
    trainable_setup = configure_model_trainable(
        module=model,
        mode=args.trainable_mode,
        lora_targets=args.lora_targets,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_init_std=args.lora_init_std,
        lora_init_seed=args.lora_init_seed,
    )
    lora_init_fingerprint = lora_parameter_fingerprint(model)
    stage0_input_embedding = model.get_input_embeddings() if rank == 0 else None
    module = build_stage_module(model=model, stage_id=rank, num_chunks=args.num_chunks, recorder=recorder)
    module.to(device)
    if stage0_input_embedding is not None:
        stage0_input_embedding.to(device)
        stage0_input_embedding.eval()
        for param in stage0_input_embedding.parameters():
            param.requires_grad = False
    local_param_stats = module_param_stats(module)
    memory_ledger = stage_memory_ledger(module=module, input_embedding=stage0_input_embedding)
    params = [param for param in module.parameters() if param.requires_grad]
    initial_lr = args.learning_rate or 3e-4
    optimizer = build_optimizer(
        params=params,
        optimizer_name=args.optimizer,
        learning_rate=initial_lr,
        sgd_momentum=args.sgd_momentum,
        sgd_dampening=args.sgd_dampening,
        sgd_weight_decay=args.sgd_weight_decay,
        sgd_nesterov=args.sgd_nesterov,
    )
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    print(
        f"[rank {rank}] ready stage={rank} trainable_mode={trainable_setup.mode} "
        f"lora_modules={trainable_setup.lora_modules} "
        f"all_trainable={trainable_setup.trainable_params} "
        f"frozen={trainable_setup.frozen_params} "
        f"local_trainable={local_param_stats.trainable_params} "
        f"lora_init={lora_init_fingerprint[:12]}",
        flush=True,
    )

    if args.recovery_policy == "restart_from_last_commit":
        capture_started = time.perf_counter()
        initial_checkpoint, initial_checkpoint_bytes = capture_trainable_checkpoint(module, optimizer)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        initial_checkpoint_ms = (time.perf_counter() - capture_started) * 1000.0
    else:
        initial_checkpoint = None
        initial_checkpoint_bytes = 0
        initial_checkpoint_ms = 0.0

    pipeline_stage = PipelineStage(module, rank, args.num_chunks, device)
    loss_fn = lambda logits, labels: causal_lm_loss(  # noqa: E731
        logits,
        labels,
        label_smoothing=args.label_smoothing,
    )
    schedule_cls = (
        Schedule1F1B
        if resolved_pipeline_schedule == "1f1b"
        else ScheduleGPipe
    )
    train_schedule = schedule_cls(
        pipeline_stage,
        n_microbatches=args.microbatches,
        loss_fn=loss_fn,
    )
    eval_schedule = _ScheduleForwardOnly(pipeline_stage, n_microbatches=args.microbatches)

    base_train_records = read_manifest(args.train_manifest, args.train_limit)
    train_records = base_train_records * args.train_epochs
    args.transient_offline_windows = load_transient_dropout_mask(
        args.transient_dropout_mask,
        num_stages=args.num_chunks,
        window_size=batch_size,
    )
    logical_windows = (len(train_records) + batch_size - 1) // batch_size
    out_of_range_windows = sorted(
        {
            window_id
            for windows in args.transient_offline_windows.values()
            for window_id in windows
            if window_id >= logical_windows
        }
    )
    if out_of_range_windows:
        raise ValueError(
            f"Transient dropout mask references windows outside 0..{logical_windows - 1}: "
            f"{out_of_range_windows[:8]}"
        )
    eval_records = read_manifest(args.eval_manifest, args.eval_limit)
    validation_records = (
        read_manifest(args.validation_manifest, args.validation_limit)
        if args.validation_manifest is not None
        else None
    )
    if args.validation_interval_steps < 0:
        raise ValueError("--validation_interval_steps must be non-negative")
    if args.validation_interval_steps > 0 and validation_records is None:
        raise ValueError("--validation_interval_steps requires --validation_manifest")
    if validation_records is not None and args.validation_interval_steps == 0:
        raise ValueError("--validation_manifest requires --validation_interval_steps > 0")
    if validation_records is not None and not args.skip_eval_before:
        raise ValueError("Use --skip_eval_before with --validation_manifest to avoid evaluating two dev sets at step 0")
    if validation_records is not None and len(validation_records) % batch_size != 0:
        raise ValueError("Validation records must be divisible by --batch_size")
    if args.validation_interval_steps > 0:
        interval_records = args.validation_interval_steps * batch_size
        if interval_records > len(train_records):
            raise ValueError("--validation_interval_steps exceeds the train budget")
    if args.offline_end_seq is not None and args.offline_end_seq > len(train_records):
        raise ValueError("--offline_end_seq must not exceed the training record count")
    total_train_batches = len(train_records) // batch_size if train_records else 0
    if args.failure_batch_seq is not None and args.failure_batch_seq >= total_train_batches:
        raise ValueError("--failure_batch_seq must be smaller than the train batch count")

    phases: list[dict[str, Any]] = []
    all_metrics: list[dict[str, Any]] = []
    all_timeline: list[dict[str, Any]] = []
    validation_curve: list[dict[str, Any]] = []
    recovery_state = RecoveryRuntimeState(
        policy=args.recovery_policy,
        failure_stage=args.failure_stage,
        failure_batch_seq=args.failure_batch_seq,
        failure_microbatch_index=args.failure_microbatch_index,
        checkpoint_interval_batches=args.checkpoint_interval_batches,
        worker_rejoin_delay_ms=float(args.worker_rejoin_delay_ms),
    )
    if initial_checkpoint is not None:
        recovery_state.latest_checkpoint = initial_checkpoint
        recovery_state.latest_checkpoint_batch_seq = -1
        recovery_state.latest_checkpoint_version = 1
        recovery_state.latest_checkpoint_bytes = initial_checkpoint_bytes
        recovery_state.initial_checkpoint_bytes = initial_checkpoint_bytes
        recovery_state.initial_checkpoint_ms = initial_checkpoint_ms
    rank_cfg = RankConfig(
        rank,
        rank,
        world_size,
        device,
        devices[rank],
        dtype,
        args.label_smoothing,
        hidden_size,
        vocab_size,
        args.trainable_mode,
        local_param_stats.params,
        local_param_stats.trainable_params,
        local_param_stats.bytes,
        local_param_stats.trainable_bytes,
        memory_ledger["resident_model_param_bytes"],
        memory_ledger["resident_frozen_param_bytes"],
        memory_ledger["base_shard_param_bytes"],
        memory_ledger["base_shard_trainable_param_bytes"],
        memory_ledger["local_readout_param_bytes"],
        memory_ledger["local_readout_trainable_param_bytes"],
        memory_ledger["input_embedding_param_bytes"],
        memory_ledger["input_embedding_trainable_param_bytes"],
    )

    def execute_phase(name: str, mode: str, records: list[dict[str, Any]], manifest_dir: Path) -> dict[str, Any]:
        summary, metrics, timeline = run_phase(
            name=name,
            mode=mode,
            records=records,
            manifest_dir=manifest_dir,
            cfg=rank_cfg,
            stage=pipeline_stage,
            module=module,
            train_schedule=train_schedule,
            eval_schedule=eval_schedule,
            optimizer=optimizer,
            args=args,
            batch_size=batch_size,
            output_dir=args.output_dir,
            stage0_input_embedding=stage0_input_embedding,
            recovery_state=recovery_state if mode == "train" else None,
        )
        phases.append(summary)
        all_metrics.extend(metrics)
        all_timeline.extend(timeline)
        return summary

    def record_validation(optimizer_step: int, train_samples_seen: int, summary: dict[str, Any]) -> None:
        validation_curve.append(
            {
                "optimizer_step": optimizer_step,
                "train_samples_seen": train_samples_seen,
                "phase": summary.get("phase", ""),
                "validation_records": summary.get("completed_records", 0),
                "choice_correct": summary.get("choice_correct", 0),
                "choice_count": summary.get("choice_count", 0),
                "choice_accuracy": summary.get("choice_accuracy", 0.0),
                "avg_loss": summary.get("avg_loss", 0.0),
                "wall_ms": summary.get("wall_ms", 0.0),
            }
        )

    try:
        if validation_records is not None:
            record_validation(
                0,
                0,
                execute_phase(
                    "validation_step000000",
                    "eval",
                    validation_records,
                    args.validation_manifest.parent,
                ),
            )
            interval_records = args.validation_interval_steps * batch_size
            completed_records = 0
            train_segments: list[dict[str, Any]] = []
            while completed_records < len(train_records):
                next_completed = min(completed_records + interval_records, len(train_records))
                segment = train_records[completed_records:next_completed]
                train_segments.append(
                    execute_phase(
                        f"train_to_{next_completed:06d}",
                        "train",
                        segment,
                        args.train_manifest.parent,
                    )
                )
                completed_records = next_completed
                record_validation(
                    completed_records // batch_size,
                    completed_records,
                    execute_phase(
                        f"validation_step{completed_records // batch_size:06d}",
                        "eval",
                        validation_records,
                        args.validation_manifest.parent,
                    ),
                )
            phases.append(aggregate_train_segments(train_segments))
        else:
            if not args.skip_eval_before:
                execute_phase("eval_before", "eval", eval_records, args.eval_manifest.parent)
            execute_phase("train", "train", train_records, args.train_manifest.parent)

        if not args.skip_eval_after:
            execute_phase("eval_after", "eval", eval_records, args.eval_manifest.parent)
    finally:
        write_csv(args.output_dir / f"rank{rank}_stage_metrics.csv", metric_fieldnames(), all_metrics)
        write_csv(args.output_dir / f"rank{rank}_timeline_events.csv", timeline_fieldnames(), all_timeline)
        if args.trainable_mode == "lora":
            save_lora_state(module, args.output_dir / f"stage{rank}_lora_state.pt")
        dist.barrier()
        if rank == 0:
            merge_rank_csvs(
                args.output_dir,
                prefix="stage_metrics",
                fieldnames=metric_fieldnames(),
                world_size=world_size,
                merged_name="stage_metrics.csv",
            )
            merge_rank_csvs(
                args.output_dir,
                prefix="timeline_events",
                fieldnames=timeline_fieldnames(),
                world_size=world_size,
                merged_name="timeline_events.csv",
            )
            if validation_curve:
                write_csv(
                    args.output_dir / "validation_curve.csv",
                    validation_curve_fieldnames(),
                    validation_curve,
                )

    if rank == 0:
        phase_by_name = {phase["phase"]: phase for phase in phases}
        main_phase = phase_by_name.get("eval_after") or phase_by_name.get("train") or (phases[-1] if phases else {})
        train_phase = phase_by_name.get("train", {}) if isinstance(phase_by_name.get("train", {}), dict) else {}
        recovery_rank_summaries = train_phase.get("recovery_rank_summaries", []) if isinstance(train_phase, dict) else []
        failure_rank_summary = next(
            (
                item
                for item in recovery_rank_summaries
                if isinstance(item, dict) and int(item.get("stage_id", -1)) == int(args.failure_stage or -1)
            ),
            {},
        )
        terminal_rank_summary = next(
            (
                item
                for item in recovery_rank_summaries
                if isinstance(item, dict) and int(item.get("stage_id", -1)) == world_size - 1
            ),
            {},
        )
        failure_detected_ms = float(failure_rank_summary.get("failure_detected_ms") or 0.0)
        recovery_unit_done_ms = float(terminal_rank_summary.get("recovery_unit_done_ms") or 0.0)
        replay_start_ms = float(failure_rank_summary.get("replay_start_ms") or 0.0)
        replay_done_ms = float(terminal_rank_summary.get("replay_done_ms") or 0.0)
        recovery_latency = {}
        if failure_detected_ms > 0 and recovery_unit_done_ms > 0:
            failure_batch_seq = failure_rank_summary.get("failure_batch_seq")
            batch_start_seq = (
                int(failure_batch_seq) * batch_size
                if failure_batch_seq not in (None, "")
                else None
            )
            batch_end_seq = (
                batch_start_seq + batch_size - 1
                if batch_start_seq is not None
                else None
            )
            recovery_latency = {
                "failure_detected_ms": failure_detected_ms,
                "replay_start_ms": replay_start_ms or None,
                "replay_done_ms": replay_done_ms or None,
                "recovery_unit_done_ms": recovery_unit_done_ms,
                "recovery_unit_latency_ms": recovery_unit_done_ms - failure_detected_ms,
                "recovery_scope": failure_rank_summary.get("recovery_scope", ""),
                "recovery_records": int(failure_rank_summary.get("recovery_records", 0)),
                "replayed_batches": int(failure_rank_summary.get("replayed_batches", 0)),
                "replayed_records": int(failure_rank_summary.get("replayed_records", 0)),
                "failure_stage": failure_rank_summary.get("failure_stage"),
                "failure_batch_seq": failure_batch_seq,
                "failure_microbatch_index": failure_rank_summary.get("failure_microbatch_index"),
                "failure_event": failure_rank_summary.get("failure_event", "batch_boundary"),
                "window_start_seq": batch_start_seq,
                "window_end_seq": batch_end_seq,
                "failure_position_in_window": failure_rank_summary.get("failure_microbatch_index"),
                "checkpoint_interval_batches": failure_rank_summary.get("checkpoint_interval_batches"),
                "global_checkpoint_restore_bytes": int(
                    failure_rank_summary.get("checkpoint_restore_bytes_total", 0)
                ),
                "global_window_committed": bool(terminal_rank_summary.get("global_window_committed", False)),
                "timing_boundary": (
                    "failure_detected_ms=target backward event on injected stage; "
                    "recovery_unit_done_ms=global batch commit after restore/replay"
                    if args.failure_microbatch_index is not None
                    else (
                        "failure_detected_ms=injected batch-boundary interruption on target stage; "
                        "recovery_unit_done_ms=terminal completion of interrupted batch after replay"
                    )
                ),
            }
        if resolved_pipeline_schedule == "gpipe":
            phase_metrics = train_phase.get("pipeline_phase_metrics")
            if isinstance(phase_metrics, dict):
                phase_metrics.update(
                    {
                        "phase_semantics": "gpipe_batch_transaction_fill_drain",
                        "phase_alignment_note": (
                            "Shared fill/drain report fields follow native GPipe "
                            "all-forward/all-backward batch-transaction semantics."
                        ),
                    }
                )

        summary = {
            "runner": f"{resolved_pipeline_schedule}_lora_pipeline",
            "transport": "nccl-pipeline",
            "transport_details": (
                "torch.distributed.pipelining.Schedule1F1B"
                if resolved_pipeline_schedule == "1f1b"
                else "torch.distributed.pipelining.ScheduleGPipe"
            ),
            "pipeline_schedule": resolved_pipeline_schedule,
            "pipeline_phase_semantics": (
                "1f1b_batch_transaction_fill_steady_drain"
                if resolved_pipeline_schedule == "1f1b"
                else "gpipe_batch_transaction_fill_drain"
            ),
            "pipeline_phase_alignment_note": (
                "Shared fill/drain report fields follow native 1F1B "
                "batch-transaction pipeline fill/steady/drain semantics."
                if resolved_pipeline_schedule == "1f1b"
                else (
                    "Shared fill/drain report fields follow native GPipe "
                    "all-forward/all-backward batch-transaction semantics."
                )
            ),
            "model_name": args.model_name,
            "resolved_model": resolved_model,
            "num_chunks": args.num_chunks,
            "stage_devices": devices,
            "microbatches": args.microbatches,
            "batch_size": batch_size,
            "physical_request_batch": batch_size // args.microbatches,
            "effective_optimizer_batch": batch_size,
            "gradient_accumulation_steps": 1,
            "stage0_input": stage0_tensor_name(base_train_records[0]),
            "activation_tracking_enabled": args.track_activation_memory,
            "gc_interval_batches": args.gc_interval_batches,
            "train_epochs": args.train_epochs,
            "unique_train_records": len(base_train_records),
            "train_records": len(train_records),
            "eval_records": len(eval_records),
            "validation_records": len(validation_records) if validation_records is not None else 0,
            "validation_interval_steps": args.validation_interval_steps,
            "validation_curve_csv": (
                str(args.output_dir / "validation_curve.csv") if validation_curve else ""
            ),
            "offline_window": {
                "stage_id": args.offline_stage,
                "start_seq": args.offline_start_seq,
                "end_seq": args.offline_end_seq,
                "semantics": (
                    f"strict_batch_drop_before_{resolved_pipeline_schedule}"
                    if args.recovery_policy == "strict_skip"
                    else "batch_boundary_failure_for_exact_bp_recovery_control"
                ),
            },
            "transient_dropout": {
                "mask_path": (
                    str(args.transient_dropout_mask.resolve())
                    if args.transient_dropout_mask is not None
                    else ""
                ),
                "policy": args.transient_dropout_policy,
                "window_size": batch_size,
                "offline_windows_by_stage": {
                    str(stage_id): sorted(windows)
                    for stage_id, windows in sorted(args.transient_offline_windows.items())
                },
                "stage_event_counts": {
                    str(stage_id): len(windows)
                    for stage_id, windows in sorted(args.transient_offline_windows.items())
                },
            },
            "recovery_baseline": {
                "policy": args.recovery_policy,
                "failure_stage": args.failure_stage,
                "failure_batch_seq": args.failure_batch_seq,
                "failure_microbatch_index": args.failure_microbatch_index,
                "checkpoint_interval_batches": args.checkpoint_interval_batches,
                "worker_rejoin_delay_ms": float(args.worker_rejoin_delay_ms),
                "semantics": (
                    "historical_strict_no_recovery_control"
                    if args.recovery_policy == "strict_skip"
                    else (
                        "cooperative_backward_microbatch_global_window_restart_exact_bp_control"
                        if args.failure_microbatch_index is not None
                        else "batch_boundary_checkpoint_restart_exact_bp_control"
                    )
                ),
                "notes": (
                    (
                        "The injected rank marks a target backward microbatch inside one global 1F1B batch. "
                        "The batch is treated as uncommitted, gradients are discarded before optimizer.step, "
                        "and restart_from_last_commit restores the latest committed checkpoint before replay."
                    )
                    if args.failure_microbatch_index is not None
                    else (
                        "Interrupted in-flight batches are not recovered as valid exact-BP updates. "
                        "restart_from_last_commit restores the latest committed batch-boundary snapshot and replays "
                        "later committed batches before continuing."
                    )
                ),
            },
            "learning_rate": args.learning_rate,
            "optimizer": args.optimizer,
            "label_smoothing": args.label_smoothing,
            "trainable_mode": args.trainable_mode,
            "completed_records": int(train_phase.get("completed_records", 0)),
            "skipped_records": int(train_phase.get("skipped_records", 0)),
            "optimizer_steps": int(train_phase.get("optimizer_steps", 0)),
            "lora": {
                "rank": args.lora_rank,
                "alpha": args.lora_alpha,
                "targets": args.lora_targets,
                "init_std": args.lora_init_std,
                "init_seed": args.lora_init_seed,
                "initialization_fingerprint": lora_init_fingerprint,
                "modules": trainable_setup.lora_modules,
                "trainable_params": trainable_setup.trainable_params,
            },
            "dtype": args.dtype,
            "seed": args.seed,
            "completed": int(main_phase.get("rows", 0)),
            "failed": 0,
            "choice_correct": int(main_phase.get("choice_correct", 0)),
            "choice_count": int(main_phase.get("choice_count", 0)),
            "choice_accuracy": float(main_phase.get("choice_accuracy", 0.0)),
            "avg_loss": float(main_phase.get("avg_loss", 0.0)),
            "wall_ms": float(main_phase.get("wall_ms", 0.0)),
            "throughput_per_s": float(main_phase.get("throughput_per_s", 0.0)),
            "workers": [
                {"rank": index, "stage_id": index, "device": devices[index]}
                for index in range(world_size)
            ],
            "stage_lora_state_files": (
                {
                    str(index): str(args.output_dir / f"stage{index}_lora_state.pt")
                    for index in range(world_size)
                }
                if args.trainable_mode == "lora"
                else {}
            ),
            "phases": phases,
            "stage_metrics_csv": str(args.output_dir / "stage_metrics.csv"),
            "timeline_events_csv": str(args.output_dir / "timeline_events.csv"),
            "recovery_rank_summaries": [phase.get("recovery", {}) for phase in phases if phase.get("recovery")],
            "recovery_latency": recovery_latency,
        }
        if "eval_before" in phase_by_name and "eval_after" in phase_by_name:
            summary["delta"] = {
                "choice_accuracy": phase_by_name["eval_after"]["choice_accuracy"]
                - phase_by_name["eval_before"]["choice_accuracy"],
                "avg_loss": phase_by_name["eval_after"]["avg_loss"] - phase_by_name["eval_before"]["avg_loss"],
            }
        train_phase = phase_by_name.get("train")
        if isinstance(train_phase, dict) and isinstance(train_phase.get("pipeline_phase_metrics"), dict):
            summary["pipeline_phase_metrics"] = train_phase["pipeline_phase_metrics"]
        summary_path = args.output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
        print(f"Wrote {summary_path}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
