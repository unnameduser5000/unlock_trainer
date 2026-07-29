#!/usr/bin/env python3
"""Orchestrated BP-free stage runtime with scheduling and recovery policies.

This is the server control-plane runtime, not the fixed FIFO phone runner or the
fixed torch.distributed send/recv runner.

The architecture is:

    central scheduler
        dispatches StageTask(request_id, stage_id, attempt, input_state)
            to a pool of stage workers

    stage worker
        owns one stage chunk on one GPU
        runs real PyTorch local forward/loss/backward/optimizer
        returns either a boundary tensor or a failure event

The goal is to prototype aggressive scheduling and recovery policies on a
server where runs are faster and failures can be injected deterministically.

Boundary tensors currently pass through CPU multiprocessing queues on purpose:
the scheduler needs to cache, retry, and reroute them. This lab is about
correctness and policy exploration first; a faster transport can replace the
queue layer after the policy stabilizes.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import random
import time
import traceback
from collections import defaultdict, deque
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM

from sg_exe_trainer.metrics.activation_memory import SavedTensorTracker
from sg_exe_trainer.artifacts.lora_export import stage_lora_export_payload
from sg_exe_trainer.tasks.label_experiment import (
    label_choice_details,
    label_choice_metrics,
    lora_parameter_fingerprint,
    load_tensor,
    one_token_choice_ids,
    parse_train_chunks,
    read_manifest,
    resolve_dtype,
    resolve_model_name,
    stage0_tensor_name,
)
from sg_exe_trainer.runtime.bpfree.model_runtime import (
    build_optimizer,
    build_stage_chunk,
    normalize_belief_transport_mode,
    tensor_to_cpu,
)
try:
    from sg_exe_trainer.common.trainable_modes import configure_model_trainable, module_param_stats, optimizer_state_nbytes
except ModuleNotFoundError:
    from sg_exe_trainer.common.trainable_modes import configure_model_trainable, module_param_stats, optimizer_state_nbytes


@dataclass(frozen=True)
class WorkerSpec:
    worker_id: str
    stage_id: int
    device: str


@dataclass
class FailureRule:
    mode: str
    random_rate: float
    fail_stage: Optional[int]
    fail_seq: Optional[int]
    fail_attempt: int
    fail_point: str
    delay_ms: float
    seed: int
    offline_stage: Optional[int]
    offline_start_seq: Optional[int]
    offline_end_seq: Optional[int]
    transient_mask_path: str
    transient_window_size: int
    transient_offline_windows: dict[int, frozenset[int]] = field(repr=False)


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
        raise ValueError(
            "Transient dropout mask window_size must match --gradient_accumulation_steps"
        )
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


@dataclass
class LabConfig:
    resolved_model: str
    num_chunks: int
    train_chunks: set[int]
    trainable_mode: str
    dtype_name: str
    belief_transport_mode: str
    alpha: float
    label_smoothing: float
    lora_rank: int
    lora_alpha: float
    lora_targets: str
    lora_init_std: float
    lora_init_seed: Optional[int]
    local_readout_adapter_bottleneck: int
    local_readout_adapter_stages: str
    learning_rate: Optional[float]
    grad_clip: float
    gradient_accumulation_steps: int
    optimizer: str
    sgd_momentum: float
    sgd_dampening: float
    sgd_weight_decay: float
    sgd_nesterov: bool
    stage_update_policy: str
    stage_train_strides: dict[int, int]
    stage_update_queue_thresholds: dict[int, int]
    seed: int
    progress_interval: int


def validate_topology(worker_specs: list[WorkerSpec], num_chunks: int, topology: str) -> None:
    counts: dict[int, int] = defaultdict(int)
    for spec in worker_specs:
        counts[spec.stage_id] += 1

    missing = [stage_id for stage_id in range(num_chunks) if counts[stage_id] == 0]
    if missing:
        raise ValueError(f"No worker configured for stages: {missing}")

    if topology == "phone_fixed":
        replicated = {
            stage_id: count
            for stage_id, count in sorted(counts.items())
            if count != 1
        }
        if replicated:
            raise ValueError(
                "phone_fixed topology requires exactly one worker per stage; "
                f"got {replicated}. Use --topology worker_pool for replica experiments."
            )
    elif topology != "worker_pool":
        raise ValueError(f"Unsupported topology: {topology}")


class FailureInjector:
    def __init__(self, rule: FailureRule) -> None:
        self.rule = rule
        self.random = random.Random(rule.seed)
        self.triggered_once: set[tuple[int, int, int]] = set()

    def offline_active(self, *, seq: int, stage_id: int, attempt: int) -> bool:
        contiguous_outage = (
            self.rule.offline_stage is not None
            and self.rule.offline_start_seq is not None
            and self.rule.offline_end_seq is not None
            and stage_id == self.rule.offline_stage
            and self.rule.offline_start_seq <= seq < self.rule.offline_end_seq
        )
        transient_outage = False
        if attempt == 0 and self.rule.transient_window_size > 0:
            window_id = seq // self.rule.transient_window_size
            transient_outage = window_id in self.rule.transient_offline_windows.get(
                stage_id, frozenset()
            )
        return contiguous_outage or transient_outage

    def choose_fail_point(self, *, seq: int, stage_id: int, attempt: int) -> str:
        # This is an availability interval, not a one-off injected exception.
        if self.offline_active(seq=seq, stage_id=stage_id, attempt=attempt):
            return "offline_before_execute"
        if self.rule.mode == "none":
            return "none"
        if self.rule.fail_stage is not None and stage_id != self.rule.fail_stage:
            return "none"
        if self.rule.fail_seq is not None and seq != self.rule.fail_seq:
            return "none"
        if self.rule.fail_attempt >= 0 and attempt != self.rule.fail_attempt:
            return "none"

        if self.rule.mode == "once":
            key = (seq, stage_id, attempt)
            if key in self.triggered_once:
                return "none"
            self.triggered_once.add(key)
            return self.rule.fail_point

        if self.rule.mode == "random":
            return self.rule.fail_point if self.random.random() < self.rule.random_rate else "none"

        raise ValueError(f"Unsupported failure mode: {self.rule.mode}")


def parse_worker_specs(raw: str, num_chunks: int, stage_devices: Optional[str]) -> list[WorkerSpec]:
    if raw:
        specs: list[WorkerSpec] = []
        for index, item in enumerate(raw.split(",")):
            item = item.strip()
            if not item:
                continue
            if ":" not in item:
                raise ValueError(f"Worker spec must be STAGE:DEVICE, got {item!r}")
            stage_raw, device = item.split(":", 1)
            stage_id = int(stage_raw)
            if stage_id < 0 or stage_id >= num_chunks:
                raise ValueError(f"Invalid stage_id in worker spec: {item!r}")
            specs.append(WorkerSpec(worker_id=f"w{index}-s{stage_id}", stage_id=stage_id, device=device))
        if not specs:
            raise ValueError("--workers did not contain any worker specs.")
        return specs

    if not stage_devices:
        raise ValueError("Provide either --workers or --stage_devices.")
    devices = [item.strip() for item in stage_devices.split(",") if item.strip()]
    if len(devices) != num_chunks:
        raise ValueError(f"--stage_devices must contain {num_chunks} devices.")
    return [
        WorkerSpec(worker_id=f"w{stage_id}-s{stage_id}", stage_id=stage_id, device=device)
        for stage_id, device in enumerate(devices)
    ]


def parse_stage_train_strides(raw: str, num_chunks: int) -> dict[int, int]:
    strides = {stage_id: 1 for stage_id in range(num_chunks)}
    if not raw:
        return strides
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Stage train stride must be STAGE:STRIDE, got {item!r}")
        stage_raw, stride_raw = item.split(":", 1)
        stage_id = int(stage_raw)
        stride = int(stride_raw)
        if stage_id < 0 or stage_id >= num_chunks:
            raise ValueError(f"Invalid stage id in --stage_train_strides: {stage_id}")
        if stride <= 0:
            raise ValueError(f"Stage train stride must be positive, got {stride}")
        strides[stage_id] = stride
    return strides


def parse_stage_int_map(raw: str, num_chunks: int, *, name: str, minimum: int = 0) -> dict[int, int]:
    values: dict[int, int] = {}
    if not raw:
        return values
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"{name} must be STAGE:VALUE, got {item!r}")
        stage_raw, value_raw = item.split(":", 1)
        stage_id = int(stage_raw)
        value = int(value_raw)
        if stage_id < 0 or stage_id >= num_chunks:
            raise ValueError(f"Invalid stage id in {name}: {stage_id}")
        if value < minimum:
            raise ValueError(f"{name} value must be >= {minimum}, got {value}")
        values[stage_id] = value
    return values


def load_initial_state(record: dict[str, Any], manifest_dir: Path) -> dict[str, Any]:
    tensors = record["tensors"]
    stage0_kind = stage0_tensor_name(record)
    return {
        "hidden": load_tensor(manifest_dir, tensors["hidden_states"]) if stage0_kind == "hidden_states" else None,
        "input_ids": load_tensor(manifest_dir, tensors["input_ids"]) if stage0_kind == "input_ids" else None,
        "attention_mask": load_tensor(manifest_dir, tensors["attention_mask"]),
        "position_ids": load_tensor(manifest_dir, tensors["position_ids"]),
        "labels": load_tensor(manifest_dir, tensors["labels"]),
        "prev_log_probs": None,
    }


def move_state_to_device(
    state: dict[str, Any],
    device: torch.device,
    compute_dtype: Optional[torch.dtype] = None,
) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key in ("hidden", "input_ids", "attention_mask", "position_ids", "labels", "prev_log_probs"):
        value = state.get(key)
        if not isinstance(value, torch.Tensor):
            moved[key] = None
        elif key in {"hidden", "attention_mask"} and compute_dtype is not None:
            moved[key] = value.to(device=device, dtype=compute_dtype, non_blocking=False)
        elif key == "input_ids":
            moved[key] = value.to(device=device, dtype=torch.long, non_blocking=False)
        elif key in {"position_ids", "labels"}:
            moved[key] = value.to(device=device, dtype=torch.long, non_blocking=False)
        elif key == "prev_log_probs":
            moved[key] = value.to(device=device, dtype=torch.float32, non_blocking=False)
        else:
            moved[key] = value.to(device=device, non_blocking=False)
    return moved


def state_bytes(state: Optional[dict[str, Any]]) -> int:
    if state is None:
        return 0
    total = 0
    for value in state.values():
        if isinstance(value, torch.Tensor):
            total += int(value.numel() * value.element_size())
    return total


def bytes_to_mib(value: int) -> float:
    return round(value / (1024.0 * 1024.0), 2)


def optional_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    return float(value)


def accumulation_window_id(seq: int, grad_accum_steps: int) -> int:
    return int(seq) // max(1, int(grad_accum_steps))


def recovery_entry_window_id(entry: dict[str, Any], grad_accum_steps: int) -> int:
    window_id = entry.get("window_id")
    if window_id not in (None, ""):
        return int(window_id)
    seqs = entry.get("seqs")
    if seqs:
        return accumulation_window_id(int(seqs[0]), grad_accum_steps)
    seq = entry.get("seq")
    if seq is not None:
        return accumulation_window_id(int(seq), grad_accum_steps)
    return 0


def recovery_entry_seqs(entry: dict[str, Any]) -> list[int]:
    seqs = entry.get("seqs")
    if seqs:
        return [int(seq) for seq in seqs]
    seq = entry.get("seq")
    if seq is None:
        return []
    return [int(seq)]


def build_worker_optimizer(params: list[torch.nn.Parameter], cfg: LabConfig) -> torch.optim.Optimizer:
    return build_optimizer(
        params=params,
        cfg=SimpleNamespace(
            learning_rate=cfg.learning_rate,
            optimizer=cfg.optimizer,
            sgd_momentum=cfg.sgd_momentum,
            sgd_dampening=cfg.sgd_dampening,
            sgd_weight_decay=cfg.sgd_weight_decay,
            sgd_nesterov=cfg.sgd_nesterov,
        ),
    )


def clone_to_cpu(value: Any) -> Any:
    """Clone a nested optimizer payload without retaining CUDA storage."""
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
    """Return the small mutable state needed to move a LoRA stage worker."""
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
    """Restore trainable parameters and optimizer moments onto a warm replica."""
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


def stage_worker_main(
    *,
    spec: WorkerSpec,
    input_queue: mp.Queue,
    result_queue: mp.Queue,
    cfg: LabConfig,
) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.manual_seed(cfg.seed + spec.stage_id)
    np.random.seed(cfg.seed + spec.stage_id)
    device = torch.device(spec.device)
    dtype = resolve_dtype(cfg.dtype_name)

    try:
        if device.type == "cuda":
            torch.cuda.set_device(device)
        print(
            f"[{spec.worker_id}] loading stage={spec.stage_id} device={device} model={cfg.resolved_model}",
            flush=True,
        )
        model = AutoModelForCausalLM.from_pretrained(cfg.resolved_model, torch_dtype=dtype)
        stage0_input_embedding = model.get_input_embeddings() if spec.stage_id == 0 else None
        trainable_setup = configure_model_trainable(
            module=model,
            mode=cfg.trainable_mode,
            lora_targets=cfg.lora_targets,
            lora_rank=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_init_std=cfg.lora_init_std,
            lora_init_seed=cfg.lora_init_seed,
        )
        lora_init_fingerprint = lora_parameter_fingerprint(model)
        chunk = build_stage_chunk(
            model=model,
            stage_id=spec.stage_id,
            num_chunks=cfg.num_chunks,
            belief_transport_mode=cfg.belief_transport_mode,
            alpha=cfg.alpha,
            label_smoothing=cfg.label_smoothing,
            local_readout_adapter_bottleneck=cfg.local_readout_adapter_bottleneck,
            local_readout_adapter_stages=cfg.local_readout_adapter_stages,
        )
        chunk.to(device)
        if stage0_input_embedding is not None:
            stage0_input_embedding.to(device)
            stage0_input_embedding.eval()
            for param in stage0_input_embedding.parameters():
                param.requires_grad = False
        local_params = [param for param in chunk.parameters() if param.requires_grad]
        local_param_stats = module_param_stats(chunk)
        local_readout_adapter = getattr(chunk, "local_readout_adapter", None)
        local_readout_adapter_trainable_params = (
            sum(param.numel() for param in local_readout_adapter.parameters() if param.requires_grad)
            if local_readout_adapter is not None
            else 0
        )
        optimizer = (
            build_worker_optimizer(local_params, cfg)
            if spec.stage_id in cfg.train_chunks
            else None
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        cuda_allocated = int(torch.cuda.memory_allocated(device)) if device.type == "cuda" else 0
        cuda_reserved = int(torch.cuda.memory_reserved(device)) if device.type == "cuda" else 0
        print(
            f"[{spec.worker_id}] ready trainable_mode={trainable_setup.mode} "
            f"lora_modules={trainable_setup.lora_modules} "
            f"all_trainable={trainable_setup.trainable_params} "
            f"frozen={trainable_setup.frozen_params} "
            f"local_trainable={local_param_stats.trainable_params} "
            f"local_readout_adapter_trainable={local_readout_adapter_trainable_params} "
            f"lora_init={lora_init_fingerprint[:12]} "
            f"compute_dtype={chunk.compute_dtype()} cuda_allocated_mib={bytes_to_mib(cuda_allocated)} "
            f"cuda_reserved_mib={bytes_to_mib(cuda_reserved)}",
            flush=True,
        )
        result_queue.put(
            {
                "kind": "worker_ready",
                "worker_id": spec.worker_id,
                "stage_id": spec.stage_id,
                "device": spec.device,
                "compute_dtype": str(chunk.compute_dtype()),
                "trainable_mode": trainable_setup.mode,
                "lora_modules": trainable_setup.lora_modules,
                "all_trainable_params": trainable_setup.trainable_params,
                "all_frozen_params": trainable_setup.frozen_params,
                "lora_init_seed": cfg.lora_init_seed,
                "lora_initialization_fingerprint": lora_init_fingerprint,
                "local_params": local_param_stats.params,
                "local_trainable_params": local_param_stats.trainable_params,
                "local_param_bytes": local_param_stats.bytes,
                "local_trainable_param_bytes": local_param_stats.trainable_bytes,
                "local_readout_adapter_trainable_params": local_readout_adapter_trainable_params,
                "cuda_allocated": cuda_allocated,
                "cuda_reserved": cuda_reserved,
            }
        )

        processed = 0
        grad_accum_count = 0
        grad_accum_steps = max(1, int(cfg.gradient_accumulation_steps))
        optimizer_steps_completed = 0
        pending_window_seqs: list[int] = []
        pending_window_states: list[dict[str, Any]] = []
        activation_tracker = SavedTensorTracker()
        while True:
            task = input_queue.get()
            if task is None:
                result_queue.put({"kind": "worker_stopped", "worker_id": spec.worker_id})
                break
            if task.get("kind") == "export_lora_state":
                export_payload = stage_lora_export_payload(
                    stage_id=spec.stage_id,
                    trainable_mode=trainable_setup.mode,
                    lora_init_seed=cfg.lora_init_seed,
                    lora_initialization_fingerprint=lora_init_fingerprint,
                    module=chunk,
                )
                if local_readout_adapter is not None:
                    export_payload["local_readout_adapter_state"] = clone_to_cpu(
                        local_readout_adapter.state_dict()
                    )
                result_queue.put(
                    {
                        **export_payload,
                        "worker_id": spec.worker_id,
                    }
                )
                continue

            task_started = time.perf_counter()
            worker_start_epoch_ms = time.time() * 1000.0
            worker_queue_ms = (task_started - float(task.get("dispatch_perf", task_started))) * 1000.0
            fail_point = task.get("fail_point", "none")
            if fail_point in {"before_execute", "offline_before_execute"}:
                result_queue.put(
                    failure_result(
                        task=task,
                        spec=spec,
                        message=(
                            "Injected stage offline window before execution."
                            if fail_point == "offline_before_execute"
                            else "Injected failure before execution."
                        ),
                        update_applied=False,
                        started_at=task_started,
                    )
                )
                continue
            if fail_point == "delay_before_execute":
                time.sleep(float(task.get("failure_delay_ms", 0.0)) / 1000.0)

            checkpoint_restore_bytes = 0
            checkpoint_restore_ms = 0.0
            checkpoint_version = task.get("stage_checkpoint_version")
            stage_checkpoint = task.get("stage_checkpoint")
            if stage_checkpoint is not None:
                restore_started = time.perf_counter()
                restore_trainable_checkpoint(
                    chunk=chunk,
                    optimizer=optimizer,
                    checkpoint=stage_checkpoint,
                    device=device,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                checkpoint_restore_ms = (time.perf_counter() - restore_started) * 1000.0
                checkpoint_restore_bytes = payload_nbytes(stage_checkpoint)
                try:
                    restored_checkpoint_version = int(checkpoint_version or 0)
                except (TypeError, ValueError):
                    restored_checkpoint_version = 0
                optimizer_steps_completed = max(optimizer_steps_completed, restored_checkpoint_version)
                grad_accum_count = 0
                pending_window_seqs = []
                pending_window_states = []
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)

            train_this_stage = bool(task.get("stage_train", task["mode"] == "train" and spec.stage_id in cfg.train_chunks))
            state = move_state_to_device(task["input_state"], device, chunk.compute_dtype())
            if state["hidden"] is None:
                if state.get("input_ids") is None:
                    raise RuntimeError("Stage input must contain either hidden or input_ids.")
                if stage0_input_embedding is None or spec.stage_id != 0:
                    raise RuntimeError("Only stage 0 can materialize hidden states from input_ids.")
                with torch.no_grad():
                    state["hidden"] = stage0_input_embedding(state["input_ids"]).detach().to(
                        dtype=chunk.compute_dtype()
                    )
                state["input_ids"] = None
            activation_tracker.configure(
                hidden_size=int(state["hidden"].shape[-1]),
                vocab_size=int(chunk.vocab_size),
            )

            lr = cfg.learning_rate
            if lr is None:
                lr = task["record"].get("learning_rate")
            if train_this_stage:
                assert optimizer is not None
                if lr is not None:
                    for group in optimizer.param_groups:
                        group["lr"] = float(lr)
                if grad_accum_count == 0:
                    optimizer.zero_grad(set_to_none=True)
                    pending_window_seqs = []
                    pending_window_states = []
            chunk.train(train_this_stage)

            execute_started = time.perf_counter()
            optimizer_ms = 0.0
            optimizer_step_applied = False
            catchup_window = task.get("recovery_window")
            activation_tracker.reset()
            hook_context = (
                torch.autograd.graph.saved_tensors_hooks(activation_tracker.pack, activation_tracker.unpack)
                if train_this_stage
                else nullcontext()
            )
            if catchup_window is not None:
                if not train_this_stage:
                    raise RuntimeError("Checkpoint catchup must run as a trainable reconstruction task.")
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                grad_accum_count = 0
                last_loss = None
                next_hidden = None
                next_log_probs = None
                final_state_for_metric = state
                window_input_states = list(catchup_window.get("input_states", []))
                window_records = list(task.get("window_records", []))
                pending_window_seqs = [int(seq) for seq in catchup_window.get("seqs", [])]
                pending_window_states = list(window_input_states)
                expected_committed_update_version = catchup_window.get("committed_update_version")
                if len(window_input_states) != len(window_records):
                    raise RuntimeError(
                        "Catchup window input/record length mismatch: "
                        f"input_states={len(window_input_states)} records={len(window_records)}"
                    )
                if len(pending_window_seqs) != len(window_input_states):
                    raise RuntimeError(
                        "Catchup window seq/input length mismatch: "
                        f"seqs={len(pending_window_seqs)} input_states={len(window_input_states)}"
                    )
                if len(window_input_states) != grad_accum_steps:
                    raise RuntimeError(
                        "Accumulated checkpoint catchup must replay one full accumulation window: "
                        f"got {len(window_input_states)} records, expected {grad_accum_steps}."
                    )
                with torch.set_grad_enabled(train_this_stage), hook_context:
                    for window_state_cpu, window_record in zip(window_input_states, window_records):
                        window_state = move_state_to_device(window_state_cpu, device, chunk.compute_dtype())
                        if window_state["hidden"] is None:
                            if window_state.get("input_ids") is None:
                                raise RuntimeError("Catchup stage input must contain either hidden or input_ids.")
                            if stage0_input_embedding is None or spec.stage_id != 0:
                                raise RuntimeError("Only stage 0 can materialize hidden states from input_ids.")
                            with torch.no_grad():
                                window_state["hidden"] = stage0_input_embedding(window_state["input_ids"]).detach().to(
                                    dtype=chunk.compute_dtype()
                                )
                            window_state["input_ids"] = None
                        choice_ids = (
                            one_token_choice_ids(window_record)
                            if spec.stage_id == cfg.num_chunks - 1 and window_record.get("label_choices")
                            else None
                        )
                        loss, next_hidden, next_log_probs = chunk(
                            hidden_states=window_state["hidden"],
                            attention_mask=window_state["attention_mask"],
                            position_ids=window_state["position_ids"],
                            labels=window_state["labels"],
                            prev_log_probs=window_state["prev_log_probs"],
                            choice_ids=choice_ids,
                        )
                        last_loss = loss
                        final_state_for_metric = window_state
                        if train_this_stage:
                            (loss / grad_accum_steps).backward()
                    if train_this_stage:
                        if cfg.grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(chunk.parameters(), cfg.grad_clip)
                        optimizer_started = time.perf_counter()
                        optimizer.step()
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        optimizer_ms = (time.perf_counter() - optimizer_started) * 1000.0
                        optimizer.zero_grad(set_to_none=True)
                        grad_accum_count = 0
                        optimizer_step_applied = True
                        optimizer_steps_completed += 1
                        if expected_committed_update_version not in (None, ""):
                            expected_version = int(expected_committed_update_version)
                            if optimizer_steps_completed != expected_version:
                                raise RuntimeError(
                                    "Catchup committed update version mismatch after checkpoint restore: "
                                    f"got {optimizer_steps_completed}, expected {expected_version}. "
                                    "This usually means checkpoint version was not synced into "
                                    "optimizer_steps_completed on the promoted worker."
                                )
                state = final_state_for_metric
                if last_loss is None:
                    raise RuntimeError("Catchup window execution produced no loss.")
                loss = last_loss
            else:
                choice_ids = (
                    one_token_choice_ids(task["record"])
                    if spec.stage_id == cfg.num_chunks - 1 and task["record"].get("label_choices")
                    else None
                )
                with torch.set_grad_enabled(train_this_stage), hook_context:
                    loss, next_hidden, next_log_probs = chunk(
                        hidden_states=state["hidden"],
                        attention_mask=state["attention_mask"],
                        position_ids=state["position_ids"],
                        labels=state["labels"],
                        prev_log_probs=state["prev_log_probs"],
                        choice_ids=choice_ids,
                    )
                    if train_this_stage:
                        pending_window_seqs.append(int(task["seq"]))
                        pending_window_states.append(task["input_state"])
                        (loss / grad_accum_steps).backward()
                        assert optimizer is not None
                        grad_accum_count += 1
                        if grad_accum_count >= grad_accum_steps:
                            if cfg.grad_clip > 0:
                                torch.nn.utils.clip_grad_norm_(chunk.parameters(), cfg.grad_clip)
                            optimizer_started = time.perf_counter()
                            optimizer.step()
                            if device.type == "cuda":
                                torch.cuda.synchronize(device)
                            optimizer_ms = (time.perf_counter() - optimizer_started) * 1000.0
                            optimizer.zero_grad(set_to_none=True)
                            grad_accum_count = 0
                            optimizer_step_applied = True
                            optimizer_steps_completed += 1
            activation_stats = activation_tracker.snapshot()
            opt_state_bytes = optimizer_state_nbytes(optimizer)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            execute_ms = (time.perf_counter() - execute_started) * 1000.0

            local_loss = float(loss.detach().cpu().item())
            loss_components = chunk.last_loss_components
            update_applied = bool(optimizer_step_applied)
            stage_checkpoint = None
            checkpoint_captured_bytes = 0
            committed_window = None
            committed_update_version = optimizer_steps_completed if update_applied else None
            if update_applied:
                if not pending_window_seqs:
                    raise RuntimeError("optimizer.step() applied but pending_window_seqs is empty.")
                if train_this_stage and len(pending_window_seqs) != grad_accum_steps:
                    raise RuntimeError(
                        "Committed accumulation window has wrong size: "
                        f"got {len(pending_window_seqs)} seqs, expected {grad_accum_steps}."
                    )
                committed_window = {
                    "stage_id": spec.stage_id,
                    "window_id": accumulation_window_id(
                        pending_window_seqs[0],
                        grad_accum_steps,
                    ) if pending_window_seqs else accumulation_window_id(int(task["seq"]), grad_accum_steps),
                    "committed_update_version": committed_update_version,
                    "seqs": list(pending_window_seqs),
                    "input_states": list(pending_window_states),
                }
            checkpoint_interval = int(task.get("checkpoint_interval", 1) or 1)
            if (
                task.get("checkpoint_requested")
                and update_applied
                and committed_update_version is not None
                and committed_update_version % checkpoint_interval == 0
            ):
                stage_checkpoint, checkpoint_captured_bytes = capture_trainable_checkpoint(chunk, optimizer)
            if fail_point == "delay_after_update":
                time.sleep(float(task.get("failure_delay_ms", 0.0)) / 1000.0)
            if fail_point == "after_update":
                result_queue.put(
                    failure_result(
                        task=task,
                        spec=spec,
                        message="Injected failure after local update; boundary output lost.",
                        update_applied=update_applied,
                        started_at=task_started,
                        local_loss=local_loss,
                    )
                )
                continue

            if spec.stage_id == cfg.num_chunks - 1:
                output_state = None
                final_log_probs = tensor_to_cpu(next_log_probs)
                choice_metrics = chunk.last_choice_metrics
                choice_details = chunk.last_choice_details
            else:
                output_state = {
                    "hidden": tensor_to_cpu(next_hidden),
                    "input_ids": None,
                    "attention_mask": tensor_to_cpu(state["attention_mask"]),
                    "position_ids": tensor_to_cpu(state["position_ids"]),
                    "labels": tensor_to_cpu(state["labels"]),
                    "prev_log_probs": (
                        tensor_to_cpu(next_log_probs)
                        if cfg.belief_transport_mode == "full"
                        else None
                    ),
                }
                final_log_probs = None
                choice_metrics = None
                choice_details = None

            metric = {
                "phase": task["phase"],
                "seq": task["seq"],
                "request_id": task["request_id"],
                "stage_id": spec.stage_id,
                "worker_id": spec.worker_id,
                "device": spec.device,
                "attempt": task["attempt"],
                "mode": task["mode"],
                "train": train_this_stage,
                "update_decision": task.get("update_decision", ""),
                "update_applied": update_applied,
                "optimizer_step_applied": optimizer_step_applied,
                "gradient_accumulation_steps": grad_accum_steps,
                "gradient_accumulation_count": grad_accum_count,
                "optimizer_step": optimizer_steps_completed,
                "local_loss": local_loss,
                "loss_ce": loss_components.get("ce_loss", local_loss),
                "loss_belief_kl": loss_components.get("belief_kl_loss", 0.0),
                "loss_total": loss_components.get("total_loss", local_loss),
                "queue_enter_epoch_ms": float(task.get("queue_enter_epoch_ms", 0.0)),
                "dispatch_epoch_ms": float(task.get("dispatch_epoch_ms", 0.0)),
                "worker_start_epoch_ms": worker_start_epoch_ms,
                "worker_end_epoch_ms": time.time() * 1000.0,
                "scheduler_queue_ms": float(task.get("scheduler_queue_ms", 0.0)),
                "worker_queue_ms": worker_queue_ms,
                "execute_ms": execute_ms,
                "optimizer_ms": optimizer_ms,
                "stage_total_ms": (time.perf_counter() - task_started) * 1000.0,
                "trainable_mode": trainable_setup.mode,
                "local_params": local_param_stats.params,
                "local_trainable_params": local_param_stats.trainable_params,
                "local_param_bytes": local_param_stats.bytes,
                "local_trainable_param_bytes": local_param_stats.trainable_bytes,
                "optimizer_state_bytes": opt_state_bytes,
                "checkpoint_version": checkpoint_version if checkpoint_version is not None else "",
                "checkpoint_captured_bytes": checkpoint_captured_bytes,
                "checkpoint_restore_bytes": checkpoint_restore_bytes,
                "checkpoint_restore_ms": checkpoint_restore_ms,
                "window_id": (
                    committed_window["window_id"]
                    if committed_window is not None
                    else accumulation_window_id(int(task["seq"]), grad_accum_steps)
                ),
                "committed_update_version": committed_update_version if committed_update_version is not None else "",
                "input_state_bytes": state_bytes(task["input_state"]),
                "output_state_bytes": state_bytes(output_state),
                "output_log_probs_bytes": (
                    int(final_log_probs.numel() * final_log_probs.element_size())
                    if isinstance(final_log_probs, torch.Tensor)
                    else 0
                ),
                **activation_stats,
                "result_status": "normal",
                "cuda_peak_memory_allocated": (
                    int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
                ),
                "cuda_peak_memory_reserved": (
                    int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
                ),
            }
            result_queue.put(
                {
                    "kind": "stage_result",
                    "success": True,
                    "task_id": task["task_id"],
                    "seq": task["seq"],
                    "request_id": task["request_id"],
                    "record": task["record"],
                    "stage_id": spec.stage_id,
                    "worker_id": spec.worker_id,
                    "attempt": task["attempt"],
                    "mode": task["mode"],
                    "local_loss": local_loss,
                    "output_state": output_state,
                    "final_log_probs": final_log_probs,
                    "choice_metrics": choice_metrics,
                    "choice_details": choice_details,
                    "labels": tensor_to_cpu(state["labels"]),
                    "metric": metric,
                    "update_applied": update_applied,
                    "stage_checkpoint": stage_checkpoint,
                    "checkpoint_captured_bytes": checkpoint_captured_bytes,
                    "committed_window": committed_window,
                }
            )
            processed += 1
            if cfg.progress_interval > 0 and processed % cfg.progress_interval == 0:
                print(
                    f"[{spec.worker_id}] processed={processed} stage={spec.stage_id} "
                    f"loss={local_loss:.4f}",
                    flush=True,
                )

    except Exception as exc:  # pragma: no cover - server/runtime failure path.
        result_queue.put(
            {
                "kind": "worker_error",
                "worker_id": spec.worker_id,
                "stage_id": spec.stage_id,
                "device": spec.device,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        raise


def failure_result(
    *,
    task: dict[str, Any],
    spec: WorkerSpec,
    message: str,
    update_applied: bool,
    started_at: float,
    worker_start_epoch_ms: Optional[float] = None,
    local_loss: Optional[float] = None,
) -> dict[str, Any]:
    if worker_start_epoch_ms is None:
        worker_start_epoch_ms = time.time() * 1000.0
    return {
        "kind": "stage_result",
        "success": False,
        "task_id": task["task_id"],
        "seq": task["seq"],
        "request_id": task["request_id"],
        "record": task["record"],
        "stage_id": spec.stage_id,
        "worker_id": spec.worker_id,
        "attempt": task["attempt"],
        "mode": task["mode"],
        "message": message,
        "update_applied": update_applied,
        "local_loss": local_loss,
        "metric": {
            "phase": task["phase"],
            "seq": task["seq"],
            "request_id": task["request_id"],
            "stage_id": spec.stage_id,
            "worker_id": spec.worker_id,
            "device": spec.device,
            "attempt": task["attempt"],
            "mode": task["mode"],
            "train": bool(task.get("stage_train", task["mode"] == "train")),
            "update_decision": task.get("update_decision", ""),
            "update_applied": update_applied,
            "optimizer_step_applied": update_applied,
            "gradient_accumulation_steps": "",
            "gradient_accumulation_count": "",
            "local_loss": local_loss if local_loss is not None else "",
            "queue_enter_epoch_ms": float(task.get("queue_enter_epoch_ms", 0.0)),
            "dispatch_epoch_ms": float(task.get("dispatch_epoch_ms", 0.0)),
            "worker_start_epoch_ms": worker_start_epoch_ms,
            "worker_end_epoch_ms": time.time() * 1000.0,
            "scheduler_queue_ms": float(task.get("scheduler_queue_ms", 0.0)),
            "worker_queue_ms": (started_at - float(task.get("dispatch_perf", started_at))) * 1000.0,
            "execute_ms": "",
            "optimizer_ms": "",
            "stage_total_ms": (time.perf_counter() - started_at) * 1000.0,
            "checkpoint_version": task.get("stage_checkpoint_version", ""),
            "checkpoint_captured_bytes": "",
            "checkpoint_restore_bytes": "",
            "checkpoint_restore_ms": "",
            "window_id": accumulation_window_id(
                int(task["seq"]),
                int(task.get("gradient_accumulation_steps", 1) or 1),
            ),
            "committed_update_version": "",
            "input_state_bytes": state_bytes(task["input_state"]),
            "output_state_bytes": 0,
            "output_log_probs_bytes": 0,
            "result_status": "normal",
            "cuda_peak_memory_allocated": "",
            "cuda_peak_memory_reserved": "",
            "failure": message,
        },
    }


def result_fieldnames() -> list[str]:
    return [
        "phase",
        "seq",
        "request_id",
        "dataset_index",
        "response",
        "predicted_response",
        "predicted_token_id",
        "target_token_id",
        "mode",
        "status",
        "loss",
        "choice_correct",
        "choice_count",
        "choice_accuracy",
        "choice_loss",
        "attempts_json",
        "message",
    ]


def metric_fieldnames() -> list[str]:
    return [
        "phase",
        "seq",
        "request_id",
        "stage_id",
        "worker_id",
        "device",
        "attempt",
        "mode",
        "train",
        "update_decision",
        "update_applied",
        "optimizer_step_applied",
        "gradient_accumulation_steps",
        "gradient_accumulation_count",
        "optimizer_step",
        "local_loss",
        "loss_ce",
        "loss_belief_kl",
        "loss_total",
        "queue_enter_epoch_ms",
        "dispatch_epoch_ms",
        "worker_start_epoch_ms",
        "worker_end_epoch_ms",
        "scheduler_queue_ms",
        "worker_queue_ms",
        "execute_ms",
        "optimizer_ms",
        "stage_total_ms",
        "trainable_mode",
        "local_params",
        "local_trainable_params",
        "local_param_bytes",
        "local_trainable_param_bytes",
        "optimizer_state_bytes",
        "checkpoint_version",
        "checkpoint_captured_bytes",
        "checkpoint_restore_bytes",
        "checkpoint_restore_ms",
        "window_id",
        "committed_update_version",
        "input_state_bytes",
        "output_state_bytes",
        "output_log_probs_bytes",
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
        "result_status",
        "cuda_peak_memory_allocated",
        "cuda_peak_memory_reserved",
        "failure",
    ]


def ledger_fieldnames() -> list[str]:
    return [
        "event_seq",
        "event_type",
        "seq",
        "request_id",
        "stage_id",
        "worker_id",
        "attempt",
        "success",
        "update_applied",
        "message",
    ]


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


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


class SchedulerLab:
    def __init__(
        self,
        *,
        records: list[dict[str, Any]],
        eval_records: Optional[list[dict[str, Any]]],
        validation_records: Optional[list[dict[str, Any]]],
        manifest_dir: Path,
        eval_manifest_dir: Optional[Path],
        validation_manifest_dir: Optional[Path],
        worker_specs: list[WorkerSpec],
        cfg: LabConfig,
        output_dir: Path,
        max_inflight: int,
        scheduler_policy: str,
        recovery_policy: str,
        topology: str,
        task_timeout_ms: float,
        timeout_policy: str,
        max_attempts: int,
        failure_injector: FailureInjector,
        standby_worker_ids: set[str],
        worker_rejoin_delay_ms: float,
        checkpoint_interval: int,
        request_prefix: str,
        validation_interval_steps: int,
    ) -> None:
        self.records = records
        self.train_records = records
        self.eval_records = eval_records
        self.validation_records = validation_records
        self.manifest_dir = manifest_dir
        self.train_manifest_dir = manifest_dir
        self.eval_manifest_dir = eval_manifest_dir
        self.validation_manifest_dir = validation_manifest_dir
        self.worker_specs = worker_specs
        self.cfg = cfg
        self.output_dir = output_dir
        self.max_inflight = max_inflight
        self.scheduler_policy = scheduler_policy
        self.recovery_policy = recovery_policy
        self.topology = topology
        self.task_timeout_ms = task_timeout_ms
        self.timeout_policy = timeout_policy
        self.max_attempts = max_attempts
        self.failure_injector = failure_injector
        self.standby_worker_ids = standby_worker_ids
        self.worker_rejoin_delay_ms = worker_rejoin_delay_ms
        self.checkpoint_interval = checkpoint_interval
        self.request_prefix = request_prefix
        self.validation_interval_steps = validation_interval_steps
        self.current_phase = "train"
        self.current_mode = "train"
        self.phase_summaries: list[dict[str, Any]] = []

        self.worker_queues: dict[str, mp.Queue] = {
            spec.worker_id: mp.Queue(maxsize=1) for spec in worker_specs
        }
        self.result_queue: mp.Queue = mp.Queue()
        self.worker_by_id = {spec.worker_id: spec for spec in worker_specs}
        self.workers_by_stage: dict[int, list[WorkerSpec]] = defaultdict(list)
        for spec in worker_specs:
            self.workers_by_stage[spec.stage_id].append(spec)
        for stage_id in range(cfg.num_chunks):
            if not self.workers_by_stage[stage_id]:
                raise ValueError(f"No worker configured for stage {stage_id}.")

        self.processes: list[mp.Process] = []
        self.ready: dict[int, deque[dict[str, Any]]] = {
            stage_id: deque() for stage_id in range(cfg.num_chunks)
        }
        self.worker_busy: dict[str, bool] = {spec.worker_id: False for spec in worker_specs}
        self.inflight_by_worker: dict[str, dict[str, Any]] = {}
        self.dispatched_tasks_by_id: dict[int, dict[str, Any]] = {}
        self.resolved_task_ids: set[int] = set()
        self.boundary_cache: dict[tuple[int, int], dict[str, Any]] = {}
        self.completed: dict[int, dict[str, Any]] = {}
        self.failed: dict[int, dict[str, Any]] = {}
        self.inflight_requests: set[int] = set()
        self.next_record_index = 0
        self.next_task_id = 0
        self.next_event_seq = 0
        self.stage_attempts: dict[tuple[int, int], int] = defaultdict(int)
        self.request_attempts: dict[int, list[dict[str, Any]]] = defaultdict(list)
        self.worker_metric_accum: dict[str, dict[str, Any]] = {}
        self.worker_ready_info: dict[str, dict[str, Any]] = {}
        self.worker_lora_exports: dict[int, dict[str, Any]] = {}
        self.timeout_events = 0
        self.late_results = 0
        self.dropped_late_results = 0
        self.cancelled_timeout_retries = 0
        self.unavailable_workers: set[str] = set()
        self.temporary_unavailable_until: dict[str, float] = {}
        self.rejoin_events: list[dict[str, Any]] = []
        self.stage_checkpoints: dict[int, dict[str, Any]] = {}
        self.stage_checkpoint_versions: dict[int, int] = defaultdict(int)
        self.worker_checkpoint_versions: dict[str, int] = {
            spec.worker_id: 0 for spec in worker_specs
        }
        self.stage_recovery_journal: dict[int, deque[tuple[int, dict[str, Any]]]] = defaultdict(deque)
        self.stage_journal_next_index: dict[int, int] = defaultdict(int)
        self.migration_events: list[dict[str, Any]] = []
        self.recovery_latency_state: dict[str, Any] = {
            "failure_detected_ms": None,
            "replay_or_catchup_start_ms": None,
            "replay_or_catchup_done_ms": None,
            "recovery_unit_done_ms": None,
            "recovery_scope": "",
            "recovery_records": 0,
        }

        self.results_path = output_dir / "scheduler_results.csv"
        self.metrics_path = output_dir / "scheduler_stage_metrics.csv"
        self.ledger_path = output_dir / "scheduler_ledger.csv"

    def start_workers(self) -> None:
        for spec in self.worker_specs:
            process = mp.Process(
                target=stage_worker_main,
                kwargs={
                    "spec": spec,
                    "input_queue": self.worker_queues[spec.worker_id],
                    "result_queue": self.result_queue,
                    "cfg": self.cfg,
                },
                name=f"{spec.worker_id}-{spec.device}",
            )
            process.start()
            self.processes.append(process)

    def stop_workers(self) -> None:
        for spec in self.worker_specs:
            self.worker_queues[spec.worker_id].put(None)
        deadline = time.time() + 120
        for process in self.processes:
            timeout = max(0.0, deadline - time.time())
            process.join(timeout=timeout)
        for process in self.processes:
            if process.is_alive():
                process.terminate()

    def wait_for_workers_ready(self) -> None:
        while len(self.worker_ready_info) < len(self.worker_specs):
            try:
                result = self.result_queue.get(timeout=0.1)
            except queue.Empty:
                self.check_worker_errors()
                continue
            if result.get("kind") == "worker_ready":
                self.worker_ready_info[result["worker_id"]] = result
                print(
                    f"[scheduler] worker_ready {result['worker_id']} "
                    f"stage={result['stage_id']} device={result['device']} "
                    f"compute_dtype={result['compute_dtype']} "
                    f"cuda_allocated_mib={bytes_to_mib(int(result['cuda_allocated']))} "
                    f"cuda_reserved_mib={bytes_to_mib(int(result['cuda_reserved']))}",
                    flush=True,
                )
            elif result.get("kind") == "worker_error":
                raise RuntimeError(f"worker error during startup: {result}")
            else:
                raise RuntimeError(f"Unexpected startup result before all workers ready: {result}")
            self.check_worker_errors()

    def reset_phase_state(
        self,
        *,
        phase: str,
        mode: str,
        records: list[dict[str, Any]],
        manifest_dir: Path,
    ) -> None:
        self.current_phase = phase
        self.current_mode = mode
        self.records = records
        self.manifest_dir = manifest_dir
        self.ready = {stage_id: deque() for stage_id in range(self.cfg.num_chunks)}
        self.inflight_by_worker.clear()
        self.dispatched_tasks_by_id.clear()
        self.resolved_task_ids.clear()
        self.boundary_cache.clear()
        self.completed.clear()
        self.failed.clear()
        self.inflight_requests.clear()
        self.next_record_index = 0
        self.stage_attempts = defaultdict(int)
        self.request_attempts = defaultdict(list)
        for worker_id in self.worker_busy:
            self.worker_busy[worker_id] = False

    def make_task(
        self,
        *,
        seq: int,
        stage_id: int,
        input_state: dict[str, Any],
        attempt: int,
        mode: str = "train",
        preferred_worker_id: Optional[str] = None,
    ) -> dict[str, Any]:
        fail_point = (
            self.failure_injector.choose_fail_point(seq=seq, stage_id=stage_id, attempt=attempt)
            if mode == "train"
            else "none"
        )
        task = {
            "task_id": self.next_task_id,
            "phase": self.current_phase,
            "seq": seq,
            "request_id": f"{self.request_prefix}-{self.current_phase}-{seq:06d}",
            "record": self.records[seq],
            "manifest_dir": str(self.manifest_dir),
            "mode": mode,
            "request_mode": self.current_mode,
            "stage_id": stage_id,
            "attempt": attempt,
            "input_state": input_state,
            "fail_point": fail_point,
            "failure_delay_ms": self.failure_injector.rule.delay_ms,
            "preferred_worker_id": preferred_worker_id,
            "gradient_accumulation_steps": self.cfg.gradient_accumulation_steps,
            "checkpoint_interval": self.checkpoint_interval,
            "window_id": accumulation_window_id(seq, self.cfg.gradient_accumulation_steps),
        }
        task["stage_train"], task["update_decision"] = self.stage_update_decision(
            seq=seq,
            stage_id=stage_id,
            mode=mode,
        )
        closes_accumulation_window = ((seq + 1) % self.cfg.gradient_accumulation_steps) == 0
        committed_update_version = (
            ((seq + 1) // self.cfg.gradient_accumulation_steps)
            if task["stage_train"] and closes_accumulation_window
            else None
        )
        task["committed_update_version_candidate"] = (
            committed_update_version if committed_update_version is not None else ""
        )
        task["checkpoint_requested"] = bool(
            self.recovery_policy == "migrate_from_boundary"
            and self.topology == "worker_pool"
            and mode == "train"
            and stage_id == self.failure_injector.rule.fail_stage
            and task["stage_train"]
            and (
                (
                    self.cfg.gradient_accumulation_steps == 1
                    and seq % self.checkpoint_interval == 0
                )
                or (
                    self.cfg.gradient_accumulation_steps > 1
                    and committed_update_version is not None
                    and committed_update_version % self.checkpoint_interval == 0
                )
            )
        )
        task["checkpoint_journal_enabled"] = bool(
            self.recovery_policy == "migrate_from_boundary"
            and self.topology == "worker_pool"
            and mode == "train"
            and stage_id == self.failure_injector.rule.fail_stage
            and task["stage_train"]
        )
        self.next_task_id += 1
        return task

    def stage_update_decision(self, *, seq: int, stage_id: int, mode: str) -> tuple[bool, str]:
        if mode != "train":
            return False, mode
        if stage_id not in self.cfg.train_chunks:
            return False, "stage_not_trainable"
        stride = self.cfg.stage_train_strides.get(stage_id, 1)
        if stride <= 1:
            return True, "train"
        if seq % stride == 0:
            return True, f"stride_{stride}_train"
        return False, f"stride_{stride}_forward_only"

    def apply_dispatch_update_policy(self, task: dict[str, Any], stage_backlog: int) -> None:
        if task.get("recovery_reconstruction"):
            task["stage_train"] = True
            task["update_decision"] = "checkpoint_catchup"
            return
        base_train, base_decision = self.stage_update_decision(
            seq=task["seq"],
            stage_id=task["stage_id"],
            mode=task["mode"],
        )
        task["stage_train"] = base_train
        task["update_decision"] = base_decision

        if self.cfg.stage_update_policy != "queue_gated":
            return
        if not base_train:
            return
        threshold = self.cfg.stage_update_queue_thresholds.get(task["stage_id"])
        if threshold is None:
            return
        if stage_backlog > threshold:
            task["stage_train"] = False
            task["update_decision"] = f"queue_gt_{threshold}_forward_only"
        else:
            task["update_decision"] = f"queue_le_{threshold}_train"

    def enqueue_task(self, task: dict[str, Any], recovery: bool = False, front: bool = False) -> None:
        stage_id = task["stage_id"]
        task["queue_enter_perf"] = time.perf_counter()
        task["queue_enter_epoch_ms"] = time.time() * 1000.0
        if front or (recovery and self.scheduler_policy == "recovery_first"):
            self.ready[stage_id].appendleft(task)
        else:
            self.ready[stage_id].append(task)

    def admit_requests(self) -> None:
        while (
            self.next_record_index < len(self.records)
            and len(self.inflight_requests) < self.max_inflight
        ):
            seq = self.next_record_index
            self.next_record_index += 1
            self.inflight_requests.add(seq)
            initial_state = load_initial_state(self.records[seq], self.manifest_dir)
            task = self.make_task(
                seq=seq,
                stage_id=0,
                input_state=initial_state,
                attempt=0,
                mode=self.current_mode,
            )
            self.enqueue_task(task)
            self.write_ledger(
                event_type="admit",
                task=task,
                worker_id="",
                success=True,
                update_applied=False,
                message="request admitted",
            )

    def dispatch_ready(self) -> None:
        for stage_id in range(self.cfg.num_chunks):
            if not self.ready[stage_id]:
                continue
            for spec in self.workers_by_stage[stage_id]:
                if self.worker_busy[spec.worker_id] or not self.worker_is_available(spec.worker_id):
                    continue
                if spec.worker_id in self.standby_worker_ids:
                    if not any(task.get("preferred_worker_id") == spec.worker_id for task in self.ready[stage_id]):
                        continue
                if not self.ready[stage_id]:
                    break
                task = self.pop_dispatchable_task(stage_id, spec.worker_id)
                if task is None:
                    continue
                self.apply_dispatch_update_policy(task, stage_backlog=len(self.ready[stage_id]))
                self.attach_checkpoint_if_needed(task, worker_id=spec.worker_id)
                dispatch_perf = time.perf_counter()
                task["dispatch_perf"] = dispatch_perf
                task["dispatch_epoch_ms"] = time.time() * 1000.0
                task["scheduler_queue_ms"] = (
                    dispatch_perf - float(task.get("queue_enter_perf", dispatch_perf))
                ) * 1000.0
                self.worker_busy[spec.worker_id] = True
                self.inflight_by_worker[spec.worker_id] = task
                self.dispatched_tasks_by_id[task["task_id"]] = task
                self.worker_queues[spec.worker_id].put(task)
                self.write_ledger(
                    event_type="dispatch",
                    task=task,
                    worker_id=spec.worker_id,
                    success=True,
                    update_applied=False,
                    message=f"dispatched to {spec.device} mode={task['mode']}",
                )

    def pop_dispatchable_task(self, stage_id: int, worker_id: str) -> Optional[dict[str, Any]]:
        queue_for_stage = self.ready[stage_id]
        for index, task in enumerate(queue_for_stage):
            preferred_worker_id = task.get("preferred_worker_id")
            if preferred_worker_id not in (None, "", worker_id):
                continue
            if index == 0:
                return queue_for_stage.popleft()
            queue_for_stage.rotate(-index)
            selected = queue_for_stage.popleft()
            queue_for_stage.rotate(index)
            return selected
        return None

    def attach_checkpoint_if_needed(self, task: dict[str, Any], *, worker_id: str) -> None:
        stage_id = int(task["stage_id"])
        checkpoint = self.stage_checkpoints.get(stage_id)
        if checkpoint is None:
            return
        checkpoint_version = int(checkpoint["version"])
        if self.worker_checkpoint_versions.get(worker_id, 0) >= checkpoint_version:
            return
        task["stage_checkpoint"] = checkpoint["payload"]
        task["stage_checkpoint_version"] = checkpoint_version
        self.worker_checkpoint_versions[worker_id] = checkpoint_version

    def worker_is_available(self, worker_id: str) -> bool:
        if worker_id in self.unavailable_workers:
            return False
        unavailable_until = self.temporary_unavailable_until.get(worker_id)
        if unavailable_until is None:
            return True
        if time.perf_counter() < unavailable_until:
            return False
        del self.temporary_unavailable_until[worker_id]
        return True

    def store_stage_checkpoint(self, result: dict[str, Any]) -> None:
        payload = result.get("stage_checkpoint")
        if payload is None:
            return
        stage_id = int(result["stage_id"])
        if self.cfg.gradient_accumulation_steps == 1:
            version = self.stage_checkpoint_versions[stage_id] + 1
            self.stage_checkpoint_versions[stage_id] = version
            self.stage_checkpoints[stage_id] = {
                "version": version,
                "source_worker_id": result["worker_id"],
                "payload": payload,
                "bytes": int(result.get("checkpoint_captured_bytes", 0)),
                "journal_index": self.stage_journal_next_index[stage_id],
            }
            self.worker_checkpoint_versions[result["worker_id"]] = version
            journal = self.stage_recovery_journal[stage_id]
            while journal and journal[0][0] <= self.stage_journal_next_index[stage_id]:
                journal.popleft()
            return
        committed_window = result.get("committed_window") or {}
        version = int(
            committed_window.get("committed_update_version")
            or (self.stage_checkpoint_versions[stage_id] + 1)
        )
        current_version = int(self.stage_checkpoint_versions.get(stage_id, 0))
        if version < current_version:
            raise RuntimeError(
                "Refusing to move stage checkpoint version backwards: "
                f"stage={stage_id} current={current_version} incoming={version}"
            )
        self.stage_checkpoint_versions[stage_id] = version
        self.stage_journal_next_index[stage_id] = max(
            int(self.stage_journal_next_index.get(stage_id, 0)),
            version,
        )
        self.stage_checkpoints[stage_id] = {
            "version": version,
            "source_worker_id": result["worker_id"],
            "payload": payload,
            "bytes": int(result.get("checkpoint_captured_bytes", 0)),
            "journal_index": version,
            "window_id": recovery_entry_window_id(committed_window, self.cfg.gradient_accumulation_steps),
            "seqs": list(committed_window.get("seqs", [])),
        }
        self.worker_checkpoint_versions[result["worker_id"]] = version
        journal = self.stage_recovery_journal[stage_id]
        while journal and journal[0][0] <= version:
            journal.popleft()

    def record_recovery_journal(self, task: dict[str, Any], result: dict[str, Any]) -> None:
        if not task.get("checkpoint_journal_enabled") or not result.get("update_applied"):
            return
        if self.cfg.gradient_accumulation_steps == 1:
            stage_id = int(task["stage_id"])
            next_index = self.stage_journal_next_index[stage_id] + 1
            self.stage_journal_next_index[stage_id] = next_index
            self.stage_recovery_journal[stage_id].append(
                (
                    next_index,
                    {
                        "seq": int(task["seq"]),
                        "input_state": task["input_state"],
                    },
                )
            )
            return
        committed_window = result.get("committed_window")
        if not committed_window:
            return
        stage_id = int(task["stage_id"])
        committed_update_version = int(committed_window["committed_update_version"])
        seqs = recovery_entry_seqs(committed_window)
        input_states = list(committed_window["input_states"])
        if len(seqs) != self.cfg.gradient_accumulation_steps:
            raise RuntimeError(
                "Recovery journal only accepts full committed accumulation windows: "
                f"stage={stage_id} version={committed_update_version} "
                f"seqs={len(seqs)} expected={self.cfg.gradient_accumulation_steps}"
            )
        if len(input_states) != len(seqs):
            raise RuntimeError(
                "Recovery journal seq/input length mismatch: "
                f"stage={stage_id} version={committed_update_version} "
                f"seqs={len(seqs)} input_states={len(input_states)}"
            )
        self.stage_journal_next_index[stage_id] = committed_update_version
        self.stage_recovery_journal[stage_id].append(
            (
                committed_update_version,
                {
                    "stage_id": stage_id,
                    "window_id": recovery_entry_window_id(committed_window, self.cfg.gradient_accumulation_steps),
                    "committed_update_version": committed_update_version,
                    "seqs": seqs,
                    "input_states": input_states,
                },
            )
        )

    def make_checkpoint_catchup_task(
        self,
        *,
        stage_id: int,
        entry: dict[str, Any],
        preferred_worker_id: str,
    ) -> dict[str, Any]:
        if self.cfg.gradient_accumulation_steps == 1:
            task = self.make_task(
                seq=int(entry["seq"]),
                stage_id=stage_id,
                input_state=entry["input_state"],
                attempt=0,
                mode="catchup",
                preferred_worker_id=preferred_worker_id,
            )
            task["stage_train"] = True
            task["update_decision"] = "checkpoint_catchup"
            task["checkpoint_requested"] = False
            task["checkpoint_journal_enabled"] = False
            task["recovery_reconstruction"] = True
            return task
        seqs = [int(seq) for seq in entry["seqs"]]
        input_states = list(entry["input_states"])
        if len(seqs) != self.cfg.gradient_accumulation_steps:
            raise RuntimeError(
                "Accumulated catchup task requires a full accumulation window: "
                f"stage={stage_id} window={recovery_entry_window_id(entry, self.cfg.gradient_accumulation_steps)} "
                f"seqs={len(seqs)} expected={self.cfg.gradient_accumulation_steps}"
            )
        if len(input_states) != len(seqs):
            raise RuntimeError(
                "Accumulated catchup task seq/input mismatch: "
                f"stage={stage_id} window={recovery_entry_window_id(entry, self.cfg.gradient_accumulation_steps)} "
                f"seqs={len(seqs)} input_states={len(input_states)}"
            )
        task = self.make_task(
            seq=seqs[0],
            stage_id=stage_id,
            input_state=input_states[0],
            attempt=0,
            mode="catchup",
            preferred_worker_id=preferred_worker_id,
        )
        task["stage_train"] = True
        task["update_decision"] = "checkpoint_catchup"
        task["checkpoint_requested"] = False
        task["checkpoint_journal_enabled"] = False
        task["recovery_reconstruction"] = True
        task["recovery_window"] = {
            "stage_id": stage_id,
            "window_id": recovery_entry_window_id(entry, self.cfg.gradient_accumulation_steps),
            "committed_update_version": int(entry["committed_update_version"]),
            "seqs": seqs,
            "input_states": input_states,
        }
        task["window_records"] = [self.records[seq] for seq in seqs]
        return task

    def healthy_fallback_worker(self, *, stage_id: int, failed_worker_id: str) -> Optional[WorkerSpec]:
        for spec in self.workers_by_stage[stage_id]:
            if spec.worker_id == failed_worker_id:
                continue
            if not self.worker_is_available(spec.worker_id):
                continue
            if self.worker_busy[spec.worker_id]:
                continue
            return spec
        return None

    def check_task_timeouts(self) -> None:
        if self.task_timeout_ms <= 0:
            return
        now = time.perf_counter()
        for worker_id, task in list(self.inflight_by_worker.items()):
            if task.get("timeout_reported"):
                continue
            dispatch_perf = task.get("dispatch_perf")
            if dispatch_perf is None:
                continue
            elapsed_ms = (now - float(dispatch_perf)) * 1000.0
            if elapsed_ms < self.task_timeout_ms:
                continue

            task["timeout_reported"] = True
            task["timeout_elapsed_ms"] = elapsed_ms
            self.timeout_events += 1
            self.record_timeout_attempt(task=task, worker_id=worker_id, elapsed_ms=elapsed_ms)
            self.write_ledger(
                event_type="stage_timeout",
                task=task,
                worker_id=worker_id,
                success=False,
                update_applied=False,
                message=f"stage task timed out after {elapsed_ms:.1f} ms",
            )

            if self.timeout_policy == "observe":
                continue
            if self.timeout_policy in {"retry_stage", "cancel_retry_on_late"}:
                seq = task["seq"]
                stage_id = task["stage_id"]
                next_attempt = task["attempt"] + 1
                self.stage_attempts[(seq, stage_id)] = max(
                    self.stage_attempts[(seq, stage_id)],
                    next_attempt,
                )
                if next_attempt >= self.max_attempts:
                    continue
                retry_task = self.make_task(
                    seq=seq,
                    stage_id=stage_id,
                    input_state=task["input_state"],
                    attempt=next_attempt,
                    mode=task.get("mode", self.current_mode),
                    preferred_worker_id=worker_id,
                )
                task["timeout_retry_task_id"] = retry_task["task_id"]
                if self.timeout_policy == "retry_stage":
                    task["timeout_superseded"] = True
                self.enqueue_task(retry_task, recovery=True)
                self.write_ledger(
                    event_type="timeout_retry_stage",
                    task=retry_task,
                    worker_id="",
                    success=True,
                    update_applied=False,
                    message=(
                        f"retrying stage {stage_id} after timeout; late result "
                        f"from task {task['task_id']} will be dropped"
                    ),
                )
                continue
            raise ValueError(f"Unsupported timeout_policy: {self.timeout_policy}")

    def cancel_ready_task(self, task_id: int) -> Optional[dict[str, Any]]:
        for stage_id, queue_for_stage in self.ready.items():
            for index, task in enumerate(queue_for_stage):
                if int(task.get("task_id", -1)) != task_id:
                    continue
                queue_for_stage.rotate(-index)
                selected = queue_for_stage.popleft()
                queue_for_stage.rotate(index)
                selected["cancelled"] = True
                return selected
        return None

    def forget_dispatched_task(self, task_id: int) -> None:
        task = self.dispatched_tasks_by_id.pop(task_id, None)
        if task is not None:
            task.pop("input_state", None)

    def clear_request_boundaries(self, seq: int) -> None:
        for key in list(self.boundary_cache.keys()):
            if key[0] == seq:
                del self.boundary_cache[key]

    def record_timeout_attempt(self, *, task: dict[str, Any], worker_id: str, elapsed_ms: float) -> None:
        self.request_attempts[task["seq"]].append(
            {
                "stage_id": task["stage_id"],
                "attempt": task["attempt"],
                "mode": task.get("mode"),
                "update_decision": task.get("update_decision"),
                "worker_id": worker_id,
                "preferred_worker_id": task.get("preferred_worker_id"),
                "success": False,
                "update_applied": False,
                "timeout_reported": True,
                "late": False,
                "dropped": False,
                "message": f"timeout after {elapsed_ms:.1f} ms",
            }
        )

    def handle_stage_result(self, result: dict[str, Any]) -> None:
        worker_id = result.get("worker_id", "")
        result_task_id = int(result.get("task_id", -1))
        if result_task_id in self.resolved_task_ids:
            task = self.dispatched_tasks_by_id.get(result_task_id)
            if task is not None:
                self.write_ledger(
                    event_type="duplicate_result",
                    task=task,
                    worker_id=worker_id,
                    success=bool(result.get("success")),
                    update_applied=bool(result.get("update_applied", False)),
                    message="dropping duplicate result for an already resolved task",
                )
            return

        current_task = self.inflight_by_worker.get(worker_id)
        if current_task is None or current_task.get("task_id") != result_task_id:
            task = self.dispatched_tasks_by_id.get(result_task_id)
            if task is None:
                raise RuntimeError(f"Got result for unknown task from {worker_id}: {result}")
            self.late_results += 1
            self.dropped_late_results += 1
            metric = result.get("metric", {})
            metric["result_status"] = "late_dropped"
            self.write_metric(metric)
            self.record_attempt(
                task=task,
                result=result,
                worker_id=worker_id,
                late=True,
                dropped=True,
            )
            self.resolved_task_ids.add(result_task_id)
            self.write_ledger(
                event_type="late_result_dropped",
                task=task,
                worker_id=worker_id,
                success=bool(result.get("success")),
                update_applied=bool(result.get("update_applied", False)),
                message="dropping late result after timeout recovery was scheduled",
            )
            self.forget_dispatched_task(result_task_id)
            return

        if (
            self.timeout_policy == "cancel_retry_on_late"
            and current_task.get("timeout_reported")
            and current_task.get("timeout_retry_task_id") is not None
        ):
            retry_task_id = int(current_task["timeout_retry_task_id"])
            cancelled = self.cancel_ready_task(retry_task_id)
            if cancelled is not None:
                self.cancelled_timeout_retries += 1
                self.write_ledger(
                    event_type="timeout_retry_cancelled",
                    task=cancelled,
                    worker_id="",
                    success=True,
                    update_applied=False,
                    message=(
                        f"cancelled timeout retry task {retry_task_id} because "
                        f"late result for task {result_task_id} arrived first"
                    ),
                )

        if current_task.get("timeout_superseded"):
            task = self.inflight_by_worker.pop(worker_id)
            if worker_id in self.worker_busy:
                self.worker_busy[worker_id] = False
            self.late_results += 1
            self.dropped_late_results += 1
            result.setdefault("metric", {})["result_status"] = "late_dropped"
            self.write_metric(result.get("metric", {}))
            self.record_attempt(
                task=task,
                result=result,
                worker_id=worker_id,
                late=True,
                dropped=True,
            )
            self.resolved_task_ids.add(result_task_id)
            self.write_ledger(
                event_type="late_result_dropped",
                task=task,
                worker_id=worker_id,
                success=bool(result.get("success")),
                update_applied=bool(result.get("update_applied", False)),
                message="dropping late result after timeout retry was scheduled",
            )
            self.forget_dispatched_task(result_task_id)
            return

        task = self.inflight_by_worker.pop(worker_id)
        if worker_id in self.worker_busy:
            self.worker_busy[worker_id] = False
        if task.get("timeout_reported"):
            self.late_results += 1
            result.setdefault("metric", {})["result_status"] = "late_accepted"
            self.write_ledger(
                event_type="late_result_accepted",
                task=task,
                worker_id=worker_id,
                success=bool(result.get("success")),
                update_applied=bool(result.get("update_applied", False)),
                message="accepting late result because no timeout recovery superseded it",
            )

        self.write_metric(result.get("metric", {}))
        if not task.get("recovery_reconstruction"):
            self.record_attempt(
                task=task,
                result=result,
                worker_id=worker_id,
                late=bool(task.get("timeout_reported")),
                dropped=False,
            )
        self.resolved_task_ids.add(result_task_id)

        if result["success"]:
            self.record_recovery_journal(task, result)
            self.store_stage_checkpoint(result)
            if task.get("recovery_reconstruction"):
                if (
                    task.get("mode") == "catchup"
                    and task.get("recovery_window") is not None
                    and int(task.get("stage_id", -1)) == int(self.failure_injector.rule.fail_stage or -1)
                ):
                    window = task.get("recovery_window") or {}
                    seqs = [int(seq) for seq in window.get("seqs", [])]
                    if seqs:
                        self.recovery_latency_state["replay_or_catchup_start_ms"] = (
                            self.recovery_latency_state["replay_or_catchup_start_ms"]
                            or float(result.get("metric", {}).get("worker_start_epoch_ms", 0.0) or 0.0)
                        )
                        self.recovery_latency_state["replay_or_catchup_done_ms"] = float(
                            result.get("metric", {}).get("worker_end_epoch_ms", 0.0) or 0.0
                        )
                self.write_ledger(
                    event_type="checkpoint_catchup_success",
                    task=task,
                    worker_id=worker_id,
                    success=True,
                    update_applied=True,
                    message="reconstructed stage state on promoted replica",
                )
                self.forget_dispatched_task(result_task_id)
                return
            self.write_ledger(
                event_type="stage_success",
                task=task,
                worker_id=worker_id,
                success=True,
                update_applied=result.get("update_applied", False),
                message="stage completed",
            )
            self.handle_success(task, result)
        else:
            self.write_ledger(
                event_type="stage_failure",
                task=task,
                worker_id=worker_id,
                success=False,
                update_applied=result.get("update_applied", False),
                message=result.get("message", "stage failed"),
            )
            self.handle_failure(task, result)
        self.forget_dispatched_task(result_task_id)

    def record_attempt(
        self,
        *,
        task: dict[str, Any],
        result: dict[str, Any],
        worker_id: str,
        late: bool,
        dropped: bool,
    ) -> None:
        self.request_attempts[result["seq"]].append(
            {
                "stage_id": result["stage_id"],
                "attempt": result["attempt"],
                "mode": result.get("mode", task.get("mode")),
                "update_decision": task.get("update_decision"),
                "worker_id": worker_id,
                "preferred_worker_id": task.get("preferred_worker_id"),
                "success": result["success"],
                "update_applied": result.get("update_applied", False),
                "timeout_reported": bool(task.get("timeout_reported")),
                "late": late,
                "dropped": dropped,
                "message": result.get("message", ""),
            }
        )

    def handle_success(self, task: dict[str, Any], result: dict[str, Any]) -> None:
        seq = task["seq"]
        stage_id = task["stage_id"]
        if task.get("recovery_reconstruction"):
            return
        if stage_id == self.cfg.num_chunks - 1:
            if (
                self.failure_injector.rule.fail_seq is not None
                and int(seq) == int(self.failure_injector.rule.fail_seq)
                and self.recovery_latency_state["recovery_unit_done_ms"] is None
            ):
                self.recovery_latency_state["recovery_unit_done_ms"] = float(
                    result.get("metric", {}).get("worker_end_epoch_ms", 0.0) or 0.0
                )
            row = self.finish_request(task, result)
            self.completed[seq] = row
            self.inflight_requests.discard(seq)
            self.clear_request_boundaries(seq)
            self.write_result(row)
            return

        output_state = result["output_state"]
        self.boundary_cache[(seq, stage_id)] = output_state
        next_stage_id = stage_id + 1
        attempt = self.stage_attempts[(seq, next_stage_id)]
        next_task = self.make_task(
            seq=seq,
            stage_id=next_stage_id,
            input_state=output_state,
            attempt=attempt,
            mode=task.get("mode", self.current_mode),
        )
        self.enqueue_task(next_task)

    def handle_failure(self, task: dict[str, Any], result: dict[str, Any]) -> None:
        seq = task["seq"]
        stage_id = task["stage_id"]
        next_attempt = task["attempt"] + 1
        self.stage_attempts[(seq, stage_id)] = next_attempt
        if (
            self.recovery_latency_state["failure_detected_ms"] is None
            and self.failure_injector.rule.fail_stage is not None
            and self.failure_injector.rule.fail_seq is not None
            and int(stage_id) == int(self.failure_injector.rule.fail_stage)
            and int(seq) == int(self.failure_injector.rule.fail_seq)
        ):
            self.recovery_latency_state["failure_detected_ms"] = float(
                result.get("metric", {}).get("worker_end_epoch_ms", 0.0) or 0.0
            )

        if self.recovery_policy == "wait_for_rejoin":
            failed_worker_id = str(result.get("worker_id") or "")
            if not failed_worker_id or self.worker_rejoin_delay_ms <= 0:
                raise RuntimeError(
                    "wait_for_rejoin requires a failed worker and --worker_rejoin_delay_ms > 0."
                )
            if next_attempt >= self.max_attempts:
                row = self.failed_row(task, result, "max attempts reached while waiting for worker rejoin")
                self.failed[seq] = row
                self.inflight_requests.discard(seq)
                self.clear_request_boundaries(seq)
                self.write_result(row)
                return
            rejoin_at = time.perf_counter() + (self.worker_rejoin_delay_ms / 1000.0)
            self.temporary_unavailable_until[failed_worker_id] = rejoin_at
            update_applied = bool(result.get("update_applied"))
            retry_task = self.make_task(
                seq=seq,
                stage_id=stage_id,
                input_state=task["input_state"],
                attempt=next_attempt,
                mode="replay" if update_applied else task.get("mode", self.current_mode),
                preferred_worker_id=failed_worker_id,
            )
            self.rejoin_events.append(
                {
                    "seq": seq,
                    "stage_id": stage_id,
                    "worker_id": failed_worker_id,
                    "delay_ms": self.worker_rejoin_delay_ms,
                }
            )
            self.enqueue_task(retry_task, recovery=True)
            self.write_ledger(
                event_type="wait_for_worker_rejoin",
                task=retry_task,
                worker_id=failed_worker_id,
                success=True,
                update_applied=False,
                message=(
                    (
                        f"replaying stage {stage_id} from boundary after "
                        f"{self.worker_rejoin_delay_ms:.1f} ms rejoin delay "
                        "to regenerate the lost downstream boundary"
                    )
                    if update_applied
                    else (
                        f"retrying stage {stage_id} from boundary after "
                        f"{self.worker_rejoin_delay_ms:.1f} ms rejoin delay"
                    )
                ),
            )
            return

        if self.recovery_policy == "migrate_from_boundary":
            failed_worker_id = str(result.get("worker_id") or "")
            if failed_worker_id:
                self.unavailable_workers.add(failed_worker_id)
                self.write_ledger(
                    event_type="worker_marked_unavailable",
                    task=task,
                    worker_id=failed_worker_id,
                    success=False,
                    update_applied=False,
                    message="worker removed from scheduling after stage failure",
                )
            fallback = self.healthy_fallback_worker(
                stage_id=stage_id,
                failed_worker_id=failed_worker_id,
            )
            if fallback is not None and next_attempt < self.max_attempts:
                self.standby_worker_ids.discard(fallback.worker_id)
                retry_task = self.make_task(
                    seq=seq,
                    stage_id=stage_id,
                    input_state=task["input_state"],
                    attempt=next_attempt,
                    mode=task.get("mode", self.current_mode),
                    preferred_worker_id=fallback.worker_id,
                )
                checkpoint = self.stage_checkpoints.get(stage_id)
                checkpoint_version = int(checkpoint["version"]) if checkpoint is not None else 0
                checkpoint_journal_index = int(checkpoint["journal_index"]) if checkpoint is not None else 0
                catchup_entries = [
                    entry
                    for journal_index, entry in self.stage_recovery_journal[stage_id]
                    if journal_index > checkpoint_journal_index
                ]
                catchup_tasks = [
                    self.make_checkpoint_catchup_task(
                        stage_id=stage_id,
                        entry=entry,
                        preferred_worker_id=fallback.worker_id,
                    )
                    for entry in catchup_entries
                ]
                self.migration_events.append(
                    {
                        "seq": seq,
                        "stage_id": stage_id,
                        "failed_worker_id": failed_worker_id,
                        "fallback_worker_id": fallback.worker_id,
                        "checkpoint_version": checkpoint_version,
                        "checkpoint_bytes": int(checkpoint["bytes"]) if checkpoint is not None else 0,
                        "checkpoint_journal_index": checkpoint_journal_index,
                        "checkpoint_window_id": checkpoint.get("window_id", "") if checkpoint is not None else "",
                        "checkpoint_window_seqs": list(checkpoint.get("seqs", [])) if checkpoint is not None else [],
                        "catchup_updates": len(catchup_tasks),
                        "catchup_windows": len(catchup_entries),
                        "catchup_window_seq_ranges": [
                            {
                                "window_id": recovery_entry_window_id(entry, self.cfg.gradient_accumulation_steps),
                                "seqs": recovery_entry_seqs(entry),
                            }
                            for entry in catchup_entries
                        ],
                        "catchup_input_bytes": sum(
                            sum(state_bytes(item) for item in catchup_task.get("recovery_window", {}).get("input_states", []))
                            for catchup_task in catchup_tasks
                        ),
                    }
                )
                if (
                    self.failure_injector.rule.fail_stage is not None
                    and self.failure_injector.rule.fail_seq is not None
                    and int(stage_id) == int(self.failure_injector.rule.fail_stage)
                    and int(seq) == int(self.failure_injector.rule.fail_seq)
                ):
                    recovery_scope = "failed stage local accumulated window"
                    if catchup_entries:
                        seqs = recovery_entry_seqs(catchup_entries[-1])
                        if seqs:
                            recovery_scope = (
                                f"failed stage local accumulated window {seqs[0]}..{seqs[-1]}"
                            )
                    self.recovery_latency_state["recovery_scope"] = recovery_scope
                    self.recovery_latency_state["recovery_records"] = sum(
                        len(recovery_entry_seqs(entry)) for entry in catchup_entries
                    )
                    if catchup_tasks:
                        self.recovery_latency_state["replay_or_catchup_start_ms"] = (
                            self.recovery_latency_state["replay_or_catchup_start_ms"]
                            or float(time.time() * 1000.0)
                        )
                for scheduled_task in reversed([*catchup_tasks, retry_task]):
                    self.enqueue_task(scheduled_task, recovery=True, front=True)
                for catchup_task in catchup_tasks:
                    self.write_ledger(
                        event_type="checkpoint_catchup",
                        task=catchup_task,
                        worker_id=fallback.worker_id,
                        success=True,
                        update_applied=False,
                        message="replaying committed accumulation window on promoted replica",
                    )
                self.write_ledger(
                    event_type="migrate_from_boundary",
                    task=retry_task,
                    worker_id=fallback.worker_id,
                    success=True,
                    update_applied=False,
                    message=(
                        f"rerouting stage {stage_id} from boundary after {failed_worker_id} failure; "
                        f"checkpoint_version={checkpoint_version} catchup_updates={len(catchup_tasks)}"
                    ),
                )
                return
            row = self.failed_row(task, result, "migration requested but no healthy stage replica is available")
            self.failed[seq] = row
            self.inflight_requests.discard(seq)
            self.clear_request_boundaries(seq)
            self.write_result(row)
            return

        if next_attempt >= self.max_attempts or self.recovery_policy == "skip":
            row = self.failed_row(task, result, "max attempts reached or skip policy")
            self.failed[seq] = row
            self.inflight_requests.discard(seq)
            self.clear_request_boundaries(seq)
            self.write_result(row)
            return

        if self.recovery_policy == "replay_after_update" and result.get("update_applied"):
            replay_task = self.make_task(
                seq=seq,
                stage_id=stage_id,
                input_state=task["input_state"],
                attempt=next_attempt,
                mode="replay",
                preferred_worker_id=result.get("worker_id"),
            )
            self.enqueue_task(replay_task, recovery=True)
            self.write_ledger(
                event_type="replay_after_update",
                task=replay_task,
                worker_id="",
                success=True,
                update_applied=False,
                message=(
                    f"replaying stage {stage_id} on {result.get('worker_id')} "
                    "to regenerate lost boundary without another optimizer step"
                ),
            )
            return

        if self.recovery_policy in {"retry_stage", "retry_from_boundary", "replay_after_update"}:
            retry_task = self.make_task(
                seq=seq,
                stage_id=stage_id,
                input_state=task["input_state"],
                attempt=next_attempt,
                mode=task.get("mode", self.current_mode),
            )
            self.enqueue_task(retry_task, recovery=True)
            self.write_ledger(
                event_type="retry_stage",
                task=retry_task,
                worker_id="",
                success=True,
                update_applied=False,
                message=f"retrying stage {stage_id} from existing input boundary",
            )
            return

        if self.recovery_policy == "retry_from_zero":
            self.clear_request_boundaries(seq)
            self.stage_attempts[(seq, 0)] += 1
            retry_task = self.make_task(
                seq=seq,
                stage_id=0,
                input_state=load_initial_state(self.records[seq], self.manifest_dir),
                attempt=self.stage_attempts[(seq, 0)],
                mode=task.get("mode", self.current_mode),
            )
            self.enqueue_task(retry_task, recovery=True)
            self.write_ledger(
                event_type="retry_from_zero",
                task=retry_task,
                worker_id="",
                success=True,
                update_applied=False,
                message="restarting request from stage 0",
            )
            return

        raise ValueError(f"Unsupported recovery_policy: {self.recovery_policy}")

    def finish_request(self, task: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        final_log_probs = result.get("final_log_probs")
        labels = result.get("labels")
        choice_details = result.get("choice_details")
        if isinstance(choice_details, dict):
            correct = int(choice_details.get("choice_correct", 0))
            count = int(choice_details.get("choice_count", 0))
            choice_loss = float(choice_details.get("choice_loss", 0.0))
            predicted_token_id = choice_details.get("predicted_token_id", "")
            target_token_id = choice_details.get("target_token_id", "")
            predicted_response = ""
            for choice in task["record"].get("label_choices") or []:
                token_ids = choice.get("token_ids") or []
                if len(token_ids) == 1 and predicted_token_id != "" and int(token_ids[0]) == int(predicted_token_id):
                    predicted_response = str(choice.get("text", "")).strip()
                    break
        elif (
            isinstance(final_log_probs, torch.Tensor)
            and isinstance(labels, torch.Tensor)
            and task["record"].get("label_choices")
        ):
            choice_info = label_choice_details(task["record"], final_log_probs, labels)
            correct = int(choice_info["choice_correct"])
            count = int(choice_info["choice_count"])
            choice_loss = float(choice_info["choice_loss"])
            predicted_response = str(choice_info["predicted_response"])
            predicted_token_id = choice_info["predicted_token_id"]
            target_token_id = choice_info["target_token_id"]
        else:
            correct, count, choice_loss = 0, 0, 0.0
            predicted_response = ""
            predicted_token_id = ""
            target_token_id = ""
        response = (task["record"].get("text") or {}).get("response", "").strip()
        return {
            "phase": task["phase"],
            "seq": task["seq"],
            "request_id": task["request_id"],
            "dataset_index": int(task["record"].get("dataset_index", -1)),
            "response": response,
            "predicted_response": predicted_response,
            "predicted_token_id": predicted_token_id,
            "target_token_id": target_token_id,
            "mode": task.get("request_mode", task["mode"]),
            "status": "completed",
            "loss": result.get("local_loss", 0.0),
            "choice_correct": correct,
            "choice_count": count,
            "choice_accuracy": (correct / count) if count else 0.0,
            "choice_loss": choice_loss,
            "attempts_json": json.dumps(self.request_attempts[task["seq"]]),
            "message": "",
        }

    def failed_row(self, task: dict[str, Any], result: dict[str, Any], message: str) -> dict[str, Any]:
        return {
            "phase": task["phase"],
            "seq": task["seq"],
            "request_id": task["request_id"],
            "dataset_index": int(task["record"].get("dataset_index", -1)),
            "response": (task["record"].get("text") or {}).get("response", "").strip(),
            "predicted_response": "",
            "predicted_token_id": "",
            "target_token_id": "",
            "mode": task.get("request_mode", task["mode"]),
            "status": "failed",
            "loss": result.get("local_loss", ""),
            "choice_correct": 0,
            "choice_count": 0,
            "choice_accuracy": 0.0,
            "choice_loss": 0.0,
            "attempts_json": json.dumps(self.request_attempts[task["seq"]]),
            "message": message,
        }

    def run_phase(
        self,
        *,
        phase: str,
        mode: str,
        records: list[dict[str, Any]],
        manifest_dir: Path,
    ) -> dict[str, Any]:
        self.reset_phase_state(phase=phase, mode=mode, records=records, manifest_dir=manifest_dir)
        started = time.perf_counter()
        timeout_start = self.timeout_events
        late_start = self.late_results
        dropped_start = self.dropped_late_results
        cancelled_start = self.cancelled_timeout_retries
        while len(self.completed) + len(self.failed) < len(self.records):
            self.admit_requests()
            self.dispatch_ready()
            try:
                result = self.result_queue.get(timeout=0.1)
            except queue.Empty:
                self.check_task_timeouts()
                self.check_worker_errors()
                continue
            if result.get("kind") == "stage_result":
                self.handle_stage_result(result)
            elif result.get("kind") == "worker_error":
                raise RuntimeError(f"worker error: {result}")
            elif result.get("kind") == "worker_ready":
                self.worker_ready_info[result["worker_id"]] = result
            elif result.get("kind") == "worker_stopped":
                continue
            else:
                raise RuntimeError(f"Unexpected scheduler result: {result}")
            self.check_task_timeouts()
            self.check_worker_errors()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        phase_summary = self.summarize_current_phase(
            phase=phase,
            mode=mode,
            elapsed_ms=elapsed_ms,
            timeout_events=self.timeout_events - timeout_start,
            late_results=self.late_results - late_start,
            dropped_late_results=self.dropped_late_results - dropped_start,
            cancelled_timeout_retries=self.cancelled_timeout_retries - cancelled_start,
        )
        self.phase_summaries.append(phase_summary)
        return phase_summary

    def summarize_current_phase(
        self,
        *,
        phase: str,
        mode: str,
        elapsed_ms: float,
        timeout_events: int,
        late_results: int,
        dropped_late_results: int,
        cancelled_timeout_retries: int,
    ) -> dict[str, Any]:
        correct = sum(int(row["choice_correct"]) for row in self.completed.values())
        count = sum(int(row["choice_count"]) for row in self.completed.values())
        losses = [float(row["loss"]) for row in self.completed.values() if row["loss"] != ""]
        return {
            "phase": phase,
            "mode": mode,
            "records": len(self.records),
            "completed": len(self.completed),
            "failed": len(self.failed),
            "choice_correct": correct,
            "choice_count": count,
            "choice_accuracy": (correct / count) if count else 0.0,
            "avg_loss": sum(losses) / len(losses) if losses else 0.0,
            "wall_ms": elapsed_ms,
            "throughput_per_s": len(self.completed) / (elapsed_ms / 1000.0) if elapsed_ms > 0 else 0.0,
            "timeout_events": timeout_events,
            "late_results": late_results,
            "dropped_late_results": dropped_late_results,
            "cancelled_timeout_retries": cancelled_timeout_retries,
            "update_consistency": self.update_consistency_summary(),
            "retained_progress": self.retained_progress_summary(),
        }

    def run_training_with_validation(self) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if self.validation_records is None:
            return (
                self.run_phase(
                    phase="train",
                    mode="train",
                    records=self.train_records,
                    manifest_dir=self.train_manifest_dir,
                ),
                [],
            )

        if self.validation_manifest_dir is None:
            raise RuntimeError("validation_manifest_dir is required with validation records")
        if self.validation_interval_steps <= 0:
            raise RuntimeError("validation_interval_steps must be positive with validation records")

        records_per_step = self.cfg.gradient_accumulation_steps
        interval_records = self.validation_interval_steps * records_per_step
        validation_curve: list[dict[str, Any]] = []

        def append_validation(optimizer_step: int, train_samples_seen: int) -> None:
            summary = self.run_phase(
                phase=f"validation_step{optimizer_step:06d}",
                mode="eval",
                records=self.validation_records or [],
                manifest_dir=self.validation_manifest_dir or self.train_manifest_dir,
            )
            validation_curve.append(
                {
                    "optimizer_step": optimizer_step,
                    "train_samples_seen": train_samples_seen,
                    "phase": summary["phase"],
                    "validation_records": summary["completed"],
                    "choice_correct": summary["choice_correct"],
                    "choice_count": summary["choice_count"],
                    "choice_accuracy": summary["choice_accuracy"],
                    "avg_loss": summary["avg_loss"],
                    "wall_ms": summary["wall_ms"],
                }
            )

        append_validation(0, 0)
        completed_records = 0
        train_segments: list[dict[str, Any]] = []
        while completed_records < len(self.train_records):
            next_completed = min(completed_records + interval_records, len(self.train_records))
            segment = self.train_records[completed_records:next_completed]
            train_segments.append(
                self.run_phase(
                    phase=f"train_to_{next_completed:06d}",
                    mode="train",
                    records=segment,
                    manifest_dir=self.train_manifest_dir,
                )
            )
            completed_records = next_completed
            append_validation(
                completed_records // records_per_step,
                completed_records,
            )

        total_records = sum(int(item["records"]) for item in train_segments)
        total_completed = sum(int(item["completed"]) for item in train_segments)
        total_wall_ms = sum(float(item["wall_ms"]) for item in train_segments)
        avg_loss = (
            sum(float(item["avg_loss"]) * int(item["completed"]) for item in train_segments) / total_completed
            if total_completed
            else 0.0
        )
        aggregate = {
            "phase": "train",
            "mode": "train",
            "records": total_records,
            "completed": total_completed,
            "failed": sum(int(item["failed"]) for item in train_segments),
            "choice_correct": 0,
            "choice_count": 0,
            "choice_accuracy": 0.0,
            "avg_loss": avg_loss,
            "wall_ms": total_wall_ms,
            "throughput_per_s": total_completed / (total_wall_ms / 1000.0) if total_wall_ms > 0 else 0.0,
            "timeout_events": sum(int(item["timeout_events"]) for item in train_segments),
            "late_results": sum(int(item["late_results"]) for item in train_segments),
            "dropped_late_results": sum(int(item["dropped_late_results"]) for item in train_segments),
            "cancelled_timeout_retries": sum(
                int(item["cancelled_timeout_retries"]) for item in train_segments
            ),
            "update_consistency": train_segments[-1]["update_consistency"],
            "retained_progress": train_segments[-1]["retained_progress"],
            "segment_phases": [str(item["phase"]) for item in train_segments],
        }
        self.phase_summaries.append(aggregate)
        return aggregate, validation_curve

    def run(self) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.result_handle = self.results_path.open("w", newline="", encoding="utf-8")
        self.metric_handle = self.metrics_path.open("w", newline="", encoding="utf-8")
        self.ledger_handle = self.ledger_path.open("w", newline="", encoding="utf-8")
        self.result_writer = csv.DictWriter(self.result_handle, fieldnames=result_fieldnames())
        self.metric_writer = csv.DictWriter(self.metric_handle, fieldnames=metric_fieldnames())
        self.ledger_writer = csv.DictWriter(self.ledger_handle, fieldnames=ledger_fieldnames())
        self.result_writer.writeheader()
        self.metric_writer.writeheader()
        self.ledger_writer.writeheader()

        startup_started = time.perf_counter()
        self.start_workers()
        try:
            self.wait_for_workers_ready()
            startup_ms = (time.perf_counter() - startup_started) * 1000.0
            train_summary, validation_curve = self.run_training_with_validation()
            primary_summary = train_summary
            if self.eval_records is not None:
                assert self.eval_manifest_dir is not None
                primary_summary = self.run_phase(
                    phase="eval",
                    mode="eval",
                    records=self.eval_records,
                    manifest_dir=self.eval_manifest_dir,
                )
            stage_lora_exports = self.export_stage_lora_states()
        finally:
            self.stop_workers()
            self.result_handle.close()
            self.metric_handle.close()
            self.ledger_handle.close()

        validation_curve_path = self.output_dir / "validation_curve.csv"
        if validation_curve:
            write_csv_rows(validation_curve_path, validation_curve_fieldnames(), validation_curve)

        summary = {
            "runner": "bpfree-orchestrated-runtime-v1",
            "transport": "cpu-mp-queue",
            "transport_details": "torch.multiprocessing.Queue CPU boundary payloads",
            "model_name": self.cfg.resolved_model,
            "topology": self.topology,
            "stage_devices": [spec.device for spec in self.worker_specs],
            "physical_request_batch": 1,
            "effective_optimizer_batch": self.cfg.gradient_accumulation_steps,
            "update_unit": (
                "stage_local_accumulated_optimizer_step"
                if self.cfg.gradient_accumulation_steps > 1
                else "stage_local_request_optimizer_step"
            ),
            "recovery_alignment_note": (
                "When gradient_accumulation_steps > 1, one committed BP-free update "
                "is one stage-local accumulated optimizer step. Checkpoints and recovery "
                "journal entries are versioned by committed local optimizer steps, not raw request seq."
            ),
            "records": primary_summary["records"],
            "completed": primary_summary["completed"],
            "failed": primary_summary["failed"],
            "choice_correct": primary_summary["choice_correct"],
            "choice_count": primary_summary["choice_count"],
            "choice_accuracy": primary_summary["choice_accuracy"],
            "avg_loss": primary_summary["avg_loss"],
            "startup_ms": startup_ms,
            "wall_ms": primary_summary["wall_ms"],
            "throughput_per_s": primary_summary["throughput_per_s"],
            "scheduler_policy": self.scheduler_policy,
            "recovery_policy": self.recovery_policy,
            "trainable_mode": self.cfg.trainable_mode,
            "dtype": self.cfg.dtype_name,
            "learning_rate": self.cfg.learning_rate,
            "belief_transport_mode": self.cfg.belief_transport_mode,
            "alpha": self.cfg.alpha,
            "lora": {
                "rank": self.cfg.lora_rank,
                "alpha": self.cfg.lora_alpha,
                "targets": self.cfg.lora_targets,
                "init_std": self.cfg.lora_init_std,
                "init_seed": self.cfg.lora_init_seed,
                "initialization_fingerprints": sorted(
                    {
                        str(info.get("lora_initialization_fingerprint", ""))
                        for info in self.worker_ready_info.values()
                        if info.get("lora_initialization_fingerprint")
                    }
                ),
            },
            "local_readout_adapter": {
                "bottleneck": self.cfg.local_readout_adapter_bottleneck,
                "stages": self.cfg.local_readout_adapter_stages,
                "trainable_params_by_worker": {
                    worker_id: int(info.get("local_readout_adapter_trainable_params", 0))
                    for worker_id, info in sorted(self.worker_ready_info.items())
                },
            },
            "gradient_accumulation_steps": self.cfg.gradient_accumulation_steps,
            "failure_mode": self.failure_injector.rule.mode,
            "failure_stage": self.failure_injector.rule.fail_stage,
            "failure_seq": self.failure_injector.rule.fail_seq,
            "failure_attempt": self.failure_injector.rule.fail_attempt,
            "failure_point": self.failure_injector.rule.fail_point,
            "validation_records": len(self.validation_records) if self.validation_records is not None else 0,
            "validation_interval_steps": self.validation_interval_steps,
            "validation_curve_csv": str(validation_curve_path) if validation_curve else "",
            "stage_update_policy": self.cfg.stage_update_policy,
            "stage_train_strides": self.cfg.stage_train_strides,
            "stage_update_queue_thresholds": self.cfg.stage_update_queue_thresholds,
            "task_timeout_ms": self.task_timeout_ms,
            "timeout_policy": self.timeout_policy,
            "timeout_events": self.timeout_events,
            "late_results": self.late_results,
            "dropped_late_results": self.dropped_late_results,
            "cancelled_timeout_retries": self.cancelled_timeout_retries,
            "max_inflight": self.max_inflight,
            "max_attempts": self.max_attempts,
            "offline_window": {
                "stage_id": self.failure_injector.rule.offline_stage,
                "start_seq": self.failure_injector.rule.offline_start_seq,
                "end_seq": self.failure_injector.rule.offline_end_seq,
            },
            "transient_dropout_mask": {
                "path": self.failure_injector.rule.transient_mask_path,
                "window_size": self.failure_injector.rule.transient_window_size,
                "offline_windows_by_stage": {
                    str(stage_id): sorted(windows)
                    for stage_id, windows in sorted(
                        self.failure_injector.rule.transient_offline_windows.items()
                    )
                },
                "stage_event_counts": {
                    str(stage_id): len(windows)
                    for stage_id, windows in sorted(
                        self.failure_injector.rule.transient_offline_windows.items()
                    )
                },
            },
            "workers": [spec.__dict__ for spec in self.worker_specs],
            "standby_worker_ids": sorted(self.standby_worker_ids),
            "migration": {
                "checkpoint_interval": self.checkpoint_interval,
                "unavailable_workers": sorted(self.unavailable_workers),
                "events": self.migration_events,
                "latest_checkpoint_versions": dict(sorted(self.stage_checkpoint_versions.items())),
                "journal_entries_waiting": {
                    str(stage_id): len(entries)
                    for stage_id, entries in sorted(self.stage_recovery_journal.items())
                },
            },
            "rejoin": {
                "delay_ms": self.worker_rejoin_delay_ms,
                "events": self.rejoin_events,
            },
            "gpu_metrics_by_worker": self.gpu_metrics_by_worker(),
            "update_consistency": train_summary["update_consistency"],
            "retained_progress": train_summary["retained_progress"],
            "phase_summaries": self.phase_summaries,
            "results_csv": str(self.results_path),
            "metrics_csv": str(self.metrics_path),
            "ledger_csv": str(self.ledger_path),
        }
        failure_detected_ms = float(self.recovery_latency_state.get("failure_detected_ms") or 0.0)
        recovery_unit_done_ms = float(self.recovery_latency_state.get("recovery_unit_done_ms") or 0.0)
        if failure_detected_ms > 0 and recovery_unit_done_ms > 0:
            summary["recovery_latency"] = {
                "failure_detected_ms": failure_detected_ms,
                "replay_or_catchup_start_ms": (
                    float(self.recovery_latency_state.get("replay_or_catchup_start_ms") or 0.0) or None
                ),
                "replay_or_catchup_done_ms": (
                    float(self.recovery_latency_state.get("replay_or_catchup_done_ms") or 0.0) or None
                ),
                "recovery_unit_done_ms": recovery_unit_done_ms,
                "recovery_unit_latency_ms": recovery_unit_done_ms - failure_detected_ms,
                "recovery_scope": self.recovery_latency_state.get("recovery_scope", ""),
                "recovery_records": int(self.recovery_latency_state.get("recovery_records", 0)),
                "failure_stage": self.failure_injector.rule.fail_stage,
                "failure_seq": self.failure_injector.rule.fail_seq,
                "checkpoint_interval": self.checkpoint_interval,
                "timing_boundary": "failure_detected_ms=stage1 seq200 failure confirmed; recovery_unit_done_ms=terminal completion of seq200",
            }
        if self.cfg.trainable_mode == "lora":
            for stage_id, payload in sorted(stage_lora_exports.items()):
                torch.save(payload["lora_state"], self.output_dir / f"stage{stage_id}_lora_state.pt")
            summary["stage_lora_state_files"] = {
                str(stage_id): str(self.output_dir / f"stage{stage_id}_lora_state.pt")
                for stage_id in sorted(stage_lora_exports)
            }
            adapter_state_files: dict[str, str] = {}
            for stage_id, payload in sorted(stage_lora_exports.items()):
                adapter_state = payload.get("local_readout_adapter_state")
                if adapter_state is None:
                    continue
                adapter_path = self.output_dir / f"stage{stage_id}_readout_adapter_state.pt"
                torch.save(adapter_state, adapter_path)
                adapter_state_files[str(stage_id)] = str(adapter_path)
            if adapter_state_files:
                summary["stage_readout_adapter_state_files"] = adapter_state_files
        summary_path = self.output_dir / "scheduler_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
        print(f"Wrote {summary_path}", flush=True)
        return summary

    def export_stage_lora_states(self) -> dict[int, dict[str, Any]]:
        if self.cfg.trainable_mode != "lora":
            return {}
        self.worker_lora_exports = {}
        export_specs: list[WorkerSpec] = []
        for stage_id in range(self.cfg.num_chunks):
            for spec in self.workers_by_stage[stage_id]:
                if spec.worker_id in self.standby_worker_ids:
                    continue
                if spec.worker_id in self.unavailable_workers:
                    continue
                export_specs.append(spec)
                break
            else:
                raise RuntimeError(f"No exportable worker available for stage {stage_id}.")
        for spec in export_specs:
            self.worker_queues[spec.worker_id].put({"kind": "export_lora_state"})
        expected = len(export_specs)
        while len(self.worker_lora_exports) < expected:
            try:
                result = self.result_queue.get(timeout=0.1)
            except queue.Empty:
                self.check_worker_errors()
                continue
            kind = result.get("kind")
            if kind == "stage_lora_export":
                self.worker_lora_exports[int(result["stage_id"])] = result
            elif kind == "worker_error":
                raise RuntimeError(f"worker error during lora export: {result}")
            elif kind == "worker_ready":
                self.worker_ready_info[result["worker_id"]] = result
            else:
                raise RuntimeError(f"Unexpected result during lora export: {result}")
        return dict(sorted(self.worker_lora_exports.items()))

    def check_worker_errors(self) -> None:
        for process in self.processes:
            if process.exitcode not in (None, 0):
                raise RuntimeError(f"Worker process {process.name} exited with code {process.exitcode}")

    def write_result(self, row: dict[str, Any]) -> None:
        self.result_writer.writerow(row)
        self.result_handle.flush()

    def write_metric(self, metric: dict[str, Any]) -> None:
        self.accumulate_metric(metric)
        self.metric_writer.writerow({name: metric.get(name, "") for name in metric_fieldnames()})
        self.metric_handle.flush()

    def accumulate_metric(self, metric: dict[str, Any]) -> None:
        worker_id = str(metric.get("worker_id") or "")
        if not worker_id:
            return
        stats = self.worker_metric_accum.setdefault(
            worker_id,
            {
                "worker_id": worker_id,
                "stage_id": metric.get("stage_id"),
                "device": metric.get("device"),
                "tasks": 0,
                "train_tasks": 0,
                "updates_applied": 0,
                "failures": 0,
                "execute_ms_sum": 0.0,
                "execute_ms_count": 0,
                "scheduler_queue_ms_sum": 0.0,
                "scheduler_queue_ms_count": 0,
                "worker_queue_ms_sum": 0.0,
                "worker_queue_ms_count": 0,
                "optimizer_ms_sum": 0.0,
                "optimizer_ms_count": 0,
                "stage_total_ms_sum": 0.0,
                "stage_total_ms_count": 0,
                "max_cuda_peak_memory_allocated": 0,
                "max_cuda_peak_memory_reserved": 0,
                "max_local_params": 0,
                "max_local_trainable_params": 0,
                "max_local_param_bytes": 0,
                "max_local_trainable_param_bytes": 0,
                "max_optimizer_state_bytes": 0,
                "max_autograd_saved_cuda_bytes_peak": 0,
                "max_autograd_saved_cuda_nonleaf_bytes_peak": 0,
                "max_autograd_saved_cuda_leaf_bytes_peak": 0,
                "max_autograd_saved_cuda_unique_bytes_peak": 0,
                "max_autograd_saved_cuda_nonleaf_unique_bytes_peak": 0,
                "max_autograd_saved_cuda_leaf_unique_bytes_peak": 0,
                "max_autograd_saved_cuda_nonleaf_unique_hidden_bytes_peak": 0,
                "max_autograd_saved_cuda_nonleaf_unique_vocab_bytes_peak": 0,
                "max_autograd_saved_cuda_nonleaf_unique_attention_bytes_peak": 0,
                "max_autograd_saved_cuda_nonleaf_unique_other_bytes_peak": 0,
                "max_autograd_saved_cuda_bytes_live_final": 0,
                "max_autograd_saved_cuda_unique_bytes_live_final": 0,
            },
        )
        stats["tasks"] += 1
        if bool(metric.get("train")):
            stats["train_tasks"] += 1
        if bool(metric.get("update_applied")):
            stats["updates_applied"] += 1
        if metric.get("failure"):
            stats["failures"] += 1
        for source, total_key, count_key in (
            ("execute_ms", "execute_ms_sum", "execute_ms_count"),
            ("scheduler_queue_ms", "scheduler_queue_ms_sum", "scheduler_queue_ms_count"),
            ("worker_queue_ms", "worker_queue_ms_sum", "worker_queue_ms_count"),
            ("optimizer_ms", "optimizer_ms_sum", "optimizer_ms_count"),
            ("stage_total_ms", "stage_total_ms_sum", "stage_total_ms_count"),
        ):
            parsed = optional_float(metric.get(source))
            if parsed is not None:
                stats[total_key] += parsed
                stats[count_key] += 1
        allocated = optional_float(metric.get("cuda_peak_memory_allocated"))
        reserved = optional_float(metric.get("cuda_peak_memory_reserved"))
        if allocated is not None:
            stats["max_cuda_peak_memory_allocated"] = max(
                int(stats["max_cuda_peak_memory_allocated"]),
                int(allocated),
            )
        if reserved is not None:
            stats["max_cuda_peak_memory_reserved"] = max(
                int(stats["max_cuda_peak_memory_reserved"]),
                int(reserved),
            )
        for source, target in (
            ("local_params", "max_local_params"),
            ("local_trainable_params", "max_local_trainable_params"),
            ("local_param_bytes", "max_local_param_bytes"),
            ("local_trainable_param_bytes", "max_local_trainable_param_bytes"),
            ("optimizer_state_bytes", "max_optimizer_state_bytes"),
        ):
            parsed = optional_float(metric.get(source))
            if parsed is not None:
                stats[target] = max(int(stats[target]), int(parsed))
        for source, target in (
            ("autograd_saved_cuda_bytes_peak", "max_autograd_saved_cuda_bytes_peak"),
            ("autograd_saved_cuda_nonleaf_bytes_peak", "max_autograd_saved_cuda_nonleaf_bytes_peak"),
            ("autograd_saved_cuda_leaf_bytes_peak", "max_autograd_saved_cuda_leaf_bytes_peak"),
            ("autograd_saved_cuda_unique_bytes_peak", "max_autograd_saved_cuda_unique_bytes_peak"),
            ("autograd_saved_cuda_nonleaf_unique_bytes_peak", "max_autograd_saved_cuda_nonleaf_unique_bytes_peak"),
            ("autograd_saved_cuda_leaf_unique_bytes_peak", "max_autograd_saved_cuda_leaf_unique_bytes_peak"),
            (
                "autograd_saved_cuda_nonleaf_unique_hidden_bytes_peak",
                "max_autograd_saved_cuda_nonleaf_unique_hidden_bytes_peak",
            ),
            (
                "autograd_saved_cuda_nonleaf_unique_vocab_bytes_peak",
                "max_autograd_saved_cuda_nonleaf_unique_vocab_bytes_peak",
            ),
            (
                "autograd_saved_cuda_nonleaf_unique_attention_bytes_peak",
                "max_autograd_saved_cuda_nonleaf_unique_attention_bytes_peak",
            ),
            (
                "autograd_saved_cuda_nonleaf_unique_other_bytes_peak",
                "max_autograd_saved_cuda_nonleaf_unique_other_bytes_peak",
            ),
            ("autograd_saved_cuda_bytes_live_final", "max_autograd_saved_cuda_bytes_live_final"),
            ("autograd_saved_cuda_unique_bytes_live_final", "max_autograd_saved_cuda_unique_bytes_live_final"),
        ):
            parsed = optional_float(metric.get(source))
            if parsed is not None:
                stats[target] = max(int(stats[target]), int(parsed))

    def gpu_metrics_by_worker(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for worker_id in sorted(self.worker_metric_accum):
            stats = self.worker_metric_accum[worker_id]
            execute_count = int(stats["execute_ms_count"])
            scheduler_queue_count = int(stats["scheduler_queue_ms_count"])
            worker_queue_count = int(stats["worker_queue_ms_count"])
            optimizer_count = int(stats["optimizer_ms_count"])
            total_count = int(stats["stage_total_ms_count"])
            rows.append(
                {
                    "worker_id": stats["worker_id"],
                    "stage_id": stats["stage_id"],
                    "device": stats["device"],
                    "tasks": stats["tasks"],
                    "train_tasks": stats["train_tasks"],
                    "updates_applied": stats["updates_applied"],
                    "failures": stats["failures"],
                    "avg_execute_ms": (
                        stats["execute_ms_sum"] / execute_count if execute_count else 0.0
                    ),
                    "avg_scheduler_queue_ms": (
                        stats["scheduler_queue_ms_sum"] / scheduler_queue_count
                        if scheduler_queue_count
                        else 0.0
                    ),
                    "avg_worker_queue_ms": (
                        stats["worker_queue_ms_sum"] / worker_queue_count
                        if worker_queue_count
                        else 0.0
                    ),
                    "avg_optimizer_ms": (
                        stats["optimizer_ms_sum"] / optimizer_count if optimizer_count else 0.0
                    ),
                    "avg_stage_total_ms": (
                        stats["stage_total_ms_sum"] / total_count if total_count else 0.0
                    ),
                    "max_cuda_peak_memory_allocated_mib": bytes_to_mib(
                        int(stats["max_cuda_peak_memory_allocated"])
                    ),
                    "max_cuda_peak_memory_reserved_mib": bytes_to_mib(
                        int(stats["max_cuda_peak_memory_reserved"])
                    ),
                    "max_local_params": int(stats["max_local_params"]),
                    "max_local_trainable_params": int(stats["max_local_trainable_params"]),
                    "max_local_param_mib": bytes_to_mib(int(stats["max_local_param_bytes"])),
                    "max_local_trainable_param_mib": bytes_to_mib(
                        int(stats["max_local_trainable_param_bytes"])
                    ),
                    "max_optimizer_state_mib": bytes_to_mib(int(stats["max_optimizer_state_bytes"])),
                    "max_autograd_saved_cuda_peak_mib": bytes_to_mib(
                        int(stats["max_autograd_saved_cuda_bytes_peak"])
                    ),
                    "max_autograd_saved_cuda_nonleaf_peak_mib": bytes_to_mib(
                        int(stats["max_autograd_saved_cuda_nonleaf_bytes_peak"])
                    ),
                    "max_autograd_saved_cuda_leaf_peak_mib": bytes_to_mib(
                        int(stats["max_autograd_saved_cuda_leaf_bytes_peak"])
                    ),
                    "max_autograd_saved_cuda_unique_peak_mib": bytes_to_mib(
                        int(stats["max_autograd_saved_cuda_unique_bytes_peak"])
                    ),
                    "max_autograd_saved_cuda_nonleaf_unique_peak_mib": bytes_to_mib(
                        int(stats["max_autograd_saved_cuda_nonleaf_unique_bytes_peak"])
                    ),
                    "max_autograd_saved_cuda_leaf_unique_peak_mib": bytes_to_mib(
                        int(stats["max_autograd_saved_cuda_leaf_unique_bytes_peak"])
                    ),
                    "max_autograd_saved_cuda_nonleaf_unique_hidden_peak_mib": bytes_to_mib(
                        int(stats["max_autograd_saved_cuda_nonleaf_unique_hidden_bytes_peak"])
                    ),
                    "max_autograd_saved_cuda_nonleaf_unique_vocab_peak_mib": bytes_to_mib(
                        int(stats["max_autograd_saved_cuda_nonleaf_unique_vocab_bytes_peak"])
                    ),
                    "max_autograd_saved_cuda_nonleaf_unique_attention_peak_mib": bytes_to_mib(
                        int(stats["max_autograd_saved_cuda_nonleaf_unique_attention_bytes_peak"])
                    ),
                    "max_autograd_saved_cuda_nonleaf_unique_other_peak_mib": bytes_to_mib(
                        int(stats["max_autograd_saved_cuda_nonleaf_unique_other_bytes_peak"])
                    ),
                    "max_autograd_saved_cuda_live_final_mib": bytes_to_mib(
                        int(stats["max_autograd_saved_cuda_bytes_live_final"])
                    ),
                    "max_autograd_saved_cuda_unique_live_final_mib": bytes_to_mib(
                        int(stats["max_autograd_saved_cuda_unique_bytes_live_final"])
                    ),
                }
            )
        return rows

    def update_consistency_summary(self) -> dict[str, Any]:
        per_stage: dict[str, dict[str, int]] = defaultdict(
            lambda: {"update_events": 0, "duplicate_update_events": 0}
        )
        duplicate_details: list[dict[str, Any]] = []
        replay_events = 0
        for seq, attempts in sorted(self.request_attempts.items()):
            updates_by_stage: dict[int, int] = defaultdict(int)
            for attempt in attempts:
                stage_id = int(attempt["stage_id"])
                if attempt.get("mode") == "replay":
                    replay_events += 1
                if attempt.get("update_applied"):
                    updates_by_stage[stage_id] += 1
                    per_stage[str(stage_id)]["update_events"] += 1
            for stage_id, count in sorted(updates_by_stage.items()):
                duplicates = max(0, count - 1)
                if duplicates:
                    per_stage[str(stage_id)]["duplicate_update_events"] += duplicates
                    duplicate_details.append(
                        {
                            "seq": seq,
                            "stage_id": stage_id,
                            "updates_applied": count,
                            "duplicate_update_events": duplicates,
                        }
                    )
        duplicate_update_events = sum(
            stats["duplicate_update_events"] for stats in per_stage.values()
        )
        return {
            "update_unit": (
                "stage_local_accumulated_optimizer_step"
                if self.cfg.gradient_accumulation_steps > 1
                else "stage_local_request_optimizer_step"
            ),
            "effective_optimizer_batch": self.cfg.gradient_accumulation_steps,
            "note": (
                "For gradient_accumulation_steps > 1, update_events count committed "
                "accumulation windows, not individual requests."
            ),
            "duplicate_update_events": duplicate_update_events,
            "replay_events": replay_events,
            "per_stage": dict(sorted(per_stage.items(), key=lambda item: int(item[0]))),
            "duplicate_details": duplicate_details[:20],
            "duplicate_details_truncated": max(0, len(duplicate_details) - 20),
        }

    def retained_progress_summary(self) -> dict[str, Any]:
        """Count local updates kept even though the end-to-end request failed."""
        failed_seqs = set(self.failed)
        per_stage: dict[str, dict[str, int]] = {
            str(stage_id): {
                "update_events": 0,
                "updates_on_completed_requests": 0,
                "retained_updates_on_failed_requests": 0,
            }
            for stage_id in range(self.cfg.num_chunks)
        }
        for seq, attempts in self.request_attempts.items():
            for attempt in attempts:
                if not attempt.get("update_applied"):
                    continue
                stage = per_stage[str(int(attempt["stage_id"]))]
                stage["update_events"] += 1
                if seq in failed_seqs:
                    stage["retained_updates_on_failed_requests"] += 1
                else:
                    stage["updates_on_completed_requests"] += 1

        total_updates = sum(row["update_events"] for row in per_stage.values())
        retained_updates = sum(row["retained_updates_on_failed_requests"] for row in per_stage.values())
        return {
            "update_unit": (
                "stage_local_accumulated_optimizer_step"
                if self.cfg.gradient_accumulation_steps > 1
                else "stage_local_request_optimizer_step"
            ),
            "effective_optimizer_batch": self.cfg.gradient_accumulation_steps,
            "failed_requests": len(failed_seqs),
            "completed_requests": len(self.completed),
            "total_update_events": total_updates,
            "retained_updates_on_failed_requests": retained_updates,
            "retained_update_fraction": (retained_updates / total_updates) if total_updates else 0.0,
            "per_stage": per_stage,
        }

    def write_ledger(
        self,
        *,
        event_type: str,
        task: dict[str, Any],
        worker_id: str,
        success: bool,
        update_applied: bool,
        message: str,
    ) -> None:
        self.ledger_writer.writerow(
            {
                "event_seq": self.next_event_seq,
                "event_type": event_type,
                "seq": task["seq"],
                "request_id": task["request_id"],
                "stage_id": task["stage_id"],
                "worker_id": worker_id,
                "attempt": task["attempt"],
                "success": success,
                "update_applied": update_applied,
                "message": message,
            }
        )
        self.next_event_seq += 1
        self.ledger_handle.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Central scheduler lab for real BP-free stage training and recovery policy experiments."
    )
    parser.add_argument("--model_name", default="tinyllama")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--eval_manifest", type=Path, default=None)
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
    parser.add_argument("--stage_devices", default=None, help="One device per stage, e.g. cuda:0,cuda:1,cuda:2.")
    parser.add_argument(
        "--topology",
        choices=["phone_fixed", "worker_pool"],
        default="phone_fixed",
        help="phone_fixed keeps one worker/state per stage; worker_pool allows stage replicas.",
    )
    parser.add_argument(
        "--workers",
        default="",
        help="Comma-separated STAGE:DEVICE specs. Allows replicas, e.g. 0:cuda:0,1:cuda:1,2:cuda:2,2:cuda:3.",
    )
    parser.add_argument(
        "--standby_worker_ids",
        default="",
        help="Comma-separated worker ids kept warm but idle until explicitly selected, e.g. w3-s1.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--eval_limit", type=int, default=None)
    parser.add_argument("--max_inflight", type=int, default=6)
    parser.add_argument("--scheduler_policy", choices=["fifo", "recovery_first"], default="recovery_first")
    parser.add_argument("--task_timeout_ms", type=float, default=0.0)
    parser.add_argument(
        "--timeout_policy",
        choices=["observe", "retry_stage", "cancel_retry_on_late"],
        default="observe",
    )
    parser.add_argument(
        "--recovery_policy",
        choices=[
            "retry_stage",
            "retry_from_boundary",
            "retry_from_zero",
            "replay_after_update",
            "migrate_from_boundary",
            "wait_for_rejoin",
            "skip",
        ],
        default="retry_stage",
    )
    parser.add_argument("--max_attempts", type=int, default=3)
    parser.add_argument(
        "--worker_rejoin_delay_ms",
        type=float,
        default=0.0,
        help="For wait_for_rejoin: deterministic unavailable interval after a worker failure.",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=1,
        help="For migrate_from_boundary: capture stage state every K local updates; later updates are journaled.",
    )
    parser.add_argument("--failure_mode", choices=["none", "once", "random"], default="none")
    parser.add_argument("--failure_rate", type=float, default=0.0)
    parser.add_argument("--failure_stage", type=int, default=None)
    parser.add_argument("--failure_seq", type=int, default=None)
    parser.add_argument("--failure_attempt", type=int, default=0)
    parser.add_argument(
        "--failure_point",
        choices=["before_execute", "after_update", "delay_before_execute", "delay_after_update"],
        default="before_execute",
    )
    parser.add_argument("--failure_delay_ms", type=float, default=0.0)
    parser.add_argument(
        "--offline_stage",
        type=int,
        default=None,
        help="Make this stage reject all train tasks in the configured sequence window.",
    )
    parser.add_argument("--offline_start_seq", type=int, default=None)
    parser.add_argument("--offline_end_seq", type=int, default=None)
    parser.add_argument(
        "--transient_dropout_mask",
        type=Path,
        default=None,
        help=(
            "JSON availability mask keyed by logical update window and stage. "
            "Selected stages reject attempt 0 and are available to later attempts."
        ),
    )
    parser.add_argument("--train_chunks", default="all")
    parser.add_argument(
        "--stage_train_strides",
        default="",
        help="Optional STAGE:STRIDE list. A stride >1 makes that stage forward-only except when seq % stride == 0.",
    )
    parser.add_argument(
        "--stage_update_policy",
        choices=["stride", "queue_gated"],
        default="stride",
        help="stride uses only --stage_train_strides; queue_gated also skips updates when a stage backlog is high.",
    )
    parser.add_argument(
        "--stage_update_queue_thresholds",
        default="",
        help="Optional STAGE:DEPTH list for queue_gated. If backlog after dispatch is above DEPTH, run forward-only.",
    )
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=(
            "Accumulate this many train requests per stage before optimizer.step(). "
            "Use this for optimizer-batch matched 1F1B comparisons."
        ),
    )
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--sgd_momentum", type=float, default=0.0)
    parser.add_argument("--sgd_dampening", type=float, default=0.0)
    parser.add_argument("--sgd_weight_decay", type=float, default=0.0)
    parser.add_argument("--sgd_nesterov", action="store_true")
    parser.add_argument("--belief_transport_mode", default="terminal", choices=["full", "terminal", "none"])
    parser.add_argument("--alpha", type=float, default=1.0)
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
    parser.add_argument("--local_readout_adapter_bottleneck", type=int, default=0)
    parser.add_argument(
        "--local_readout_adapter_stages",
        choices=["none", "middle", "all"],
        default="none",
    )
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--request_prefix", default="bpfree-runtime")
    parser.add_argument("--progress_interval", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_chunks <= 0:
        raise ValueError("--num_chunks must be positive.")
    if args.max_inflight <= 0:
        raise ValueError("--max_inflight must be positive.")
    if args.max_attempts <= 0:
        raise ValueError("--max_attempts must be positive.")
    if args.worker_rejoin_delay_ms < 0:
        raise ValueError("--worker_rejoin_delay_ms must be non-negative.")
    if args.checkpoint_interval <= 0:
        raise ValueError("--checkpoint_interval must be positive.")
    if args.task_timeout_ms < 0:
        raise ValueError("--task_timeout_ms must be non-negative.")
    if args.failure_delay_ms < 0:
        raise ValueError("--failure_delay_ms must be non-negative.")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient_accumulation_steps must be positive.")
    if args.local_readout_adapter_bottleneck < 0:
        raise ValueError("--local_readout_adapter_bottleneck must be non-negative.")
    adapter_enabled = args.local_readout_adapter_stages != "none"
    if adapter_enabled != (args.local_readout_adapter_bottleneck > 0):
        raise ValueError(
            "Enable the local readout adapter with both a positive bottleneck and non-none stages."
        )
    if args.failure_mode == "random" and not (0.0 <= args.failure_rate <= 1.0):
        raise ValueError("--failure_rate must be in [0, 1].")
    offline_values = (args.offline_stage, args.offline_start_seq, args.offline_end_seq)
    if args.transient_dropout_mask is not None and any(
        value is not None for value in offline_values
    ):
        raise ValueError("Use either a contiguous offline window or --transient_dropout_mask")
    if any(value is not None for value in offline_values):
        if any(value is None for value in offline_values):
            raise ValueError(
                "--offline_stage, --offline_start_seq, and --offline_end_seq must be supplied together."
            )
        assert args.offline_stage is not None
        assert args.offline_start_seq is not None
        assert args.offline_end_seq is not None
        if args.offline_stage < 0 or args.offline_stage >= args.num_chunks:
            raise ValueError("--offline_stage must name a configured stage.")
        if args.offline_start_seq < 0 or args.offline_end_seq <= args.offline_start_seq:
            raise ValueError("offline window must satisfy 0 <= start < end.")

    mp.set_start_method("spawn", force=True)
    resolved_model = resolve_model_name(args.model_name)
    records = read_manifest(args.manifest, args.limit)
    transient_offline_windows = load_transient_dropout_mask(
        args.transient_dropout_mask,
        num_stages=args.num_chunks,
        window_size=args.gradient_accumulation_steps,
    )
    logical_windows = (
        len(records) + args.gradient_accumulation_steps - 1
    ) // args.gradient_accumulation_steps
    out_of_range_windows = sorted(
        {
            window_id
            for windows in transient_offline_windows.values()
            for window_id in windows
            if window_id >= logical_windows
        }
    )
    if out_of_range_windows:
        raise ValueError(
            f"Transient dropout mask references windows outside 0..{logical_windows - 1}: "
            f"{out_of_range_windows[:8]}"
        )
    eval_records = read_manifest(args.eval_manifest, args.eval_limit) if args.eval_manifest is not None else None
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
    if args.gradient_accumulation_steps > 1:
        if args.task_timeout_ms > 0:
            raise ValueError("--gradient_accumulation_steps > 1 recovery currently requires --task_timeout_ms 0.")
        if len(records) % args.gradient_accumulation_steps != 0:
            raise ValueError(
                f"Training records ({len(records)}) must be divisible by "
                f"--gradient_accumulation_steps ({args.gradient_accumulation_steps})."
            )
    train_chunks = parse_train_chunks(args.train_chunks, args.num_chunks)
    stage_train_strides = parse_stage_train_strides(args.stage_train_strides, args.num_chunks)
    stage_update_queue_thresholds = parse_stage_int_map(
        args.stage_update_queue_thresholds,
        args.num_chunks,
        name="--stage_update_queue_thresholds",
        minimum=0,
    )
    if args.stage_update_policy == "queue_gated" and not stage_update_queue_thresholds:
        raise ValueError("--stage_update_policy queue_gated requires --stage_update_queue_thresholds.")
    if args.gradient_accumulation_steps > 1 and (
        args.stage_update_policy != "stride" or any(stride != 1 for stride in stage_train_strides.values())
    ):
        raise ValueError(
            "--gradient_accumulation_steps > 1 currently requires --stage_update_policy stride "
            "and all --stage_train_strides = 1."
        )
    if args.gradient_accumulation_steps > 1:
        supported_window_boundary_recovery = (
            args.failure_mode == "none"
            or (
                args.failure_mode == "once"
                and args.failure_point == "before_execute"
                and args.failure_seq is not None
                and args.failure_seq % args.gradient_accumulation_steps == 0
                and args.task_timeout_ms == 0
                and args.recovery_policy in {"wait_for_rejoin", "migrate_from_boundary"}
                and args.stage_update_policy == "stride"
                and all(stride == 1 for stride in stage_train_strides.values())
            )
        )
        if not supported_window_boundary_recovery:
            raise ValueError(
                "--gradient_accumulation_steps > 1 currently supports only the window-boundary recovery case: "
                "failure_mode once, failure_point before_execute, failure_seq divisible by gradient_accumulation_steps, "
                "task_timeout_ms 0, recovery_policy wait_for_rejoin|migrate_from_boundary, "
                "stage_update_policy stride, and all stage_train_strides = 1."
            )
    if validation_records is not None:
        interval_records = args.validation_interval_steps * args.gradient_accumulation_steps
        if interval_records > len(records):
            raise ValueError("--validation_interval_steps exceeds the training budget")
        if len(validation_records) == 0:
            raise ValueError("Validation manifest is empty")
        if args.failure_mode != "none" or args.task_timeout_ms > 0:
            raise ValueError("Intermediate validation currently requires a failure-free, timeout-free training run")
    worker_specs = parse_worker_specs(args.workers, args.num_chunks, args.stage_devices)
    validate_topology(worker_specs, args.num_chunks, args.topology)
    standby_worker_ids = {item.strip() for item in args.standby_worker_ids.split(",") if item.strip()}
    known_worker_ids = {spec.worker_id for spec in worker_specs}
    unknown_standby = sorted(standby_worker_ids - known_worker_ids)
    if unknown_standby:
        raise ValueError(f"Unknown --standby_worker_ids: {unknown_standby}")
    if args.recovery_policy == "migrate_from_boundary":
        if args.topology != "worker_pool":
            raise ValueError("migrate_from_boundary requires --topology worker_pool with a warm stage replica.")
        if args.failure_mode != "once" or args.failure_stage is None:
            raise ValueError(
                "migrate_from_boundary currently requires --failure_mode once and --failure_stage so "
                "the scheduler can checkpoint and replace one failed stage worker."
            )
        replicas = [spec for spec in worker_specs if spec.stage_id == args.failure_stage]
        if len(replicas) < 2:
            raise ValueError("migrate_from_boundary requires at least two workers for --failure_stage.")
        if not any(spec.worker_id in standby_worker_ids for spec in replicas):
            raise ValueError(
                "migrate_from_boundary requires one replica for --failure_stage in --standby_worker_ids "
                "so checkpoints have a stable primary source before failover."
            )
    if args.recovery_policy == "wait_for_rejoin" and args.worker_rejoin_delay_ms <= 0:
        raise ValueError("wait_for_rejoin requires --worker_rejoin_delay_ms > 0.")
    cfg = LabConfig(
        resolved_model=resolved_model,
        num_chunks=args.num_chunks,
        train_chunks=train_chunks,
        trainable_mode=args.trainable_mode,
        dtype_name=args.dtype,
        belief_transport_mode=normalize_belief_transport_mode(args.belief_transport_mode),
        alpha=args.alpha,
        label_smoothing=args.label_smoothing,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_targets=args.lora_targets,
        lora_init_std=args.lora_init_std,
        lora_init_seed=args.lora_init_seed,
        local_readout_adapter_bottleneck=args.local_readout_adapter_bottleneck,
        local_readout_adapter_stages=args.local_readout_adapter_stages,
        learning_rate=args.learning_rate,
        grad_clip=args.grad_clip,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optimizer=args.optimizer,
        sgd_momentum=args.sgd_momentum,
        sgd_dampening=args.sgd_dampening,
        sgd_weight_decay=args.sgd_weight_decay,
        sgd_nesterov=args.sgd_nesterov,
        stage_update_policy=args.stage_update_policy,
        stage_train_strides=stage_train_strides,
        stage_update_queue_thresholds=stage_update_queue_thresholds,
        seed=args.seed,
        progress_interval=args.progress_interval,
    )
    failure_rule = FailureRule(
        mode=args.failure_mode,
        random_rate=args.failure_rate,
        fail_stage=args.failure_stage,
        fail_seq=args.failure_seq,
        fail_attempt=args.failure_attempt,
        fail_point=args.failure_point,
        delay_ms=args.failure_delay_ms,
        seed=args.seed + 1009,
        offline_stage=args.offline_stage,
        offline_start_seq=args.offline_start_seq,
        offline_end_seq=args.offline_end_seq,
        transient_mask_path=(
            str(args.transient_dropout_mask.resolve())
            if args.transient_dropout_mask is not None
            else ""
        ),
        transient_window_size=args.gradient_accumulation_steps,
        transient_offline_windows=transient_offline_windows,
    )
    print(
        f"Starting scheduler lab model={resolved_model} records={len(records)} "
        f"eval_records={len(eval_records) if eval_records is not None else 0} "
        f"workers={[spec.__dict__ for spec in worker_specs]} "
        f"topology={args.topology} policy={args.scheduler_policy} "
        f"recovery={args.recovery_policy} timeout={args.task_timeout_ms}ms/{args.timeout_policy} "
        f"failure={failure_rule}",
        flush=True,
    )
    lab = SchedulerLab(
        records=records,
        eval_records=eval_records,
        validation_records=validation_records,
        manifest_dir=args.manifest.parent,
        eval_manifest_dir=args.eval_manifest.parent if args.eval_manifest is not None else None,
        validation_manifest_dir=(
            args.validation_manifest.parent if args.validation_manifest is not None else None
        ),
        worker_specs=worker_specs,
        cfg=cfg,
        output_dir=args.output_dir,
        max_inflight=args.max_inflight,
        scheduler_policy=args.scheduler_policy,
        recovery_policy=args.recovery_policy,
        topology=args.topology,
        task_timeout_ms=args.task_timeout_ms,
        timeout_policy=args.timeout_policy,
        max_attempts=args.max_attempts,
        failure_injector=FailureInjector(failure_rule),
        standby_worker_ids=standby_worker_ids,
        worker_rejoin_delay_ms=args.worker_rejoin_delay_ms,
        checkpoint_interval=args.checkpoint_interval,
        request_prefix=args.request_prefix,
        validation_interval_steps=args.validation_interval_steps,
    )
    lab.run()


if __name__ == "__main__":
    main()
