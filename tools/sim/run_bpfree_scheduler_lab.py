#!/usr/bin/env python3
"""Server scheduler lab for BP-free stage training and recovery experiments.

This is the experimental scheduler, not the fixed FIFO phone runner and not the
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
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM

from run_bpfree_lora_label_experiment import (
    configure_lora_trainable,
    inject_lora_adapters,
    label_choice_metrics,
    load_tensor,
    one_token_choice_ids,
    parse_train_chunks,
    read_manifest,
    resolve_dtype,
    resolve_model_name,
)
from run_bpfree_lora_pipeline_multigpu import (
    build_optimizer,
    build_stage_chunk,
    normalize_belief_transport_mode,
    tensor_to_cpu,
)


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
    seed: int


@dataclass
class LabConfig:
    resolved_model: str
    num_chunks: int
    train_chunks: set[int]
    dtype_name: str
    belief_transport_mode: str
    alpha: float
    label_smoothing: float
    lora_rank: int
    lora_alpha: float
    lora_targets: str
    lora_init_std: float
    learning_rate: Optional[float]
    grad_clip: float
    optimizer: str
    sgd_momentum: float
    sgd_dampening: float
    sgd_weight_decay: float
    sgd_nesterov: bool
    seed: int
    progress_interval: int


class FailureInjector:
    def __init__(self, rule: FailureRule) -> None:
        self.rule = rule
        self.random = random.Random(rule.seed)
        self.triggered_once: set[tuple[int, int, int]] = set()

    def choose_fail_point(self, *, seq: int, stage_id: int, attempt: int) -> str:
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


def load_initial_state(record: dict[str, Any], manifest_dir: Path) -> dict[str, Any]:
    tensors = record["tensors"]
    return {
        "hidden": load_tensor(manifest_dir, tensors["hidden_states"]),
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
    for key in ("hidden", "attention_mask", "position_ids", "labels", "prev_log_probs"):
        value = state.get(key)
        if not isinstance(value, torch.Tensor):
            moved[key] = None
        elif key in {"hidden", "attention_mask"} and compute_dtype is not None:
            moved[key] = value.to(device=device, dtype=compute_dtype, non_blocking=False)
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
        injected = inject_lora_adapters(
            module=model,
            target_names={item.strip() for item in cfg.lora_targets.split(",") if item.strip()},
            rank=cfg.lora_rank,
            alpha=cfg.lora_alpha,
            init_std=cfg.lora_init_std,
        )
        trainable, frozen = configure_lora_trainable(model)
        chunk = build_stage_chunk(
            model=model,
            stage_id=spec.stage_id,
            num_chunks=cfg.num_chunks,
            belief_transport_mode=cfg.belief_transport_mode,
            alpha=cfg.alpha,
            label_smoothing=cfg.label_smoothing,
        )
        chunk.to(device)
        local_params = [param for param in chunk.parameters() if param.requires_grad]
        optimizer = (
            build_worker_optimizer(local_params, cfg)
            if spec.stage_id in cfg.train_chunks
            else None
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        print(
            f"[{spec.worker_id}] ready lora_modules={injected} all_trainable={trainable} "
            f"frozen={frozen} local_trainable={sum(p.numel() for p in local_params)}",
            flush=True,
        )

        processed = 0
        while True:
            task = input_queue.get()
            if task is None:
                result_queue.put({"kind": "worker_stopped", "worker_id": spec.worker_id})
                break

            task_started = time.perf_counter()
            fail_point = task.get("fail_point", "none")
            if fail_point == "before_execute":
                result_queue.put(
                    failure_result(
                        task=task,
                        spec=spec,
                        message="Injected failure before execution.",
                        update_applied=False,
                        started_at=task_started,
                    )
                )
                continue

            train_this_stage = task["mode"] == "train" and spec.stage_id in cfg.train_chunks
            state = move_state_to_device(task["input_state"], device, chunk.compute_dtype())

            lr = cfg.learning_rate
            if lr is None:
                lr = task["record"].get("learning_rate")
            if train_this_stage:
                assert optimizer is not None
                if lr is not None:
                    for group in optimizer.param_groups:
                        group["lr"] = float(lr)
                optimizer.zero_grad(set_to_none=True)
            chunk.train(train_this_stage)

            execute_started = time.perf_counter()
            optimizer_ms = 0.0
            with torch.set_grad_enabled(train_this_stage):
                loss, next_hidden, next_log_probs = chunk(
                    hidden_states=state["hidden"],
                    attention_mask=state["attention_mask"],
                    position_ids=state["position_ids"],
                    labels=state["labels"],
                    prev_log_probs=state["prev_log_probs"],
                )
                if train_this_stage:
                    loss.backward()
                    if cfg.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(chunk.parameters(), cfg.grad_clip)
                    optimizer_started = time.perf_counter()
                    assert optimizer is not None
                    optimizer.step()
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    optimizer_ms = (time.perf_counter() - optimizer_started) * 1000.0
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            execute_ms = (time.perf_counter() - execute_started) * 1000.0

            local_loss = float(loss.detach().cpu().item())
            update_applied = bool(train_this_stage)
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
            else:
                output_state = {
                    "hidden": tensor_to_cpu(next_hidden),
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
                "local_loss": local_loss,
                "execute_ms": execute_ms,
                "optimizer_ms": optimizer_ms,
                "stage_total_ms": (time.perf_counter() - task_started) * 1000.0,
                "input_state_bytes": state_bytes(task["input_state"]),
                "output_state_bytes": state_bytes(output_state),
                "output_log_probs_bytes": (
                    int(final_log_probs.numel() * final_log_probs.element_size())
                    if isinstance(final_log_probs, torch.Tensor)
                    else 0
                ),
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
                    "labels": tensor_to_cpu(state["labels"]),
                    "metric": metric,
                    "update_applied": update_applied,
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
    local_loss: Optional[float] = None,
) -> dict[str, Any]:
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
            "train": task["mode"] == "train",
            "local_loss": local_loss if local_loss is not None else "",
            "execute_ms": "",
            "optimizer_ms": "",
            "stage_total_ms": (time.perf_counter() - started_at) * 1000.0,
            "input_state_bytes": state_bytes(task["input_state"]),
            "output_state_bytes": 0,
            "output_log_probs_bytes": 0,
            "cuda_peak_memory_allocated": "",
            "cuda_peak_memory_reserved": "",
            "failure": message,
        },
    }


def result_fieldnames() -> list[str]:
    return [
        "seq",
        "request_id",
        "dataset_index",
        "response",
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
        "local_loss",
        "execute_ms",
        "optimizer_ms",
        "stage_total_ms",
        "input_state_bytes",
        "output_state_bytes",
        "output_log_probs_bytes",
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


class SchedulerLab:
    def __init__(
        self,
        *,
        records: list[dict[str, Any]],
        manifest_dir: Path,
        worker_specs: list[WorkerSpec],
        cfg: LabConfig,
        output_dir: Path,
        max_inflight: int,
        scheduler_policy: str,
        recovery_policy: str,
        max_attempts: int,
        failure_injector: FailureInjector,
        request_prefix: str,
    ) -> None:
        self.records = records
        self.manifest_dir = manifest_dir
        self.worker_specs = worker_specs
        self.cfg = cfg
        self.output_dir = output_dir
        self.max_inflight = max_inflight
        self.scheduler_policy = scheduler_policy
        self.recovery_policy = recovery_policy
        self.max_attempts = max_attempts
        self.failure_injector = failure_injector
        self.request_prefix = request_prefix

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
        self.boundary_cache: dict[tuple[int, int], dict[str, Any]] = {}
        self.completed: dict[int, dict[str, Any]] = {}
        self.failed: dict[int, dict[str, Any]] = {}
        self.inflight_requests: set[int] = set()
        self.next_record_index = 0
        self.next_task_id = 0
        self.next_event_seq = 0
        self.stage_attempts: dict[tuple[int, int], int] = defaultdict(int)
        self.request_attempts: dict[int, list[dict[str, Any]]] = defaultdict(list)

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

    def make_task(self, *, seq: int, stage_id: int, input_state: dict[str, Any], attempt: int) -> dict[str, Any]:
        fail_point = self.failure_injector.choose_fail_point(seq=seq, stage_id=stage_id, attempt=attempt)
        task = {
            "task_id": self.next_task_id,
            "phase": "train",
            "seq": seq,
            "request_id": f"{self.request_prefix}-{seq:06d}",
            "record": self.records[seq],
            "manifest_dir": str(self.manifest_dir),
            "mode": "train",
            "stage_id": stage_id,
            "attempt": attempt,
            "input_state": input_state,
            "fail_point": fail_point,
        }
        self.next_task_id += 1
        return task

    def enqueue_task(self, task: dict[str, Any], recovery: bool = False) -> None:
        stage_id = task["stage_id"]
        if recovery and self.scheduler_policy == "recovery_first":
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
            task = self.make_task(seq=seq, stage_id=0, input_state=initial_state, attempt=0)
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
                if self.worker_busy[spec.worker_id]:
                    continue
                if not self.ready[stage_id]:
                    break
                task = self.ready[stage_id].popleft()
                self.worker_busy[spec.worker_id] = True
                self.inflight_by_worker[spec.worker_id] = task
                self.worker_queues[spec.worker_id].put(task)
                self.write_ledger(
                    event_type="dispatch",
                    task=task,
                    worker_id=spec.worker_id,
                    success=True,
                    update_applied=False,
                    message=f"dispatched to {spec.device}",
                )

    def handle_stage_result(self, result: dict[str, Any]) -> None:
        worker_id = result.get("worker_id", "")
        if worker_id in self.worker_busy:
            self.worker_busy[worker_id] = False
        task = self.inflight_by_worker.pop(worker_id, None)
        if task is None:
            raise RuntimeError(f"Got result from idle/unknown worker {worker_id}: {result}")

        self.write_metric(result.get("metric", {}))
        self.request_attempts[result["seq"]].append(
            {
                "stage_id": result["stage_id"],
                "attempt": result["attempt"],
                "worker_id": worker_id,
                "success": result["success"],
                "update_applied": result.get("update_applied", False),
                "message": result.get("message", ""),
            }
        )

        if result["success"]:
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

    def handle_success(self, task: dict[str, Any], result: dict[str, Any]) -> None:
        seq = task["seq"]
        stage_id = task["stage_id"]
        if stage_id == self.cfg.num_chunks - 1:
            row = self.finish_request(task, result)
            self.completed[seq] = row
            self.inflight_requests.discard(seq)
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
        )
        self.enqueue_task(next_task)

    def handle_failure(self, task: dict[str, Any], result: dict[str, Any]) -> None:
        seq = task["seq"]
        stage_id = task["stage_id"]
        next_attempt = task["attempt"] + 1
        self.stage_attempts[(seq, stage_id)] = next_attempt

        if next_attempt >= self.max_attempts or self.recovery_policy == "skip":
            row = self.failed_row(task, result, "max attempts reached or skip policy")
            self.failed[seq] = row
            self.inflight_requests.discard(seq)
            self.write_result(row)
            return

        if self.recovery_policy in {"retry_stage", "retry_from_boundary"}:
            retry_task = self.make_task(
                seq=seq,
                stage_id=stage_id,
                input_state=task["input_state"],
                attempt=next_attempt,
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
            for key in list(self.boundary_cache.keys()):
                if key[0] == seq:
                    del self.boundary_cache[key]
            self.stage_attempts[(seq, 0)] += 1
            retry_task = self.make_task(
                seq=seq,
                stage_id=0,
                input_state=load_initial_state(self.records[seq], self.manifest_dir),
                attempt=self.stage_attempts[(seq, 0)],
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
        if isinstance(final_log_probs, torch.Tensor) and isinstance(labels, torch.Tensor):
            choice_ids = one_token_choice_ids(task["record"])
            correct, count, choice_loss = label_choice_metrics(final_log_probs, labels, choice_ids)
        else:
            correct, count, choice_loss = 0, 0, 0.0
        response = (task["record"].get("text") or {}).get("response", "").strip()
        return {
            "seq": task["seq"],
            "request_id": task["request_id"],
            "dataset_index": int(task["record"].get("dataset_index", -1)),
            "response": response,
            "mode": task["mode"],
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
            "seq": task["seq"],
            "request_id": task["request_id"],
            "dataset_index": int(task["record"].get("dataset_index", -1)),
            "response": (task["record"].get("text") or {}).get("response", "").strip(),
            "mode": task["mode"],
            "status": "failed",
            "loss": result.get("local_loss", ""),
            "choice_correct": 0,
            "choice_count": 0,
            "choice_accuracy": 0.0,
            "choice_loss": 0.0,
            "attempts_json": json.dumps(self.request_attempts[task["seq"]]),
            "message": message,
        }

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

        started = time.perf_counter()
        self.start_workers()
        try:
            while len(self.completed) + len(self.failed) < len(self.records):
                self.admit_requests()
                self.dispatch_ready()
                try:
                    result = self.result_queue.get(timeout=0.1)
                except queue.Empty:
                    self.check_worker_errors()
                    continue
                if result.get("kind") == "stage_result":
                    self.handle_stage_result(result)
                elif result.get("kind") == "worker_error":
                    raise RuntimeError(f"worker error: {result}")
                elif result.get("kind") == "worker_stopped":
                    continue
                else:
                    raise RuntimeError(f"Unexpected scheduler result: {result}")
                self.check_worker_errors()
        finally:
            self.stop_workers()
            self.result_handle.close()
            self.metric_handle.close()
            self.ledger_handle.close()

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        correct = sum(int(row["choice_correct"]) for row in self.completed.values())
        count = sum(int(row["choice_count"]) for row in self.completed.values())
        losses = [float(row["loss"]) for row in self.completed.values() if row["loss"] != ""]
        summary = {
            "runner": "scheduler_lab",
            "records": len(self.records),
            "completed": len(self.completed),
            "failed": len(self.failed),
            "choice_correct": correct,
            "choice_count": count,
            "choice_accuracy": (correct / count) if count else 0.0,
            "avg_loss": sum(losses) / len(losses) if losses else 0.0,
            "wall_ms": elapsed_ms,
            "throughput_per_s": len(self.completed) / (elapsed_ms / 1000.0) if elapsed_ms > 0 else 0.0,
            "scheduler_policy": self.scheduler_policy,
            "recovery_policy": self.recovery_policy,
            "max_inflight": self.max_inflight,
            "max_attempts": self.max_attempts,
            "workers": [spec.__dict__ for spec in self.worker_specs],
            "results_csv": str(self.results_path),
            "metrics_csv": str(self.metrics_path),
            "ledger_csv": str(self.ledger_path),
        }
        summary_path = self.output_dir / "scheduler_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
        print(f"Wrote {summary_path}", flush=True)
        return summary

    def check_worker_errors(self) -> None:
        for process in self.processes:
            if process.exitcode not in (None, 0):
                raise RuntimeError(f"Worker process {process.name} exited with code {process.exitcode}")

    def write_result(self, row: dict[str, Any]) -> None:
        self.result_writer.writerow(row)
        self.result_handle.flush()

    def write_metric(self, metric: dict[str, Any]) -> None:
        self.metric_writer.writerow({name: metric.get(name, "") for name in metric_fieldnames()})
        self.metric_handle.flush()

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
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_chunks", type=int, default=3)
    parser.add_argument("--stage_devices", default=None, help="One device per stage, e.g. cuda:0,cuda:1,cuda:2.")
    parser.add_argument(
        "--workers",
        default="",
        help="Comma-separated STAGE:DEVICE specs. Allows replicas, e.g. 0:cuda:0,1:cuda:1,2:cuda:2,2:cuda:3.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_inflight", type=int, default=6)
    parser.add_argument("--scheduler_policy", choices=["fifo", "recovery_first"], default="recovery_first")
    parser.add_argument(
        "--recovery_policy",
        choices=["retry_stage", "retry_from_boundary", "retry_from_zero", "skip"],
        default="retry_stage",
    )
    parser.add_argument("--max_attempts", type=int, default=3)
    parser.add_argument("--failure_mode", choices=["none", "once", "random"], default="none")
    parser.add_argument("--failure_rate", type=float, default=0.0)
    parser.add_argument("--failure_stage", type=int, default=None)
    parser.add_argument("--failure_seq", type=int, default=None)
    parser.add_argument("--failure_attempt", type=int, default=0)
    parser.add_argument("--failure_point", choices=["before_execute", "after_update"], default="before_execute")
    parser.add_argument("--train_chunks", default="all")
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--sgd_momentum", type=float, default=0.0)
    parser.add_argument("--sgd_dampening", type=float, default=0.0)
    parser.add_argument("--sgd_weight_decay", type=float, default=0.0)
    parser.add_argument("--sgd_nesterov", action="store_true")
    parser.add_argument("--belief_transport_mode", default="terminal", choices=["full", "terminal", "none"])
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_targets", default="q_proj,v_proj")
    parser.add_argument("--lora_init_std", type=float, default=0.01)
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--request_prefix", default="scheduler-lab")
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
    if args.failure_mode == "random" and not (0.0 <= args.failure_rate <= 1.0):
        raise ValueError("--failure_rate must be in [0, 1].")

    mp.set_start_method("spawn", force=True)
    resolved_model = resolve_model_name(args.model_name)
    records = read_manifest(args.manifest, args.limit)
    train_chunks = parse_train_chunks(args.train_chunks, args.num_chunks)
    worker_specs = parse_worker_specs(args.workers, args.num_chunks, args.stage_devices)
    cfg = LabConfig(
        resolved_model=resolved_model,
        num_chunks=args.num_chunks,
        train_chunks=train_chunks,
        dtype_name=args.dtype,
        belief_transport_mode=normalize_belief_transport_mode(args.belief_transport_mode),
        alpha=args.alpha,
        label_smoothing=args.label_smoothing,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_targets=args.lora_targets,
        lora_init_std=args.lora_init_std,
        learning_rate=args.learning_rate,
        grad_clip=args.grad_clip,
        optimizer=args.optimizer,
        sgd_momentum=args.sgd_momentum,
        sgd_dampening=args.sgd_dampening,
        sgd_weight_decay=args.sgd_weight_decay,
        sgd_nesterov=args.sgd_nesterov,
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
        seed=args.seed + 1009,
    )
    print(
        f"Starting scheduler lab model={resolved_model} records={len(records)} "
        f"workers={[spec.__dict__ for spec in worker_specs]} "
        f"policy={args.scheduler_policy} recovery={args.recovery_policy} failure={failure_rule}",
        flush=True,
    )
    lab = SchedulerLab(
        records=records,
        manifest_dir=args.manifest.parent,
        worker_specs=worker_specs,
        cfg=cfg,
        output_dir=args.output_dir,
        max_inflight=args.max_inflight,
        scheduler_policy=args.scheduler_policy,
        recovery_policy=args.recovery_policy,
        max_attempts=args.max_attempts,
        failure_injector=FailureInjector(failure_rule),
        request_prefix=args.request_prefix,
    )
    lab.run()


if __name__ == "__main__":
    main()
