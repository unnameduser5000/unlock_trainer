#!/usr/bin/env python3
"""Run real BP-free chunk-local LoRA training as a multi-GPU stage pipeline.

This is the server-side counterpart of the Android stage pipeline:

    stage0(request i) -> stage1(request i) -> ... -> terminal

Each stage is a long-lived Python process pinned to one device such as cuda:0.
The stage owns its local chunk weights and optimizer. It performs real PyTorch
forward, local CE/KD loss, local backward, and optimizer step when mode=train.
Boundary tensors are moved through multiprocessing queues as CPU tensors, which
keeps the runner simple and makes the GPU placement explicit.

This script is intentionally not a PipeDream 1F1B runner. There is no
cross-stage backward edge. The scheduled unit is one atomic local stage task.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import queue
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM

from run_bpfree_lora_label_experiment import (
    configure_lora_trainable,
    get_model_parts,
    infer_module_compute_dtype,
    inject_lora_adapters,
    label_choice_metrics,
    load_tensor,
    one_token_choice_ids,
    parse_train_chunks,
    read_manifest,
    resolve_dtype,
    resolve_model_name,
)


@dataclass
class StageWorkerConfig:
    model_name: str
    resolved_model: str
    num_chunks: int
    train_chunks: list[int]
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


class ServerBpfreeChunk(nn.Module):
    def __init__(
        self,
        *,
        chunk_idx: int,
        layers: list[nn.Module],
        final_norm: nn.Module,
        lm_head: nn.Module,
        vocab_size: int,
        rotary_emb: Optional[nn.Module],
        is_terminal_chunk: bool,
        belief_transport_mode: str,
        alpha: float,
        label_smoothing: float,
    ) -> None:
        super().__init__()
        self.chunk_idx = chunk_idx
        self.layers = nn.ModuleList(layers)
        self.final_norm = final_norm
        self.lm_head = lm_head
        self.vocab_size = vocab_size
        self.rotary_emb = rotary_emb
        self.is_terminal_chunk = is_terminal_chunk
        self.belief_transport_mode = normalize_belief_transport_mode(belief_transport_mode)
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    @property
    def consumes_prev_log_probs(self) -> bool:
        return self.chunk_idx > 0 and self.belief_transport_mode == "full"

    @property
    def uses_belief_loss(self) -> bool:
        return self.consumes_prev_log_probs and self.alpha < 1.0

    @property
    def returns_full_log_probs(self) -> bool:
        return self.belief_transport_mode == "full" or (
            self.belief_transport_mode == "terminal" and self.is_terminal_chunk
        )

    def compute_dtype(self) -> torch.dtype:
        return infer_module_compute_dtype(self)

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        labels: torch.Tensor,
        prev_log_probs: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        dtype = self.compute_dtype()
        hidden_states = hidden_states.to(dtype=dtype)
        attention_mask = attention_mask.to(dtype=dtype)
        position_ids = position_ids.long()
        labels = labels.long()

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

        logits = self.lm_head(self.final_norm(curr_hidden))
        shift_logits = logits[..., :-1, :].float()
        shift_labels = labels[..., 1:]
        valid_mask = (shift_labels != -100).float()
        valid_count = valid_mask.sum().clamp_min(1.0)
        safe_labels = torch.where(shift_labels != -100, shift_labels, torch.zeros_like(shift_labels))

        loss_ce_unmasked = F.cross_entropy(
            shift_logits.reshape(-1, self.vocab_size),
            safe_labels.reshape(-1),
            reduction="none",
            label_smoothing=self.label_smoothing,
        ).reshape_as(shift_labels)
        loss_ce = (loss_ce_unmasked * valid_mask).sum() / valid_count

        if not self.uses_belief_loss:
            total_loss = loss_ce
        else:
            if prev_log_probs is None:
                raise RuntimeError("prev_log_probs is required in full belief mode.")
            teacher_log_probs = prev_log_probs[..., :-1, :].float()
            student_log_probs = F.log_softmax(shift_logits, dim=-1)
            loss_kl_unmasked = F.kl_div(
                student_log_probs,
                teacher_log_probs,
                reduction="none",
                log_target=True,
            ).sum(dim=-1)
            loss_kl = (loss_kl_unmasked * valid_mask).sum() / valid_count
            total_loss = self.alpha * loss_ce + (1.0 - self.alpha) * loss_kl

        output_log_probs = F.log_softmax(logits.float(), dim=-1) if self.returns_full_log_probs else None
        return total_loss, curr_hidden.detach(), output_log_probs.detach() if output_log_probs is not None else None


def normalize_belief_transport_mode(raw_mode: str) -> str:
    normalized = raw_mode.strip().lower()
    if normalized in {"", "full", "dense"}:
        return "full"
    if normalized in {"terminal", "terminal_only", "final", "final_only"}:
        return "terminal"
    if normalized in {"none", "off", "disabled", "false"}:
        return "none"
    raise ValueError(f"Unsupported belief transport mode: {raw_mode}. Use full, terminal, or none.")


def parse_devices(raw: str, expected: int) -> list[str]:
    devices = [item.strip() for item in raw.split(",") if item.strip()]
    if len(devices) != expected:
        raise ValueError(f"--stage_devices must contain {expected} devices, got {devices}.")
    return devices


def build_stage_chunk(
    *,
    model: nn.Module,
    stage_id: int,
    num_chunks: int,
    belief_transport_mode: str,
    alpha: float,
    label_smoothing: float,
) -> ServerBpfreeChunk:
    layers, final_norm, lm_head, vocab_size, rotary_emb = get_model_parts(model)
    total_layers = len(layers)
    chunk_size = total_layers // num_chunks
    start = stage_id * chunk_size
    end = (stage_id + 1) * chunk_size if stage_id < num_chunks - 1 else total_layers
    print(f"stage {stage_id}: layers=[{start}, {end - 1}]", flush=True)
    return ServerBpfreeChunk(
        chunk_idx=stage_id,
        layers=[layers[i] for i in range(start, end)],
        final_norm=final_norm,
        lm_head=lm_head,
        vocab_size=vocab_size,
        rotary_emb=rotary_emb,
        is_terminal_chunk=stage_id == num_chunks - 1,
        belief_transport_mode=belief_transport_mode,
        alpha=alpha,
        label_smoothing=label_smoothing,
    )


def build_optimizer(
    *,
    params: list[nn.Parameter],
    cfg: StageWorkerConfig,
) -> torch.optim.Optimizer:
    learning_rate = cfg.learning_rate or 3e-4
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=learning_rate)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(
            params,
            lr=learning_rate,
            momentum=cfg.sgd_momentum,
            dampening=cfg.sgd_dampening,
            weight_decay=cfg.sgd_weight_decay,
            nesterov=cfg.sgd_nesterov,
        )
    raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")


def tensor_to_cpu(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if value is None:
        return None
    return value.detach().to("cpu")


def load_stage0_state_cpu(record: dict[str, Any], manifest_dir: str) -> dict[str, Any]:
    base_dir = Path(manifest_dir)
    tensors = record["tensors"]
    return {
        "hidden": load_tensor(base_dir, tensors["hidden_states"]),
        "attention_mask": load_tensor(base_dir, tensors["attention_mask"]),
        "position_ids": load_tensor(base_dir, tensors["position_ids"]),
        "labels": load_tensor(base_dir, tensors["labels"]),
        "prev_log_probs": None,
    }


def move_state_to_device(
    state: dict[str, Any],
    device: torch.device,
    compute_dtype: Optional[torch.dtype] = None,
) -> dict[str, Any]:
    moved = {}
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


def cpu_state_from_task(task: dict[str, Any]) -> dict[str, Any]:
    state = task.get("state")
    if state is not None:
        return state
    return load_stage0_state_cpu(task["record"], task["manifest_dir"])


def stage_worker_main(
    *,
    stage_id: int,
    device_name: str,
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    cfg: StageWorkerConfig,
) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.manual_seed(cfg.seed + stage_id)
    np.random.seed(cfg.seed + stage_id)
    device = torch.device(device_name)
    dtype = resolve_dtype(cfg.dtype_name)
    train_chunks = set(cfg.train_chunks)

    try:
        if device.type == "cuda":
            torch.cuda.set_device(device)
        print(
            f"[stage {stage_id}] loading model={cfg.resolved_model} dtype={dtype} device={device}",
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
            stage_id=stage_id,
            num_chunks=cfg.num_chunks,
            belief_transport_mode=cfg.belief_transport_mode,
            alpha=cfg.alpha,
            label_smoothing=cfg.label_smoothing,
        )
        chunk.to(device)
        chunk_params = [param for param in chunk.parameters() if param.requires_grad]
        optimizer = build_optimizer(params=chunk_params, cfg=cfg) if stage_id in train_chunks else None

        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        print(
            f"[stage {stage_id}] ready device={device} lora_modules={injected} "
            f"all_trainable={trainable} frozen={frozen} local_trainable={sum(p.numel() for p in chunk_params)}",
            flush=True,
        )

        processed = 0
        while True:
            task = input_queue.get()
            if task is None:
                output_queue.put(None)
                break
            if isinstance(task, dict) and task.get("kind") == "worker_error":
                output_queue.put(task)
                break

            task_started = time.perf_counter()
            queue_wait_ms = (task_started - float(task.get("queue_enter_perf", task_started))) * 1000.0
            mode = task["mode"]
            train_this_chunk = mode == "train" and stage_id in train_chunks
            chunk.train(train_this_chunk)

            state_cpu = cpu_state_from_task(task)
            h2d_started = time.perf_counter()
            state = move_state_to_device(state_cpu, device, chunk.compute_dtype())
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            h2d_ms = (time.perf_counter() - h2d_started) * 1000.0

            lr = cfg.learning_rate
            if lr is None:
                lr = task["record"].get("learning_rate")
            optimizer_ms = 0.0
            if train_this_chunk:
                assert optimizer is not None
                if lr is not None:
                    for group in optimizer.param_groups:
                        group["lr"] = float(lr)
                optimizer.zero_grad(set_to_none=True)

            execute_started = time.perf_counter()
            with torch.set_grad_enabled(train_this_chunk):
                loss, next_hidden, next_log_probs = chunk(
                    hidden_states=state["hidden"],
                    attention_mask=state["attention_mask"],
                    position_ids=state["position_ids"],
                    labels=state["labels"],
                    prev_log_probs=state["prev_log_probs"],
                )
                if train_this_chunk:
                    loss.backward()
                    if cfg.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(chunk.parameters(), cfg.grad_clip)
                    optimizer_step_started = time.perf_counter()
                    assert optimizer is not None
                    optimizer.step()
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    optimizer_ms = (time.perf_counter() - optimizer_step_started) * 1000.0
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            execute_ms = (time.perf_counter() - execute_started) * 1000.0

            cpu_started = time.perf_counter()
            next_state_cpu = {
                "hidden": tensor_to_cpu(next_hidden),
                "attention_mask": tensor_to_cpu(state["attention_mask"]),
                "position_ids": tensor_to_cpu(state["position_ids"]),
                "labels": tensor_to_cpu(state["labels"]),
                "prev_log_probs": tensor_to_cpu(next_log_probs)
                if next_log_probs is not None and normalize_belief_transport_mode(cfg.belief_transport_mode) == "full"
                else None,
            }
            final_log_probs_cpu = tensor_to_cpu(next_log_probs) if stage_id == cfg.num_chunks - 1 else None
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            cpu_transfer_ms = (time.perf_counter() - cpu_started) * 1000.0

            local_loss = float(loss.detach().cpu().item())
            metric = {
                "phase": task["phase"],
                "seq": task["seq"],
                "request_id": task["request_id"],
                "stage_id": stage_id,
                "device": device_name,
                "mode": mode,
                "train": train_this_chunk,
                "local_loss": local_loss,
                "queue_wait_ms": queue_wait_ms,
                "h2d_ms": h2d_ms,
                "execute_ms": execute_ms,
                "optimizer_ms": optimizer_ms,
                "cpu_transfer_ms": cpu_transfer_ms,
                "stage_total_ms": (time.perf_counter() - task_started) * 1000.0,
                "output_hidden_bytes": int(next_state_cpu["hidden"].numel() * next_state_cpu["hidden"].element_size()),
                "output_log_probs_bytes": (
                    int(final_log_probs_cpu.numel() * final_log_probs_cpu.element_size())
                    if final_log_probs_cpu is not None
                    else (
                        int(next_state_cpu["prev_log_probs"].numel() * next_state_cpu["prev_log_probs"].element_size())
                        if next_state_cpu["prev_log_probs"] is not None
                        else 0
                    )
                ),
                "cuda_peak_memory_allocated": (
                    int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
                ),
                "cuda_peak_memory_reserved": (
                    int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
                ),
            }
            task["losses"].append(local_loss)
            task["stage_metrics"].append(metric)

            if stage_id == cfg.num_chunks - 1:
                result = finish_task_result(
                    task=task,
                    final_log_probs=final_log_probs_cpu,
                    labels=next_state_cpu["labels"],
                    completed_perf=time.perf_counter(),
                )
                output_queue.put(result)
            else:
                task["state"] = next_state_cpu
                task["queue_enter_perf"] = time.perf_counter()
                output_queue.put(task)

            processed += 1
            if cfg.progress_interval > 0 and processed % cfg.progress_interval == 0:
                print(
                    f"[stage {stage_id}] processed={processed} phase={task['phase']} "
                    f"loss={local_loss:.4f} train={train_this_chunk}",
                    flush=True,
                )

    except Exception as exc:  # pragma: no cover - exercised on server failure
        error = {
            "kind": "worker_error",
            "stage_id": stage_id,
            "device": device_name,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        try:
            output_queue.put(error)
        finally:
            raise


def finish_task_result(
    *,
    task: dict[str, Any],
    final_log_probs: Optional[torch.Tensor],
    labels: torch.Tensor,
    completed_perf: float,
) -> dict[str, Any]:
    if final_log_probs is None:
        choice_correct, choice_count, choice_loss = 0, 0, 0.0
    else:
        choice_ids = one_token_choice_ids(task["record"])
        choice_correct, choice_count, choice_loss = label_choice_metrics(final_log_probs, labels, choice_ids)
    response = (task["record"].get("text") or {}).get("response", "").strip()
    return {
        "kind": "result",
        "phase": task["phase"],
        "seq": task["seq"],
        "request_id": task["request_id"],
        "dataset_index": int(task["record"].get("dataset_index", -1)),
        "response": response,
        "mode": task["mode"],
        "loss": task["losses"][-1],
        "chunk_losses": task["losses"],
        "choice_correct": choice_correct,
        "choice_count": choice_count,
        "choice_accuracy": (choice_correct / choice_count) if choice_count else 0.0,
        "choice_loss": choice_loss,
        "stage_metrics": task["stage_metrics"],
        "elapsed_ms": (completed_perf - float(task["admitted_perf"])) * 1000.0,
    }


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
        "chunk_losses_json",
    ]


def stage_metric_fieldnames() -> list[str]:
    return [
        "phase",
        "seq",
        "request_id",
        "stage_id",
        "device",
        "mode",
        "train",
        "local_loss",
        "queue_wait_ms",
        "h2d_ms",
        "execute_ms",
        "optimizer_ms",
        "cpu_transfer_ms",
        "stage_total_ms",
        "output_hidden_bytes",
        "output_log_probs_bytes",
        "cuda_peak_memory_allocated",
        "cuda_peak_memory_reserved",
    ]


def write_result_row(writer: csv.DictWriter, result: dict[str, Any]) -> None:
    writer.writerow(
        {
            "seq": result["seq"],
            "request_id": result["request_id"],
            "dataset_index": result["dataset_index"],
            "response": result["response"],
            "mode": result["mode"],
            "loss": result["loss"],
            "choice_correct": result["choice_correct"],
            "choice_count": result["choice_count"],
            "choice_accuracy": result["choice_accuracy"],
            "choice_loss": result["choice_loss"],
            "elapsed_ms": result["elapsed_ms"],
            "chunk_losses_json": json.dumps(result["chunk_losses"]),
        }
    )


def write_stage_metric_rows(writer: csv.DictWriter, result: dict[str, Any]) -> None:
    for metric in result["stage_metrics"]:
        writer.writerow({name: metric.get(name, "") for name in stage_metric_fieldnames()})


def summarize_phase(name: str, results: list[dict[str, Any]], output_csv: Path) -> dict[str, Any]:
    correct = sum(int(result["choice_correct"]) for result in results)
    count = sum(int(result["choice_count"]) for result in results)
    losses = [float(result["loss"]) for result in results]
    per_class: dict[str, dict[str, float]] = {}
    for result in results:
        label = result["response"] or "unknown"
        stats = per_class.setdefault(label, {"correct": 0.0, "count": 0.0, "loss_sum": 0.0})
        stats["correct"] += result["choice_correct"]
        stats["count"] += result["choice_count"]
        stats["loss_sum"] += result["loss"]
    class_rows = []
    for label, stats in sorted(per_class.items()):
        class_count = int(stats["count"])
        class_rows.append(
            {
                "label": label,
                "correct": int(stats["correct"]),
                "count": class_count,
                "accuracy": (stats["correct"] / class_count) if class_count else 0.0,
                "avg_loss": stats["loss_sum"] / class_count if class_count else 0.0,
            }
        )
    return {
        "phase": name,
        "rows": len(results),
        "choice_correct": correct,
        "choice_count": count,
        "choice_accuracy": (correct / count) if count else 0.0,
        "avg_loss": sum(losses) / len(losses) if losses else 0.0,
        "avg_elapsed_ms": (
            sum(float(result["elapsed_ms"]) for result in results) / len(results) if results else 0.0
        ),
        "per_class": class_rows,
        "csv": str(output_csv),
    }


def make_task(
    *,
    phase: str,
    seq: int,
    record: dict[str, Any],
    manifest_dir: Path,
    mode: str,
    request_prefix: str,
) -> dict[str, Any]:
    source_request_id = record.get("request_id") or f"record-{seq:06d}"
    request_id = f"{request_prefix}-{phase}-{seq:06d}"
    return {
        "kind": "work",
        "phase": phase,
        "seq": seq,
        "request_id": request_id,
        "source_request_id": source_request_id,
        "record": record,
        "manifest_dir": str(manifest_dir),
        "mode": mode,
        "state": None,
        "losses": [],
        "stage_metrics": [],
        "admitted_perf": time.perf_counter(),
        "queue_enter_perf": time.perf_counter(),
    }


def run_phase(
    *,
    name: str,
    records: list[dict[str, Any]],
    manifest_dir: Path,
    mode: str,
    request_prefix: str,
    input_queue: mp.Queue,
    result_queue: mp.Queue,
    output_csv: Path,
    stage_metrics_csv: Path,
    progress_interval: int,
) -> dict[str, Any]:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    stage_metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    def producer() -> None:
        for seq, record in enumerate(records):
            task = make_task(
                phase=name,
                seq=seq,
                record=record,
                manifest_dir=manifest_dir,
                mode=mode,
                request_prefix=request_prefix,
            )
            input_queue.put(task)

    producer_thread = threading.Thread(target=producer, name=f"{name}-producer", daemon=True)
    producer_thread.start()

    with output_csv.open("w", newline="", encoding="utf-8") as result_handle, stage_metrics_csv.open(
        "w", newline="", encoding="utf-8"
    ) as metrics_handle:
        result_writer = csv.DictWriter(result_handle, fieldnames=result_fieldnames())
        metric_writer = csv.DictWriter(metrics_handle, fieldnames=stage_metric_fieldnames())
        result_writer.writeheader()
        metric_writer.writeheader()

        while len(results) + len(errors) < len(records):
            item = result_queue.get()
            if item is None:
                raise RuntimeError("Stage pipeline terminated before phase completed.")
            if item.get("kind") == "worker_error":
                errors.append(item)
                print(json.dumps(item, indent=2), flush=True)
                break
            if item.get("kind") != "result":
                raise RuntimeError(f"Unexpected result queue item: {item}")
            results.append(item)
            write_result_row(result_writer, item)
            write_stage_metric_rows(metric_writer, item)
            if progress_interval > 0 and (len(results) % progress_interval == 0 or len(results) == len(records)):
                correct = sum(int(result["choice_correct"]) for result in results)
                count = sum(int(result["choice_count"]) for result in results)
                loss = sum(float(result["loss"]) for result in results) / len(results)
                acc = correct / count if count else 0.0
                print(f"{name}: {len(results)}/{len(records)} acc={acc:.4f} loss={loss:.4f}", flush=True)

    producer_thread.join(timeout=None if not errors else 5.0)
    if errors:
        raise RuntimeError(f"{name} failed with worker error: {errors[0]['message']}")
    return summarize_phase(name, results, output_csv)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Real server-side multi-GPU BP-free stage pipeline training. "
            "Each stage process is pinned to one --stage_devices entry."
        )
    )
    parser.add_argument("--model_name", default="tinyllama")
    parser.add_argument("--train_manifest", type=Path, required=True)
    parser.add_argument("--eval_manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_chunks", type=int, default=3)
    parser.add_argument(
        "--stage_devices",
        required=True,
        help="Comma-separated devices, one per stage, e.g. cuda:0,cuda:1,cuda:2.",
    )
    parser.add_argument("--max_buffered_per_stage", type=int, default=3)
    parser.add_argument("--train_chunks", default="all")
    parser.add_argument("--train_limit", type=int, default=None)
    parser.add_argument("--train_epochs", type=int, default=1)
    parser.add_argument("--eval_limit", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--sgd_momentum", type=float, default=0.0)
    parser.add_argument("--sgd_dampening", type=float, default=0.0)
    parser.add_argument("--sgd_weight_decay", type=float, default=0.0)
    parser.add_argument("--sgd_nesterov", action="store_true")
    parser.add_argument(
        "--belief_transport_mode",
        default="terminal",
        choices=["full", "terminal", "none"],
        help="terminal matches the current CE-only AG News phone mainline.",
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_targets", default="q_proj,v_proj")
    parser.add_argument("--lora_init_std", type=float, default=0.01)
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--request_prefix", default="server-pipeline")
    parser.add_argument("--progress_interval", type=int, default=16)
    parser.add_argument("--skip_eval_before", action="store_true")
    parser.add_argument("--skip_eval_after", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_chunks <= 0:
        raise ValueError("--num_chunks must be positive.")
    if args.train_epochs <= 0:
        raise ValueError("--train_epochs must be positive.")
    if args.max_buffered_per_stage <= 0:
        raise ValueError("--max_buffered_per_stage must be positive.")

    devices = parse_devices(args.stage_devices, args.num_chunks)
    resolved_model = resolve_model_name(args.model_name)
    train_chunks = parse_train_chunks(args.train_chunks, args.num_chunks)
    base_train_records = read_manifest(args.train_manifest, args.train_limit)
    train_records = base_train_records * args.train_epochs
    eval_records = read_manifest(args.eval_manifest, args.eval_limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = StageWorkerConfig(
        model_name=args.model_name,
        resolved_model=resolved_model,
        num_chunks=args.num_chunks,
        train_chunks=sorted(train_chunks),
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

    print(
        f"Starting multi-GPU pipeline model={resolved_model} chunks={args.num_chunks} "
        f"devices={devices} buffer={args.max_buffered_per_stage} "
        f"belief_transport_mode={cfg.belief_transport_mode}",
        flush=True,
    )
    mp.set_start_method("spawn", force=True)
    stage_queues = [mp.Queue(maxsize=args.max_buffered_per_stage) for _ in range(args.num_chunks)]
    result_queue: mp.Queue = mp.Queue(maxsize=args.max_buffered_per_stage)
    processes: list[mp.Process] = []

    for stage_id in range(args.num_chunks):
        output_queue = result_queue if stage_id == args.num_chunks - 1 else stage_queues[stage_id + 1]
        process = mp.Process(
            target=stage_worker_main,
            kwargs={
                "stage_id": stage_id,
                "device_name": devices[stage_id],
                "input_queue": stage_queues[stage_id],
                "output_queue": output_queue,
                "cfg": cfg,
            },
            name=f"stage-{stage_id}-{devices[stage_id]}",
        )
        process.start()
        processes.append(process)

    phases: list[dict[str, Any]] = []
    try:
        if not args.skip_eval_before:
            phases.append(
                run_phase(
                    name="eval_before",
                    records=eval_records,
                    manifest_dir=args.eval_manifest.parent,
                    mode="eval",
                    request_prefix=args.request_prefix,
                    input_queue=stage_queues[0],
                    result_queue=result_queue,
                    output_csv=args.output_dir / "eval_before.csv",
                    stage_metrics_csv=args.output_dir / "eval_before.stage_metrics.csv",
                    progress_interval=args.progress_interval,
                )
            )
        phases.append(
            run_phase(
                name="train",
                records=train_records,
                manifest_dir=args.train_manifest.parent,
                mode="train",
                request_prefix=args.request_prefix,
                input_queue=stage_queues[0],
                result_queue=result_queue,
                output_csv=args.output_dir / "train.csv",
                stage_metrics_csv=args.output_dir / "train.stage_metrics.csv",
                progress_interval=args.progress_interval,
            )
        )
        if not args.skip_eval_after:
            phases.append(
                run_phase(
                    name="eval_after",
                    records=eval_records,
                    manifest_dir=args.eval_manifest.parent,
                    mode="eval",
                    request_prefix=args.request_prefix,
                    input_queue=stage_queues[0],
                    result_queue=result_queue,
                    output_csv=args.output_dir / "eval_after.csv",
                    stage_metrics_csv=args.output_dir / "eval_after.stage_metrics.csv",
                    progress_interval=args.progress_interval,
                )
            )
    finally:
        stage_queues[0].put(None)
        for process in processes:
            process.join(timeout=120)
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            if process.exitcode not in (0, None):
                print(f"worker {process.name} exited with code {process.exitcode}", flush=True)

    summary = {
        "model_name": args.model_name,
        "resolved_model": resolved_model,
        "num_chunks": args.num_chunks,
        "stage_devices": devices,
        "max_buffered_per_stage": args.max_buffered_per_stage,
        "train_chunks": sorted(train_chunks),
        "train_epochs": args.train_epochs,
        "unique_train_records": len(base_train_records),
        "train_steps": len(train_records),
        "learning_rate": args.learning_rate,
        "optimizer": args.optimizer,
        "belief_transport_mode": cfg.belief_transport_mode,
        "alpha": args.alpha,
        "label_smoothing": args.label_smoothing,
        "lora": {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "targets": args.lora_targets,
            "init_std": args.lora_init_std,
        },
        "dtype": args.dtype,
        "seed": args.seed,
        "phases": phases,
    }
    phase_by_name = {phase["phase"]: phase for phase in phases}
    if "eval_before" in phase_by_name and "eval_after" in phase_by_name:
        summary["delta"] = {
            "choice_accuracy": phase_by_name["eval_after"]["choice_accuracy"]
            - phase_by_name["eval_before"]["choice_accuracy"],
            "avg_loss": phase_by_name["eval_after"]["avg_loss"] - phase_by_name["eval_before"]["avg_loss"],
        }

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"Wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
