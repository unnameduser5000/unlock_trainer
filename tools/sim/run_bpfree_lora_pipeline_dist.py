#!/usr/bin/env python3
"""Real multi-GPU BP-free stage pipeline training with torch.distributed.

This is the server-speed runner. It maps each BP-free stage to one GPU/rank and
uses point-to-point distributed tensor communication between adjacent stages:

    rank0/cuda:0 --send hidden--> rank1/cuda:1 --send hidden--> rank2/cuda:2

There is still no cross-stage backward edge. Each rank owns its local chunk,
local LoRA optimizer, and local update rule.

Compared with run_bpfree_lora_pipeline_multigpu.py, this avoids Python
multiprocessing Queue CPU tensor handoff for the stage boundary hidden states.
It is the runner to use when the goal is faster server-side training.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoConfig, AutoModelForCausalLM

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
)


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


def load_common_tensors(
    *,
    record: dict[str, Any],
    manifest_dir: Path,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tensors = record["tensors"]
    attention_mask = load_tensor(manifest_dir, tensors["attention_mask"]).to(device)
    position_ids = load_tensor(manifest_dir, tensors["position_ids"]).to(device)
    labels = load_tensor(manifest_dir, tensors["labels"]).to(device)
    return attention_mask, position_ids, labels


def load_stage0_hidden(
    *,
    record: dict[str, Any],
    manifest_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    hidden = load_tensor(manifest_dir, record["tensors"]["hidden_states"]).to(device)
    return hidden.to(dtype=dtype)


def hidden_shape(record: dict[str, Any]) -> list[int]:
    return list(record["tensors"]["hidden_states"]["shape"])


def log_probs_shape(record: dict[str, Any], vocab_size: int) -> list[int]:
    label_shape = list(record["tensors"]["labels"]["shape"])
    if len(label_shape) != 2:
        raise ValueError(f"Expected labels shape [batch, seq], got {label_shape}")
    return [label_shape[0], label_shape[1], vocab_size]


def recv_tensor(shape: list[int], dtype: torch.dtype, device: torch.device, src: int) -> tuple[torch.Tensor, float]:
    tensor = torch.empty(tuple(shape), dtype=dtype, device=device)
    started = time.perf_counter()
    dist.recv(tensor=tensor, src=src)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return tensor, (time.perf_counter() - started) * 1000.0


def send_tensor(tensor: torch.Tensor, dst: int, device: torch.device) -> float:
    started = time.perf_counter()
    dist.send(tensor=tensor.contiguous(), dst=dst)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (time.perf_counter() - started) * 1000.0


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
        "stage_id",
        "device",
        "mode",
        "train",
        "local_loss",
        "recv_hidden_ms",
        "recv_log_probs_ms",
        "load_input_ms",
        "execute_ms",
        "optimizer_ms",
        "send_hidden_ms",
        "send_log_probs_ms",
        "stage_total_ms",
        "output_hidden_bytes",
        "output_log_probs_bytes",
        "cuda_peak_memory_allocated",
        "cuda_peak_memory_reserved",
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
) -> dict[str, Any]:
    if final_log_probs is None:
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


def run_phase_distributed(
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
    output_dir: Path,
    progress_interval: int,
) -> Optional[dict[str, Any]]:
    train_this_rank = mode == "train" and rank in train_chunks
    result_rows: list[dict[str, Any]] = []
    phase_started = time.perf_counter()
    metrics_path = output_dir / f"{phase}.stage{rank}.metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    result_path = output_dir / f"{phase}.csv"
    result_handle = None
    result_writer = None
    if rank == world_size - 1:
        result_handle = result_path.open("w", newline="", encoding="utf-8")
        result_writer = csv.DictWriter(result_handle, fieldnames=result_fieldnames())
        result_writer.writeheader()

    try:
        with metrics_path.open("w", newline="", encoding="utf-8") as metrics_handle:
            metric_writer = csv.DictWriter(metrics_handle, fieldnames=metric_fieldnames())
            metric_writer.writeheader()
            for seq, record in enumerate(records):
                request_id = request_id_for(request_prefix, phase, seq)
                record_started = time.perf_counter()
                stage_started = record_started

                recv_hidden_ms = 0.0
                recv_log_probs_ms = 0.0
                if rank == 0:
                    hidden = load_stage0_hidden(
                        record=record,
                        manifest_dir=manifest_dir,
                        device=device,
                        dtype=dtype,
                    )
                    prev_log_probs = None
                else:
                    hidden, recv_hidden_ms = recv_tensor(
                        hidden_shape(record),
                        dtype,
                        device,
                        src=rank - 1,
                    )
                    if belief_transport_mode == "full":
                        prev_log_probs, recv_log_probs_ms = recv_tensor(
                            log_probs_shape(record, vocab_size),
                            torch.float32,
                            device,
                            src=rank - 1,
                        )
                    else:
                        prev_log_probs = None

                load_started = time.perf_counter()
                attention_mask, position_ids, labels = load_common_tensors(
                    record=record,
                    manifest_dir=manifest_dir,
                    device=device,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                load_input_ms = (time.perf_counter() - load_started) * 1000.0

                if train_this_rank:
                    assert optimizer is not None
                    lr = learning_rate_override
                    if lr is None:
                        lr = record.get("learning_rate")
                    if lr is not None:
                        for group in optimizer.param_groups:
                            group["lr"] = float(lr)
                    optimizer.zero_grad(set_to_none=True)
                chunk.train(train_this_rank)

                optimizer_ms = 0.0
                execute_started = time.perf_counter()
                with torch.set_grad_enabled(train_this_rank):
                    loss, next_hidden, next_log_probs = chunk(
                        hidden_states=hidden,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        labels=labels,
                        prev_log_probs=prev_log_probs,
                    )
                    if train_this_rank:
                        loss.backward()
                        if grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(chunk.parameters(), grad_clip)
                        optimizer_started = time.perf_counter()
                        assert optimizer is not None
                        optimizer.step()
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        optimizer_ms = (time.perf_counter() - optimizer_started) * 1000.0
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                execute_ms = (time.perf_counter() - execute_started) * 1000.0
                loss_value = float(loss.detach().cpu().item())

                send_hidden_ms = 0.0
                send_log_probs_ms = 0.0
                if rank < world_size - 1:
                    send_hidden_ms = send_tensor(next_hidden.detach(), dst=rank + 1, device=device)
                    if belief_transport_mode == "full":
                        if next_log_probs is None:
                            raise RuntimeError("full belief mode requires log-prob output from every stage.")
                        send_log_probs_ms = send_tensor(next_log_probs.detach().float(), dst=rank + 1, device=device)
                else:
                    assert result_writer is not None
                    result = make_terminal_result(
                        phase=phase,
                        seq=seq,
                        request_id=request_id,
                        record=record,
                        mode=mode,
                        loss_value=loss_value,
                        final_log_probs=next_log_probs,
                        labels=labels,
                        elapsed_ms=(time.perf_counter() - record_started) * 1000.0,
                    )
                    result_rows.append(result)
                    write_result_row(result_writer, result)

                metric = {
                    "phase": phase,
                    "seq": seq,
                    "request_id": request_id,
                    "stage_id": rank,
                    "device": str(device),
                    "mode": mode,
                    "train": train_this_rank,
                    "local_loss": loss_value,
                    "recv_hidden_ms": recv_hidden_ms,
                    "recv_log_probs_ms": recv_log_probs_ms,
                    "load_input_ms": load_input_ms,
                    "execute_ms": execute_ms,
                    "optimizer_ms": optimizer_ms,
                    "send_hidden_ms": send_hidden_ms,
                    "send_log_probs_ms": send_log_probs_ms,
                    "stage_total_ms": (time.perf_counter() - stage_started) * 1000.0,
                    "output_hidden_bytes": tensor_nbytes(next_hidden),
                    "output_log_probs_bytes": tensor_nbytes(next_log_probs),
                    "cuda_peak_memory_allocated": (
                        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
                    ),
                    "cuda_peak_memory_reserved": (
                        int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
                    ),
                }
                write_metric_row(metric_writer, metric)

                if progress_interval > 0 and (seq + 1) % progress_interval == 0:
                    print(
                        f"[rank {rank}] {phase}: {seq + 1}/{len(records)} "
                        f"loss={loss_value:.4f} train={train_this_rank}",
                        flush=True,
                    )

    finally:
        if result_handle is not None:
            result_handle.close()

    dist.barrier()
    if rank == world_size - 1:
        summary = summarize_phase(phase, result_rows, result_path)
        summary["wall_ms"] = (time.perf_counter() - phase_started) * 1000.0
        return summary
    return None


def distributed_worker(rank: int, world_size: int, cfg: dict[str, Any]) -> None:
    devices = cfg["stage_devices"]
    device = torch.device(devices[rank])
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group(
        backend=cfg["dist_backend"],
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )

    torch.manual_seed(int(cfg["seed"]) + rank)
    np.random.seed(int(cfg["seed"]) + rank)
    dtype = resolve_dtype(cfg["dtype"])
    train_chunks = set(cfg["train_chunks"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_model = cfg["resolved_model"]

    print(f"[rank {rank}] loading model={resolved_model} device={device} dtype={dtype}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(resolved_model, torch_dtype=dtype)
    injected = inject_lora_adapters(
        module=model,
        target_names={item.strip() for item in cfg["lora_targets"].split(",") if item.strip()},
        rank=int(cfg["lora_rank"]),
        alpha=float(cfg["lora_alpha"]),
        init_std=float(cfg["lora_init_std"]),
    )
    trainable, frozen = configure_lora_trainable(model)
    chunk = build_stage_chunk(
        model=model,
        stage_id=rank,
        num_chunks=world_size,
        belief_transport_mode=cfg["belief_transport_mode"],
        alpha=float(cfg["alpha"]),
        label_smoothing=float(cfg["label_smoothing"]),
    )
    chunk.to(device)
    local_params = [param for param in chunk.parameters() if param.requires_grad]

    optimizer = None
    if rank in train_chunks:
        optimizer = build_optimizer(
            params=local_params,
            cfg=argparse.Namespace(
                learning_rate=cfg["learning_rate"],
                optimizer=cfg["optimizer"],
                sgd_momentum=cfg["sgd_momentum"],
                sgd_dampening=cfg["sgd_dampening"],
                sgd_weight_decay=cfg["sgd_weight_decay"],
                sgd_nesterov=cfg["sgd_nesterov"],
            ),
        )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    print(
        f"[rank {rank}] ready lora_modules={injected} all_trainable={trainable} "
        f"frozen={frozen} local_trainable={sum(param.numel() for param in local_params)}",
        flush=True,
    )
    dist.barrier()

    train_manifest = Path(cfg["train_manifest"])
    eval_manifest = Path(cfg["eval_manifest"])
    train_records = read_manifest(train_manifest, cfg["train_limit"]) * int(cfg["train_epochs"])
    eval_records = read_manifest(eval_manifest, cfg["eval_limit"])
    phases: list[dict[str, Any]] = []

    if not cfg["skip_eval_before"]:
        phase_summary = run_phase_distributed(
            rank=rank,
            world_size=world_size,
            phase="eval_before",
            records=eval_records,
            manifest_dir=eval_manifest.parent,
            mode="eval",
            request_prefix=cfg["request_prefix"],
            chunk=chunk,
            optimizer=optimizer,
            train_chunks=train_chunks,
            dtype=dtype,
            device=device,
            belief_transport_mode=cfg["belief_transport_mode"],
            grad_clip=float(cfg["grad_clip"]),
            learning_rate_override=cfg["learning_rate"],
            vocab_size=int(cfg["vocab_size"]),
            output_dir=output_dir,
            progress_interval=int(cfg["progress_interval"]),
        )
        if phase_summary is not None:
            phases.append(phase_summary)

    phase_summary = run_phase_distributed(
        rank=rank,
        world_size=world_size,
        phase="train",
        records=train_records,
        manifest_dir=train_manifest.parent,
        mode="train",
        request_prefix=cfg["request_prefix"],
        chunk=chunk,
        optimizer=optimizer,
        train_chunks=train_chunks,
        dtype=dtype,
        device=device,
        belief_transport_mode=cfg["belief_transport_mode"],
        grad_clip=float(cfg["grad_clip"]),
        learning_rate_override=cfg["learning_rate"],
        vocab_size=int(cfg["vocab_size"]),
        output_dir=output_dir,
        progress_interval=int(cfg["progress_interval"]),
    )
    if phase_summary is not None:
        phases.append(phase_summary)

    if not cfg["skip_eval_after"]:
        phase_summary = run_phase_distributed(
            rank=rank,
            world_size=world_size,
            phase="eval_after",
            records=eval_records,
            manifest_dir=eval_manifest.parent,
            mode="eval",
            request_prefix=cfg["request_prefix"],
            chunk=chunk,
            optimizer=optimizer,
            train_chunks=train_chunks,
            dtype=dtype,
            device=device,
            belief_transport_mode=cfg["belief_transport_mode"],
            grad_clip=float(cfg["grad_clip"]),
            learning_rate_override=cfg["learning_rate"],
            vocab_size=int(cfg["vocab_size"]),
            output_dir=output_dir,
            progress_interval=int(cfg["progress_interval"]),
        )
        if phase_summary is not None:
            phases.append(phase_summary)

    if rank == world_size - 1:
        phase_by_name = {phase["phase"]: phase for phase in phases}
        summary: dict[str, Any] = {
            "runner": "torch.distributed-sendrecv",
            "model_name": cfg["model_name"],
            "resolved_model": resolved_model,
            "num_chunks": world_size,
            "stage_devices": devices,
            "train_chunks": sorted(train_chunks),
            "train_epochs": cfg["train_epochs"],
            "unique_train_records": len(read_manifest(train_manifest, cfg["train_limit"])),
            "train_steps": len(train_records),
            "learning_rate": cfg["learning_rate"],
            "optimizer": cfg["optimizer"],
            "belief_transport_mode": cfg["belief_transport_mode"],
            "alpha": cfg["alpha"],
            "label_smoothing": cfg["label_smoothing"],
            "dtype": cfg["dtype"],
            "seed": cfg["seed"],
            "phases": phases,
        }
        if "eval_before" in phase_by_name and "eval_after" in phase_by_name:
            summary["delta"] = {
                "choice_accuracy": phase_by_name["eval_after"]["choice_accuracy"]
                - phase_by_name["eval_before"]["choice_accuracy"],
                "avg_loss": phase_by_name["eval_after"]["avg_loss"] - phase_by_name["eval_before"]["avg_loss"],
            }
        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
        print(f"Wrote {summary_path}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real server-side BP-free stage pipeline training with torch.distributed send/recv."
    )
    parser.add_argument("--model_name", default="tinyllama")
    parser.add_argument("--train_manifest", type=Path, required=True)
    parser.add_argument("--eval_manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_chunks", type=int, default=3)
    parser.add_argument("--stage_devices", required=True, help="Example: cuda:0,cuda:1,cuda:2")
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
    parser.add_argument("--belief_transport_mode", default="terminal", choices=["full", "terminal", "none"])
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_targets", default="q_proj,v_proj")
    parser.add_argument("--lora_init_std", type=float, default=0.01)
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--request_prefix", default="server-dist")
    parser.add_argument("--progress_interval", type=int, default=16)
    parser.add_argument("--skip_eval_before", action="store_true")
    parser.add_argument("--skip_eval_after", action="store_true")
    parser.add_argument("--master_addr", default="127.0.0.1")
    parser.add_argument("--master_port", default="29531")
    parser.add_argument("--dist_backend", default="nccl", choices=["nccl", "gloo"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_chunks <= 0:
        raise ValueError("--num_chunks must be positive.")
    if args.train_epochs <= 0:
        raise ValueError("--train_epochs must be positive.")
    devices = parse_devices(args.stage_devices, args.num_chunks)
    train_chunks = parse_train_chunks(args.train_chunks, args.num_chunks)
    resolved_model = resolve_model_name(args.model_name)

    # Load config once on the parent to get vocab size and fail fast on missing model metadata.
    model_config = AutoConfig.from_pretrained(resolved_model)
    vocab_size = int(model_config.vocab_size)
    del model_config

    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = str(args.master_port)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    cfg = {
        "model_name": args.model_name,
        "resolved_model": resolved_model,
        "train_manifest": str(args.train_manifest),
        "eval_manifest": str(args.eval_manifest),
        "output_dir": str(args.output_dir),
        "num_chunks": args.num_chunks,
        "stage_devices": devices,
        "train_chunks": sorted(train_chunks),
        "train_limit": args.train_limit,
        "train_epochs": args.train_epochs,
        "eval_limit": args.eval_limit,
        "learning_rate": args.learning_rate,
        "grad_clip": args.grad_clip,
        "optimizer": args.optimizer,
        "sgd_momentum": args.sgd_momentum,
        "sgd_dampening": args.sgd_dampening,
        "sgd_weight_decay": args.sgd_weight_decay,
        "sgd_nesterov": args.sgd_nesterov,
        "belief_transport_mode": normalize_belief_transport_mode(args.belief_transport_mode),
        "alpha": args.alpha,
        "label_smoothing": args.label_smoothing,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_targets": args.lora_targets,
        "lora_init_std": args.lora_init_std,
        "dtype": args.dtype,
        "seed": args.seed,
        "request_prefix": args.request_prefix,
        "progress_interval": args.progress_interval,
        "skip_eval_before": args.skip_eval_before,
        "skip_eval_after": args.skip_eval_after,
        "dist_backend": args.dist_backend,
        "vocab_size": vocab_size,
    }
    print(
        f"Starting distributed stage pipeline model={resolved_model} "
        f"devices={devices} backend={args.dist_backend} transport=sendrecv",
        flush=True,
    )
    mp.spawn(distributed_worker, args=(args.num_chunks, cfg), nprocs=args.num_chunks, join=True)


if __name__ == "__main__":
    main()
