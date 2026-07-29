#!/usr/bin/env python3
"""Launch the BPFree GPU/NCCL runtime.

This module owns process-group initialization, per-rank model construction,
stage placement, and train/evaluation phase execution. Paper experiments select
its arguments but do not implement the GPU runtime.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoConfig, AutoModelForCausalLM

from sg_exe_trainer.common.trainable_modes import (
    configure_model_trainable,
    module_param_stats,
)
from sg_exe_trainer.runtime.bpfree.pipeline_support import stage_memory_ledger
from sg_exe_trainer.runtime.bpfree.model_runtime import (
    build_optimizer,
    build_stage_chunk,
    normalize_belief_transport_mode,
)
from sg_exe_trainer.runtime.bpfree.gpu_phase import run_phase_schedule_v3
from sg_exe_trainer.tasks.label_experiment import (
    lora_parameter_fingerprint,
    read_manifest,
    resolve_dtype,
    resolve_model_name,
)


def parse_devices(raw: str, expected: int) -> list[str]:
    devices = [item.strip() for item in raw.split(",") if item.strip()]
    if len(devices) != expected:
        raise ValueError(f"--stage_devices must contain {expected} devices, got {devices}.")
    return devices


def parse_train_chunks(raw: str, world_size: int) -> set[int]:
    raw = str(raw or "all").strip()
    if raw in {"", "all", "*"}:
        return set(range(world_size))
    chunks = {int(item.strip()) for item in raw.split(",") if item.strip()}
    bad = [idx for idx in chunks if idx < 0 or idx >= world_size]
    if bad:
        raise ValueError(f"--train_chunks contains invalid stage ids {bad}; world_size={world_size}.")
    return chunks


def seed_everything(seed: int, rank: int) -> None:
    actual = int(seed) + int(rank)
    random.seed(actual)
    np.random.seed(actual % (2**32 - 1))
    torch.manual_seed(actual)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(actual)


def barrier_on_device(device: torch.device) -> None:
    if device.type == "cuda":
        if device.index is None:
            raise RuntimeError(f"CUDA device must have an explicit index: {device}")
        dist.barrier(device_ids=[device.index])
    else:
        dist.barrier()


def distributed_worker(rank: int, world_size: int, cfg: dict[str, Any]) -> None:
    os.environ["MASTER_ADDR"] = str(cfg["master_addr"])
    os.environ["MASTER_PORT"] = str(cfg["master_port"])
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    devices = list(cfg["stage_devices"])
    device = torch.device(devices[rank])
    if device.type == "cuda":
        torch.cuda.set_device(device)

    init_process_group_kwargs: dict[str, Any] = {
        "backend": str(cfg["backend"]),
        "rank": rank,
        "world_size": world_size,
    }
    if device.type == "cuda":
        init_process_group_kwargs["device_id"] = device

    dist.init_process_group(**init_process_group_kwargs)

    try:
        seed_everything(int(cfg["seed"]), rank)

        output_dir = Path(cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        dtype = resolve_dtype(str(cfg["dtype"]))
        resolved_model = str(cfg["resolved_model"])
        train_chunks = parse_train_chunks(str(cfg["train_chunks"]), world_size)

        print(
            f"[rank {rank}] loading model={resolved_model} device={device} dtype={dtype}",
            flush=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            resolved_model,
            torch_dtype=dtype,
        )
        hidden_size = int(getattr(model.config, "hidden_size", 0) or getattr(model.config, "n_embd", 0) or 0)
        if hidden_size <= 0:
            raise ValueError(f"Could not resolve a positive hidden_size for model={resolved_model}")

        trainable_setup = configure_model_trainable(
            module=model,
            mode=str(cfg["trainable_mode"]),
            lora_targets=str(cfg["lora_targets"]),
            lora_rank=int(cfg["lora_rank"]),
            lora_alpha=float(cfg["lora_alpha"]),
            lora_init_std=float(cfg["lora_init_std"]),
            lora_init_seed=cfg["lora_init_seed"],
        )

        lora_init_fingerprint = (
            lora_parameter_fingerprint(model)
            if trainable_setup.mode == "lora"
            else ""
        )

        chunk = build_stage_chunk(
            model=model,
            stage_id=rank,
            num_chunks=world_size,
            belief_transport_mode=str(cfg["belief_transport_mode"]),
            alpha=float(cfg["alpha"]),
            label_smoothing=float(cfg["label_smoothing"]),
        )
        chunk.to(device)

        local_params = [param for param in chunk.parameters() if param.requires_grad]
        local_param_stats = module_param_stats(chunk)
        memory_ledger = stage_memory_ledger(chunk)

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
            f"[rank {rank}] ready trainable_mode={trainable_setup.mode} "
            f"lora_modules={trainable_setup.lora_modules} "
            f"all_trainable={trainable_setup.trainable_params} "
            f"frozen={trainable_setup.frozen_params} "
            f"local_trainable={local_param_stats.trainable_params} "
            f"lora_init={lora_init_fingerprint[:12]}",
            flush=True,
        )

        barrier_on_device(device)

        train_manifest = Path(cfg["train_manifest"])
        eval_manifest = Path(cfg["eval_manifest"])
        train_records = read_manifest(train_manifest, cfg["train_limit"]) * int(cfg["train_epochs"])
        eval_records = read_manifest(eval_manifest, cfg["eval_limit"])

        phases: list[dict[str, Any]] = []

        if not bool(cfg["skip_eval_before"]):
            phase_summary = run_phase_schedule_v3(
                rank=rank,
                world_size=world_size,
                phase="eval_before",
                records=eval_records,
                manifest_dir=eval_manifest.parent,
                mode="eval",
                request_prefix=str(cfg["request_prefix"]),
                chunk=chunk,
                optimizer=optimizer,
                train_chunks=train_chunks,
                dtype=dtype,
                device=device,
                belief_transport_mode=str(cfg["belief_transport_mode"]),
                grad_clip=float(cfg["grad_clip"]),
                learning_rate_override=cfg["learning_rate"],
                hidden_size=hidden_size,
                vocab_size=int(cfg["vocab_size"]),
                output_dir=output_dir,
                progress_interval=int(cfg["progress_interval"]),
                physical_batch_size=int(cfg["physical_batch_size"]),
                track_activation_memory=bool(cfg["track_activation_memory"]),
                gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
                enable_action_trace=bool(cfg["enable_action_trace"]),
                sync_action_trace=bool(cfg["sync_action_trace"]),
                perf_minimal_metrics=bool(cfg["perf_minimal_metrics"]),
                recv_inflight_depth=int(cfg.get("recv_inflight_depth", 1)),
                window_input_staging=bool(cfg.get("window_input_staging", False)),
                local_param_stats=local_param_stats,
                memory_ledger=memory_ledger,
            )
            if phase_summary is not None:
                phases.append(phase_summary)

        phase_summary = run_phase_schedule_v3(
            rank=rank,
            world_size=world_size,
            phase="train",
            records=train_records,
            manifest_dir=train_manifest.parent,
            mode="train",
            request_prefix=str(cfg["request_prefix"]),
            chunk=chunk,
            optimizer=optimizer,
            train_chunks=train_chunks,
            dtype=dtype,
            device=device,
            belief_transport_mode=str(cfg["belief_transport_mode"]),
            grad_clip=float(cfg["grad_clip"]),
            learning_rate_override=cfg["learning_rate"],
            hidden_size=hidden_size,
            vocab_size=int(cfg["vocab_size"]),
            output_dir=output_dir,
            progress_interval=int(cfg["progress_interval"]),
            physical_batch_size=int(cfg["physical_batch_size"]),
            track_activation_memory=bool(cfg["track_activation_memory"]),
            gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
            enable_action_trace=bool(cfg["enable_action_trace"]),
            sync_action_trace=bool(cfg["sync_action_trace"]),
            perf_minimal_metrics=bool(cfg["perf_minimal_metrics"]),
            recv_inflight_depth=int(cfg.get("recv_inflight_depth", 1)),
            window_input_staging=bool(cfg.get("window_input_staging", False)),
            local_param_stats=local_param_stats,
            memory_ledger=memory_ledger,
        )
        if phase_summary is not None:
            phases.append(phase_summary)

        if not bool(cfg["skip_eval_after"]):
            phase_summary = run_phase_schedule_v3(
                rank=rank,
                world_size=world_size,
                phase="eval_after",
                records=eval_records,
                manifest_dir=eval_manifest.parent,
                mode="eval",
                request_prefix=str(cfg["request_prefix"]),
                chunk=chunk,
                optimizer=optimizer,
                train_chunks=train_chunks,
                dtype=dtype,
                device=device,
                belief_transport_mode=str(cfg["belief_transport_mode"]),
                grad_clip=float(cfg["grad_clip"]),
                learning_rate_override=cfg["learning_rate"],
                hidden_size=hidden_size,
                vocab_size=int(cfg["vocab_size"]),
                output_dir=output_dir,
                progress_interval=int(cfg["progress_interval"]),
                physical_batch_size=int(cfg["physical_batch_size"]),
                track_activation_memory=bool(cfg["track_activation_memory"]),
                gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
                enable_action_trace=bool(cfg["enable_action_trace"]),
                sync_action_trace=bool(cfg["sync_action_trace"]),
                perf_minimal_metrics=bool(cfg["perf_minimal_metrics"]),
                recv_inflight_depth=int(cfg.get("recv_inflight_depth", 1)),
                window_input_staging=bool(cfg.get("window_input_staging", False)),
                local_param_stats=local_param_stats,
                memory_ledger=memory_ledger,
            )
            if phase_summary is not None:
                phases.append(phase_summary)

        if rank == world_size - 1:
            phase_by_name = {phase["phase"]: phase for phase in phases}
            train_phase = phase_by_name.get("train", {})

            summary: dict[str, Any] = {
                "runner": "bpfree-clean-schedule-v5-minimal-perf",
                "transport": "nccl-batch-isend-irecv-clean-v5-minimal-perf",
                "recv_inflight_depth": int(cfg.get("recv_inflight_depth", 1)),
                "window_input_staging": bool(cfg.get("window_input_staging", False)),
                "schedule_semantics": "local_update_window_split_into_physical_microbatches",
                "model_name": cfg["model_name"],
                "resolved_model": resolved_model,
                "num_chunks": world_size,
                "stage_devices": devices,
                "train_chunks": sorted(train_chunks),
                "train_epochs": cfg["train_epochs"],
                "physical_request_batch": cfg["physical_batch_size"],
                "gradient_accumulation_steps": cfg["gradient_accumulation_steps"],
                "microbatches": cfg["gradient_accumulation_steps"],
                "effective_optimizer_batch": int(cfg["physical_batch_size"])
                * int(cfg["gradient_accumulation_steps"]),
                "activation_tracking_enabled": cfg["track_activation_memory"],
                "action_trace_enabled": cfg["enable_action_trace"],
                "sync_action_trace": cfg["sync_action_trace"],
                "perf_minimal_metrics": cfg["perf_minimal_metrics"],
                "unique_train_records": len(read_manifest(train_manifest, cfg["train_limit"])),
                "train_steps": len(train_records),
                "learning_rate": cfg["learning_rate"],
                "optimizer": cfg["optimizer"],
                "belief_transport_mode": cfg["belief_transport_mode"],
                "alpha": cfg["alpha"],
                "label_smoothing": cfg["label_smoothing"],
                "dtype": cfg["dtype"],
                "seed": cfg["seed"],
                "trainable_mode": cfg["trainable_mode"],
                "lora": {
                    "rank": cfg["lora_rank"],
                    "alpha": cfg["lora_alpha"],
                    "targets": cfg["lora_targets"],
                    "init_std": cfg["lora_init_std"],
                    "init_seed": cfg["lora_init_seed"],
                    "initialization_fingerprint": lora_init_fingerprint,
                    "modules": trainable_setup.lora_modules,
                    "trainable_params": trainable_setup.trainable_params,
                },
                "completed_records": int(train_phase.get("completed_records", 0)),
                "optimizer_steps": int(train_phase.get("optimizer_steps", 0)),
                "phases": phases,
            }

            if "eval_before" in phase_by_name and "eval_after" in phase_by_name:
                summary["delta"] = {
                    "choice_accuracy": phase_by_name["eval_after"]["choice_accuracy"]
                    - phase_by_name["eval_before"]["choice_accuracy"],
                    "avg_loss": phase_by_name["eval_after"]["avg_loss"]
                    - phase_by_name["eval_before"]["avg_loss"],
                }

            if isinstance(train_phase.get("pipeline_phase_metrics"), dict):
                summary["pipeline_phase_metrics"] = train_phase["pipeline_phase_metrics"]

            summary_path = output_dir / "summary.json"
            summary_path.write_text(
                json.dumps(summary, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
            print(f"Wrote {summary_path}", flush=True)

        barrier_on_device(device)

    finally:
        dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean BP-free schedule runner v0")

    parser.add_argument("--model_name", default="tinyllama")
    parser.add_argument("--train_manifest", required=True)
    parser.add_argument("--eval_manifest", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)

    parser.add_argument("--num_chunks", type=int, default=3)
    parser.add_argument("--stage_devices", default="cuda:0,cuda:1,cuda:2")
    parser.add_argument("--backend", default="nccl")
    parser.add_argument("--master_addr", default="127.0.0.1")
    parser.add_argument("--master_port", type=int, default=29671)

    parser.add_argument("--train_limit", type=int, default=None)
    parser.add_argument("--eval_limit", type=int, default=None)
    parser.add_argument("--train_epochs", type=int, default=1)

    parser.add_argument("--physical_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=3)

    parser.add_argument("--train_chunks", default="all")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--request_prefix", default="agnews")

    parser.add_argument("--belief_transport_mode", default="terminal")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)

    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--optimizer", default="adamw")
    parser.add_argument("--sgd_momentum", type=float, default=0.0)
    parser.add_argument("--sgd_dampening", type=float, default=0.0)
    parser.add_argument("--sgd_weight_decay", type=float, default=0.0)
    parser.add_argument("--sgd_nesterov", action="store_true")
    parser.add_argument("--grad_clip", type=float, default=0.0)

    parser.add_argument("--trainable_mode", default="lora")
    parser.add_argument("--lora_targets", default="q_proj,v_proj")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_init_std", type=float, default=0.01)
    parser.add_argument("--lora_init_seed", type=int, default=None)

    parser.add_argument("--progress_interval", type=int, default=16)
    parser.add_argument("--enable_action_trace", action="store_true")
    parser.add_argument("--sync_action_trace", action="store_true")
    parser.add_argument("--perf_minimal_metrics", action="store_true")
    parser.add_argument("--recv_inflight_depth", type=int, default=1)

    window_staging_group = parser.add_mutually_exclusive_group()
    window_staging_group.add_argument(
        "--window_input_staging",
        dest="window_input_staging",
        action="store_true",
        help=(
            "Load common tensors once per optimizer window and slice "
            "GPU views per microbatch."
        ),
    )
    window_staging_group.add_argument(
        "--no-window_input_staging",
        dest="window_input_staging",
        action="store_false",
        help="Load common tensors separately for every physical microbatch.",
    )
    parser.set_defaults(window_input_staging=False)

    parser.add_argument("--skip_eval_before", action="store_true")
    parser.add_argument("--skip_eval_after", action="store_true")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--track_activation_memory", dest="track_activation_memory", action="store_true")
    group.add_argument("--no-track_activation_memory", dest="track_activation_memory", action="store_false")
    parser.set_defaults(track_activation_memory=False)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.num_chunks <= 0:
        raise ValueError("--num_chunks must be positive.")
    if args.physical_batch_size <= 0:
        raise ValueError("--physical_batch_size must be positive.")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient_accumulation_steps must be positive.")
    if args.train_epochs <= 0:
        raise ValueError("--train_epochs must be positive.")

    resolved_model = resolve_model_name(args.model_name)
    model_config = AutoConfig.from_pretrained(resolved_model)
    vocab_size = int(model_config.vocab_size)

    cfg = {
        "model_name": args.model_name,
        "resolved_model": resolved_model,
        "train_manifest": str(args.train_manifest),
        "eval_manifest": str(args.eval_manifest),
        "output_dir": str(args.output_dir),
        "num_chunks": args.num_chunks,
        "stage_devices": parse_devices(args.stage_devices, args.num_chunks),
        "backend": args.backend,
        "master_addr": args.master_addr,
        "master_port": args.master_port,
        "train_limit": args.train_limit,
        "eval_limit": args.eval_limit,
        "train_epochs": args.train_epochs,
        "physical_batch_size": args.physical_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "train_chunks": args.train_chunks,
        "dtype": args.dtype,
        "seed": args.seed,
        "request_prefix": args.request_prefix,
        "belief_transport_mode": normalize_belief_transport_mode(args.belief_transport_mode),
        "alpha": args.alpha,
        "label_smoothing": args.label_smoothing,
        "learning_rate": args.learning_rate,
        "optimizer": args.optimizer,
        "sgd_momentum": args.sgd_momentum,
        "sgd_dampening": args.sgd_dampening,
        "sgd_weight_decay": args.sgd_weight_decay,
        "sgd_nesterov": args.sgd_nesterov,
        "grad_clip": args.grad_clip,
        "trainable_mode": args.trainable_mode,
        "lora_targets": args.lora_targets,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_init_std": args.lora_init_std,
        "lora_init_seed": args.lora_init_seed,
        "progress_interval": args.progress_interval,
        "skip_eval_before": args.skip_eval_before,
        "skip_eval_after": args.skip_eval_after,
        "track_activation_memory": args.track_activation_memory,
        "enable_action_trace": args.enable_action_trace,
        "sync_action_trace": args.sync_action_trace,
        "perf_minimal_metrics": args.perf_minimal_metrics,
        "window_input_staging": args.window_input_staging,
        "vocab_size": vocab_size,
    }

    print(
        "Starting clean BP-free schedule runner "
        f"model={resolved_model} devices={cfg['stage_devices']} "
        f"backend={args.backend} physical_batch={args.physical_batch_size} "
        f"microbatches={args.gradient_accumulation_steps} "
        f"effective_batch={args.physical_batch_size * args.gradient_accumulation_steps} "
        f"action_trace={args.enable_action_trace} "
        f"sync_action_trace={args.sync_action_trace} "
        f"window_input_staging={args.window_input_staging}",
        flush=True,
    )

    cfg["recv_inflight_depth"] = int(getattr(args, "recv_inflight_depth", 1))

    mp.spawn(
        distributed_worker,
        args=(args.num_chunks, cfg),
        nprocs=args.num_chunks,
        join=True,
    )


if __name__ == "__main__":
    main()
