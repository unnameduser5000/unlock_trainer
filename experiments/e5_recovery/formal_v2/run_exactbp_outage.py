#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sg_exe_trainer.runtime.bpfree.schedule import (
    BPFreeUpdateWindow,
    split_records_into_update_windows,
)
from sg_exe_trainer.runtime.exactbp import distributed_runtime as f1b

from sg_exe_trainer.runtime.recovery.checkpoint_store import StageCheckpointStore
from sg_exe_trainer.runtime.recovery.durable_io import atomic_write_json
from sg_exe_trainer.runtime.recovery.event_log import (
    RecoveryEventName,
    RecoveryEventRecorder,
    RecoveryTimeline,
)
from sg_exe_trainer.runtime.recovery.state_contract import StageCommitLedger

from .exactbp_contract import ExactBPBacklogContract
from .exactbp_runtime import ExactBPWindowRuntime
from .protocol import E5OutageProtocol


def _shared_protocol_payload(protocol: E5OutageProtocol) -> dict[str, Any]:
    payload = protocol.to_dict()
    for method_specific in (
        "belief_transport_mode",
        "catchup_policy",
        "phase_actions",
        "expected_invariants",
    ):
        payload.pop(method_specific, None)
    return payload


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


def _run_windows(
    *,
    phase: str,
    windows: list[BPFreeUpdateWindow],
    runtime: ExactBPWindowRuntime,
    checkpoint_store: StageCheckpointStore | None = None,
    commit_ledger: StageCommitLedger | None = None,
) -> list[dict[str, Any]]:
    return [
        asdict(
            runtime.run_window(
                phase=phase,
                window=window,
                checkpoint_store=checkpoint_store,
                commit_ledger=commit_ledger,
            )
        )
        for window in windows
    ]


def _init_process_group(
    *,
    backend: str,
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    kwargs: dict[str, Any] = {
        "backend": backend,
        "rank": rank,
        "world_size": world_size,
    }
    if backend == "nccl":
        kwargs["device_id"] = device
    try:
        dist.init_process_group(**kwargs)
    except TypeError:
        kwargs.pop("device_id", None)
        dist.init_process_group(**kwargs)


def _worker(
    rank: int,
    world_size: int,
    cfg: dict[str, Any],
    runtime_cls: type[ExactBPWindowRuntime],
) -> None:
    os.environ["MASTER_ADDR"] = str(cfg["master_addr"])
    os.environ["MASTER_PORT"] = str(cfg["master_port"])
    device = torch.device(cfg["stage_devices"][rank])
    if device.type == "cuda":
        torch.cuda.set_device(device)
    _init_process_group(
        backend=str(cfg["backend"]),
        rank=rank,
        world_size=world_size,
        device=device,
    )

    try:
        _seed_everything(int(cfg["seed"]), rank)
        output_dir = Path(cfg["output_dir"])
        state_root = output_dir / "durable_state"
        protocol = E5OutageProtocol(**cfg["protocol_constructor"])
        contract = ExactBPBacklogContract(protocol)
        recovery_state_mode = str(cfg["recovery_state_mode"])
        durable_recovery = recovery_state_mode == "durable"
        event_recorder = RecoveryEventRecorder(
            state_root,
            run_id=protocol.run_id,
            rank=rank,
            stage_id=rank,
        )
        ledger = StageCommitLedger(state_root, protocol.run_id) if durable_recovery else None
        checkpoint_store = (
            StageCheckpointStore(state_root, protocol.run_id) if durable_recovery else None
        )

        runtime = runtime_cls.build(
            rank=rank,
            world_size=world_size,
            device=device,
            device_name=str(device),
            manifest_dir=Path(cfg["train_manifest"]).parent,
            cfg=cfg,
        )

        required_records = protocol.total_logical_windows * protocol.effective_batch_size
        records = f1b.read_manifest(Path(cfg["train_manifest"]), required_records)
        if len(records) < required_records:
            raise ValueError(f"need {required_records} records, found {len(records)}")
        windows = split_records_into_update_windows(
            records=records,
            effective_batch_size=protocol.effective_batch_size,
            n_microbatches=protocol.microbatches_per_window,
            drop_last=True,
        )
        prelude = windows[: protocol.outage_start_window]
        outage = windows[
            protocol.outage_start_window : protocol.outage_end_window_exclusive
        ]
        resumed = windows[
            protocol.outage_end_window_exclusive : protocol.total_logical_windows
        ]

        dist.barrier()
        prelude_metrics = _run_windows(
            phase="prelude",
            windows=prelude,
            runtime=runtime,
        )
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

        commits_at_rejoin = (
            {
                stage_id: len(ledger.list_stage(stage_id))
                for stage_id in range(world_size)
            }
            if ledger is not None
            else {stage_id: 0 for stage_id in range(world_size)}
        )
        if rank == protocol.failure_stage:
            event_recorder.record(
                RecoveryEventName.STAGE_REJOINED,
                window_id=protocol.outage_end_window_exclusive,
                details={
                    "queued_windows": protocol.outage_windows,
                    "queued_records": protocol.outage_windows
                    * protocol.effective_batch_size,
                },
            )
        # The measured interval starts when the rejoin is published. Catch-up
        # cannot begin until every pipeline rank has observed that transition.
        dist.barrier()

        catchup_metrics = _run_windows(
            phase="catchup_full_1f1b",
            windows=outage,
            runtime=runtime,
            checkpoint_store=checkpoint_store,
            commit_ledger=ledger,
        )
        if rank == world_size - 1:
            event_recorder.record(
                RecoveryEventName.TERMINAL_TARGET_REACHED,
                window_id=protocol.outage_end_window_exclusive - 1,
            )
        dist.barrier()
        if rank == 0:
            event_recorder.record(
                RecoveryEventName.LIVE_P2P_RESUMED,
                window_id=protocol.outage_end_window_exclusive,
            )
        dist.barrier()

        resumed_metrics = _run_windows(
            phase="resumed",
            windows=resumed,
            runtime=runtime,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        rank_payload = {
            "rank": rank,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
            ),
            "optimizer_steps": runtime.optimizer_steps,
            "trainable_mode": runtime.trainable_mode,
            "transport_summary": (
                runtime.transport_summary()
                if hasattr(runtime, "transport_summary")
                else {}
            ),
            "peak_cuda_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            ),
            "prelude_metrics": prelude_metrics,
            "catchup_metrics": catchup_metrics,
            "resumed_metrics": resumed_metrics,
        }
        atomic_write_json(output_dir / f"rank{rank}.summary.json", rank_payload)
        dist.barrier()

        if rank == 0:
            final_completion_counts = (
                {
                    stage_id: len(ledger.list_stage(stage_id))
                    for stage_id in range(world_size)
                }
                if ledger is not None
                else {
                    stage_id: protocol.outage_windows
                    for stage_id in range(world_size)
                }
            )
            contract.validate(
                commits_at_rejoin=commits_at_rejoin,
                final_commit_counts=final_completion_counts,
            )
            timeline = RecoveryTimeline.load(state_root, protocol.run_id)
            rank_summaries = [
                json.loads(
                    (output_dir / f"rank{stage_id}.summary.json").read_text(
                        encoding="utf-8"
                    )
                )
                for stage_id in range(world_size)
            ]
            checkpoint_file_bytes = sum(
                int(metric["checkpoint"]["payload_file_bytes"])
                for summary in rank_summaries
                for metric in summary["catchup_metrics"]
                if metric["checkpoint"] is not None
            )
            recovery_timing = {
                "outage_duration_ms": timeline.duration_ms(
                    RecoveryEventName.OUTAGE_INJECTED,
                    RecoveryEventName.STAGE_REJOINED,
                ),
                **timeline.common_recovery_summary(),
            }
            summary = {
                "runner": (
                    "e5_formal_v2_exactbp_outage"
                    if durable_recovery
                    else "e5_formal_v2_exactbp_volatile_outage"
                ),
                "transport": cfg["transport_name"],
                "transport_details": cfg["transport_details"],
                "recovery_state_mode": recovery_state_mode,
                "protocol": _shared_protocol_payload(protocol),
                "execution_plan": contract.execution_plan(),
                "expected_invariants": contract.expected_invariants(),
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
                "recovery_timing": recovery_timing,
                "backlog_at_rejoin": {
                    "windows": protocol.outage_windows,
                    "records": protocol.outage_windows * protocol.effective_batch_size,
                    "stage_commit_counts": {
                        str(key): value for key, value in commits_at_rejoin.items()
                    },
                },
                "progress_at_rejoin": {
                    "local_optimizer_windows": {
                        str(key): value for key, value in commits_at_rejoin.items()
                    }
                },
                "completed_stage_windows": {
                    str(key): value for key, value in final_completion_counts.items()
                },
                **(
                    {
                        "stage_commit_counts": {
                            str(key): value
                            for key, value in final_completion_counts.items()
                        }
                    }
                    if durable_recovery
                    else {}
                ),
                "durable_state": {
                    "checkpoint_file_bytes": checkpoint_file_bytes,
                    "boundary_file_bytes": 0,
                    "total_file_bytes": checkpoint_file_bytes,
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
    default_recovery_state_mode: str = "durable",
    default_backend: str = "nccl",
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="E5 formal-v2 Exact-BP backlog recovery with real Schedule1F1B"
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="tinyllama")
    parser.add_argument("--stage-devices", default="cuda:0,cuda:1,cuda:2")
    parser.add_argument("--backend", default=default_backend)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29771)
    parser.add_argument("--prelude-windows", type=int, default=4)
    parser.add_argument("--outage-windows", type=int, default=4)
    parser.add_argument("--resumed-windows", type=int, default=2)
    parser.add_argument("--physical-batch-size", type=int, default=1)
    parser.add_argument("--microbatches-per-window", type=int, default=8)
    parser.add_argument("--max-pending-windows", type=int, default=4)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--trainable-mode", default="lora")
    parser.add_argument("--lora-targets", default="q_proj,v_proj")
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-init-std", type=float, default=0.01)
    parser.add_argument("--lora-init-seed", type=int, default=20260531)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument(
        "--recovery-state-mode",
        choices=("durable", "volatile"),
        default=default_recovery_state_mode,
        help="durable writes checkpoints; volatile measures transient-outage compute only",
    )
    return parser


def main(
    *,
    default_recovery_state_mode: str = "durable",
    default_backend: str = "nccl",
    transport_name: str = "nccl-gpu-p2p",
    transport_details: str = "PyTorch Schedule1F1B GPU point-to-point transport",
    runtime_cls: type[ExactBPWindowRuntime] = ExactBPWindowRuntime,
) -> None:
    args = build_parser(
        default_recovery_state_mode=default_recovery_state_mode,
        default_backend=default_backend,
    ).parse_args()
    protocol = E5OutageProtocol(
        run_id=args.run_id,
        prelude_windows=args.prelude_windows,
        outage_windows=args.outage_windows,
        resumed_windows=args.resumed_windows,
        physical_batch_size=args.physical_batch_size,
        microbatches_per_window=args.microbatches_per_window,
        max_pending_windows=args.max_pending_windows,
    )
    ExactBPBacklogContract(protocol)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    atomic_write_json(args.output_dir / "protocol.json", _shared_protocol_payload(protocol))

    resolved_model = f1b.resolve_model_name(args.model_name)
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
        "label_smoothing": args.label_smoothing,
        "microbatches_per_window": args.microbatches_per_window,
        "seed": args.seed,
        "recovery_state_mode": args.recovery_state_mode,
    }
    mp.spawn(
        _worker,
        args=(protocol.num_stages, cfg, runtime_cls),
        nprocs=protocol.num_stages,
        join=True,
    )


if __name__ == "__main__":
    main()
