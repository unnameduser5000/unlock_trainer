#!/usr/bin/env python3
"""Run the formal E4.1 BP-free vs Exact-BP GPU-scaling matrix."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from experiments.e4_throughput.provenance import (
    artifact_hashes,
    capture_provenance,
)


DEFAULT_CONFIG = (
    "experiments/e4_throughput/configs/e4_1_gpu_scaling.json"
)
GPU_SCALING_SOURCE_PATHS = (
    "experiments/e4_throughput/provenance.py",
    "experiments/e4_throughput/run_e4_1_gpu_scaling.py",
    "src/sg_exe_trainer/runtime/bpfree/chunk_split.py",
    "src/sg_exe_trainer/runtime/bpfree/gpu_phase.py",
    "src/sg_exe_trainer/runtime/bpfree/gpu_runner.py",
    "src/sg_exe_trainer/runtime/bpfree/gpu_stage.py",
    "src/sg_exe_trainer/runtime/bpfree/gpu_transport.py",
    "src/sg_exe_trainer/runtime/bpfree/model_runtime.py",
    "src/sg_exe_trainer/runtime/bpfree/schedule_runtime.py",
    "src/sg_exe_trainer/runtime/exactbp/distributed_runtime.py",
)


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def validate_config(cfg: dict[str, Any], repo_root: Path) -> None:
    gpu_counts = cfg["gpu_counts"]
    methods = cfg["methods"]
    physical = int(cfg["physical_request_batch"])
    microbatches = int(cfg["microbatches_per_update"])
    effective = int(cfg["effective_optimizer_batch"])
    train_limit = int(cfg["train_limit"])

    if not gpu_counts or any(int(value) < 2 for value in gpu_counts):
        raise ValueError("gpu_counts must contain integers >= 2")
    if not set(methods).issubset({"bpfree", "exactbp"}):
        raise ValueError("methods may only contain bpfree and exactbp")
    if physical * microbatches != effective:
        raise ValueError(
            "physical_request_batch * microbatches_per_update "
            "must equal effective_optimizer_batch"
        )
    if any(microbatches < int(stages) for stages in gpu_counts):
        raise ValueError(
            "microbatches_per_update must be >= every GPU/stage count"
        )
    if train_limit % effective != 0:
        raise ValueError(
            "train_limit must be divisible by effective_optimizer_batch"
        )
    for key in ("train_manifest", "eval_manifest"):
        path = repo_root / cfg[key]
        if not path.is_file():
            raise FileNotFoundError(f"missing {key}: {path}")


def common_runtime_args(
    cfg: dict[str, Any],
    output_dir: Path,
    gpu_count: int,
    train_limit: int,
) -> list[str]:
    devices = ",".join(f"cuda:{index}" for index in range(gpu_count))
    args = [
        "--model_name", str(cfg["model_name"]),
        "--train_manifest", str(cfg["train_manifest"]),
        "--eval_manifest", str(cfg["eval_manifest"]),
        "--output_dir", str(output_dir),
        "--num_chunks", str(gpu_count),
        "--stage_devices", devices,
        "--train_limit", str(train_limit),
        "--eval_limit", str(cfg["eval_limit"]),
        "--train_epochs", str(cfg["train_epochs"]),
        "--dtype", str(cfg["dtype"]),
        "--learning_rate", str(cfg["learning_rate"]),
        "--optimizer", str(cfg["optimizer"]),
        "--grad_clip", str(cfg["grad_clip"]),
        "--seed", str(cfg["seed"]),
        "--trainable_mode", "lora",
        "--lora_targets", str(cfg["lora_targets"]),
        "--lora_rank", str(cfg["lora_rank"]),
        "--lora_alpha", str(cfg["lora_alpha"]),
        "--lora_init_std", str(cfg["lora_init_std"]),
        "--lora_init_seed", str(cfg["lora_init_seed"]),
        "--progress_interval", "0",
    ]
    if cfg.get("skip_eval_before", False):
        args.append("--skip_eval_before")
    if cfg.get("skip_eval_after", False):
        args.append("--skip_eval_after")
    return args


def build_command(
    method: str,
    cfg: dict[str, Any],
    output_dir: Path,
    gpu_count: int,
    train_limit: int,
    master_port: int,
) -> list[str]:
    common = common_runtime_args(
        cfg=cfg,
        output_dir=output_dir,
        gpu_count=gpu_count,
        train_limit=train_limit,
    )
    microbatches = int(cfg["microbatches_per_update"])
    physical = int(cfg["physical_request_batch"])

    if method == "bpfree":
        command = [
            sys.executable,
            "-m",
            "sg_exe_trainer.runtime.bpfree.gpu_runner",
            *common,
            "--backend", "nccl",
            "--master_port", str(master_port),
            "--physical_batch_size", str(physical),
            "--gradient_accumulation_steps", str(microbatches),
            "--train_chunks", "all",
            "--belief_transport_mode", str(cfg["belief_transport_mode"]),
            "--recv_inflight_depth", str(cfg["recv_inflight_depth"]),
            "--no-track_activation_memory",
        ]
        if cfg.get("perf_minimal_metrics", False):
            command.append("--perf_minimal_metrics")
        return command

    if method == "exactbp":
        return [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={gpu_count}",
            "--module",
            "sg_exe_trainer.runtime.exactbp.distributed_runtime",
            *common,
            "--batch_size", str(cfg["effective_optimizer_batch"]),
            "--microbatches", str(microbatches),
            "--gc_interval_batches", str(cfg["gc_interval_batches"]),
            "--no-track_activation_memory",
            "--no-record_timeline",
        ]

    raise ValueError(f"unsupported method: {method}")


def train_phase(summary: dict[str, Any]) -> dict[str, Any]:
    phases = summary.get("phases", [])
    for phase in phases:
        if isinstance(phase, dict) and phase.get("phase") == "train":
            return phase
    raise ValueError("summary has no train phase")


def validate_summary(
    summary_path: Path,
    cfg: dict[str, Any],
    gpu_count: int,
    train_limit: int,
) -> dict[str, Any]:
    summary = read_json(summary_path)
    phase = train_phase(summary)
    expected_steps = train_limit // int(cfg["effective_optimizer_batch"])

    checks = {
        "num_chunks": summary.get("num_chunks") == gpu_count,
        "physical_request_batch": (
            summary.get("physical_request_batch")
            == int(cfg["physical_request_batch"])
        ),
        "microbatches": (
            summary.get("microbatches")
            == int(cfg["microbatches_per_update"])
        ),
        "effective_optimizer_batch": (
            summary.get("effective_optimizer_batch")
            == int(cfg["effective_optimizer_batch"])
        ),
        "activation_tracking_disabled": (
            summary.get("activation_tracking_enabled") is False
        ),
        "completed_records": (
            int(phase.get("completed_records", -1)) == train_limit
        ),
        "optimizer_steps": (
            int(phase.get("optimizer_steps", -1)) == expected_steps
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            f"summary contract failed at {summary_path}: {failed}"
        )
    return {
        "checks": checks,
        "throughput_per_s": phase.get("throughput_per_s"),
        "wall_ms": phase.get("wall_ms"),
        "completed_records": phase.get("completed_records"),
        "optimizer_steps": phase.get("optimizer_steps"),
    }


def run_and_tee(
    command: list[str],
    env: dict[str, str],
    cwd: Path,
    log_path: Path,
) -> int:
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_handle.write(line)
            log_handle.flush()
        return process.wait()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--gpu-counts")
    parser.add_argument("--methods")
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--rep-indices")
    parser.add_argument("--output-root")
    parser.add_argument("--base-port", type=int, default=29710)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    cfg = read_json(config_path)
    validate_config(cfg, repo_root)

    gpu_counts = (
        parse_csv_ints(args.gpu_counts)
        if args.gpu_counts
        else [int(value) for value in cfg["gpu_counts"]]
    )
    methods = (
        parse_csv_strings(args.methods)
        if args.methods
        else list(cfg["methods"])
    )
    repetitions = (
        args.repetitions
        if args.repetitions is not None
        else int(cfg["repetitions"])
    )
    rep_indices = (
        parse_csv_ints(args.rep_indices)
        if args.rep_indices
        else list(range(repetitions))
    )

    if not set(gpu_counts).issubset(set(int(v) for v in cfg["gpu_counts"])):
        raise ValueError("requested GPU count is outside the frozen config")
    if not set(methods).issubset(set(cfg["methods"])):
        raise ValueError("requested method is outside the frozen config")
    if any(index < 0 for index in rep_indices):
        raise ValueError("rep indices must be non-negative")

    output_root = Path(args.output_root or cfg["output_root"])
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    if args.smoke:
        output_root = output_root / "_launcher_smoke"
        train_limit = int(cfg["effective_optimizer_batch"]) * 2
        rep_indices = [0]
    else:
        train_limit = int(cfg["train_limit"])

    jobs: list[tuple[int, str, int, Path, int, list[str]]] = []
    for gpu_count in gpu_counts:
        for rep_index in rep_indices:
            # Alternate method order across repetitions to reduce systematic
            # warm-cache, temperature, and execution-order bias.
            ordered_methods = (
                methods
                if rep_index % 2 == 0
                else list(reversed(methods))
            )
            for method in ordered_methods:
                method_index = methods.index(method)
                output_dir = (
                    output_root
                    / f"gpu_{gpu_count}"
                    / method
                    / f"rep_{rep_index:02d}"
                )
                port = (
                    args.base_port
                    + gpu_count * 100
                    + method_index * 10
                    + rep_index
                )
                command = build_command(
                    method=method,
                    cfg=cfg,
                    output_dir=output_dir,
                    gpu_count=gpu_count,
                    train_limit=train_limit,
                    master_port=port,
                )
                jobs.append(
                    (
                        gpu_count,
                        method,
                        rep_index,
                        output_dir,
                        port,
                        command,
                    )
                )

    print(f"config={config_path}")
    print(f"output_root={output_root}")
    print(f"jobs={len(jobs)}")
    print(f"train_limit={train_limit}")
    print()

    for index, job in enumerate(jobs, start=1):
        gpu_count, method, rep_index, output_dir, port, command = job
        visible_devices = ",".join(str(i) for i in range(gpu_count))
        print(
            f"[{index}/{len(jobs)}] gpu={gpu_count} method={method} "
            f"rep={rep_index} port={port}"
        )
        print(f"CUDA_VISIBLE_DEVICES={visible_devices} {shlex.join(command)}")
        print(f"output={output_dir}")
        print()

    if args.dry_run:
        return

    failures: list[str] = []

    for index, job in enumerate(jobs, start=1):
        gpu_count, method, rep_index, output_dir, port, command = job
        summary_path = output_dir / "summary.json"
        job_name = f"gpu={gpu_count} method={method} rep={rep_index}"

        if args.resume and summary_path.is_file():
            try:
                result = validate_summary(
                    summary_path=summary_path,
                    cfg=cfg,
                    gpu_count=gpu_count,
                    train_limit=train_limit,
                )
                print(f"[{index}/{len(jobs)}] SKIP valid {job_name}: {result}")
                continue
            except Exception as error:
                print(f"[{index}/{len(jobs)}] rerun invalid {job_name}: {error}")

        if output_dir.exists():
            if args.overwrite or args.resume:
                shutil.rmtree(output_dir)
            else:
                raise FileExistsError(
                    f"{output_dir} exists; use --resume or --overwrite"
                )
        output_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(index) for index in range(gpu_count)
        )
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        execution_environment = {
            "CUDA_VISIBLE_DEVICES": env["CUDA_VISIBLE_DEVICES"],
        }

        started_at = time.time()
        metadata = {
            "experiment_id": cfg["experiment_id"],
            "gpu_count": gpu_count,
            "method": method,
            "rep_index": rep_index,
            "train_limit": train_limit,
            "master_port": port,
            "command": command,
            "command_shell": shlex.join(command),
            "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"],
            "python_executable": sys.executable,
            "config_path": str(config_path),
            "config": cfg,
            "provenance": capture_provenance(
                repo_root=repo_root,
                config_path=config_path,
                cfg=cfg,
                command=command,
                execution_environment=execution_environment,
                source_paths=GPU_SCALING_SOURCE_PATHS,
            ),
            "started_at_epoch_s": started_at,
            "status": "running",
        }
        write_json(output_dir / "run_metadata.json", metadata)

        print(f"[{index}/{len(jobs)}] START {job_name}", flush=True)
        return_code = run_and_tee(
            command=command,
            env=env,
            cwd=repo_root,
            log_path=output_dir / "run.log",
        )
        metadata["finished_at_epoch_s"] = time.time()
        metadata["elapsed_s"] = (
            metadata["finished_at_epoch_s"] - started_at
        )
        metadata["return_code"] = return_code

        try:
            if return_code != 0:
                raise RuntimeError(f"process returned {return_code}")
            result = validate_summary(
                summary_path=summary_path,
                cfg=cfg,
                gpu_count=gpu_count,
                train_limit=train_limit,
            )
            metadata["summary_validation"] = result
            metadata["status"] = "complete"
            print(f"[{index}/{len(jobs)}] PASS {job_name}: {result}")
        except Exception as error:
            metadata["status"] = "failed"
            metadata["error"] = str(error)
            failures.append(f"{job_name}: {error}")
            print(f"[{index}/{len(jobs)}] FAIL {job_name}: {error}")
        finally:
            metadata["artifacts"] = artifact_hashes(
                [
                    output_dir / "run.log",
                    output_dir / "summary.json",
                ],
                repo_root=repo_root,
            )
            write_json(output_dir / "run_metadata.json", metadata)

        if failures and not args.keep_going:
            break

    if failures:
        print("\nFailures:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)

    print("\nE4.1 matrix completed successfully.")


if __name__ == "__main__":
    main()
