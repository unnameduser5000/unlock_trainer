#!/usr/bin/env python3
"""Run repeated full-backward 1F1B LoRA baseline experiments."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER_SCRIPT = REPO_ROOT / "src" / "sg_exe_trainer" / "runtime" / "exactbp" / "distributed_runtime.py"


def parse_csv_ints(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def phase_by_name(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {phase["phase"]: phase for phase in summary.get("phases", [])}


def flatten_summary(*, seed: int, output_dir: Path, summary: dict[str, Any], elapsed_s: float) -> dict[str, Any]:
    phases = phase_by_name(summary)
    train = phases.get("train", {})
    eval_after = phases.get("eval_after", {})
    return {
        "runner": summary.get("runner", "1f1b_lora_pipeline"),
        "policy": "1f1b",
        "seed": seed,
        "output_dir": str(output_dir),
        "elapsed_s": elapsed_s,
        "train_records": summary.get("train_records", train.get("rows", "")),
        "unique_train_records": summary.get("unique_train_records", ""),
        "train_epochs": summary.get("train_epochs", ""),
        "eval_records": summary.get("eval_records", eval_after.get("rows", "")),
        "microbatches": summary.get("microbatches", ""),
        "batch_size": summary.get("batch_size", ""),
        "train_batches": train.get("batches", ""),
        "train_completed_records": train.get("completed_records", train.get("rows", "")),
        "train_skipped_records": train.get("skipped_records", 0),
        "train_skipped_batches": train.get("skipped_batches", 0),
        "train_optimizer_steps": train.get("optimizer_steps", train.get("batches", "")),
        "train_wall_ms": train.get("wall_ms", ""),
        "train_throughput_per_s": train.get("throughput_per_s", ""),
        "train_avg_loss": train.get("avg_loss", ""),
        "eval_wall_ms": eval_after.get("wall_ms", ""),
        "eval_throughput_per_s": eval_after.get("throughput_per_s", ""),
        "eval_choice_accuracy": eval_after.get("choice_accuracy", ""),
        "eval_avg_loss": eval_after.get("avg_loss", ""),
        "choice_correct": eval_after.get("choice_correct", summary.get("choice_correct", "")),
        "choice_count": eval_after.get("choice_count", summary.get("choice_count", "")),
        "dtype": summary.get("dtype", ""),
        "learning_rate": summary.get("learning_rate", ""),
        "optimizer": summary.get("optimizer", ""),
        "recovery_policy": summary.get("recovery_baseline", {}).get("policy", "strict_skip"),
        "recovery_semantics": summary.get("recovery_baseline", {}).get("semantics", ""),
        "failure_stage": summary.get("recovery_baseline", {}).get("failure_stage", ""),
        "failure_batch_seq": summary.get("recovery_baseline", {}).get("failure_batch_seq", ""),
        "failure_microbatch_index": summary.get("recovery_baseline", {}).get("failure_microbatch_index", ""),
        "checkpoint_interval_batches": summary.get("recovery_baseline", {}).get("checkpoint_interval_batches", ""),
        "recovery_wait_ms": summary.get("recovery_baseline", {}).get("worker_rejoin_delay_ms", ""),
    }


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    numeric_keys = [
        "train_records",
        "unique_train_records",
        "train_epochs",
        "eval_records",
        "microbatches",
        "batch_size",
        "train_batches",
        "train_completed_records",
        "train_skipped_records",
        "train_skipped_batches",
        "train_optimizer_steps",
        "train_wall_ms",
        "train_throughput_per_s",
        "train_avg_loss",
        "eval_wall_ms",
        "eval_throughput_per_s",
        "eval_choice_accuracy",
        "eval_avg_loss",
    ]
    if not rows:
        return []
    out: dict[str, Any] = {"policy": "1f1b", "runs": len(rows)}
    for key in numeric_keys:
        values = []
        for row in rows:
            value = row.get(key)
            if value in ("", None):
                continue
            values.append(float(value))
        if not values:
            continue
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        out[f"{key}_mean"] = mean
        out[f"{key}_std"] = variance ** 0.5
    return [out]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def runner_command(args: argparse.Namespace, *, seed: int, output_dir: Path) -> list[str]:
    torchrun = shutil.which("torchrun")
    if torchrun:
        cmd = [torchrun]
    else:
        cmd = [sys.executable, "-m", "torch.distributed.run"]
    cmd.extend(
        [
            "--standalone",
            f"--nproc_per_node={args.num_chunks}",
            str(RUNNER_SCRIPT),
            "--model_name",
            args.model_name,
            "--train_manifest",
            str(args.train_manifest),
            "--eval_manifest",
            str(args.eval_manifest),
            "--output_dir",
            str(output_dir),
            "--num_chunks",
            str(args.num_chunks),
            "--stage_devices",
            args.stage_devices,
            "--train_limit",
            str(args.train_limit),
            "--eval_limit",
            str(args.eval_limit),
            "--train_epochs",
            str(args.train_epochs),
            "--microbatches",
            str(args.microbatches),
            "--batch_size",
            str(args.batch_size),
            "--dtype",
            args.dtype,
            "--optimizer",
            args.optimizer,
            "--label_smoothing",
            str(args.label_smoothing),
            "--lora_rank",
            str(args.lora_rank),
            "--lora_alpha",
            str(args.lora_alpha),
            "--lora_targets",
            args.lora_targets,
            "--lora_init_std",
            str(args.lora_init_std),
            "--seed",
            str(seed),
            "--progress_interval",
            str(args.progress_interval),
            ]
        )
    if args.learning_rate is not None:
        cmd.extend(["--learning_rate", str(args.learning_rate)])
    if args.recovery_policy is not None:
        cmd.extend(["--recovery_policy", args.recovery_policy])
    if args.failure_stage is not None:
        cmd.extend(["--failure_stage", str(args.failure_stage)])
    if args.failure_batch_seq is not None:
        cmd.extend(["--failure_batch_seq", str(args.failure_batch_seq)])
    if args.failure_microbatch_index is not None:
        cmd.extend(["--failure_microbatch_index", str(args.failure_microbatch_index)])
    if args.checkpoint_interval_batches is not None:
        cmd.extend(["--checkpoint_interval_batches", str(args.checkpoint_interval_batches)])
    if args.worker_rejoin_delay_ms is not None:
        cmd.extend(["--worker_rejoin_delay_ms", str(args.worker_rejoin_delay_ms)])
    if args.offline_stage is not None:
        cmd.extend(
            [
                "--offline_stage",
                str(args.offline_stage),
                "--offline_start_seq",
                str(args.offline_start_seq),
                "--offline_end_seq",
                str(args.offline_end_seq),
            ]
        )
    if args.grad_clip is not None:
        cmd.extend(["--grad_clip", str(args.grad_clip)])
    if args.skip_eval_before:
        cmd.append("--skip_eval_before")
    if args.skip_eval_after:
        cmd.append("--skip_eval_after")
    if not args.record_timeline:
        cmd.append("--no-record_timeline")
    return cmd


def run_seed(args: argparse.Namespace, *, seed: int) -> dict[str, Any]:
    output_dir = args.output_root / f"1f1b_seed{seed}"
    summary_path = output_dir / "summary.json"
    log_path = output_dir / "run.log"
    output_dir.mkdir(parents=True, exist_ok=True)
    if summary_path.is_file() and not args.force:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return flatten_summary(seed=seed, output_dir=output_dir, summary=summary, elapsed_s=0.0)

    cmd = runner_command(args, seed=seed, output_dir=output_dir)
    env = os.environ.copy()
    env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    env.setdefault("TQDM_DISABLE", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("TERM", "dumb")

    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write("$ " + shlex.join(cmd) + "\n")
        log_handle.flush()
        process = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            check=False,
        )
    elapsed_s = time.perf_counter() - started
    (output_dir / "exit_code.txt").write_text(f"{process.returncode}\n", encoding="utf-8")
    if process.returncode != 0:
        raise RuntimeError(f"1F1B seed {seed} failed; see {log_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return flatten_summary(seed=seed, output_dir=output_dir, summary=summary, elapsed_s=elapsed_s)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeated 1F1B baseline seeds.")
    parser.add_argument("--model_name", default="tinyllama")
    parser.add_argument("--train_manifest", type=Path, required=True)
    parser.add_argument("--eval_manifest", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--seeds", default="20260531,20260532,20260533")
    parser.add_argument("--num_chunks", type=int, default=3)
    parser.add_argument("--stage_devices", default="cuda:0,cuda:1,cuda:2")
    parser.add_argument("--train_limit", type=int, default=512)
    parser.add_argument("--eval_limit", type=int, default=256)
    parser.add_argument("--train_epochs", type=int, default=1)
    parser.add_argument("--microbatches", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_targets", default="q_proj,v_proj")
    parser.add_argument("--lora_init_std", type=float, default=0.01)
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--progress_interval", type=int, default=128)
    parser.add_argument("--offline_stage", type=int, default=None)
    parser.add_argument("--offline_start_seq", type=int, default=None)
    parser.add_argument("--offline_end_seq", type=int, default=None)
    parser.add_argument(
        "--recovery_policy",
        default="strict_skip",
        choices=["strict_skip", "wait_for_rejoin_batch_boundary", "restart_from_last_commit"],
    )
    parser.add_argument("--failure_stage", type=int, default=None)
    parser.add_argument("--failure_batch_seq", type=int, default=None)
    parser.add_argument("--failure_microbatch_index", type=int, default=None)
    parser.add_argument("--checkpoint_interval_batches", type=int, default=1)
    parser.add_argument("--worker_rejoin_delay_ms", type=float, default=2000.0)
    parser.add_argument("--skip_eval_before", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip_eval_after", action="store_true")
    parser.add_argument("--record_timeline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = parse_csv_ints(args.seeds)
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = [run_seed(args, seed=seed) for seed in seeds]
    summary_rows = aggregate(rows)
    write_csv(args.output_root / "1f1b_runs.csv", rows)
    write_csv(args.output_root / "1f1b_summary.csv", summary_rows)
    (args.output_root / "1f1b_summary.json").write_text(
        json.dumps({"runs": rows, "aggregate": summary_rows}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary_rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
