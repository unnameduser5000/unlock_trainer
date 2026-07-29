#!/usr/bin/env python3
"""Measure the three-seed GPipe validation curve with one training run per seed."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from run_pipedream_curve import (
    DEFAULT_SEEDS,
    DEFAULT_STEPS,
    TRAIN_MANIFEST,
    VALIDATION_MANIFEST,
    eval_command,
    legacy_step_zero,
    read_json,
    run_logged,
    validate_eval,
    write_json,
)


DEFAULT_OUTPUT_ROOT = "results/e1_quality/raw/e1_gpipe_curve_b8_20260727"


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("expected a non-empty list of unique integers")
    return values


def train_command(
    *,
    repo_root: Path,
    output_dir: Path,
    seed: int,
    port: int,
    checkpoint_interval: int,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=3",
        f"--master_port={port}",
        "-m", "sg_exe_trainer.runtime.exactbp.cpu_runner",
        "--model_name",
        "tinyllama",
        "--train_manifest",
        str(repo_root / TRAIN_MANIFEST),
        "--eval_manifest",
        str(repo_root / VALIDATION_MANIFEST),
        "--output_dir",
        str(output_dir),
        "--num_chunks",
        "3",
        "--stage_devices",
        "cuda:0,cuda:1,cuda:2",
        "--train_limit",
        "10000",
        "--train_epochs",
        "1",
        "--physical_batch_size",
        "1",
        "--gradient_accumulation_steps",
        "8",
        "--learning_rate",
        "0.0001",
        "--optimizer",
        "adamw",
        "--grad_clip",
        "1.0",
        "--dtype",
        "bfloat16",
        "--label_smoothing",
        "0.0",
        "--trainable_mode",
        "lora",
        "--lora_rank",
        "4",
        "--lora_alpha",
        "16.0",
        "--lora_targets",
        "q_proj,v_proj",
        "--lora_init_std",
        "0.01",
        "--lora_init_seed",
        str(seed),
        "--seed",
        str(seed),
        "--pipeline_schedule",
        "gpipe",
        "--recv_prepost_depth",
        "4",
        "--max_pending_send_bytes",
        "67108864",
        "--max_posted_recv_bytes",
        "67108864",
        "--eval_limit",
        "1",
        "--progress_interval",
        "1000",
        "--skip_eval_before",
        "--skip_eval_after",
        "--no-track_activation_memory",
        "--perf_minimal_metrics",
        "--save_trainable_state",
        "--trainable_checkpoint_interval",
        str(checkpoint_interval),
    ]


def checkpoint_dir(train_dir: Path, step: int) -> Path:
    return train_dir / "trainable_checkpoints" / f"step_{step:04d}"


def validate_checkpoints(train_dir: Path, steps: list[int]) -> None:
    missing = [
        checkpoint_dir(train_dir, step) / f"stage{stage}.trainable.pt"
        for step in steps
        for stage in range(3)
        if not (checkpoint_dir(train_dir, step) / f"stage{stage}.trainable.pt").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"missing periodic trainable checkpoints: {missing[:6]}")


def run_seed(
    *,
    repo_root: Path,
    output_root: Path,
    seed: int,
    steps: list[int],
    port: int,
    resume: bool,
) -> None:
    seed_dir = output_root / f"seed{seed}"
    train_dir = seed_dir / "train"
    train_summary = train_dir / "summary.json"
    if not (resume and train_summary.is_file()):
        if seed_dir.exists() and not resume:
            raise FileExistsError(seed_dir)
        train_dir.mkdir(parents=True, exist_ok=True)
        command = train_command(
            repo_root=repo_root,
            output_dir=train_dir,
            seed=seed,
            port=port,
            checkpoint_interval=min(steps),
        )
        returncode = run_logged(
            command,
            repo_root=repo_root,
            log_path=seed_dir / "train.log",
        )
        if returncode:
            raise RuntimeError(f"GPipe training failed for seed {seed}")
    validate_checkpoints(train_dir, steps)

    for step in steps:
        eval_dir = seed_dir / f"validation_step_{step:04d}"
        eval_summary = eval_dir / "summary.json"
        if resume and eval_summary.is_file():
            validate_eval(eval_summary)
            continue
        eval_dir.mkdir(exist_ok=True)
        returncode = run_logged(
            eval_command(
                repo_root=repo_root,
                train_dir=checkpoint_dir(train_dir, step),
                output_dir=eval_dir,
            ),
            repo_root=repo_root,
            log_path=seed_dir / f"validation_step_{step:04d}.log",
        )
        if returncode:
            raise RuntimeError(f"GPipe validation failed for seed {seed}, step {step}")
        validate_eval(eval_summary)
        print(f"[ok] seed={seed} step={step}", flush=True)


def aggregate(
    *,
    repo_root: Path,
    output_root: Path,
    seeds: list[int],
    steps: list[int],
) -> None:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        rows.append(legacy_step_zero(repo_root, seed))
        for step in steps:
            summary_path = output_root / f"seed{seed}" / f"validation_step_{step:04d}" / "summary.json"
            evaluation = validate_eval(summary_path)
            rows.append(
                {
                    "seed": seed,
                    "optimizer_step": step,
                    "train_samples_seen": step * 8,
                    "validation_records": int(evaluation["records"]),
                    "choice_correct": int(evaluation["choice_correct"]),
                    "choice_accuracy": float(evaluation["choice_accuracy"]),
                    "choice_loss": float(evaluation["avg_loss"]),
                    "source": str(summary_path.relative_to(repo_root)),
                }
            )
    rows.sort(key=lambda row: (int(row["seed"]), int(row["optimizer_step"])))
    with (output_root / "gpipe_curve_raw.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary_rows: list[dict[str, Any]] = []
    for step in [0, *steps]:
        selected = [row for row in rows if int(row["optimizer_step"]) == step]
        accuracies = [float(row["choice_accuracy"]) for row in selected]
        losses = [float(row["choice_loss"]) for row in selected]
        summary_rows.append(
            {
                "suite": "e1_quality",
                "method": "gpipe_3gpu",
                "optimizer_step": step,
                "train_samples_seen": step * 8,
                "runs": len(selected),
                "choice_accuracy_mean": statistics.mean(accuracies),
                "choice_accuracy_std": (
                    statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
                ),
                "choice_loss_mean": statistics.mean(losses),
                "choice_loss_std": statistics.stdev(losses) if len(losses) > 1 else 0.0,
            }
        )
    with (output_root / "gpipe_curve_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=parse_int_list, default=list(DEFAULT_SEEDS))
    parser.add_argument("--steps", type=parse_int_list, default=list(DEFAULT_STEPS))
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    output_root = repo_root / args.output_root
    seeds = list(args.seeds)
    steps = sorted(args.steps)
    if steps != list(range(min(steps), max(steps) + min(steps), min(steps))):
        raise ValueError("steps must be a contiguous multiple of the checkpoint interval")
    if steps[-1] != 1250 or steps[0] != 125:
        raise ValueError("the formal E1 protocol requires steps 125..1250")
    for path in (repo_root / TRAIN_MANIFEST, repo_root / VALIDATION_MANIFEST):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(
        output_root / "experiment_manifest.json",
        {
            "experiment_id": "e1_gpipe_curve_b8_20260727",
            "method": "gpipe_3gpu",
            "seeds": seeds,
            "optimizer_steps": [0, *steps],
            "physical_batch_size": 1,
            "microbatches_per_update": 8,
            "effective_batch": 8,
            "train_manifest": TRAIN_MANIFEST,
            "validation_manifest": VALIDATION_MANIFEST,
            "created_epoch_s": time.time(),
        },
    )
    for index, seed in enumerate(seeds):
        run_seed(
            repo_root=repo_root,
            output_root=output_root,
            seed=seed,
            steps=steps,
            port=31500 + index,
            resume=args.resume,
        )
        aggregate(
            repo_root=repo_root,
            output_root=output_root,
            seeds=seeds[: index + 1],
            steps=steps,
        )
    print(json.dumps({"output_root": str(output_root), "seeds": seeds}, indent=2))


if __name__ == "__main__":
    main()
