#!/usr/bin/env python3
"""Measure the PipeDream convergence curve under the legacy E1 protocol."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_SEEDS = (20260531, 20260532, 20260533)
DEFAULT_STEPS = tuple(range(125, 1251, 125))
TRAIN_MANIFEST = (
    "data/sft_requests/"
    "tinyllama_agnews128_label_train10000_seed20260531/requests.jsonl"
)
VALIDATION_MANIFEST = (
    "data/sft_requests/"
    "tinyllama_agnews128_label_valid1000_nooverlap_seed20260531/requests.jsonl"
)
LEGACY_E1_ROOT = (
    "results/e1_quality/raw/legacy/root_debug_runs/"
    "agnews_e1e2_formal_v1/e1_quality"
)
DEFAULT_OUTPUT_ROOT = "results/e1_quality/raw/e1_pipedream_curve_b8_20260722"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("expected a non-empty list of unique integers")
    return values


def run_logged(command: list[str], *, repo_root: Path, log_path: Path) -> int:
    env = os.environ.copy()
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "TRANSFORMERS_VERBOSITY": "error",
            "OMP_NUM_THREADS": "4",
        }
    )
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"started_epoch_s={started}\n")
        log.write(f"command={shlex.join(command)}\n")
        log.flush()
        returncode = subprocess.run(
            command,
            cwd=repo_root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode
        log.write(f"ended_epoch_s={time.time()}\n")
        log.write(f"returncode={returncode}\n")
    return returncode


def train_command(
    *, repo_root: Path, output_dir: Path, seed: int, step: int, port: int
) -> list[str]:
    train_limit = step * 8
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=3",
        f"--master_port={port}",
        "-m", "experiments.shared.baselines.pipedream_cpu",
        "--model_name",
        "tinyllama",
        "--train_manifest",
        str(repo_root / TRAIN_MANIFEST),
        "--output_dir",
        str(output_dir),
        "--num_chunks",
        "3",
        "--stage_devices",
        "cuda:0,cuda:1,cuda:2",
        "--train_limit",
        str(train_limit),
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
        "--recv_prepost_depth",
        "4",
        "--max_pending_send_bytes",
        "67108864",
        "--max_posted_recv_bytes",
        "67108864",
        "--perf_minimal_metrics",
    ]


def eval_command(
    *, repo_root: Path, train_dir: Path, output_dir: Path
) -> list[str]:
    command = [
        sys.executable,
        "experiments/e4_throughput/pipedream_async/evaluate_pipeline_lora_state.py",
        "--manifest",
        str(repo_root / VALIDATION_MANIFEST),
    ]
    for stage in range(3):
        command.extend(
            ["--lora_state", str(train_dir / f"stage{stage}.trainable.pt")]
        )
    command.extend(
        [
            "--output_dir",
            str(output_dir),
            "--model_name",
            "tinyllama",
            "--num_chunks",
            "3",
            "--eval_limit",
            "1000",
            "--batch_size",
            "32",
            "--device",
            "cuda:3",
            "--dtype",
            "bfloat16",
            "--lora_targets",
            "q_proj,v_proj",
            "--lora_rank",
            "4",
            "--lora_alpha",
            "16.0",
            "--lora_init_std",
            "0.01",
        ]
    )
    return command


def validate_train(path: Path, *, step: int) -> dict[str, Any]:
    summary = read_json(path)
    train_limit = step * 8
    checks = {
        "completed_records": summary.get("completed_records") == train_limit,
        "optimizer_steps": summary.get("optimizer_steps_per_stage") == step,
        "physical_batch": summary.get("physical_request_batch") == 1,
        "effective_batch": summary.get("effective_optimizer_batch") == 8,
        "rank_count": len(summary.get("by_rank", [])) == 3,
    }
    for rank in summary.get("by_rank", []):
        stage = int(rank["stage_id"])
        transport = rank.get("transport_budget", {})
        checks[f"stage{stage}_backward_count"] = (
            rank.get("local_backward_count") == train_limit
        )
        checks[f"stage{stage}_optimizer_steps"] = (
            rank.get("local_optimizer_steps") == step
        )
        checks[f"stage{stage}_missing_gradients"] = (
            rank.get("missing_snapshot_gradients") == 0
        )
        checks[f"stage{stage}_pending_send_bytes"] = (
            transport.get("pending_send_bytes_at_end") == 0
        )
        checks[f"stage{stage}_posted_recv_bytes"] = (
            transport.get("posted_recv_bytes_at_end") == 0
        )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"training contract failed: {failed}")
    return summary


def validate_eval(path: Path) -> dict[str, Any]:
    summary = read_json(path)
    accuracy = float(summary.get("choice_accuracy", math.nan))
    loss = float(summary.get("avg_loss", math.nan))
    checks = {
        "records": summary.get("records") == 1000,
        "choice_count": summary.get("choice_count") == 1000,
        "accuracy": math.isfinite(accuracy) and 0.0 <= accuracy <= 1.0,
        "loss": math.isfinite(loss) and 0.0 <= loss < 100.0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"evaluation contract failed: {failed}")
    return summary


def legacy_step_zero(repo_root: Path, seed: int) -> dict[str, Any]:
    curve_path = (
        repo_root
        / LEGACY_E1_ROOT
        / "1f1b_3gpu"
        / f"seed{seed}"
        / "validation_curve.csv"
    )
    with curve_path.open(newline="", encoding="utf-8") as handle:
        row = next(row for row in csv.DictReader(handle) if int(row["optimizer_step"]) == 0)
    return {
        "seed": seed,
        "optimizer_step": 0,
        "train_samples_seen": 0,
        "validation_records": int(row["validation_records"]),
        "choice_correct": int(row["choice_correct"]),
        "choice_accuracy": float(row["choice_accuracy"]),
        "choice_loss": float(row["avg_loss"]),
        "source": str(curve_path.relative_to(repo_root)),
        "train_wall_s": 0.0,
        "eval_wall_s": float(row["wall_ms"]) / 1000.0,
        "run_wall_s": 0.0,
    }


def collect_rows(
    *, repo_root: Path, output_root: Path, seeds: list[int]
) -> list[dict[str, Any]]:
    rows = [legacy_step_zero(repo_root, seed) for seed in seeds]
    for metadata_path in sorted(output_root.glob("seed*/step*/run_metadata.json")):
        metadata = read_json(metadata_path)
        run_dir = metadata_path.parent
        train_path = run_dir / "train" / "summary.json"
        eval_path = run_dir / "validation" / "summary.json"
        if not train_path.is_file() or not eval_path.is_file():
            continue
        step = int(metadata["optimizer_step"])
        train = validate_train(train_path, step=step)
        evaluation = validate_eval(eval_path)
        rows.append(
            {
                "seed": int(metadata["seed"]),
                "optimizer_step": step,
                "train_samples_seen": step * 8,
                "validation_records": int(evaluation["records"]),
                "choice_correct": int(evaluation["choice_correct"]),
                "choice_accuracy": float(evaluation["choice_accuracy"]),
                "choice_loss": float(evaluation["avg_loss"]),
                "source": str(run_dir.relative_to(repo_root)),
                "train_wall_s": float(train["wall_ms"]) / 1000.0,
                "eval_wall_s": float(evaluation["wall_ms"]) / 1000.0,
                "run_wall_s": float(metadata["ended_epoch_s"])
                - float(metadata["started_epoch_s"]),
            }
        )
    rows.sort(key=lambda row: (int(row["seed"]), int(row["optimizer_step"])))
    return rows


def aggregate(*, repo_root: Path, output_root: Path, seeds: list[int]) -> None:
    rows = collect_rows(repo_root=repo_root, output_root=output_root, seeds=seeds)
    raw_path = output_root / "pipedream_curve_raw.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary_rows: list[dict[str, Any]] = []
    for step in sorted({int(row["optimizer_step"]) for row in rows}):
        selected = [row for row in rows if int(row["optimizer_step"]) == step]
        accuracies = [float(row["choice_accuracy"]) for row in selected]
        losses = [float(row["choice_loss"]) for row in selected]
        summary_rows.append(
            {
                "suite": "e1_quality",
                "method": "pipedream",
                "optimizer_step": step,
                "train_samples_seen": step * 8,
                "runs": len(selected),
                "choice_accuracy_mean": statistics.mean(accuracies),
                "choice_accuracy_std": (
                    statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
                ),
                "choice_loss_mean": statistics.mean(losses),
                "choice_loss_std": (
                    statistics.stdev(losses) if len(losses) > 1 else 0.0
                ),
            }
        )
    with (output_root / "pipedream_curve_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)


def run_one(
    *,
    repo_root: Path,
    output_root: Path,
    seed: int,
    step: int,
    port: int,
    resume: bool,
) -> None:
    run_dir = output_root / f"seed{seed}" / f"step{step:04d}"
    train_dir = run_dir / "train"
    eval_dir = run_dir / "validation"
    metadata_path = run_dir / "run_metadata.json"
    train_summary_path = train_dir / "summary.json"
    eval_summary_path = eval_dir / "summary.json"

    if resume and train_summary_path.is_file() and eval_summary_path.is_file():
        validate_train(train_summary_path, step=step)
        validate_eval(eval_summary_path)
        print(f"[resume] seed={seed} step={step}", flush=True)
        return
    if run_dir.exists() and not resume:
        raise FileExistsError(f"output exists; pass --resume: {run_dir}")

    run_dir.mkdir(parents=True, exist_ok=True)
    train = train_command(
        repo_root=repo_root,
        output_dir=train_dir,
        seed=seed,
        step=step,
        port=port,
    )
    evaluate = eval_command(repo_root=repo_root, train_dir=train_dir, output_dir=eval_dir)
    metadata = read_json(metadata_path) if metadata_path.is_file() else {}
    metadata.update(
        {
            "seed": seed,
            "optimizer_step": step,
            "train_samples_seen": step * 8,
            "train_command": train,
            "eval_command": evaluate,
        }
    )
    metadata.setdefault("started_epoch_s", time.time())
    write_json(metadata_path, metadata)

    if not train_summary_path.is_file():
        train_dir.mkdir(exist_ok=True)
        metadata["train_returncode"] = run_logged(
            train, repo_root=repo_root, log_path=run_dir / "train.log"
        )
        write_json(metadata_path, metadata)
        if metadata["train_returncode"]:
            raise RuntimeError(f"training failed: seed={seed} step={step}")
    validate_train(train_summary_path, step=step)
    missing = [
        train_dir / f"stage{stage}.trainable.pt"
        for stage in range(3)
        if not (train_dir / f"stage{stage}.trainable.pt").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"missing stage checkpoints: {missing}")

    if not eval_summary_path.is_file():
        eval_dir.mkdir(exist_ok=True)
        metadata["eval_returncode"] = run_logged(
            evaluate, repo_root=repo_root, log_path=run_dir / "validation.log"
        )
        if metadata["eval_returncode"]:
            write_json(metadata_path, metadata)
            raise RuntimeError(f"evaluation failed: seed={seed} step={step}")
    evaluation = validate_eval(eval_summary_path)
    metadata["train_returncode"] = 0
    metadata["eval_returncode"] = 0
    metadata["ended_epoch_s"] = time.time()
    write_json(metadata_path, metadata)
    print(
        f"[ok] seed={seed} step={step} "
        f"accuracy={float(evaluation['choice_accuracy']):.4f} "
        f"nll={float(evaluation['avg_loss']):.4f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=parse_int_list, default=list(DEFAULT_SEEDS))
    parser.add_argument("--steps", type=parse_int_list, default=list(DEFAULT_STEPS))
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    output_root = repo_root / args.output_root
    seeds = list(args.seeds)
    steps = list(args.steps)
    if any(step <= 0 or step % 125 for step in steps):
        raise ValueError("all steps must be positive multiples of 125")
    if max(steps) > 1250:
        raise ValueError("the legacy E1 training budget ends at step 1250")
    for path in (repo_root / TRAIN_MANIFEST, repo_root / VALIDATION_MANIFEST):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = {
        "experiment_id": "e1_pipedream_curve_b8_20260722",
        "method": "pipedream",
        "seeds": seeds,
        "optimizer_steps": steps,
        "train_manifest": TRAIN_MANIFEST,
        "train_manifest_sha256": sha256(repo_root / TRAIN_MANIFEST),
        "validation_manifest": VALIDATION_MANIFEST,
        "validation_manifest_sha256": sha256(repo_root / VALIDATION_MANIFEST),
        "train_records_per_step": 8,
        "physical_request_batch": 1,
        "microbatches_per_update": 8,
        "effective_optimizer_batch": 8,
        "learning_rate": 0.0001,
        "grad_clip": 1.0,
        "dtype": "bfloat16",
        "validation_records": 1000,
        "python": sys.executable,
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2), flush=True)
        for seed_index, seed in enumerate(seeds):
            for step_index, step in enumerate(steps):
                command = train_command(
                    repo_root=repo_root,
                    output_dir=output_root
                    / f"seed{seed}"
                    / f"step{step:04d}"
                    / "train",
                    seed=seed,
                    step=step,
                    port=31000 + seed_index * 20 + step_index,
                )
                print(shlex.join(command), flush=True)
        return

    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "experiment_manifest.json", manifest)
    for seed_index, seed in enumerate(seeds):
        for step_index, step in enumerate(steps):
            run_one(
                repo_root=repo_root,
                output_root=output_root,
                seed=seed,
                step=step,
                port=31000 + seed_index * 20 + step_index,
                resume=args.resume,
            )
            aggregate(repo_root=repo_root, output_root=output_root, seeds=seeds)
    aggregate(repo_root=repo_root, output_root=output_root, seeds=seeds)


if __name__ == "__main__":
    main()
