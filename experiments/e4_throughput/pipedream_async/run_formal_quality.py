#!/usr/bin/env python3
"""Run three-seed quality evaluation for the four E4 schedule semantics."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from experiments.e4_throughput.pipedream_async.run_formal_comparison import (
    build_command,
    method_orders,
    metrics,
    read_json,
    validate_config,
    validate_summary,
    write_json,
)


DEFAULT_CONFIG = (
    "experiments/e4_throughput/pipedream_async/configs/quality.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_process(
    command: list[str], *, repo_root: Path, log_path: Path
) -> int:
    env = os.environ.copy()
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "TRANSFORMERS_VERBOSITY": "error",
            "OMP_NUM_THREADS": "4",
        }
    )
    with log_path.open("w", encoding="utf-8") as log:
        return subprocess.run(
            command,
            cwd=repo_root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode


def evaluator_command(
    *, cfg: dict[str, Any], repo_root: Path, train_dir: Path, eval_dir: Path
) -> list[str]:
    command = [
        sys.executable,
        "experiments/e4_throughput/pipedream_async/"
        "evaluate_pipeline_lora_state.py",
        "--manifest", str(repo_root / str(cfg["eval_manifest"])),
    ]
    for stage in range(int(cfg["num_chunks"])):
        command.extend(
            ["--lora_state", str(train_dir / f"stage{stage}.trainable.pt")]
        )
    command.extend(
        [
            "--output_dir", str(eval_dir),
            "--model_name", str(cfg["model_name"]),
            "--num_chunks", str(cfg["num_chunks"]),
            "--eval_limit", str(cfg["eval_limit"]),
            "--batch_size", str(cfg["eval_batch_size"]),
            "--device", str(cfg["eval_device"]),
            "--dtype", str(cfg["dtype"]),
            "--lora_targets", str(cfg["lora_targets"]),
            "--lora_rank", str(cfg["lora_rank"]),
            "--lora_alpha", str(cfg["lora_alpha"]),
            "--lora_init_std", str(cfg["lora_init_std"]),
        ]
    )
    return command


def validate_eval(summary: dict[str, Any], cfg: dict[str, Any]) -> None:
    expected = int(cfg["eval_limit"])
    checks = {
        "records": int(summary.get("records", -1)) == expected,
        "choice_count": int(summary.get("choice_count", -1)) == expected,
        "choice_correct_range": 0
        <= int(summary.get("choice_correct", -1))
        <= expected,
        "accuracy_range": 0.0
        <= float(summary.get("choice_accuracy", -1.0))
        <= 1.0,
        "finite_loss": 0.0 <= float(summary.get("avg_loss", -1.0)) < 100.0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"evaluation contract failed: {failed}")


def aggregate(output_root: Path) -> None:
    rows: list[dict[str, Any]] = []
    for metadata_path in sorted(output_root.glob("seed_*/*/run_metadata.json")):
        metadata = read_json(metadata_path)
        run_dir = metadata_path.parent
        train_summary_path = run_dir / "train" / "summary.json"
        eval_summary_path = run_dir / "eval_test" / "summary.json"
        if not train_summary_path.is_file() or not eval_summary_path.is_file():
            continue
        if metadata.get("train_returncode") != 0 or metadata.get("eval_returncode") != 0:
            continue
        method = str(metadata["method"])
        train = metrics(read_json(train_summary_path), method)
        evaluation = read_json(eval_summary_path)
        rows.append(
            {
                "seed": int(metadata["seed"]),
                "order_position": int(metadata["order_position"]),
                "method": method,
                "train_throughput_per_s": train["throughput_per_s"],
                "train_wall_ms": train["wall_ms"],
                "optimizer_steps": train["optimizer_steps"],
                "test_records": int(evaluation["records"]),
                "test_choice_correct": int(evaluation["choice_correct"]),
                "test_accuracy": float(evaluation["choice_accuracy"]),
                "test_choice_nll": float(evaluation["avg_loss"]),
                "lora_fingerprint": evaluation["lora_fingerprint"],
            }
        )

    if not rows:
        return
    with (output_root / "quality_runs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary_rows: list[dict[str, Any]] = []
    for method in sorted({str(row["method"]) for row in rows}):
        selected = [row for row in rows if row["method"] == method]
        accuracies = [float(row["test_accuracy"]) for row in selected]
        losses = [float(row["test_choice_nll"]) for row in selected]
        throughputs = [
            float(row["train_throughput_per_s"]) for row in selected
        ]
        summary_rows.append(
            {
                "method": method,
                "seeds": len(selected),
                "test_accuracy_mean": statistics.mean(accuracies),
                "test_accuracy_stdev": (
                    statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
                ),
                "test_nll_mean": statistics.mean(losses),
                "test_nll_stdev": (
                    statistics.stdev(losses) if len(losses) > 1 else 0.0
                ),
                "train_throughput_mean": statistics.mean(throughputs),
            }
        )
    with (output_root / "quality_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    write_json(output_root / "quality_summary.json", summary_rows)


def run_one(
    *,
    base_cfg: dict[str, Any],
    repo_root: Path,
    output_root: Path,
    seed: int,
    order_position: int,
    method: str,
    resume: bool,
    dry_run: bool,
) -> None:
    cfg = dict(base_cfg)
    cfg["seed"] = seed
    cfg["lora_init_seed"] = seed
    run_dir = output_root / f"seed_{seed}" / method
    train_dir = run_dir / "train"
    eval_dir = run_dir / "eval_test"
    train_summary_path = train_dir / "summary.json"
    eval_summary_path = eval_dir / "summary.json"
    if train_summary_path.is_file() and eval_summary_path.is_file() and resume:
        validate_summary(read_json(train_summary_path), cfg, method)
        validate_eval(read_json(eval_summary_path), cfg)
        print(f"[resume] seed={seed} method={method}", flush=True)
        return
    if run_dir.exists() and not resume and not dry_run:
        raise FileExistsError(run_dir)

    train_ready = False
    if run_dir.exists() and resume:
        if not train_summary_path.is_file():
            raise RuntimeError(
                f"cannot resume incomplete training without summary: {run_dir}"
            )
        validate_summary(read_json(train_summary_path), cfg, method)
        train_ready = True

    train_command = build_command(
        cfg=cfg,
        repo_root=repo_root,
        method=method,
        output_dir=train_dir,
        master_port=30300 + (seed - 20260531) * 10 + order_position,
    )
    if method == "bpfree":
        train_command.append("--save_trainable_state")
    eval_command = evaluator_command(
        cfg=cfg,
        repo_root=repo_root,
        train_dir=train_dir,
        eval_dir=eval_dir,
    )
    print("$", shlex.join(train_command), flush=True)
    print("$", shlex.join(eval_command), flush=True)
    if dry_run:
        return

    run_dir.mkdir(parents=True, exist_ok=train_ready)
    metadata_path = run_dir / "run_metadata.json"
    metadata = read_json(metadata_path) if metadata_path.is_file() else {
        "experiment_id": cfg["experiment_id"],
        "seed": seed,
        "order_position": order_position,
        "method": method,
        "train_command": train_command,
        "eval_command": eval_command,
        "started_epoch_s": time.time(),
    }
    if train_ready:
        metadata.setdefault("train_returncode", 0)
    write_json(run_dir / "run_metadata.json", metadata)
    if not train_ready:
        train_dir.mkdir()
        metadata["train_returncode"] = run_process(
            train_command,
            repo_root=repo_root,
            log_path=run_dir / "train.log",
        )
        write_json(run_dir / "run_metadata.json", metadata)
        if metadata["train_returncode"]:
            raise RuntimeError(f"training failed: seed={seed} method={method}")
        validate_summary(read_json(train_summary_path), cfg, method)
    missing = [
        train_dir / f"stage{stage}.trainable.pt"
        for stage in range(int(cfg["num_chunks"]))
        if not (train_dir / f"stage{stage}.trainable.pt").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"missing stage states: {missing}")

    eval_dir.mkdir(exist_ok=True)
    metadata["eval_returncode"] = run_process(
        eval_command,
        repo_root=repo_root,
        log_path=run_dir / "eval.log",
    )
    metadata["ended_epoch_s"] = time.time()
    write_json(run_dir / "run_metadata.json", metadata)
    if metadata["eval_returncode"]:
        raise RuntimeError(f"evaluation failed: seed={seed} method={method}")
    validate_eval(read_json(eval_summary_path), cfg)
    print(f"[ok] seed={seed} method={method}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    cfg = read_json(repo_root / args.config)
    validate_config(cfg, repo_root)
    output_root = repo_root / str(cfg["output_root"])
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    elif output_root.exists() and not args.resume and not args.dry_run:
        raise FileExistsError(
            f"output exists; use --resume or --overwrite: {output_root}"
        )

    seeds = [int(seed) for seed in cfg["seeds"]]
    orders = method_orders(
        [str(method) for method in cfg["method_order"]], len(seeds)
    )
    manifest = {
        "config": cfg,
        "train_manifest_sha256": sha256(
            repo_root / str(cfg["train_manifest"])
        ),
        "eval_manifest_sha256": sha256(
            repo_root / str(cfg["eval_manifest"])
        ),
        "effective_optimizer_batch": int(cfg["physical_request_batch"])
        * int(cfg["microbatches_per_update"]),
        "optimizer_steps_per_method": int(cfg["train_limit"])
        // (
            int(cfg["physical_request_batch"])
            * int(cfg["microbatches_per_update"])
        ),
        "seed_method_orders": {
            str(seed): order for seed, order in zip(seeds, orders)
        },
        "python": sys.executable,
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2), flush=True)
    else:
        write_json(output_root / "experiment_manifest.json", manifest)

    for seed, order in zip(seeds, orders):
        for position, method in enumerate(order):
            run_one(
                base_cfg=cfg,
                repo_root=repo_root,
                output_root=output_root,
                seed=seed,
                order_position=position,
                method=method,
                resume=args.resume,
                dry_run=args.dry_run,
            )
            if not args.dry_run:
                aggregate(output_root)
                time.sleep(float(cfg["cooldown_seconds"]))
    if not args.dry_run:
        aggregate(output_root)


if __name__ == "__main__":
    main()
