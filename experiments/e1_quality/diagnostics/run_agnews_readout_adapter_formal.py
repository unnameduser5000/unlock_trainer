#!/usr/bin/env python3
"""Run the paired-seed AG News E1 readout-adapter ablation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = "agnews-e1-readout-adapter-v1"
METHOD = "bpfree_ce_adapter_3gpu"
RUNTIME_MODULE = "sg_exe_trainer.runtime.bpfree.orchestrated_runtime"
RUNTIME_SOURCE = Path("src/sg_exe_trainer/runtime/bpfree/orchestrated_runtime.py")


def parse_seeds(raw: str) -> list[int]:
    seeds = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    return seeds


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(project_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def stream_command(command: list[str], *, cwd: Path, log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_handle.write(line)
            log_handle.flush()
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def validation_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    best = max(rows, key=lambda row: float(row["choice_accuracy"]))
    final = rows[-1]
    return {
        "final_validation_accuracy": float(final["choice_accuracy"]),
        "final_validation_nll": float(final["avg_loss"]),
        "best_validation_accuracy": float(best["choice_accuracy"]),
        "best_validation_step": int(best["optimizer_step"]),
    }


def write_aggregate(output_root: Path, seeds: list[int]) -> None:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        run_dir = output_root / METHOD / f"seed{seed}"
        summary_path = run_dir / "scheduler_summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        row = {
            "seed": seed,
            "test_accuracy": float(summary["choice_accuracy"]),
            "test_nll": float(summary["avg_loss"]),
            "test_records": int(summary["choice_count"]),
            **validation_stats(run_dir / "validation_curve.csv"),
        }
        rows.append(row)

    aggregate_csv = output_root / "aggregate_seeds.csv"
    if rows:
        fieldnames = list(rows[0])
        with aggregate_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    accuracies = [float(row["test_accuracy"]) for row in rows]
    nlls = [float(row["test_nll"]) for row in rows]
    aggregate = {
        "protocol_version": PROTOCOL_VERSION,
        "method": METHOD,
        "requested_seeds": seeds,
        "completed_seeds": [int(row["seed"]) for row in rows],
        "test_accuracy_mean": statistics.mean(accuracies) if accuracies else None,
        "test_accuracy_std": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0 if accuracies else None,
        "test_nll_mean": statistics.mean(nlls) if nlls else None,
        "test_nll_std": statistics.stdev(nlls) if len(nlls) > 1 else 0.0 if nlls else None,
        "aggregate_csv": str(aggregate_csv),
    }
    (output_root / "aggregate_summary.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train_manifest",
        type=Path,
        default=Path("data/sft_requests/tinyllama_agnews128_label_train10000_seed20260531/requests.jsonl"),
    )
    parser.add_argument(
        "--validation_manifest",
        type=Path,
        default=Path(
            "data/sft_requests/tinyllama_agnews128_label_valid1000_nooverlap_seed20260531/requests.jsonl"
        ),
    )
    parser.add_argument(
        "--test_manifest",
        type=Path,
        default=Path("data/sft_requests/tinyllama_agnews128_label_test7600_seed20260531/requests.jsonl"),
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("results/e1_quality/raw/agnews_readout_adapter_formal_v1"),
    )
    parser.add_argument("--seeds", default="20260531,20260532,20260533")
    parser.add_argument("--stage_devices", default="cuda:0,cuda:1,cuda:2")
    parser.add_argument("--train_limit", type=int, default=10000)
    parser.add_argument("--validation_limit", type=int, default=1000)
    parser.add_argument("--test_limit", type=int, default=7600)
    parser.add_argument("--validation_interval_steps", type=int, default=125)
    parser.add_argument("--effective_batch", type=int, default=8)
    parser.add_argument("--adapter_bottleneck", type=int, default=128)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    seeds = parse_seeds(args.seeds)
    manifests = {
        "train": (project_root / args.train_manifest).resolve(),
        "validation": (project_root / args.validation_manifest).resolve(),
        "test": (project_root / args.test_manifest).resolve(),
    }
    for name, path in manifests.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} manifest does not exist: {path}")
    runtime_source = (project_root / RUNTIME_SOURCE).resolve()
    if not runtime_source.exists():
        raise FileNotFoundError(f"runtime source does not exist: {runtime_source}")
    output_root = (project_root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "method": METHOD,
        "git_commit": git_commit(project_root),
        "code": {
            "driver": {
                "path": str(Path(__file__).resolve().relative_to(project_root)),
                "sha256": sha256(Path(__file__).resolve()),
            },
            "runtime": {
                "module": RUNTIME_MODULE,
                "path": str(runtime_source.relative_to(project_root)),
                "sha256": sha256(runtime_source),
            },
        },
        "manifests": {
            name: {"path": str(path.relative_to(project_root)), "sha256": sha256(path)}
            for name, path in manifests.items()
        },
        "seeds": seeds,
        "train_limit": args.train_limit,
        "validation_limit": args.validation_limit,
        "test_limit": args.test_limit,
        "effective_batch": args.effective_batch,
        "validation_interval_steps": args.validation_interval_steps,
        "stage_devices": args.stage_devices,
        "learning_rate": 1e-4,
        "dtype": "bfloat16",
        "lora": {"rank": 4, "alpha": 16.0, "targets": "q_proj,v_proj", "init_std": 0.01},
        "local_readout_adapter": {"bottleneck": args.adapter_bottleneck, "stages": "middle"},
    }
    (output_root / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for seed in seeds:
        run_dir = output_root / METHOD / f"seed{seed}"
        summary_path = run_dir / "scheduler_summary.json"
        if summary_path.exists() and not args.force:
            print(f"Skipping completed seed {seed}: {summary_path}", flush=True)
            continue
        if run_dir.exists() and any(run_dir.iterdir()) and not args.force:
            raise FileExistsError(f"Incomplete non-empty run directory requires --force: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            RUNTIME_MODULE,
            "--manifest",
            str(manifests["train"]),
            "--output_dir",
            str(run_dir),
            "--num_chunks",
            "3",
            "--stage_devices",
            args.stage_devices,
            "--topology",
            "phone_fixed",
            "--max_inflight",
            str(args.effective_batch),
            "--scheduler_policy",
            "fifo",
            "--recovery_policy",
            "replay_after_update",
            "--failure_mode",
            "none",
            "--task_timeout_ms",
            "0",
            "--train_chunks",
            "all",
            "--stage_update_policy",
            "stride",
            "--gradient_accumulation_steps",
            str(args.effective_batch),
            "--belief_transport_mode",
            "none",
            "--alpha",
            "1.0",
            "--label_smoothing",
            "0.0",
            "--limit",
            str(args.train_limit),
            "--model_name",
            "tinyllama",
            "--learning_rate",
            "0.0001",
            "--optimizer",
            "adamw",
            "--grad_clip",
            "1.0",
            "--dtype",
            "bfloat16",
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
            "--local_readout_adapter_bottleneck",
            str(args.adapter_bottleneck),
            "--local_readout_adapter_stages",
            "middle",
            "--seed",
            str(seed),
            "--progress_interval",
            "1000",
            "--eval_manifest",
            str(manifests["test"]),
            "--eval_limit",
            str(args.test_limit),
            "--validation_manifest",
            str(manifests["validation"]),
            "--validation_limit",
            str(args.validation_limit),
            "--validation_interval_steps",
            str(args.validation_interval_steps),
        ]
        (run_dir / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
        (run_dir / "run_config.json").write_text(
            json.dumps({**protocol, "seed": seed, "command": command}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Starting seed {seed}: {run_dir}", flush=True)
        stream_command(command, cwd=project_root, log_path=run_dir / "run.log")
        write_aggregate(output_root, seeds)

    write_aggregate(output_root, seeds)
    print(f"Completed AG News readout-adapter protocol: {output_root}", flush=True)


if __name__ == "__main__":
    main()
