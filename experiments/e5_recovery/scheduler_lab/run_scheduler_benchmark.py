#!/usr/bin/env python3
"""Run a reproducible BP-free scheduler policy benchmark matrix.

This is a harness around the canonical orchestrated BP-free runtime. It keeps the topology,
train/eval manifests, limits, and seeds fixed while sweeping scheduler policies
that change the BP-free local update budget.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PolicyConfig:
    name: str
    args: list[str]


def default_policy_configs(stage_id: int) -> dict[str, PolicyConfig]:
    return {
        "all_updates": PolicyConfig(name="all_updates", args=[]),
        "stride2": PolicyConfig(name="stride2", args=["--stage_train_strides", f"{stage_id}:2"]),
        "stride3": PolicyConfig(name="stride3", args=["--stage_train_strides", f"{stage_id}:3"]),
        "queue0": PolicyConfig(
            name="queue0",
            args=[
                "--stage_update_policy",
                "queue_gated",
                "--stage_update_queue_thresholds",
                f"{stage_id}:0",
            ],
        ),
        "queue1": PolicyConfig(
            name="queue1",
            args=[
                "--stage_update_policy",
                "queue_gated",
                "--stage_update_queue_thresholds",
                f"{stage_id}:1",
            ],
        ),
        "queue2": PolicyConfig(
            name="queue2",
            args=[
                "--stage_update_policy",
                "queue_gated",
                "--stage_update_queue_thresholds",
                f"{stage_id}:2",
            ],
        ),
    }


def parse_csv_ints(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer.")
    return values


def parse_policy_names(raw: str, available: dict[str, PolicyConfig]) -> list[str]:
    if raw.strip() == "default":
        return ["all_updates", "stride2", "stride3", "queue0", "queue1", "queue2"]
    names = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [name for name in names if name not in available]
    if unknown:
        raise ValueError(f"Unknown policies {unknown}; available={sorted(available)}")
    return names


def phase_by_name(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {phase["phase"]: phase for phase in summary.get("phase_summaries", [])}


def worker_for_stage(summary: dict[str, Any], stage_id: int) -> dict[str, Any]:
    workers = [
        worker
        for worker in summary.get("gpu_metrics_by_worker", [])
        if int(worker.get("stage_id", -1)) == stage_id
    ]
    if not workers:
        return {}
    if len(workers) == 1:
        return workers[0]
    merged: dict[str, Any] = {
        "worker_id": ",".join(str(worker.get("worker_id")) for worker in workers),
        "stage_id": stage_id,
        "device": ",".join(str(worker.get("device")) for worker in workers),
    }
    for key in ("tasks", "train_tasks", "updates_applied", "failures"):
        merged[key] = sum(int(worker.get(key, 0)) for worker in workers)
    for key in (
        "avg_execute_ms",
        "avg_scheduler_queue_ms",
        "avg_worker_queue_ms",
        "avg_optimizer_ms",
        "avg_stage_total_ms",
    ):
        total_tasks = sum(int(worker.get("tasks", 0)) for worker in workers)
        merged[key] = (
            sum(float(worker.get(key, 0.0)) * int(worker.get("tasks", 0)) for worker in workers) / total_tasks
            if total_tasks
            else 0.0
        )
    for key in ("max_cuda_peak_memory_allocated_mib", "max_cuda_peak_memory_reserved_mib"):
        merged[key] = max(float(worker.get(key, 0.0)) for worker in workers)
    return merged


def decision_counts(metrics_csv: Path, *, phase: str, stage_id: int) -> dict[str, int]:
    counts: Counter[str] = Counter()
    if not metrics_csv.is_file():
        return {}
    with metrics_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("phase") == phase and int(row.get("stage_id", -1)) == stage_id:
                counts[row.get("update_decision", "")] += 1
    return dict(sorted(counts.items()))


def flatten_run_result(
    *,
    policy: str,
    seed: int,
    output_dir: Path,
    summary: dict[str, Any],
    target_stage: int,
    elapsed_s: float,
) -> dict[str, Any]:
    phases = phase_by_name(summary)
    train = phases.get("train", summary)
    eval_phase = phases.get("eval", train)
    stage_worker = worker_for_stage(summary, target_stage)
    update_consistency = summary.get("update_consistency", {})
    metrics_csv = Path(summary.get("metrics_csv", output_dir / "scheduler_stage_metrics.csv"))
    decisions = decision_counts(metrics_csv, phase="train", stage_id=target_stage)
    return {
        "policy": policy,
        "seed": seed,
        "output_dir": str(output_dir),
        "elapsed_s": elapsed_s,
        "train_completed": train.get("completed", 0),
        "train_failed": train.get("failed", 0),
        "train_throughput_per_s": train.get("throughput_per_s", 0.0),
        "train_choice_accuracy": train.get("choice_accuracy", 0.0),
        "train_avg_loss": train.get("avg_loss", 0.0),
        "eval_completed": eval_phase.get("completed", 0),
        "eval_failed": eval_phase.get("failed", 0),
        "eval_throughput_per_s": eval_phase.get("throughput_per_s", 0.0),
        "eval_choice_accuracy": eval_phase.get("choice_accuracy", 0.0),
        "eval_avg_loss": eval_phase.get("avg_loss", 0.0),
        "target_stage_tasks": stage_worker.get("tasks", 0),
        "target_stage_train_tasks": stage_worker.get("train_tasks", 0),
        "target_stage_updates": stage_worker.get("updates_applied", 0),
        "target_stage_avg_execute_ms": stage_worker.get("avg_execute_ms", 0.0),
        "target_stage_avg_queue_ms": stage_worker.get("avg_scheduler_queue_ms", 0.0),
        "target_stage_peak_alloc_mib": stage_worker.get("max_cuda_peak_memory_allocated_mib", 0.0),
        "target_stage_peak_reserved_mib": stage_worker.get("max_cuda_peak_memory_reserved_mib", 0.0),
        "duplicate_update_events": update_consistency.get("duplicate_update_events", 0),
        "timeout_events": summary.get("timeout_events", 0),
        "late_results": summary.get("late_results", 0),
        "dropped_late_results": summary.get("dropped_late_results", 0),
        "stage_update_policy": summary.get("stage_update_policy", ""),
        "stage_train_strides_json": json.dumps(summary.get("stage_train_strides", {}), sort_keys=True),
        "stage_update_queue_thresholds_json": json.dumps(
            summary.get("stage_update_queue_thresholds", {}),
            sort_keys=True,
        ),
        "target_stage_decisions_json": json.dumps(decisions, sort_keys=True),
    }


def numeric_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def numeric_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = numeric_mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    numeric_fields = [
        "train_throughput_per_s",
        "train_choice_accuracy",
        "train_avg_loss",
        "eval_throughput_per_s",
        "eval_choice_accuracy",
        "eval_avg_loss",
        "target_stage_train_tasks",
        "target_stage_updates",
        "target_stage_avg_execute_ms",
        "target_stage_avg_queue_ms",
        "target_stage_peak_alloc_mib",
        "target_stage_peak_reserved_mib",
        "duplicate_update_events",
        "timeout_events",
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["policy"])].append(row)

    aggregates: list[dict[str, Any]] = []
    for policy, policy_rows in sorted(grouped.items()):
        aggregate: dict[str, Any] = {"policy": policy, "runs": len(policy_rows)}
        for field in numeric_fields:
            values = [float(row[field]) for row in policy_rows]
            aggregate[f"{field}_mean"] = numeric_mean(values)
            aggregate[f"{field}_std"] = numeric_std(values)
        aggregate["seeds"] = ",".join(str(row["seed"]) for row in policy_rows)
        aggregate["stage_update_policy"] = policy_rows[0].get("stage_update_policy", "")
        aggregate["stage_train_strides_json"] = policy_rows[0].get("stage_train_strides_json", "{}")
        aggregate["stage_update_queue_thresholds_json"] = policy_rows[0].get(
            "stage_update_queue_thresholds_json",
            "{}",
        )
        aggregates.append(aggregate)
    return aggregates


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def scheduler_command(args: argparse.Namespace, policy: PolicyConfig, seed: int, output_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "sg_exe_trainer.runtime.bpfree.orchestrated_runtime",
        "--model_name",
        args.model_name,
        "--manifest",
        str(args.manifest),
        "--output_dir",
        str(output_dir),
        "--num_chunks",
        str(args.num_chunks),
        "--topology",
        args.topology,
        "--limit",
        str(args.train_limit),
        "--eval_limit",
        str(args.eval_limit),
        "--max_inflight",
        str(args.max_inflight),
        "--scheduler_policy",
        args.scheduler_policy,
        "--recovery_policy",
        args.recovery_policy,
        "--belief_transport_mode",
        args.belief_transport_mode,
        "--dtype",
        args.dtype,
        "--seed",
        str(seed),
        "--request_prefix",
        f"bench-{policy.name}-seed{seed}",
        "--progress_interval",
        str(args.progress_interval),
    ]
    if args.eval_manifest is not None:
        cmd.extend(["--eval_manifest", str(args.eval_manifest)])
    if args.stage_devices:
        cmd.extend(["--stage_devices", args.stage_devices])
    if args.workers:
        cmd.extend(["--workers", args.workers])
    if args.learning_rate is not None:
        cmd.extend(["--learning_rate", str(args.learning_rate)])
    if args.train_chunks:
        cmd.extend(["--train_chunks", args.train_chunks])
    cmd.extend(policy.args)
    return cmd


def run_one(args: argparse.Namespace, policy: PolicyConfig, seed: int) -> dict[str, Any]:
    output_dir = args.output_root / f"{policy.name}_seed{seed}"
    summary_path = output_dir / "scheduler_summary.json"
    log_path = output_dir / "run.log"
    output_dir.mkdir(parents=True, exist_ok=True)

    if summary_path.is_file() and not args.force:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return flatten_run_result(
            policy=policy.name,
            seed=seed,
            output_dir=output_dir,
            summary=summary,
            target_stage=args.target_stage,
            elapsed_s=0.0,
        )

    cmd = scheduler_command(args, policy, seed, output_dir)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write("$ " + " ".join(cmd) + "\n")
        log_handle.flush()
        process = subprocess.run(
            cmd,
            cwd=Path(__file__).resolve().parents[3],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    elapsed_s = time.perf_counter() - started
    if process.returncode != 0:
        raise RuntimeError(f"Policy {policy.name} seed {seed} failed; see {log_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return flatten_run_result(
        policy=policy.name,
        seed=seed,
        output_dir=output_dir,
        summary=summary,
        target_stage=args.target_stage,
        elapsed_s=elapsed_s,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scheduler lab policy benchmark matrix.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--eval_manifest", type=Path, default=None)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--model_name", default="tinyllama")
    parser.add_argument("--num_chunks", type=int, default=3)
    parser.add_argument("--stage_devices", default="cuda:0,cuda:1,cuda:2")
    parser.add_argument("--workers", default="")
    parser.add_argument("--topology", choices=["phone_fixed", "worker_pool"], default="phone_fixed")
    parser.add_argument("--target_stage", type=int, default=2)
    parser.add_argument("--policies", default="default")
    parser.add_argument("--seeds", default="20260531")
    parser.add_argument("--train_limit", type=int, default=512)
    parser.add_argument("--eval_limit", type=int, default=256)
    parser.add_argument("--max_inflight", type=int, default=8)
    parser.add_argument("--scheduler_policy", choices=["fifo", "recovery_first"], default="fifo")
    parser.add_argument("--recovery_policy", default="replay_after_update")
    parser.add_argument("--belief_transport_mode", choices=["full", "terminal", "none"], default="terminal")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--train_chunks", default="all")
    parser.add_argument("--progress_interval", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policies = default_policy_configs(args.target_stage)
    selected_policy_names = parse_policy_names(args.policies, policies)
    seeds = parse_csv_ints(args.seeds)
    args.output_root.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, Any]] = []
    for policy_name in selected_policy_names:
        policy = policies[policy_name]
        for seed in seeds:
            print(f"[benchmark] running policy={policy.name} seed={seed}", flush=True)
            row = run_one(args, policy, seed)
            run_rows.append(row)
            print(
                f"[benchmark] done policy={policy.name} seed={seed} "
                f"train_tput={float(row['train_throughput_per_s']):.3f} "
                f"eval_acc={float(row['eval_choice_accuracy']):.4f} "
                f"eval_loss={float(row['eval_avg_loss']):.4f} "
                f"target_updates={row['target_stage_updates']}",
                flush=True,
            )

    aggregate = aggregate_rows(run_rows)
    write_csv(args.output_root / "scheduler_benchmark_runs.csv", run_rows)
    write_csv(args.output_root / "scheduler_benchmark_summary.csv", aggregate)
    (args.output_root / "scheduler_benchmark_summary.json").write_text(
        json.dumps({"runs": run_rows, "aggregate": aggregate}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == "__main__":
    main()
