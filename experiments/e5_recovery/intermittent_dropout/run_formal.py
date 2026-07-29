#!/usr/bin/env python3
"""Run paired E5 accuracy experiments with independent per-device dropout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
BPFREE_MODULE = "sg_exe_trainer.runtime.bpfree.orchestrated_runtime"
BPFREE_SOURCE = REPO_ROOT / "src/sg_exe_trainer/runtime/bpfree/orchestrated_runtime.py"
EXACT_RUNNER = REPO_ROOT / "src/sg_exe_trainer/runtime/exactbp/distributed_runtime.py"
TRAIN_MANIFEST = REPO_ROOT / "data/sft_requests/tinyllama_agnews128_label_train10000_formal_v1/requests.jsonl"
TEST_MANIFEST = REPO_ROOT / "data/sft_requests/tinyllama_agnews128_label_test7600_formal_v1/requests.jsonl"

WINDOW_SIZE = 8
NUM_STAGES = 3
DEFAULT_SEEDS = (20260531, 20260532, 20260533)
DEFAULT_PROBABILITIES = (0.05, 0.10)
BASELINE_METHODS = ("bpfree_fault_free", "exact_fault_free")
DROPOUT_METHODS = (
    "bpfree_local_retain",
    "bpfree_replay",
    "bpfree_skip",
    "exact_skip",
    "exact_replay",
)
ALL_METHODS = BASELINE_METHODS + DROPOUT_METHODS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_csv(raw: str) -> list[str]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("empty comma-separated value")
    return values


def probability_tag(probability: float) -> str:
    return f"p{round(probability * 100):02d}"


def generate_mask(*, probability: float, seed: int, num_windows: int) -> dict[str, Any]:
    rng = random.Random(seed)
    offline = {
        str(stage_id): [
            window_id
            for window_id in range(num_windows)
            if rng.random() < probability
        ]
        for stage_id in range(NUM_STAGES)
    }
    any_failure = sorted({window for windows in offline.values() for window in windows})
    simultaneous = sum(
        1
        for window_id in range(num_windows)
        if sum(window_id in set(offline[str(stage)]) for stage in range(NUM_STAGES)) > 1
    )
    return {
        "schema_version": 1,
        "unit": "logical_update_window_stage_service",
        "sampling": "independent_bernoulli_per_stage_window",
        "dropout_probability": probability,
        "mask_seed": seed,
        "num_stages": NUM_STAGES,
        "num_windows": num_windows,
        "window_size": WINDOW_SIZE,
        "offline_windows_by_stage": offline,
        "realized_stage_event_counts": {
            stage: len(windows) for stage, windows in offline.items()
        },
        "realized_any_failure_windows": len(any_failure),
        "realized_simultaneous_failure_windows": simultaneous,
    }


def global_skip_mask(mask: dict[str, Any]) -> dict[str, Any]:
    any_failure = sorted(
        {
            int(window)
            for windows in mask["offline_windows_by_stage"].values()
            for window in windows
        }
    )
    payload = dict(mask)
    payload.update(
        {
            "sampling": "derived_global_skip_at_stage0",
            "source_sampling": mask["sampling"],
            "offline_windows_by_stage": {
                "0": any_failure,
                "1": [],
                "2": [],
            },
            "realized_stage_event_counts": {
                "0": len(any_failure),
                "1": 0,
                "2": 0,
            },
        }
    )
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def mask_sets(mask: dict[str, Any]) -> dict[int, set[int]]:
    return {
        stage: set(int(value) for value in mask["offline_windows_by_stage"][str(stage)])
        for stage in range(NUM_STAGES)
    }


def expected_steps(method: str, mask: dict[str, Any] | None, num_windows: int) -> list[int]:
    if method in {"bpfree_fault_free", "bpfree_replay", "exact_fault_free", "exact_replay"}:
        return [num_windows] * NUM_STAGES
    if mask is None:
        raise ValueError(f"{method} requires a dropout mask")
    offline = mask_sets(mask)
    healthy_windows = sum(
        all(window not in offline[stage] for stage in range(NUM_STAGES))
        for window in range(num_windows)
    )
    if method in {"bpfree_skip", "exact_skip"}:
        return [healthy_windows] * NUM_STAGES
    if method == "bpfree_local_retain":
        return [
            sum(
                all(window not in offline[prefix] for prefix in range(stage + 1))
                for window in range(num_windows)
            )
            for stage in range(NUM_STAGES)
        ]
    raise ValueError(method)


def common_training_args(seed: int) -> list[str]:
    return [
        "--model_name", "tinyllama",
        "--num_chunks", "3",
        "--stage_devices", "cuda:0,cuda:1,cuda:2",
        "--dtype", "bfloat16",
        "--optimizer", "adamw",
        "--learning_rate", "1e-4",
        "--lora_rank", "4",
        "--lora_alpha", "16",
        "--lora_targets", "q_proj,v_proj",
        "--lora_init_std", "0.01",
        "--lora_init_seed", str(seed),
        "--seed", str(seed),
        "--grad_clip", "1.0",
    ]


def bpfree_command(
    *,
    method: str,
    seed: int,
    train_limit: int,
    eval_limit: int,
    output_dir: Path,
    mask_path: Path | None,
    global_mask_path: Path | None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        BPFREE_MODULE,
        "--manifest", str(TRAIN_MANIFEST),
        "--eval_manifest", str(TEST_MANIFEST),
        "--output_dir", str(output_dir),
        "--limit", str(train_limit),
        "--eval_limit", str(eval_limit),
        "--max_inflight", str(WINDOW_SIZE),
        "--scheduler_policy", "recovery_first",
        "--gradient_accumulation_steps", str(WINDOW_SIZE),
        "--belief_transport_mode", "none",
        "--trainable_mode", "lora",
        "--progress_interval", "512",
        *common_training_args(seed),
    ]
    if method == "bpfree_replay":
        command += [
            "--transient_dropout_mask", str(mask_path),
            "--recovery_policy", "retry_stage",
            "--max_attempts", "2",
        ]
    elif method == "bpfree_local_retain":
        command += [
            "--transient_dropout_mask", str(mask_path),
            "--recovery_policy", "skip",
            "--max_attempts", "1",
        ]
    elif method == "bpfree_skip":
        command += [
            "--transient_dropout_mask", str(global_mask_path),
            "--recovery_policy", "skip",
            "--max_attempts", "1",
        ]
    elif method == "bpfree_fault_free":
        command += ["--recovery_policy", "skip", "--max_attempts", "1"]
    else:
        raise ValueError(method)
    return command


def exact_command(
    *,
    method: str,
    seed: int,
    train_limit: int,
    eval_limit: int,
    output_dir: Path,
    mask_path: Path | None,
) -> list[str]:
    command = [
        sys.executable,
        "-m", "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=3",
        str(EXACT_RUNNER),
        "--train_manifest", str(TRAIN_MANIFEST),
        "--eval_manifest", str(TEST_MANIFEST),
        "--output_dir", str(output_dir),
        "--train_limit", str(train_limit),
        "--eval_limit", str(eval_limit),
        "--train_epochs", "1",
        "--microbatches", str(WINDOW_SIZE),
        "--batch_size", str(WINDOW_SIZE),
        "--pipeline_schedule", "1f1b",
        "--trainable_mode", "lora",
        "--progress_interval", "512",
        "--recovery_policy", "strict_skip",
        "--skip_eval_before",
        *common_training_args(seed),
    ]
    if method == "exact_skip":
        command += [
            "--transient_dropout_mask", str(mask_path),
            "--transient_dropout_policy", "skip",
        ]
    elif method == "exact_replay":
        command += [
            "--transient_dropout_mask", str(mask_path),
            "--transient_dropout_policy", "replay",
        ]
    elif method != "exact_fault_free":
        raise ValueError(method)
    return command


def run_command(command: list[str], log_path: Path, *, dry_run: bool) -> float:
    rendered = shlex.join(command)
    print(f"$ {rendered}", flush=True)
    if dry_run:
        return 0.0
    env = dict(os.environ)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0,1,2",
            "OMP_NUM_THREADS": "4",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
        }
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"$ {rendered}\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"command failed with exit code {return_code}: {rendered}")
    return time.perf_counter() - started


def phase(summary: dict[str, Any], names: set[str]) -> dict[str, Any]:
    for item in summary.get("phase_summaries") or summary.get("phases") or []:
        if str(item.get("phase")) in names:
            return item
    raise ValueError(f"missing phase {sorted(names)}")


def summarize_run(
    *,
    method: str,
    probability: float,
    seed: int,
    train_limit: int,
    eval_limit: int,
    output_dir: Path,
    mask: dict[str, Any] | None,
    elapsed_s: float,
) -> dict[str, Any]:
    summary_name = "scheduler_summary.json" if method.startswith("bpfree") else "summary.json"
    summary = json.loads((output_dir / summary_name).read_text(encoding="utf-8"))
    train = phase(summary, {"train"})
    evaluation = phase(summary, {"eval", "eval_after"})
    if method.startswith("bpfree"):
        per_stage = train["update_consistency"]["per_stage"]
        steps = [int(per_stage.get(str(stage), {}).get("update_events", 0)) for stage in range(NUM_STAGES)]
        completed = int(train["completed"])
        failed = int(train["failed"])
    else:
        steps = [int(train["optimizer_steps"])] * NUM_STAGES
        completed = int(train["completed_records"])
        failed = int(train["skipped_records"])
    expected = expected_steps(method, mask, train_limit // WINDOW_SIZE)
    if steps != expected:
        raise AssertionError(f"{method}/p={probability}/seed={seed}: steps={steps}, expected={expected}")
    return {
        "method": method,
        "dropout_probability": probability,
        "seed": seed,
        "train_records": train_limit,
        "eval_records": int(evaluation.get("records", evaluation.get("rows", eval_limit))),
        "eval_accuracy": float(evaluation["choice_accuracy"]),
        "eval_nll": float(evaluation["avg_loss"]),
        "train_completed": completed,
        "train_failed_or_skipped": failed,
        "stage0_optimizer_steps": steps[0],
        "stage1_optimizer_steps": steps[1],
        "stage2_optimizer_steps": steps[2],
        "realized_stage0_events": len(mask_sets(mask)[0]) if mask else 0,
        "realized_stage1_events": len(mask_sets(mask)[1]) if mask else 0,
        "realized_stage2_events": len(mask_sets(mask)[2]) if mask else 0,
        "realized_any_failure_windows": int(mask.get("realized_any_failure_windows", 0)) if mask else 0,
        "elapsed_s": elapsed_s,
        "output_dir": str(output_dir),
    }


def collect_results(output_root: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in output_root.glob("runs/*/*/seed*/normalized_result.json")
    ]
    return sorted(rows, key=lambda row: (float(row["dropout_probability"]), row["method"], int(row["seed"])))


def write_progress(output_root: Path, planned: int) -> None:
    rows = collect_results(output_root)
    lines = [
        "# E5 independent intermittent dropout accuracy",
        "",
        f"- Completed: {len(rows)}/{planned}",
        "- Unit: independent Bernoulli availability for every (logical window, stage)",
        "- Config: AG News, B=8, bfloat16, LoRA r4/alpha16, AdamW lr=1e-4",
        "",
        "| p | method | seed | accuracy | NLL | steps S0/S1/S2 | failures |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        steps = "/".join(str(row[f"stage{stage}_optimizer_steps"]) for stage in range(NUM_STAGES))
        lines.append(
            f"| {100 * float(row['dropout_probability']):.0f}% | {row['method']} | {row['seed']} | "
            f"{row['eval_accuracy']:.4f} | {row['eval_nll']:.4f} | {steps} | "
            f"{row['realized_any_failure_windows']} |"
        )
    (output_root / "PROGRESS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if rows:
        with (output_root / "results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--train_limit", type=int, default=10000)
    parser.add_argument("--eval_limit", type=int, default=7600)
    parser.add_argument("--probabilities", default=",".join(str(value) for value in DEFAULT_PROBABILITIES))
    parser.add_argument("--seeds", default=",".join(str(value) for value in DEFAULT_SEEDS))
    parser.add_argument("--methods", default=",".join(ALL_METHODS))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_limit <= 0 or args.train_limit % WINDOW_SIZE:
        raise ValueError("--train_limit must be positive and divisible by 8")
    probabilities = [float(value) for value in parse_csv(args.probabilities)]
    if any(not 0.0 < value < 1.0 for value in probabilities):
        raise ValueError("dropout probabilities must be in (0, 1)")
    seeds = [int(value) for value in parse_csv(args.seeds)]
    methods = parse_csv(args.methods)
    unknown = sorted(set(methods) - set(ALL_METHODS))
    if unknown:
        raise ValueError(f"unknown methods: {unknown}")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    num_windows = args.train_limit // WINDOW_SIZE
    masks: dict[tuple[float, int], tuple[dict[str, Any], Path, Path]] = {}
    for probability in probabilities:
        for seed in seeds:
            mask_seed = seed + round(probability * 10000) * 1009
            mask = generate_mask(probability=probability, seed=mask_seed, num_windows=num_windows)
            tag = probability_tag(probability)
            mask_path = output_root / "masks" / f"{tag}_seed{seed}.json"
            global_path = output_root / "masks" / f"{tag}_seed{seed}_global_skip.json"
            write_json(mask_path, mask)
            write_json(global_path, global_skip_mask(mask))
            masks[(probability, seed)] = (mask, mask_path, global_path)

    protocol = {
        "schema_version": 1,
        "train_manifest": str(TRAIN_MANIFEST),
        "train_manifest_sha256": sha256(TRAIN_MANIFEST),
        "test_manifest": str(TEST_MANIFEST),
        "test_manifest_sha256": sha256(TEST_MANIFEST),
        "train_limit": args.train_limit,
        "eval_limit": args.eval_limit,
        "window_size": WINDOW_SIZE,
        "num_windows": num_windows,
        "num_stages": NUM_STAGES,
        "probabilities": probabilities,
        "seeds": seeds,
        "methods": methods,
        "bpfree_runner_sha256": sha256(BPFREE_SOURCE),
        "exact_runner_sha256": sha256(EXACT_RUNNER),
        "python": sys.executable,
    }
    write_json(output_root / "protocol.json", protocol)

    planned = sum(
        len(seeds) if method in BASELINE_METHODS else len(seeds) * len(probabilities)
        for method in methods
    )
    write_progress(output_root, planned)
    scopes: list[tuple[float, str]] = [
        (0.0, method) for method in methods if method in BASELINE_METHODS
    ] + [
        (probability, method)
        for probability in probabilities
        for method in methods
        if method in DROPOUT_METHODS
    ]
    for probability, method in scopes:
        for seed in seeds:
            mask: dict[str, Any] | None = None
            mask_path: Path | None = None
            global_path: Path | None = None
            if probability > 0:
                mask, mask_path, global_path = masks[(probability, seed)]
            scope = probability_tag(probability)
            output_dir = output_root / "runs" / scope / method / f"seed{seed}"
            result_path = output_dir / "normalized_result.json"
            if result_path.is_file() and not args.force:
                print(f"[skip] {result_path}", flush=True)
                continue
            output_dir.mkdir(parents=True, exist_ok=True)
            if method.startswith("bpfree"):
                command = bpfree_command(
                    method=method,
                    seed=seed,
                    train_limit=args.train_limit,
                    eval_limit=args.eval_limit,
                    output_dir=output_dir,
                    mask_path=mask_path,
                    global_mask_path=global_path,
                )
            else:
                command = exact_command(
                    method=method,
                    seed=seed,
                    train_limit=args.train_limit,
                    eval_limit=args.eval_limit,
                    output_dir=output_dir,
                    mask_path=mask_path,
                )
            elapsed_s = run_command(command, output_dir / "run.log", dry_run=args.dry_run)
            if args.dry_run:
                continue
            result = summarize_run(
                method=method,
                probability=probability,
                seed=seed,
                train_limit=args.train_limit,
                eval_limit=args.eval_limit,
                output_dir=output_dir,
                mask=mask,
                elapsed_s=elapsed_s,
            )
            write_json(result_path, result)
            write_progress(output_root, planned)
    write_progress(output_root, planned)


if __name__ == "__main__":
    main()
