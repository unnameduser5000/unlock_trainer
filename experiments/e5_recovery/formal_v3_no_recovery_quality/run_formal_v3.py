#!/usr/bin/env python3
"""Run the matched E5 no-recovery quality experiment.

The suite compares retained prefix updates against a BP-free balanced-skip
control and exact-BP strict skip at two evaluation horizons.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
BPFREE_MODULE = "sg_exe_trainer.runtime.bpfree.orchestrated_runtime"
BPFREE_SOURCE = REPO_ROOT / "src/sg_exe_trainer/runtime/bpfree/orchestrated_runtime.py"
EXACT_RUNNER = REPO_ROOT / "src" / "sg_exe_trainer" / "runtime" / "exactbp" / "distributed_runtime.py"

SOURCE_TRAIN = (
    REPO_ROOT
    / "data"
    / "sft_requests"
    / "tinyllama_agnews128_label_train10000_formal_v1"
    / "requests.jsonl"
)
TEST_MANIFEST = (
    REPO_ROOT
    / "data"
    / "sft_requests"
    / "tinyllama_agnews128_label_test7600_formal_v1"
    / "requests.jsonl"
)

OUTAGE_STAGE = 1
OUTAGE_START = 768
OUTAGE_END = 1280
EVAL_LIMIT = 7600
SEEDS = (20260531, 20260532, 20260533)


@dataclass(frozen=True)
class Horizon:
    name: str
    source_limit: int


HORIZONS = {
    "post_outage": Horizon(name="post_outage", source_limit=1280),
    "final": Horizon(name="final", source_limit=2048),
}
METHODS = (
    "bpfree_fault_free",
    "bpfree_local_retain",
    "bpfree_balanced_skip",
    "exact_fault_free",
    "exact_strict_skip",
)


def parse_csv(raw: str, *, allowed: set[str] | None = None) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("empty comma-separated value")
    if allowed is not None:
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"unsupported values {unknown}; allowed={sorted(allowed)}")
    return values


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if len(rows) >= limit:
                break
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) != limit:
        raise ValueError(f"expected {limit} records in {path}, found {len(rows)}")
    return rows


def absolutize_tensor_paths(record: dict[str, Any], source_dir: Path) -> dict[str, Any]:
    copied = json.loads(json.dumps(record))
    for spec in copied.get("tensors", {}).values():
        tensor_path = Path(spec["path"])
        if not tensor_path.is_absolute():
            spec["path"] = str((source_dir / tensor_path).resolve())
    return copied


def write_balanced_manifest(output_root: Path, horizon: Horizon) -> tuple[Path, int]:
    records = load_jsonl(SOURCE_TRAIN, horizon.source_limit)
    kept = [record for seq, record in enumerate(records) if not (OUTAGE_START <= seq < OUTAGE_END)]
    expected = horizon.source_limit - (OUTAGE_END - OUTAGE_START)
    if len(kept) != expected or len(kept) % 8 != 0:
        raise ValueError(f"invalid balanced manifest size {len(kept)} for {horizon.name}")
    manifest_dir = output_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_dir / f"agnews_{horizon.name}_balanced_skip.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for record in kept:
            normalized = absolutize_tensor_paths(record, SOURCE_TRAIN.parent)
            handle.write(json.dumps(normalized, ensure_ascii=False) + "\n")
    return path, len(kept)


def expected_stage_steps(method: str, horizon: Horizon) -> list[int]:
    full_steps = horizon.source_limit // 8
    skipped_steps = (OUTAGE_END - OUTAGE_START) // 8
    balanced_steps = full_steps - skipped_steps
    if method in {"bpfree_fault_free", "exact_fault_free"}:
        return [full_steps, full_steps, full_steps]
    if method == "bpfree_local_retain":
        return [full_steps, balanced_steps, balanced_steps]
    return [balanced_steps, balanced_steps, balanced_steps]


def bpfree_command(
    *, method: str, seed: int, horizon: Horizon, output_dir: Path, balanced_manifest: Path, balanced_limit: int
) -> list[str]:
    is_balanced = method == "bpfree_balanced_skip"
    manifest = balanced_manifest if is_balanced else SOURCE_TRAIN
    train_limit = balanced_limit if is_balanced else horizon.source_limit
    command = [
        sys.executable,
        "-m",
        BPFREE_MODULE,
        "--model_name", "tinyllama",
        "--manifest", str(manifest),
        "--eval_manifest", str(TEST_MANIFEST),
        "--output_dir", str(output_dir),
        "--num_chunks", "3",
        "--stage_devices", "cuda:0,cuda:1,cuda:2",
        "--limit", str(train_limit),
        "--eval_limit", str(EVAL_LIMIT),
        "--max_inflight", "8",
        "--scheduler_policy", "fifo",
        "--recovery_policy", "skip",
        "--max_attempts", "1",
        "--gradient_accumulation_steps", "8",
        "--belief_transport_mode", "none",
        "--trainable_mode", "lora",
        "--dtype", "bfloat16",
        "--optimizer", "adamw",
        "--learning_rate", "1e-4",
        "--lora_rank", "4",
        "--lora_alpha", "16",
        "--lora_targets", "q_proj,v_proj",
        "--lora_init_std", "0.01",
        "--lora_init_seed", str(seed),
        "--seed", str(seed),
        "--progress_interval", "128",
        "--grad_clip", "1.0",
    ]
    if method == "bpfree_local_retain":
        command.extend(
            [
                "--offline_stage", str(OUTAGE_STAGE),
                "--offline_start_seq", str(OUTAGE_START),
                "--offline_end_seq", str(OUTAGE_END),
            ]
        )
    return command


def exact_command(*, method: str, seed: int, horizon: Horizon, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "-m", "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=3",
        str(EXACT_RUNNER),
        "--model_name", "tinyllama",
        "--train_manifest", str(SOURCE_TRAIN),
        "--eval_manifest", str(TEST_MANIFEST),
        "--output_dir", str(output_dir),
        "--num_chunks", "3",
        "--stage_devices", "cuda:0,cuda:1,cuda:2",
        "--train_limit", str(horizon.source_limit),
        "--eval_limit", str(EVAL_LIMIT),
        "--train_epochs", "1",
        "--microbatches", "8",
        "--batch_size", "8",
        "--dtype", "bfloat16",
        "--optimizer", "adamw",
        "--learning_rate", "1e-4",
        "--label_smoothing", "0.0",
        "--lora_rank", "4",
        "--lora_alpha", "16",
        "--lora_targets", "q_proj,v_proj",
        "--lora_init_std", "0.01",
        "--lora_init_seed", str(seed),
        "--seed", str(seed),
        "--progress_interval", "128",
        "--recovery_policy", "strict_skip",
        "--grad_clip", "1.0",
        "--skip_eval_before",
    ]
    if method == "exact_strict_skip":
        command.extend(
            [
                "--offline_stage", str(OUTAGE_STAGE),
                "--offline_start_seq", str(OUTAGE_START),
                "--offline_end_seq", str(OUTAGE_END),
            ]
        )
    return command


def run_command(command: list[str], log_path: Path, *, dry_run: bool) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = shlex.join(command)
    print(f"$ {rendered}", flush=True)
    if dry_run:
        return 0.0
    started = time.perf_counter()
    env = dict(os.environ)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0,1,2",
            "OMP_NUM_THREADS": "4",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
        }
    )
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


def load_summary(method: str, output_dir: Path) -> dict[str, Any]:
    path = output_dir / ("scheduler_summary.json" if method.startswith("bpfree") else "summary.json")
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def phase(summary: dict[str, Any], names: set[str]) -> dict[str, Any]:
    phases = summary.get("phase_summaries") or summary.get("phases") or []
    for item in phases:
        if str(item.get("phase")) in names:
            return item
    raise ValueError(f"missing phase {sorted(names)}")


def summarize_run(method: str, horizon: Horizon, seed: int, output_dir: Path, elapsed_s: float) -> dict[str, Any]:
    summary = load_summary(method, output_dir)
    train = phase(summary, {"train"})
    evaluation = phase(summary, {"eval", "eval_after"})
    if method.startswith("bpfree"):
        per_stage = train["update_consistency"]["per_stage"]
        steps = [int(per_stage.get(str(stage), {}).get("update_events", 0)) for stage in range(3)]
        retained = train["retained_progress"]["per_stage"]
        retained_steps = [
            int(retained.get(str(stage), {}).get("retained_updates_on_failed_requests", 0))
            for stage in range(3)
        ]
        train_completed = int(train["completed"])
        train_failed = int(train["failed"])
    else:
        optimizer_steps = int(train["optimizer_steps"])
        steps = [optimizer_steps, optimizer_steps, optimizer_steps]
        retained_steps = [0, 0, 0]
        train_completed = int(train["completed_records"])
        train_failed = int(train["skipped_records"])
    expected = expected_stage_steps(method, horizon)
    if steps != expected:
        raise AssertionError(f"{method}/{horizon.name}/seed{seed}: stage steps {steps}, expected {expected}")
    return {
        "method": method,
        "horizon": horizon.name,
        "seed": seed,
        "source_train_limit": horizon.source_limit,
        "eval_records": int(evaluation.get("records", evaluation.get("rows", 0))),
        "eval_accuracy": float(evaluation["choice_accuracy"]),
        "eval_nll": float(evaluation["avg_loss"]),
        "train_completed": train_completed,
        "train_failed_or_skipped": train_failed,
        "stage0_optimizer_steps": steps[0],
        "stage1_optimizer_steps": steps[1],
        "stage2_optimizer_steps": steps[2],
        "retained_stage0_steps": retained_steps[0],
        "retained_stage1_steps": retained_steps[1],
        "retained_stage2_steps": retained_steps[2],
        "elapsed_s": elapsed_s,
        "output_dir": str(output_dir),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect_existing(output_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in output_root.glob("runs/*/*/seed*/normalized_result.json"):
        rows.append(json.loads(summary_path.read_text(encoding="utf-8")))
    return sorted(rows, key=lambda row: (row["horizon"], row["method"], int(row["seed"])))


def write_progress(output_root: Path, planned: int) -> None:
    rows = collect_existing(output_root)
    lines = [
        "# E5 no-recovery quality formal v3",
        "",
        f"- Completed: {len(rows)}/{planned}",
        f"- Outage: Stage-{OUTAGE_STAGE} samples [{OUTAGE_START},{OUTAGE_END})",
        "- Config: bfloat16, LoRA r4/alpha16, lr=1e-4, B=8, BP-free CE/none",
        "",
        "| horizon | method | seed | accuracy | NLL | steps S0/S1/S2 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        steps = "/".join(str(row[f"stage{stage}_optimizer_steps"]) for stage in range(3))
        lines.append(
            f"| {row['horizon']} | {row['method']} | {row['seed']} | "
            f"{row['eval_accuracy']:.4f} | {row['eval_nll']:.4f} | {steps} |"
        )
    (output_root / "PROGRESS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_csv(output_root / "results.csv", rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument("--horizons", default=",".join(HORIZONS))
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    seeds = [int(value) for value in parse_csv(args.seeds)]
    horizons = parse_csv(args.horizons, allowed=set(HORIZONS))
    methods = parse_csv(args.methods, allowed=set(METHODS))
    output_root.mkdir(parents=True, exist_ok=True)
    balanced: dict[str, tuple[Path, int]] = {
        name: write_balanced_manifest(output_root, HORIZONS[name]) for name in horizons
    }
    protocol = {
        "schema_version": 1,
        "source_train": str(SOURCE_TRAIN),
        "source_train_sha256": sha256(SOURCE_TRAIN),
        "test_manifest": str(TEST_MANIFEST),
        "test_manifest_sha256": sha256(TEST_MANIFEST),
        "outage_stage": OUTAGE_STAGE,
        "outage_start": OUTAGE_START,
        "outage_end": OUTAGE_END,
        "seeds": seeds,
        "horizons": horizons,
        "methods": methods,
        "bpfree_runner_sha256": sha256(BPFREE_SOURCE),
        "exact_runner_sha256": sha256(EXACT_RUNNER),
        "python": sys.executable,
    }
    (output_root / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    planned = len(seeds) * len(horizons) * len(methods)
    write_progress(output_root, planned)
    for horizon_name in horizons:
        horizon = HORIZONS[horizon_name]
        balanced_manifest, balanced_limit = balanced[horizon_name]
        for seed in seeds:
            for method in methods:
                output_dir = output_root / "runs" / horizon.name / method / f"seed{seed}"
                result_path = output_dir / "normalized_result.json"
                if result_path.is_file() and not args.force:
                    print(f"[skip] {result_path}", flush=True)
                    continue
                output_dir.mkdir(parents=True, exist_ok=True)
                if method.startswith("bpfree"):
                    command = bpfree_command(
                        method=method,
                        seed=seed,
                        horizon=horizon,
                        output_dir=output_dir,
                        balanced_manifest=balanced_manifest,
                        balanced_limit=balanced_limit,
                    )
                else:
                    command = exact_command(method=method, seed=seed, horizon=horizon, output_dir=output_dir)
                elapsed_s = run_command(command, output_dir / "run.log", dry_run=args.dry_run)
                if args.dry_run:
                    continue
                row = summarize_run(method, horizon, seed, output_dir, elapsed_s)
                result_path.write_text(
                    json.dumps(row, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
                )
                write_progress(output_root, planned)
    write_progress(output_root, planned)


if __name__ == "__main__":
    main()
