#!/usr/bin/env python3
"""Run the immutable AG News E1 quality protocol used by the paper."""

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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("configs") / "agnews_quality_v1.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results/e1_quality/raw/agnews_e1_quality_v1"
EXACTBP_SOURCE = REPO_ROOT / "src/sg_exe_trainer/runtime/exactbp/distributed_runtime.py"
BPFREE_SOURCE = REPO_ROOT / "src/sg_exe_trainer/runtime/bpfree/orchestrated_runtime.py"
BPFREE_MODULE = "sg_exe_trainer.runtime.bpfree.orchestrated_runtime"


@dataclass(frozen=True)
class Job:
    method: str
    paper_label: str
    runtime: str
    seed: int
    output_dir: str
    summary_name: str
    command: list[str]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_ints(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("expected a non-empty list of unique integers")
    return values


def parse_names(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("expected a non-empty list of unique method names")
    return values


def load_protocol(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported E1 protocol schema: {payload.get('schema_version')}")
    if payload.get("protocol_id") != "agnews-e1-quality-v1":
        raise ValueError(f"unexpected E1 protocol id: {payload.get('protocol_id')}")
    return payload


def selected_dataset_indices(path: Path, limit: int) -> list[int]:
    result: list[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if "dataset_index" not in record:
                raise ValueError(f"manifest record lacks dataset_index: {path}")
            result.append(int(record["dataset_index"]))
            if len(result) == limit:
                break
    if len(result) != limit:
        raise ValueError(f"{path} has {len(result)} selected records; expected {limit}")
    if len(set(result)) != len(result):
        raise ValueError(f"selected manifest prefix contains duplicate dataset_index values: {path}")
    return result


def validate_inputs(protocol: dict[str, Any], data_root: Path) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    indices: dict[str, list[int]] = {}
    for split, spec in protocol["manifests"].items():
        path = (data_root / spec["path"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"missing {split} manifest: {path}")
        actual_hash = sha256(path)
        if actual_hash != spec["sha256"]:
            raise ValueError(f"{split} manifest hash mismatch: {actual_hash}")
        resolved[split] = path
        indices[split] = selected_dataset_indices(path, int(spec["records"]))
    overlap = set(indices["train"]) & set(indices["validation"])
    if overlap:
        raise ValueError(f"train/validation dataset_index overlap: {len(overlap)}")
    return resolved


def common_model_args(protocol: dict[str, Any], seed: int) -> list[str]:
    model = protocol["model"]
    opt = protocol["optimization"]
    return [
        "--model_name", str(model["name"]),
        "--learning_rate", str(opt["learning_rate"]),
        "--optimizer", str(opt["optimizer"]),
        "--grad_clip", str(opt["grad_clip"]),
        "--dtype", str(model["dtype"]),
        "--trainable_mode", str(model["trainable_mode"]),
        "--lora_rank", str(model["lora_rank"]),
        "--lora_alpha", str(model["lora_alpha"]),
        "--lora_targets", str(model["lora_targets"]),
        "--lora_init_std", str(model["lora_init_std"]),
        "--lora_init_seed", str(seed),
        "--seed", str(seed),
        "--progress_interval", "1000",
    ]


def exactbp_command(
    protocol: dict[str, Any],
    method: dict[str, Any],
    manifests: dict[str, Path],
    output_dir: Path,
    seed: int,
    stage_devices: str,
) -> list[str]:
    stages = int(method["num_stages"])
    devices = stage_devices.split(",")
    if len(devices) < stages:
        raise ValueError(f"method {method['id']} requires {stages} stage devices")
    selected_devices = ",".join(devices[:stages])
    opt = protocol["optimization"]
    data = protocol["manifests"]
    return [
        sys.executable,
        "-m", "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={stages}",
        str(EXACTBP_SOURCE),
        "--train_manifest", str(manifests["train"]),
        "--eval_manifest", str(manifests["test"]),
        "--output_dir", str(output_dir),
        "--num_chunks", str(stages),
        "--stage_devices", selected_devices,
        "--train_limit", str(data["train"]["records"]),
        "--eval_limit", str(data["test"]["records"]),
        "--train_epochs", str(opt["epochs"]),
        "--batch_size", str(opt["effective_batch"]),
        "--microbatches", str(method["microbatches"]),
        "--pipeline_schedule", str(method["pipeline_schedule"]),
        "--label_smoothing", str(opt["label_smoothing"]),
        *common_model_args(protocol, seed),
        "--skip_eval_before",
        "--validation_manifest", str(manifests["validation"]),
        "--validation_limit", str(data["validation"]["records"]),
        "--validation_interval_steps", str(opt["validation_interval_steps"]),
    ]


def bpfree_command(
    protocol: dict[str, Any],
    method: dict[str, Any],
    manifests: dict[str, Path],
    output_dir: Path,
    seed: int,
    stage_devices: str,
) -> list[str]:
    opt = protocol["optimization"]
    data = protocol["manifests"]
    return [
        sys.executable,
        "-m", BPFREE_MODULE,
        "--manifest", str(manifests["train"]),
        "--output_dir", str(output_dir),
        "--num_chunks", str(protocol["model"]["num_stages"]),
        "--stage_devices", stage_devices,
        "--topology", "phone_fixed",
        "--max_inflight", str(opt["effective_batch"]),
        "--scheduler_policy", "fifo",
        "--recovery_policy", "replay_after_update",
        "--failure_mode", "none",
        "--task_timeout_ms", "0",
        "--train_chunks", "all",
        "--stage_update_policy", "stride",
        "--gradient_accumulation_steps", str(opt["effective_batch"]),
        "--belief_transport_mode", str(method["belief_transport_mode"]),
        "--alpha", str(method["alpha"]),
        "--label_smoothing", str(opt["label_smoothing"]),
        "--limit", str(data["train"]["records"]),
        *common_model_args(protocol, seed),
        "--eval_manifest", str(manifests["test"]),
        "--eval_limit", str(data["test"]["records"]),
        "--validation_manifest", str(manifests["validation"]),
        "--validation_limit", str(data["validation"]["records"]),
        "--validation_interval_steps", str(opt["validation_interval_steps"]),
    ]


def build_jobs(
    protocol: dict[str, Any],
    manifests: dict[str, Path],
    output_root: Path,
    seeds: list[int],
    methods: list[str],
    stage_devices: str,
) -> list[Job]:
    by_id = {method["id"]: method for method in protocol["methods"]}
    unknown = sorted(set(methods) - set(by_id))
    if unknown:
        raise ValueError(f"unknown E1 methods: {unknown}")
    jobs: list[Job] = []
    for seed_index, seed in enumerate(seeds):
        selected = [by_id[name] for name in methods]
        rotated = selected[seed_index % len(selected):] + selected[:seed_index % len(selected)]
        for method in rotated:
            output_dir = output_root / method["id"] / f"seed{seed}"
            if method["runtime"] == "exactbp":
                command = exactbp_command(protocol, method, manifests, output_dir, seed, stage_devices)
                summary_name = "summary.json"
            elif method["runtime"] == "bpfree_orchestrated":
                command = bpfree_command(protocol, method, manifests, output_dir, seed, stage_devices)
                summary_name = "scheduler_summary.json"
            else:
                raise ValueError(f"unsupported runtime: {method['runtime']}")
            jobs.append(Job(method["id"], method["paper_label"], method["runtime"], seed, str(output_dir), summary_name, command))
    return jobs


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def authoritative_train_phases(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Return non-overlapping training summaries from the runtime report."""
    train_phases = [
        item for item in summary.get("phase_summaries", []) if item.get("mode") == "train"
    ]
    aggregate = [item for item in train_phases if item.get("phase") == "train"]
    if len(aggregate) > 1:
        raise ValueError("runtime summary contains multiple aggregate train phases")
    return aggregate or train_phases


def audit_result(job: Job, protocol: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(job.output_dir)
    summary_path = output_dir / job.summary_name
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing summary for {job.method}: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = protocol["manifests"]
    checks: dict[str, bool] = {
        "failed_zero": int(summary.get("failed", 0)) == 0,
        "test_records": int(summary.get("completed", -1)) == int(expected["test"]["records"]),
        "validation_records": int(summary.get("validation_records", -1)) == int(expected["validation"]["records"]),
        "validation_interval": int(summary.get("validation_interval_steps", -1)) == int(protocol["optimization"]["validation_interval_steps"]),
    }
    curve_path = output_dir / "validation_curve.csv"
    if curve_path.is_file():
        with curve_path.open(newline="", encoding="utf-8") as handle:
            curve = list(csv.DictReader(handle))
        expected_steps = list(range(0, int(protocol["optimization"]["optimizer_steps"]) + 1, int(protocol["optimization"]["validation_interval_steps"])))
        checks["validation_steps"] = [int(row["optimizer_step"]) for row in curve] == expected_steps
    else:
        checks["validation_steps"] = False
    if job.runtime == "exactbp":
        checks["train_records"] = int(summary.get("train_records", -1)) == int(expected["train"]["records"])
    else:
        train_phases = authoritative_train_phases(summary)
        checks["train_records"] = sum(int(item.get("records", 0)) for item in train_phases) == int(expected["train"]["records"])
        stage_updates = {stage: 0 for stage in range(int(protocol["model"]["num_stages"]))}
        for phase in train_phases:
            per_stage = phase.get("update_consistency", {}).get("per_stage", {})
            for stage in stage_updates:
                stage_updates[stage] += int(per_stage.get(str(stage), {}).get("update_events", 0))
        checks["stage_optimizer_steps"] = all(value == int(protocol["optimization"]["optimizer_steps"]) for value in stage_updates.values())
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"E1 result contract failed for {job.method} seed={job.seed}: {failed}")
    return {"method": job.method, "seed": job.seed, "summary": str(summary_path), "checks": checks}


def run_job(job: Job, protocol: dict[str, Any], protocol_sha256: str) -> dict[str, Any]:
    output_dir = Path(job.output_dir)
    summary_path = output_dir / job.summary_name
    if summary_path.is_file():
        return {"status": "already_complete", **audit_result(job, protocol)}
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"incomplete non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "command.txt").write_text(shlex.join(job.command) + "\n", encoding="utf-8")
    write_json(output_dir / "run_config.json", {**asdict(job), "protocol_sha256": protocol_sha256})
    env = os.environ.copy()
    src = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    started = time.time()
    with (output_dir / "run.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(job.command, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    (output_dir / "exit_code.txt").write_text(f"{result.returncode}\n", encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"E1 job failed: {job.method} seed={job.seed}; see {output_dir / 'run.log'}")
    return {"status": "completed", "elapsed_s": time.time() - started, **audit_result(job, protocol)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--seeds", type=parse_ints)
    parser.add_argument("--methods", type=parse_names)
    parser.add_argument("--stage-devices")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    protocol = load_protocol(config_path)
    data_root = args.data_root.resolve()
    manifests = validate_inputs(protocol, data_root)
    seeds = args.seeds or [int(value) for value in protocol["seeds"]]
    methods = args.methods or [str(item["id"]) for item in protocol["methods"]]
    stage_devices = args.stage_devices or str(protocol["model"]["stage_devices"])
    output_root = args.output_root.resolve()
    jobs = build_jobs(protocol, manifests, output_root, seeds, methods, stage_devices)
    code = {
        "driver": {"path": str(Path(__file__).resolve().relative_to(REPO_ROOT)), "sha256": sha256(Path(__file__).resolve())},
        "config": {"path": str(config_path.relative_to(REPO_ROOT)), "sha256": sha256(config_path)},
        "exactbp_runtime": {"path": str(EXACTBP_SOURCE.relative_to(REPO_ROOT)), "sha256": sha256(EXACTBP_SOURCE)},
        "bpfree_runtime": {"module": BPFREE_MODULE, "path": str(BPFREE_SOURCE.relative_to(REPO_ROOT)), "sha256": sha256(BPFREE_SOURCE)},
    }
    run_protocol = {**protocol, "data_root": str(data_root), "resolved_stage_devices": stage_devices, "requested_seeds": seeds, "requested_methods": methods, "code": code, "python": sys.executable}
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "protocol.json", run_protocol)
    write_json(output_root / "jobs.json", [asdict(job) for job in jobs])
    if args.dry_run:
        for job in jobs:
            print(shlex.join(job.command))
        return
    status: list[dict[str, Any]] = []
    for job in jobs:
        print(f"Starting {job.method} seed={job.seed}: {job.output_dir}", flush=True)
        row = run_job(job, protocol, code["config"]["sha256"])
        status.append(row)
        write_json(output_root / "status.json", status)
        print(f"Finished {job.method} seed={job.seed}: {row['status']}", flush=True)


if __name__ == "__main__":
    main()
