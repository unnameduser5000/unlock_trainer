#!/usr/bin/env python3
"""Run the fixed E2 GPU-memory protocol used by the paper."""

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
DEFAULT_CONFIG = Path(__file__).with_name("configs") / "gpu_memory_v1.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results/e2_memory/raw/gpu_memory_v1"
DEFAULT_SMOKE_OUTPUT_ROOT = REPO_ROOT / "output/e2-gpu-memory-smoke"
METHOD_IDS = ("bpfree", "exactbp_1f1b", "exactbp_gpipe", "pipedream")
GEOMETRIES = (
    ("b1_m32", 1, 32),
    ("b2_m16", 2, 16),
    ("b4_m8", 4, 8),
    ("b8_m4", 8, 4),
)
RUNNER_MODULES = {
    "bpfree": "sg_exe_trainer.runtime.bpfree.cpu_runner",
    "exactbp": "sg_exe_trainer.runtime.exactbp.cpu_runner",
    "pipedream": "experiments.shared.baselines.pipedream_cpu",
}
RUNNER_FILES = {
    "bpfree": REPO_ROOT / "src/sg_exe_trainer/runtime/bpfree/cpu_runner.py",
    "exactbp": REPO_ROOT / "src/sg_exe_trainer/runtime/exactbp/cpu_runner.py",
    "pipedream": REPO_ROOT / "experiments/shared/baselines/pipedream_cpu.py",
}


@dataclass(frozen=True)
class Job:
    method: str
    runtime: str
    paper_label: str
    geometry: str
    physical_batch_size: int
    microbatches: int
    repetition: int
    train_windows: int
    output_dir: str
    command: list[str]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def parse_names(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError(
            "expected a non-empty comma-separated list without duplicates"
        )
    return values


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != 1:
        raise ValueError(
            f"unsupported E2 protocol schema: {protocol.get('schema_version')}"
        )
    if protocol.get("protocol_id") != "e2-gpu-memory-v1":
        raise ValueError(f"unexpected E2 protocol id: {protocol.get('protocol_id')}")

    methods = tuple(item["id"] for item in protocol["methods"])
    if methods != METHOD_IDS:
        raise ValueError(f"E2 methods changed: {methods}")
    geometries = tuple(
        (
            item["id"],
            int(item["physical_batch_size"]),
            int(item["microbatches"]),
        )
        for item in protocol["geometries"]
    )
    if geometries != GEOMETRIES:
        raise ValueError(f"E2 geometries changed: {geometries}")

    effective_batch = int(protocol["optimization"]["effective_batch"])
    if effective_batch != 32:
        raise ValueError(f"E2 effective batch changed: {effective_batch}")
    for geometry, physical_batch, microbatches in geometries:
        if physical_batch * microbatches != effective_batch:
            raise ValueError(f"{geometry} does not preserve B={effective_batch}")

    measurement = protocol["measurement"]
    if not measurement.get("activation_tracking"):
        raise ValueError("E2 requires activation-memory tracking")
    if measurement.get("terminal_output_log_probabilities"):
        raise ValueError("E2 requires terminal output log-probabilities to be disabled")
    repetitions = protocol["repetitions"]
    if (
        int(repetitions["default"]) != 1
        or int(repetitions["primary"]) != 3
        or repetitions["primary_geometry"] != "b8_m4"
    ):
        raise ValueError("E2 repetition policy changed")
    return protocol


def resolve_manifests(
    protocol: dict[str, Any],
    data_root: Path,
) -> tuple[dict[str, Path], dict[str, str]]:
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for split, spec in protocol["manifests"].items():
        path = (data_root / spec["path"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"missing E2 {split} manifest: {path}")
        actual = sha256(path)
        if actual != spec["sha256"]:
            raise ValueError(
                f"E2 {split} manifest hash mismatch: {actual}; "
                f"expected {spec['sha256']}"
            )
        paths[split] = path
        hashes[split] = actual
    return paths, hashes


def common_args(
    protocol: dict[str, Any],
    manifests: dict[str, Path],
    output_dir: Path,
    physical_batch: int,
    microbatches: int,
    train_limit: int,
    stage_devices: str,
) -> list[str]:
    model = protocol["model"]
    optimization = protocol["optimization"]
    transport = protocol["transport"]
    seed = int(optimization["seed"])
    return [
        "--model_name", str(model["name"]),
        "--train_manifest", str(manifests["train"]),
        "--output_dir", str(output_dir),
        "--num_chunks", str(model["num_stages"]),
        "--stage_devices", stage_devices,
        "--train_limit", str(train_limit),
        "--train_epochs", "1",
        "--physical_batch_size", str(physical_batch),
        "--gradient_accumulation_steps", str(microbatches),
        "--learning_rate", str(optimization["learning_rate"]),
        "--optimizer", str(optimization["optimizer"]),
        "--grad_clip", str(optimization["grad_clip"]),
        "--dtype", str(model["dtype"]),
        "--label_smoothing", str(optimization["label_smoothing"]),
        "--trainable_mode", str(model["trainable_mode"]),
        "--lora_rank", str(model["lora_rank"]),
        "--lora_alpha", str(model["lora_alpha"]),
        "--lora_targets", str(model["lora_targets"]),
        "--lora_init_std", str(model["lora_init_std"]),
        "--lora_init_seed", str(seed),
        "--seed", str(seed),
        "--recv_prepost_depth", str(transport["recv_prepost_depth"]),
        "--max_pending_send_bytes", str(transport["max_pending_send_bytes"]),
        "--max_posted_recv_bytes", str(transport["max_posted_recv_bytes"]),
        "--link_latency_ms", str(transport["link_latency_ms"]),
        "--link_bandwidth_mbps", str(transport["link_bandwidth_mbps"]),
        "--link_jitter_ms", str(transport["link_jitter_ms"]),
        "--link_emulation_seed", str(seed),
        "--track_activation_memory",
    ]


def build_command(
    protocol: dict[str, Any],
    method: dict[str, Any],
    manifests: dict[str, Path],
    output_dir: Path,
    geometry: dict[str, Any],
    train_windows: int,
    stage_devices: str,
    master_port: int,
) -> list[str]:
    physical_batch = int(geometry["physical_batch_size"])
    microbatches = int(geometry["microbatches"])
    train_limit = int(protocol["optimization"]["effective_batch"]) * train_windows
    common = common_args(
        protocol,
        manifests,
        output_dir,
        physical_batch,
        microbatches,
        train_limit,
        stage_devices,
    )
    stages = int(protocol["model"]["num_stages"])
    runtime = method["runtime"]

    if runtime == "bpfree":
        return [
            sys.executable,
            "-m", RUNNER_MODULES["bpfree"],
            *common,
            "--eval_manifest", str(manifests["eval"]),
            "--eval_limit", "1",
            "--progress_interval", "0",
            "--skip_eval_before",
            "--skip_eval_after",
            "--train_chunks", "all",
            "--belief_transport_mode", str(method["belief_transport_mode"]),
            "--alpha", "1.0",
            "--master_port", str(master_port),
        ]

    distributed = [
        sys.executable,
        "-m", "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={stages}",
        "--module",
        RUNNER_MODULES[runtime],
    ]
    if runtime == "exactbp":
        return [
            *distributed,
            *common,
            "--eval_manifest", str(manifests["eval"]),
            "--eval_limit", "1",
            "--progress_interval", "0",
            "--skip_eval_before",
            "--skip_eval_after",
            "--pipeline_schedule", str(method["pipeline_schedule"]),
            "--memory_warmup_windows",
            str(protocol["measurement"]["memory_warmup_windows"]),
        ]
    if runtime == "pipedream":
        return [*distributed, *common]
    raise ValueError(f"unsupported E2 runtime: {runtime}")


def build_jobs(
    protocol: dict[str, Any],
    manifests: dict[str, Path],
    output_root: Path,
    selected_methods: list[str],
    selected_geometries: list[str],
    stage_devices: str,
    smoke: bool,
) -> list[Job]:
    methods = {item["id"]: item for item in protocol["methods"]}
    geometries = {item["id"]: item for item in protocol["geometries"]}
    unknown_methods = sorted(set(selected_methods) - set(methods))
    unknown_geometries = sorted(set(selected_geometries) - set(geometries))
    if unknown_methods:
        raise ValueError(f"unknown E2 methods: {unknown_methods}")
    if unknown_geometries:
        raise ValueError(f"unknown E2 geometries: {unknown_geometries}")

    train_windows = (
        1 if smoke else int(protocol["optimization"]["train_windows"])
    )
    repetitions = protocol["repetitions"]
    jobs: list[Job] = []
    port = 29831
    for geometry_id in selected_geometries:
        geometry = geometries[geometry_id]
        reps = 1
        if not smoke and geometry_id == repetitions["primary_geometry"]:
            reps = int(repetitions["primary"])
        for rep in range(reps):
            for method_id in selected_methods:
                method = methods[method_id]
                output_dir = (
                    output_root / geometry_id / method_id / f"rep_{rep:02d}"
                )
                command = build_command(
                    protocol,
                    method,
                    manifests,
                    output_dir,
                    geometry,
                    train_windows,
                    stage_devices,
                    port,
                )
                jobs.append(
                    Job(
                        method=method_id,
                        runtime=str(method["runtime"]),
                        paper_label=str(method["paper_label"]),
                        geometry=geometry_id,
                        physical_batch_size=int(geometry["physical_batch_size"]),
                        microbatches=int(geometry["microbatches"]),
                        repetition=rep,
                        train_windows=train_windows,
                        output_dir=str(output_dir),
                        command=command,
                    )
                )
                port += 1
    return jobs


def is_complete(job: Job, stages: int) -> bool:
    output_dir = Path(job.output_dir)
    if not (output_dir / "summary.json").is_file():
        return False
    if job.runtime in {"exactbp", "pipedream"}:
        return all(
            (output_dir / f"rank{stage}.summary.json").is_file()
            for stage in range(stages)
        )
    return all(
        (output_dir / f"train.stage{stage}.metrics.csv").is_file()
        for stage in range(stages)
    )


def positive(value: Any) -> bool:
    try:
        return int(float(value)) > 0
    except (TypeError, ValueError):
        return False


def audit_job(job: Job, protocol: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(job.output_dir)
    stages = int(protocol["model"]["num_stages"])
    expected_records = (
        int(protocol["optimization"]["effective_batch"]) * job.train_windows
    )
    summary = json.loads(
        (output_dir / "summary.json").read_text(encoding="utf-8")
    )
    checks: dict[str, bool] = {
        "completed_records": int(summary.get("completed_records", -1))
        == expected_records,
        "effective_batch": (
            job.physical_batch_size * job.microbatches
            == int(protocol["optimization"]["effective_batch"])
        ),
    }

    stage_peaks: list[int] = []
    saved_nonleaf_peaks: list[int] = []
    if job.runtime == "bpfree":
        checks["activation_tracking"] = bool(
            summary.get("activation_tracking_enabled")
        )
        output_log_prob_peaks: list[int] = []
        for stage in range(stages):
            metrics_path = output_dir / f"train.stage{stage}.metrics.csv"
            with metrics_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            if not rows:
                raise RuntimeError(f"empty BP-free metrics: {metrics_path}")
            stage_peaks.append(
                max(int(float(row["cuda_peak_memory_allocated"])) for row in rows)
            )
            saved_nonleaf_peaks.append(
                max(
                    int(float(row.get(
                        "autograd_saved_cuda_nonleaf_unique_bytes_peak", 0
                    ) or 0))
                    for row in rows
                )
            )
            output_log_prob_peaks.append(
                max(
                    int(float(row.get("output_log_probs_bytes", 0) or 0))
                    for row in rows
                )
            )
        checks["terminal_output_log_probabilities_disabled"] = (
            max(output_log_prob_peaks, default=0) == 0
        )
    else:
        for stage in range(stages):
            rank_summary = json.loads(
                (output_dir / f"rank{stage}.summary.json").read_text(
                    encoding="utf-8"
                )
            )
            if job.runtime == "exactbp":
                profile = rank_summary["train"]["memory_profile"]
                aggregate = profile["aggregate"]
                checks[f"stage_{stage}_activation_tracking"] = bool(
                    profile["tracking_enabled"]
                )
                stage_peaks.append(
                    int(aggregate.get("peak_cuda_allocated_bytes", 0))
                )
                saved_nonleaf_peaks.append(
                    int(aggregate.get(
                        "autograd_saved_cuda_nonleaf_unique_bytes_peak", 0
                    ))
                )
            else:
                activation = rank_summary.get("activation_memory", {})
                checks[f"stage_{stage}_activation_tracking"] = bool(
                    rank_summary.get("activation_tracking_enabled")
                )
                stage_peaks.append(
                    int(rank_summary.get("peak_cuda_allocated_bytes", 0))
                )
                saved_nonleaf_peaks.append(
                    int(activation.get(
                        "autograd_saved_cuda_nonleaf_unique_bytes_peak", 0
                    ))
                )
        if job.runtime == "exactbp":
            expected_schedule = next(
                item["pipeline_schedule"]
                for item in protocol["methods"]
                if item["id"] == job.method
            )
            checks["pipeline_schedule"] = (
                summary.get("pipeline_schedule") == expected_schedule
            )

    checks["all_stage_cuda_peaks_positive"] = (
        len(stage_peaks) == stages and all(positive(value) for value in stage_peaks)
    )
    checks["all_stage_saved_nonleaf_peaks_positive"] = (
        len(saved_nonleaf_peaks) == stages
        and all(positive(value) for value in saved_nonleaf_peaks)
    )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            f"E2 result contract failed for {job.geometry}/{job.method}/"
            f"rep={job.repetition}: {failed}"
        )
    audit = {
        "method": job.method,
        "geometry": job.geometry,
        "repetition": job.repetition,
        "expected_records": expected_records,
        "stage_peak_cuda_allocated_bytes": stage_peaks,
        "stage_saved_nonleaf_unique_peak_bytes": saved_nonleaf_peaks,
        "checks": checks,
    }
    write_json(output_dir / "result_audit.json", audit)
    return audit


def run_job(
    job: Job,
    protocol: dict[str, Any],
    protocol_sha256: str,
    manifest_hashes: dict[str, str],
    runner_hashes: dict[str, str],
    resume: bool,
) -> dict[str, Any]:
    output_dir = Path(job.output_dir)
    stages = int(protocol["model"]["num_stages"])
    if resume and is_complete(job, stages):
        return {"status": "already_complete", "audit": audit_job(job, protocol)}
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"incomplete non-empty E2 output: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "command.txt").write_text(
        shlex.join(job.command) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "protocol_version": protocol["protocol_id"],
        "method": job.method,
        "geometry": {
            "id": job.geometry,
            "physical_batch_size": job.physical_batch_size,
            "microbatches": job.microbatches,
        },
        "effective_batch": int(protocol["optimization"]["effective_batch"]),
        "train_windows": job.train_windows,
        "train_limit": (
            int(protocol["optimization"]["effective_batch"]) * job.train_windows
        ),
        "rep": job.repetition,
        "stage_devices": protocol["_resolved_stage_devices"],
        "command": job.command,
        "protocol_sha256": protocol_sha256,
        "manifest_sha256": manifest_hashes,
        "runner_sha256": runner_hashes,
        "status": "running",
        "started_epoch_s": time.time(),
    }
    write_json(output_dir / "run_metadata.json", metadata)

    env = os.environ.copy()
    python_paths = [str(REPO_ROOT / "src"), str(REPO_ROOT)]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    measurement = protocol["measurement"]
    threads = str(measurement["omp_num_threads"])
    env["OMP_NUM_THREADS"] = threads
    env["MKL_NUM_THREADS"] = threads
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["BPFREE_DEFERRED_BACKWARD_GROUP_SIZE"] = str(
        measurement["bpfree_deferred_backward_group_size"]
    )
    env["BPFREE_WINDOW_INPUT_STAGING"] = str(
        measurement["bpfree_window_input_staging"]
    )

    started = time.perf_counter()
    with (output_dir / "run.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(
            job.command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    metadata["elapsed_s"] = time.perf_counter() - started
    metadata["returncode"] = result.returncode
    metadata["ended_epoch_s"] = time.time()
    if result.returncode != 0:
        metadata["status"] = "failed"
        write_json(output_dir / "run_metadata.json", metadata)
        raise RuntimeError(
            f"E2 job failed: {job.geometry}/{job.method}/"
            f"rep={job.repetition}; see {output_dir / 'run.log'}"
        )

    audit = audit_job(job, protocol)
    metadata["status"] = "complete"
    write_json(output_dir / "run_metadata.json", metadata)
    return {
        "status": "completed",
        "elapsed_s": metadata["elapsed_s"],
        "audit": audit,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--methods", type=parse_names)
    parser.add_argument("--geometries", type=parse_names)
    parser.add_argument("--stage-devices")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_path = args.config.resolve()
    protocol = load_protocol(protocol_path)
    protocol_sha256 = sha256(protocol_path)
    manifests, manifest_hashes = resolve_manifests(protocol, args.data_root)

    stage_devices = args.stage_devices or protocol["model"]["stage_devices"]
    devices = [item.strip() for item in stage_devices.split(",") if item.strip()]
    if len(devices) != int(protocol["model"]["num_stages"]):
        raise ValueError(
            f"E2 requires {protocol['model']['num_stages']} stage devices; "
            f"received {devices}"
        )
    protocol["_resolved_stage_devices"] = ",".join(devices)

    selected_methods = args.methods or list(METHOD_IDS)
    if args.geometries:
        selected_geometries = args.geometries
    elif args.smoke:
        selected_geometries = [protocol["repetitions"]["primary_geometry"]]
    else:
        selected_geometries = [item[0] for item in GEOMETRIES]

    if args.output_root:
        output_root = args.output_root.resolve()
    elif args.smoke:
        output_root = DEFAULT_SMOKE_OUTPUT_ROOT
    else:
        output_root = DEFAULT_OUTPUT_ROOT
    jobs = build_jobs(
        protocol,
        manifests,
        output_root,
        selected_methods,
        selected_geometries,
        protocol["_resolved_stage_devices"],
        args.smoke,
    )
    runner_hashes = {
        name: sha256(path) for name, path in RUNNER_FILES.items()
    }
    plan = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "mode": "smoke" if args.smoke else "formal",
        "output_root": str(output_root),
        "manifest_sha256": manifest_hashes,
        "runner_sha256": runner_hashes,
        "jobs": [asdict(job) for job in jobs],
    }

    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return

    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "protocol_snapshot.json", plan)
    results = []
    for index, job in enumerate(jobs, start=1):
        print(
            f"[{index}/{len(jobs)}] {job.geometry} {job.method} "
            f"rep={job.repetition}",
            flush=True,
        )
        results.append(
            {
                **asdict(job),
                **run_job(
                    job,
                    protocol,
                    protocol_sha256,
                    manifest_hashes,
                    runner_hashes,
                    args.resume,
                ),
            }
        )
    write_json(
        output_root / "matrix_audit.json",
        {
            "protocol_id": protocol["protocol_id"],
            "mode": "smoke" if args.smoke else "formal",
            "job_count": len(jobs),
            "results": results,
        },
    )
    print(f"Wrote {output_root / 'matrix_audit.json'}", flush=True)


if __name__ == "__main__":
    main()
