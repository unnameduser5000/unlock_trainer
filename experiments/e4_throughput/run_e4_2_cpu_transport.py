#!/usr/bin/env python3
"""Launch fair GPU-compute + pinned-CPU/Gloo E4.2 sweeps.

The methods share the same per-rank transport resource contract:
- pinned CPU staging and tagged Gloo point-to-point
- one total pending-send byte budget across all logical channels
- one total preposted-receive byte budget across all logical channels
- the same receive-prepost depth cap
- CPU MAX reduction for wall time
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from experiments.e4_throughput.provenance import (
    artifact_hashes,
    build_execution_environment,
    capture_provenance,
)

DEFAULT_CONFIG = "experiments/e4_throughput/configs/e4_2a_batch_geometry.json"
SUPPORTED_METHODS = {
    "bpfree",
    "exactbp_gpipe",
    "exactbp_1f1b",
    "pipedream",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_csv(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def train_phase(summary: dict[str, Any]) -> dict[str, Any]:
    for phase in summary.get("phases", []):
        if isinstance(phase, dict) and phase.get("phase") == "train":
            return phase
    raise ValueError("summary has no train phase")


def training_metrics(
    summary: dict[str, Any], method: str
) -> dict[str, float | int]:
    if method == "pipedream":
        return {
            "completed_records": int(summary["completed_records"]),
            "optimizer_steps": int(summary["optimizer_steps_per_stage"]),
            "wall_ms": float(summary["wall_ms"]),
            "throughput_per_s": float(summary["throughput_per_s"]),
        }
    phase = train_phase(summary)
    return {
        "completed_records": int(phase["completed_records"]),
        "optimizer_steps": int(phase["optimizer_steps"]),
        "wall_ms": float(phase["wall_ms"]),
        "throughput_per_s": float(phase["throughput_per_s"]),
    }


def budget_rows(
    summary: dict[str, Any], method: str
) -> list[dict[str, Any]]:
    if method == "pipedream":
        rows = [row.get("transport_budget") for row in summary.get("by_rank", [])]
        if rows and all(isinstance(row, dict) for row in rows):
            return rows
        raise ValueError("PipeDream summary has no per-rank transport budget")
    rows = summary.get("transport_budget_by_rank")
    if not isinstance(rows, list):
        rows = train_phase(summary).get("transport_budget_by_rank")
    if not isinstance(rows, list):
        raise ValueError("summary has no transport_budget_by_rank")
    return rows


def resolve_cases(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for raw in cfg["cases"]:
        physical = int(raw["physical_request_batch"])
        micro = int(raw["microbatches_per_update"])
        methods = [str(x) for x in raw.get("methods", cfg["methods"])]
        cases.append(
            {
                "name": str(raw["name"]),
                "physical_request_batch": physical,
                "microbatches_per_update": micro,
                "effective_optimizer_batch": physical * micro,
                "methods": methods,
            }
        )
    return cases


def validate(cfg: dict[str, Any], repo_root: Path) -> list[dict[str, Any]]:
    gpu_count = int(cfg["gpu_count"])
    devices = [x.strip() for x in str(cfg["stage_devices"]).split(",") if x.strip()]
    if gpu_count < 2 or len(devices) != gpu_count:
        raise ValueError("gpu_count/stage_devices mismatch")
    if int(cfg["recv_prepost_depth"]) < 0:
        raise ValueError("recv_prepost_depth must be non-negative")
    if int(cfg["max_pending_send_bytes"]) <= 0:
        raise ValueError("max_pending_send_bytes must be positive")
    if int(cfg["max_posted_recv_bytes"]) <= 0:
        raise ValueError("max_posted_recv_bytes must be positive")
    for key in ("train_manifest", "eval_manifest"):
        if not (repo_root / str(cfg[key])).is_file():
            raise FileNotFoundError(repo_root / str(cfg[key]))

    cases = resolve_cases(cfg)
    names: set[str] = set()
    for case in cases:
        if case["name"] in names:
            raise ValueError(f"duplicate case {case['name']}")
        names.add(case["name"])
        if case["physical_request_batch"] <= 0 or case["microbatches_per_update"] <= 0:
            raise ValueError(f"invalid batch geometry: {case}")
        if int(cfg["train_limit"]) % case["effective_optimizer_batch"] != 0:
            raise ValueError(f"train_limit not divisible for {case['name']}")
        unknown = set(case["methods"]) - SUPPORTED_METHODS
        if unknown:
            raise ValueError(f"unsupported methods for {case['name']}: {sorted(unknown)}")
        for method in ("exactbp_1f1b", "pipedream"):
            if method in case["methods"] and case["microbatches_per_update"] < gpu_count:
                raise ValueError(f"{case['name']}: {method} requires m >= P")
    return cases


def method_schedule(method: str) -> str:
    if method == "exactbp_gpipe":
        return "gpipe"
    if method == "exactbp_1f1b":
        return "1f1b"
    if method == "pipedream":
        return "pipedream"
    return "bpfree"


def build_command(
    *,
    repo_root: Path,
    cfg: dict[str, Any],
    case: dict[str, Any],
    method: str,
    output_dir: Path,
    master_port: int,
) -> list[str]:
    runner_common = [
        "--model_name", str(cfg["model_name"]),
        "--train_manifest", str(repo_root / str(cfg["train_manifest"])),
        "--output_dir", str(output_dir),
        "--num_chunks", str(cfg["gpu_count"]),
        "--stage_devices", str(cfg["stage_devices"]),
        "--train_limit", str(cfg["train_limit"]),
        "--train_epochs", str(cfg["train_epochs"]),
        "--physical_batch_size", str(case["physical_request_batch"]),
        "--gradient_accumulation_steps", str(case["microbatches_per_update"]),
        "--learning_rate", str(cfg["learning_rate"]),
        "--optimizer", str(cfg["optimizer"]),
        "--grad_clip", str(cfg["grad_clip"]),
        "--dtype", str(cfg["dtype"]),
        "--label_smoothing", str(cfg["label_smoothing"]),
        "--trainable_mode", str(cfg["trainable_mode"]),
        "--lora_rank", str(cfg["lora_rank"]),
        "--lora_alpha", str(cfg["lora_alpha"]),
        "--lora_targets", str(cfg["lora_targets"]),
        "--lora_init_std", str(cfg["lora_init_std"]),
        "--lora_init_seed", str(cfg["lora_init_seed"]),
        "--seed", str(cfg["seed"]),
        "--perf_minimal_metrics",
        "--recv_prepost_depth", str(cfg["recv_prepost_depth"]),
        "--max_pending_send_bytes", str(cfg["max_pending_send_bytes"]),
        "--max_posted_recv_bytes", str(cfg["max_posted_recv_bytes"]),
    ]
    exact_common = [
        *runner_common,
        "--eval_manifest", str(repo_root / str(cfg["eval_manifest"])),
        "--eval_limit", str(cfg["eval_limit"]),
        "--progress_interval", "0",
        "--skip_eval_before",
        "--skip_eval_after",
        "--no-track_activation_memory",
    ]

    if method == "bpfree":
        return [
            sys.executable,
            "-m", "sg_exe_trainer.runtime.bpfree.cpu_runner",
            *exact_common,
            "--backend", "gloo",
            "--master_addr", "127.0.0.1",
            "--master_port", str(master_port),
            "--belief_transport_mode", str(cfg["belief_transport_mode"]),
        ]

    if method == "pipedream":
        return [
            sys.executable,
            "-m", "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={int(cfg['gpu_count'])}",
            f"--master_port={master_port}",
            "-m", "experiments.shared.baselines.pipedream_cpu",
            *runner_common,
        ]

    schedule = method_schedule(method)
    return [
        sys.executable,
        "-m", "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={int(cfg['gpu_count'])}",
        "-m", "sg_exe_trainer.runtime.exactbp.cpu_runner",
        *exact_common,
        "--pipeline_schedule", schedule,
    ]


def validate_summary(
    summary: dict[str, Any],
    *,
    cfg: dict[str, Any],
    case: dict[str, Any],
    method: str,
) -> None:
    expected_steps = int(cfg["train_limit"]) // int(case["effective_optimizer_batch"])
    metrics = training_metrics(summary, method)
    checks: dict[str, bool] = {
        "num_chunks": int(summary.get("num_chunks", -1)) == int(cfg["gpu_count"]),
        "physical_request_batch": int(summary.get("physical_request_batch", -1)) == int(case["physical_request_batch"]),
        "effective_optimizer_batch": int(summary.get("effective_optimizer_batch", -1)) == int(case["effective_optimizer_batch"]),
        "completed_records": int(metrics["completed_records"]) == int(cfg["train_limit"]),
        "optimizer_steps": int(metrics["optimizer_steps"]) == expected_steps,
        "recv_prepost_depth": int(summary.get("recv_prepost_depth", -1)) == int(cfg["recv_prepost_depth"]),
    }
    if method == "pipedream":
        checks["microbatches"] = (
            int(summary.get("local_gradient_coalescing", -1))
            == int(case["microbatches_per_update"])
        )
    else:
        checks["microbatches"] = (
            int(summary.get("microbatches", -1))
            == int(case["microbatches_per_update"])
        )
        checks["perf_minimal_metrics"] = bool(summary.get("perf_minimal_metrics")) is True
        checks["max_pending_send_bytes"] = (
            int(summary.get("max_pending_send_bytes", -1))
            == int(cfg["max_pending_send_bytes"])
        )
        checks["max_posted_recv_bytes"] = (
            int(summary.get("max_posted_recv_bytes", -1))
            == int(cfg["max_posted_recv_bytes"])
        )
    if method == "bpfree":
        checks["transport"] = str(summary.get("transport", "")).startswith("gloo-cpu-hidden")
    else:
        checks["transport"] = str(summary.get("transport", "")).startswith("gloo-cpu-hidden-and-grad")
        if method == "pipedream":
            checks["runner"] = str(summary.get("runner", "")).startswith(
                "pipedream-continuous-1f1b"
            )
        else:
            checks["pipeline_schedule"] = summary.get("pipeline_schedule") == method_schedule(method)
    rows = budget_rows(summary, method)
    checks["transport_budget_rank_count"] = len(rows) == int(cfg["gpu_count"])
    for rank, row in enumerate(rows):
        checks[f"transport_budget_rank_{rank}"] = isinstance(row, dict) and all(
            [
                int(row.get("max_pending_send_bytes", -1))
                == int(cfg["max_pending_send_bytes"]),
                int(row.get("max_posted_recv_bytes", -1))
                == int(cfg["max_posted_recv_bytes"]),
                int(row.get("peak_pending_send_bytes", -1))
                <= int(cfg["max_pending_send_bytes"]),
                int(row.get("peak_posted_recv_bytes", -1))
                <= int(cfg["max_posted_recv_bytes"]),
                int(row.get("pending_send_bytes_at_end", -1)) == 0,
                int(row.get("posted_recv_bytes_at_end", -1)) == 0,
            ]
        )

    failed = [key for key, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"summary contract failed: {failed}")


def run_one(
    *,
    repo_root: Path,
    cfg: dict[str, Any],
    case: dict[str, Any],
    method: str,
    rep: int,
    dry_run: bool,
    resume: bool,
    overwrite: bool,
    master_port: int,
) -> None:
    output_root = repo_root / str(cfg["output_root"])
    output_dir = output_root / case["name"] / method / f"rep_{rep:02d}"
    summary_path = output_dir / "summary.json"

    if summary_path.is_file() and resume:
        try:
            validate_summary(read_json(summary_path), cfg=cfg, case=case, method=method)
            print(f"[resume] valid: {summary_path}")
            return
        except Exception as exc:
            print(f"[resume] invalid summary, rerunning: {summary_path}: {exc}")
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    elif output_dir.exists() and not dry_run and not resume:
        raise FileExistsError(f"output exists; use --resume or --overwrite: {output_dir}")

    command = build_command(
        repo_root=repo_root,
        cfg=cfg,
        case=case,
        method=method,
        output_dir=output_dir,
        master_port=master_port,
    )
    run_env, execution_environment = build_execution_environment(cfg)
    print(f"\n[env] {execution_environment}", flush=True)
    print("\n$", " ".join(command), flush=True)
    if dry_run:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "run_metadata.json"
    metadata = {
        "experiment_id": cfg["experiment_id"],
        "case": case,
        "method": method,
        "rep": rep,
        "command": command,
        "execution_environment": execution_environment,
        "started_epoch_s": time.time(),
        "transport_contract": {
            "backend": "gloo",
            "gpu_compute": True,
            "pinned_cpu_staging": True,
            "recv_prepost_depth": cfg["recv_prepost_depth"],
            "max_pending_send_bytes": cfg["max_pending_send_bytes"],
            "max_posted_recv_bytes": cfg["max_posted_recv_bytes"],
        },
        "provenance": capture_provenance(
            repo_root=repo_root,
            config_path=Path(cfg["_resolved_config_path"]),
            cfg=cfg,
            command=command,
            execution_environment=execution_environment,
        ),
    }
    write_json(metadata_path, metadata)
    with (output_dir / "run.log").open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            command,
            cwd=repo_root,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=run_env,
            check=False,
        )
    metadata["returncode"] = proc.returncode
    metadata["ended_epoch_s"] = time.time()
    metadata["duration_s"] = metadata["ended_epoch_s"] - metadata["started_epoch_s"]
    metadata["artifacts"] = artifact_hashes(
        [output_dir / "run.log", output_dir / "summary.json"],
        repo_root=repo_root,
    )
    write_json(metadata_path, metadata)
    if proc.returncode != 0:
        raise RuntimeError(f"run failed ({proc.returncode}); see {output_dir / 'run.log'}")
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    validate_summary(read_json(summary_path), cfg=cfg, case=case, method=method)
    print(f"[ok] {summary_path}")


def aggregate(repo_root: Path, cfg: dict[str, Any], cases: list[dict[str, Any]]) -> None:
    output_root = repo_root / str(cfg["output_root"])
    run_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for case in cases:
        method_values: dict[str, list[float]] = {}
        for method in case["methods"]:
            vals: list[float] = []
            method_dir = output_root / case["name"] / method
            rep_dirs = sorted(
                path
                for path in method_dir.glob("rep_*")
                if path.is_dir() and path.name.removeprefix("rep_").isdigit()
            )
            for rep_dir in rep_dirs:
                rep = int(rep_dir.name.removeprefix("rep_"))
                path = rep_dir / "summary.json"
                if not path.is_file():
                    continue
                summary = read_json(path)
                validate_summary(summary, cfg=cfg, case=case, method=method)
                metrics = training_metrics(summary, method)
                wall_ms = float(metrics["wall_ms"])
                throughput = float(metrics["throughput_per_s"])
                steps = int(metrics["optimizer_steps"])
                vals.append(throughput)
                run_rows.append(
                    {
                        "case": case["name"],
                        "physical_request_batch": case["physical_request_batch"],
                        "microbatches_per_update": case["microbatches_per_update"],
                        "effective_optimizer_batch": case["effective_optimizer_batch"],
                        "method": method,
                        "pipeline_schedule": method_schedule(method),
                        "rep": rep,
                        "throughput_per_s": throughput,
                        "wall_ms": wall_ms,
                        "optimizer_steps": steps,
                        "wall_ms_per_optimizer_step": wall_ms / steps if steps else "",
                    }
                )
            method_values[method] = vals
            if vals:
                mean = statistics.mean(vals)
                std = statistics.stdev(vals) if len(vals) > 1 else 0.0
                summary_rows.append(
                    {
                        "case": case["name"],
                        "physical_request_batch": case["physical_request_batch"],
                        "microbatches_per_update": case["microbatches_per_update"],
                        "effective_optimizer_batch": case["effective_optimizer_batch"],
                        "method": method,
                        "pipeline_schedule": method_schedule(method),
                        "n": len(vals),
                        "mean_throughput_per_s": mean,
                        "std_throughput_per_s": std,
                        "cv_percent": (100.0 * std / mean) if mean else 0.0,
                    }
                )

        if method_values.get("bpfree"):
            bp = statistics.mean(method_values["bpfree"])
            for exact_method in ("exactbp_gpipe", "exactbp_1f1b", "pipedream"):
                if method_values.get(exact_method):
                    ex = statistics.mean(method_values[exact_method])
                    summary_rows.append(
                        {
                            "case": case["name"],
                            "physical_request_batch": case["physical_request_batch"],
                            "microbatches_per_update": case["microbatches_per_update"],
                            "effective_optimizer_batch": case["effective_optimizer_batch"],
                            "method": f"ratio:bpfree/{exact_method}",
                            "pipeline_schedule": method_schedule(exact_method),
                            "n": min(len(method_values["bpfree"]), len(method_values[exact_method])),
                            "mean_throughput_per_s": bp / ex,
                            "std_throughput_per_s": "",
                            "cv_percent": "",
                        }
                    )

    output_root.mkdir(parents=True, exist_ok=True)
    run_fields = [
        "case", "physical_request_batch", "microbatches_per_update",
        "effective_optimizer_batch", "method", "pipeline_schedule", "rep",
        "throughput_per_s", "wall_ms", "optimizer_steps",
        "wall_ms_per_optimizer_step",
    ]
    with (output_root / "runs.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=run_fields)
        w.writeheader(); w.writerows(run_rows)

    summary_fields = [
        "case", "physical_request_batch", "microbatches_per_update",
        "effective_optimizer_batch", "method", "pipeline_schedule", "n",
        "mean_throughput_per_s", "std_throughput_per_s", "cv_percent",
    ]
    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader(); w.writerows(summary_rows)

    print("\n===== fair CPU-transport E4.2 results =====")
    for case in cases:
        rows = [r for r in summary_rows if r["case"] == case["name"] and not str(r["method"]).startswith("ratio:")]
        rendered = ", ".join(f"{r['method']}={float(r['mean_throughput_per_s']):.3f}" for r in rows)
        print(f"{case['name']}: {rendered}")
    print(f"Wrote {output_root / 'runs.csv'}")
    print(f"Wrote {output_root / 'summary.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--methods", help="Optional comma-separated method filter")
    parser.add_argument("--cases", help="Optional comma-separated case-name filter")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.resume and args.overwrite:
        raise ValueError("choose only one of --resume/--overwrite")
    repo_root = Path(__file__).resolve().parents[2]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    cfg = read_json(config_path)
    cfg["_resolved_config_path"] = str(config_path.resolve())
    cases = validate(cfg, repo_root)

    repetitions = args.repetitions if args.repetitions is not None else int(cfg["repetitions"])
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    method_filter = set(parse_csv(args.methods))
    case_filter = set(parse_csv(args.cases))
    if method_filter - SUPPORTED_METHODS:
        raise ValueError(f"unknown method filter: {sorted(method_filter - SUPPORTED_METHODS)}")

    selected_cases: list[dict[str, Any]] = []
    for case in cases:
        if case_filter and case["name"] not in case_filter:
            continue
        selected = dict(case)
        if method_filter:
            selected["methods"] = [m for m in case["methods"] if m in method_filter]
        if selected["methods"]:
            selected_cases.append(selected)
    if not selected_cases:
        raise ValueError("no cases/methods selected")

    _, execution_environment = build_execution_environment(cfg)
    manifest = {
        "config": cfg,
        "selected_cases": selected_cases,
        "repetitions": repetitions,
        "python": sys.executable,
        "execution_environment": execution_environment,
        "transport_fairness": {
            "shared_cpu_transport_module": "sg_exe_trainer.runtime.transport.cpu",
            "recv_prepost_depth": cfg["recv_prepost_depth"],
            "max_pending_send_bytes": cfg["max_pending_send_bytes"],
            "max_posted_recv_bytes": cfg["max_posted_recv_bytes"],
            "budget_scope": "shared_per_rank_across_all_logical_channels",
            "wall_time_reduction": "gloo_cpu_all_reduce_max",
        },
    }
    if not args.dry_run:
        write_json(repo_root / str(cfg["output_root"]) / "experiment_manifest.json", manifest)

    ordinal = 0
    for case_index, case in enumerate(selected_cases):
        for rep in range(repetitions):
            methods = list(case["methods"])
            offset = (case_index + rep) % len(methods)
            methods = methods[offset:] + methods[:offset]
            for method in methods:
                port = int(cfg.get("master_port_base", 29740)) + ordinal
                ordinal += 1
                run_one(
                    repo_root=repo_root,
                    cfg=cfg,
                    case=case,
                    method=method,
                    rep=rep,
                    dry_run=args.dry_run,
                    resume=args.resume,
                    overwrite=args.overwrite,
                    master_port=port,
                )
                if not args.dry_run:
                    time.sleep(float(cfg.get("cooldown_seconds", 0.0)))

    if not args.dry_run:
        aggregate(repo_root, cfg, cases)
        print("\nE4.2 fair CPU-transport sweep completed successfully.")


if __name__ == "__main__":
    main()
