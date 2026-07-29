#!/usr/bin/env python3
"""Capture low-perturbation CUDA timelines for the four E4 schedules."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from experiments.e4_throughput.provenance import (
    E4_RUNTIME_SOURCES,
    artifact_hashes,
    build_execution_environment,
    capture_provenance,
)
from experiments.e4_throughput.run_e4_2_cpu_transport import (
    build_command,
    read_json,
    training_metrics,
    validate,
    validate_summary,
)


EXPERIMENT_ID = "e4_gpu_timeline_nsys_v1"
METHODS = ("bpfree", "exactbp_gpipe", "exactbp_1f1b", "pipedream")
DEFAULT_CONFIG = (
    "experiments/e4_throughput/configs/e4_2a_batch_geometry.json"
)
DEFAULT_NSYS = (
    Path.home()
    / "tools/nsight-systems-2024.6.2/target-linux-x64/nsys"
)
TIMELINE_SOURCE_PATHS = (
    *E4_RUNTIME_SOURCES,
    "experiments/e4_throughput/gpu_timeline/run_nsys_timeline.py",
)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def parse_methods(raw: str) -> list[str]:
    methods = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"unknown methods: {unknown}")
    if not methods:
        raise ValueError("at least one method is required")
    return methods


def select_case(cases: list[dict[str, Any]], name: str) -> dict[str, Any]:
    matches = [case for case in cases if case["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"expected one case named {name!r}, found {len(matches)}")
    return matches[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--case", default="b8_m4")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--train-limit", type=int, default=512)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/e4_throughput/raw") / EXPERIMENT_ID,
    )
    parser.add_argument("--nsys", type=Path, default=DEFAULT_NSYS)
    parser.add_argument("--master-port-base", type=int, default=30780)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    config_path = config_path.resolve()

    if args.train_limit <= 0:
        raise ValueError("--train-limit must be positive")
    if not args.nsys.is_file():
        raise FileNotFoundError(args.nsys)

    cfg = read_json(config_path)
    cfg["_resolved_config_path"] = str(config_path)
    cfg["base_experiment_id"] = cfg["experiment_id"]
    cfg["experiment_id"] = EXPERIMENT_ID
    cfg["train_limit"] = args.train_limit
    cases = validate(cfg, repo_root)
    case = select_case(cases, args.case)
    methods = parse_methods(args.methods)
    unsupported = sorted(set(methods) - set(case["methods"]))
    if unsupported:
        raise ValueError(
            f"{case['name']} does not define methods: {unsupported}"
        )

    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    cfg["output_root"] = str(output_root.relative_to(repo_root))

    run_env, execution_environment = build_execution_environment(cfg)
    mkl_threads = str(cfg.get("mkl_num_threads", execution_environment["OMP_NUM_THREADS"]))
    run_env["MKL_NUM_THREADS"] = mkl_threads
    execution_environment["MKL_NUM_THREADS"] = mkl_threads

    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        write_json(
            output_root / "experiment_manifest.json",
            {
                "experiment_id": EXPERIMENT_ID,
                "base_config": str(config_path.relative_to(repo_root)),
                "case": case,
                "methods": methods,
                "train_limit": args.train_limit,
                "nsys": str(args.nsys.resolve()),
                "execution_environment": execution_environment,
            },
        )

    for index, method in enumerate(methods):
        method_dir = output_root / method
        if method_dir.exists() and args.overwrite and not args.dry_run:
            shutil.rmtree(method_dir)
        elif method_dir.exists() and not args.dry_run:
            raise FileExistsError(
                f"output exists; use --overwrite: {method_dir}"
            )

        method_command = build_command(
            repo_root=repo_root,
            cfg=cfg,
            case=case,
            method=method,
            output_dir=method_dir,
            master_port=args.master_port_base + index,
        )
        report_base = method_dir / "cuda_timeline"
        command = [
            str(args.nsys),
            "profile",
            "--trace=cuda,nvtx,osrt",
            "--sample=none",
            "--cpuctxsw=none",
            "--trace-fork-before-exec=false",
            "--export=sqlite",
            "--force-overwrite=true",
            "--output",
            str(report_base),
            *method_command,
        ]
        print("\n$", " ".join(command), flush=True)
        if args.dry_run:
            continue

        method_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = method_dir / "timeline_metadata.json"
        metadata: dict[str, Any] = {
            "experiment_id": EXPERIMENT_ID,
            "method": method,
            "case": case,
            "train_limit": args.train_limit,
            "action_trace_enabled": False,
            "sync_action_trace": False,
            "execution_environment": execution_environment,
            "nsys_command": command,
            "started_epoch_s": time.time(),
            "provenance": capture_provenance(
                repo_root=repo_root,
                config_path=config_path,
                cfg=cfg,
                command=command,
                execution_environment=execution_environment,
                source_paths=TIMELINE_SOURCE_PATHS,
            ),
        }
        write_json(metadata_path, metadata)

        with (method_dir / "timeline.log").open(
            "w", encoding="utf-8"
        ) as log:
            completed = subprocess.run(
                command,
                cwd=repo_root,
                env=run_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        metadata["returncode"] = completed.returncode
        metadata["ended_epoch_s"] = time.time()
        metadata["duration_s"] = (
            metadata["ended_epoch_s"] - metadata["started_epoch_s"]
        )
        if completed.returncode != 0:
            write_json(metadata_path, metadata)
            raise RuntimeError(
                f"{method} failed; see {method_dir / 'timeline.log'}"
            )

        summary_path = method_dir / "summary.json"
        sqlite_path = report_base.with_suffix(".sqlite")
        report_path = report_base.with_suffix(".nsys-rep")
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        if not sqlite_path.is_file():
            raise FileNotFoundError(sqlite_path)
        if not report_path.is_file():
            raise FileNotFoundError(report_path)

        summary = read_json(summary_path)
        validate_summary(
            summary,
            cfg=cfg,
            case=case,
            method=method,
        )
        metadata["training"] = training_metrics(summary, method)
        metadata["trace_sqlite"] = str(sqlite_path)
        metadata["trace_report"] = str(report_path)
        metadata["artifacts"] = artifact_hashes(
            [
                method_dir / "timeline.log",
                summary_path,
                sqlite_path,
                report_path,
            ],
            repo_root=repo_root,
        )
        write_json(metadata_path, metadata)
        print(f"[ok] {method}: {metadata['training']}", flush=True)


if __name__ == "__main__":
    main()
