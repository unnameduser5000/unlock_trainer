#!/usr/bin/env python3
"""Run short synchronized E4.4 traces and aggregate overhead categories."""
from __future__ import annotations

import argparse
import json
import shlex
import shutil
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

DEFAULT_CONFIG = (
    "experiments/e4_throughput/configs/"
    "e4_4_overhead_decomposition.json"
)
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
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def parse_csv(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def method_schedule(method: str) -> str:
    if method == "exactbp_gpipe":
        return "gpipe"
    if method == "exactbp_1f1b":
        return "1f1b"
    if method == "pipedream":
        return "pipedream"
    return "bpfree"


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


def resolve_cases(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for profile in cfg["profiles"]:
        for regime in cfg["regimes"]:
            physical = int(regime["physical_request_batch"])
            micro = int(regime["microbatches_per_update"])
            cases.append(
                {
                    "name": f"{profile['name']}__{regime['name']}",
                    "profile": str(profile["name"]),
                    "regime": str(regime["name"]),
                    "physical_request_batch": physical,
                    "microbatches_per_update": micro,
                    "effective_optimizer_batch": physical * micro,
                    "train_limit": int(regime["train_limit"]),
                    "methods": [str(item) for item in regime["methods"]],
                    "link_latency_ms": float(profile["link_latency_ms"]),
                    "link_bandwidth_mbps": float(profile["link_bandwidth_mbps"]),
                    "link_jitter_ms": float(profile["link_jitter_ms"]),
                }
            )
    return cases


def validate(cfg: dict[str, Any], repo_root: Path) -> list[dict[str, Any]]:
    for key in ("train_manifest", "eval_manifest"):
        path = repo_root / str(cfg[key])
        if not path.is_file():
            raise FileNotFoundError(path)
    gpu_count = int(cfg["gpu_count"])
    devices = [item.strip() for item in str(cfg["stage_devices"]).split(",") if item.strip()]
    if len(devices) != gpu_count:
        raise ValueError("gpu_count/stage_devices mismatch")
    cases = resolve_cases(cfg)
    for case in cases:
        if case["train_limit"] % case["effective_optimizer_batch"] != 0:
            raise ValueError(f"{case['name']}: train_limit not divisible by effective batch")
        unknown = set(case["methods"]) - SUPPORTED_METHODS
        if unknown:
            raise ValueError(f"{case['name']}: unknown methods {sorted(unknown)}")
        if "exactbp_1f1b" in case["methods"] and case["microbatches_per_update"] < gpu_count:
            raise ValueError(f"{case['name']}: 1F1B requires M >= stages")
    return cases


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
        "--train_limit", str(case["train_limit"]),
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
        "--enable_action_trace",
        "--sync_action_trace",
        "--action_trace_start_window", str(cfg.get("action_trace_start_window", 0)),
        "--recv_prepost_depth", str(cfg["recv_prepost_depth"]),
        "--max_pending_send_bytes", str(cfg["max_pending_send_bytes"]),
        "--max_posted_recv_bytes", str(cfg["max_posted_recv_bytes"]),
        "--link_latency_ms", str(case["link_latency_ms"]),
        "--link_bandwidth_mbps", str(case["link_bandwidth_mbps"]),
        "--link_jitter_ms", str(case["link_jitter_ms"]),
        "--link_emulation_seed", str(cfg["link_emulation_seed"]),
    ]
    if cfg.get("action_trace_end_window") is not None:
        runner_common.extend(
            [
                "--action_trace_end_window",
                str(cfg["action_trace_end_window"]),
            ]
        )
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
    return [
        sys.executable,
        "-m", "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={int(cfg['gpu_count'])}",
        "-m", "sg_exe_trainer.runtime.exactbp.cpu_runner",
        *exact_common,
        "--pipeline_schedule", method_schedule(method),
    ]


def validate_summary(
    summary: dict[str, Any], *, cfg: dict[str, Any], case: dict[str, Any], method: str
) -> None:
    metrics = training_metrics(summary, method)
    expected_steps = case["train_limit"] // case["effective_optimizer_batch"]
    checks = {
        "num_chunks": int(summary.get("num_chunks", -1)) == int(cfg["gpu_count"]),
        "physical_request_batch": int(summary.get("physical_request_batch", -1)) == case["physical_request_batch"],
        "completed_records": int(metrics["completed_records"]) == case["train_limit"],
        "optimizer_steps": int(metrics["optimizer_steps"]) == expected_steps,
    }
    if method == "pipedream":
        checks["microbatches"] = (
            int(summary.get("local_gradient_coalescing", -1))
            == case["microbatches_per_update"]
        )
        checks["runner"] = str(summary.get("runner", "")).startswith(
            "pipedream-continuous-1f1b"
        )
    else:
        checks["microbatches"] = (
            int(summary.get("microbatches", -1))
            == case["microbatches_per_update"]
        )
        checks["action_trace_enabled"] = (
            bool(summary.get("action_trace_enabled")) is True
        )
        checks["sync_action_trace"] = (
            bool(summary.get("sync_action_trace")) is True
        )
        checks["perf_minimal_metrics"] = (
            bool(summary.get("perf_minimal_metrics")) is True
        )
    if method not in {"bpfree", "pipedream"}:
        checks["pipeline_schedule"] = summary.get("pipeline_schedule") == method_schedule(method)
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"summary contract failed: {failed}")


def run_one(
    *, repo_root: Path, cfg: dict[str, Any], case: dict[str, Any], method: str,
    rep: int, dry_run: bool, resume: bool, overwrite: bool, master_port: int,
) -> None:
    output_root = repo_root / str(cfg["output_root"])
    output_dir = output_root / case["profile"] / case["regime"] / method / f"rep_{rep:02d}"
    summary_path = output_dir / "summary.json"
    if summary_path.is_file() and resume:
        try:
            validate_summary(read_json(summary_path), cfg=cfg, case=case, method=method)
            if len(list(output_dir.glob("train.stage*.actions.csv"))) != int(cfg["gpu_count"]):
                raise ValueError("missing stage action traces")
            print(f"[resume] valid: {summary_path}")
            return
        except Exception as exc:
            print(f"[resume] invalid, rerunning {summary_path}: {exc}")
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    elif output_dir.exists() and not dry_run and not resume:
        raise FileExistsError(f"output exists; use --resume or --overwrite: {output_dir}")

    command = build_command(
        repo_root=repo_root, cfg=cfg, case=case, method=method,
        output_dir=output_dir, master_port=master_port,
    )
    run_env, execution_environment = build_execution_environment(cfg)
    print(f"\n[env] {execution_environment}", flush=True)
    print("\n$", shlex.join(command), flush=True)
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
        "diagnostic_only": True,
        "note": "Synchronized traces perturb throughput; use E4.1-E4.3 for performance claims.",
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
            command, cwd=repo_root, stdout=log, stderr=subprocess.STDOUT,
            env=run_env, check=False,
        )
    metadata["ended_epoch_s"] = time.time()
    metadata["duration_s"] = metadata["ended_epoch_s"] - metadata["started_epoch_s"]
    metadata["returncode"] = proc.returncode
    metadata["artifacts"] = artifact_hashes(
        [
            output_dir / "run.log",
            output_dir / "summary.json",
            *output_dir.glob("train.stage*.actions.csv"),
        ],
        repo_root=repo_root,
    )
    write_json(metadata_path, metadata)
    if proc.returncode != 0:
        raise RuntimeError(f"run failed ({proc.returncode}); see {output_dir / 'run.log'}")
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    validate_summary(read_json(summary_path), cfg=cfg, case=case, method=method)
    traces = list(output_dir.glob("train.stage*.actions.csv"))
    if len(traces) != int(cfg["gpu_count"]):
        raise RuntimeError(f"expected {cfg['gpu_count']} traces, found {len(traces)}")
    print(f"[ok] {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--profiles")
    parser.add_argument("--regimes")
    parser.add_argument("--methods")
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
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
    profile_filter = set(parse_csv(args.profiles))
    regime_filter = set(parse_csv(args.regimes))
    method_filter = set(parse_csv(args.methods))
    selected = []
    for case in cases:
        if profile_filter and case["profile"] not in profile_filter:
            continue
        if regime_filter and case["regime"] not in regime_filter:
            continue
        copy = dict(case)
        if method_filter:
            copy["methods"] = [method for method in case["methods"] if method in method_filter]
        if copy["methods"]:
            selected.append(copy)
    if not selected:
        raise ValueError("no cases selected")
    repetitions = args.repetitions if args.repetitions is not None else int(cfg["repetitions"])

    _, execution_environment = build_execution_environment(cfg)
    if not args.dry_run:
        write_json(
            repo_root / str(cfg["output_root"]) / "experiment_manifest.json",
            {
                "config": cfg,
                "selected_cases": selected,
                "repetitions_requested": repetitions,
                "python": sys.executable,
                "execution_environment": execution_environment,
                "diagnostic_only": True,
            },
        )

    ordinal = 0
    for case_index, case in enumerate(selected):
        for rep in range(repetitions):
            methods = list(case["methods"])
            if (case_index + rep) % 2:
                methods.reverse()
            for method in methods:
                run_one(
                    repo_root=repo_root, cfg=cfg, case=case, method=method, rep=rep,
                    dry_run=args.dry_run, resume=args.resume, overwrite=args.overwrite,
                    master_port=int(cfg["master_port_base"]) + ordinal,
                )
                ordinal += 1

    if not args.dry_run and not args.skip_analysis:
        subprocess.run(
            [
                sys.executable,
                "experiments/e4_throughput/analyze_e4_4_traces.py",
                "--root", str(repo_root / str(cfg["output_root"])),
                "--output-dir", str(repo_root / str(cfg["analysis_output"])),
            ],
            cwd=repo_root,
            check=True,
        )
    if not args.dry_run:
        print("\nE4.4 overhead decomposition completed successfully.")


if __name__ == "__main__":
    main()
