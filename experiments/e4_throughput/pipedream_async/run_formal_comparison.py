#!/usr/bin/env python3
"""Run the balanced E4 BP-free/GPipe/1F1B/PipeDream comparison."""
from __future__ import annotations

import argparse
import csv
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


DEFAULT_CONFIG = (
    "experiments/e4_throughput/pipedream_async/configs/comparison.json"
)
SUPPORTED_METHODS = {
    "bpfree",
    "exactbp_gpipe",
    "exactbp_1f1b",
    "pipedream",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def train_phase(summary: dict[str, Any]) -> dict[str, Any]:
    for phase in summary.get("phases", []):
        if isinstance(phase, dict) and phase.get("phase") == "train":
            return phase
    raise ValueError("summary has no train phase")


def method_orders(base: list[str], repetitions: int) -> list[list[str]]:
    return [
        base[offset % len(base):] + base[:offset % len(base)]
        for offset in range(repetitions)
    ]


def validate_config(cfg: dict[str, Any], repo_root: Path) -> None:
    methods = [str(item) for item in cfg["method_order"]]
    if len(methods) != len(set(methods)):
        raise ValueError("method_order contains duplicates")
    unknown = set(methods) - SUPPORTED_METHODS
    if unknown:
        raise ValueError(f"unsupported methods: {sorted(unknown)}")
    if set(methods) != SUPPORTED_METHODS:
        raise ValueError("formal comparison requires all four methods")

    devices = [item.strip() for item in cfg["stage_devices"].split(",")]
    if len(devices) != int(cfg["num_chunks"]):
        raise ValueError("num_chunks/stage_devices mismatch")
    if int(cfg["microbatches_per_update"]) < int(cfg["num_chunks"]):
        raise ValueError("1F1B and PipeDream require m >= pipeline stages")

    effective_batch = (
        int(cfg["physical_request_batch"])
        * int(cfg["microbatches_per_update"])
    )
    if int(cfg["train_limit"]) % effective_batch:
        raise ValueError("train_limit must be divisible by effective batch")
    for key in ("train_manifest", "eval_manifest"):
        if not (repo_root / str(cfg[key])).is_file():
            raise FileNotFoundError(repo_root / str(cfg[key]))


def common_args(
    cfg: dict[str, Any], repo_root: Path, output_dir: Path
) -> list[str]:
    return [
        "--model_name", str(cfg["model_name"]),
        "--train_manifest", str(repo_root / str(cfg["train_manifest"])),
        "--output_dir", str(output_dir),
        "--num_chunks", str(cfg["num_chunks"]),
        "--stage_devices", str(cfg["stage_devices"]),
        "--train_limit", str(cfg["train_limit"]),
        "--train_epochs", str(cfg["train_epochs"]),
        "--physical_batch_size", str(cfg["physical_request_batch"]),
        "--gradient_accumulation_steps",
        str(cfg["microbatches_per_update"]),
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
    ]


def transport_args(cfg: dict[str, Any]) -> list[str]:
    return [
        "--recv_prepost_depth", str(cfg["recv_prepost_depth"]),
        "--max_pending_send_bytes", str(cfg["max_pending_send_bytes"]),
        "--max_posted_recv_bytes", str(cfg["max_posted_recv_bytes"]),
    ]


def build_command(
    *,
    cfg: dict[str, Any],
    repo_root: Path,
    method: str,
    output_dir: Path,
    master_port: int,
) -> list[str]:
    common = common_args(cfg, repo_root, output_dir)
    transport = transport_args(cfg)
    exact_common = [
        *common,
        "--eval_manifest", str(repo_root / str(cfg["eval_manifest"])),
        "--eval_limit", "1",
        "--progress_interval", "0",
        "--skip_eval_before",
        "--skip_eval_after",
        "--no-track_activation_memory",
        "--perf_minimal_metrics",
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
            *transport,
        ]

    if method in {"exactbp_gpipe", "exactbp_1f1b"}:
        schedule = "gpipe" if method == "exactbp_gpipe" else "1f1b"
        return [
            sys.executable,
            "-m", "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={cfg['num_chunks']}",
            f"--master_port={master_port}",
            "-m", "sg_exe_trainer.runtime.exactbp.cpu_runner",
            *exact_common,
            "--pipeline_schedule", schedule,
            "--save_trainable_state",
            *transport,
        ]

    if method == "pipedream":
        return [
            sys.executable,
            "-m", "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={cfg['num_chunks']}",
            f"--master_port={master_port}",
            "-m", "experiments.shared.baselines.pipedream_cpu",
            *common,
            "--perf_minimal_metrics",
            *transport,
        ]

    raise ValueError(method)


def budget_rows(summary: dict[str, Any], method: str) -> list[dict[str, Any]]:
    if method == "pipedream":
        return [row["transport_budget"] for row in summary["by_rank"]]
    rows = summary.get("transport_budget_by_rank")
    if not isinstance(rows, list):
        rows = train_phase(summary).get("transport_budget_by_rank")
    if not isinstance(rows, list):
        raise ValueError("summary has no per-rank transport budget")
    return rows


def validate_summary(
    summary: dict[str, Any], cfg: dict[str, Any], method: str
) -> None:
    physical = int(cfg["physical_request_batch"])
    micro = int(cfg["microbatches_per_update"])
    records = int(cfg["train_limit"])
    expected_steps = records // (physical * micro)
    expected_batches = records // physical

    if method == "pipedream":
        checks = {
            "completed_records": int(summary.get("completed_records", -1))
            == records,
            "optimizer_steps": int(
                summary.get("optimizer_steps_per_stage", -1)
            ) == expected_steps,
            "physical_batch": int(
                summary.get("physical_request_batch", -1)
            ) == physical,
            "effective_batch": int(
                summary.get("effective_optimizer_batch", -1)
            ) == physical * micro,
            "rank_count": len(summary.get("by_rank", []))
            == int(cfg["num_chunks"]),
        }
        for rank in summary.get("by_rank", []):
            stage = int(rank["stage_id"])
            checks[f"stage{stage}_backwards"] = int(
                rank.get("local_backward_count", -1)
            ) == expected_batches
            checks[f"stage{stage}_steps"] = int(
                rank.get("local_optimizer_steps", -1)
            ) == expected_steps
            checks[f"stage{stage}_missing_gradients"] = int(
                rank.get("missing_snapshot_gradients", -1)
            ) == 0
    else:
        phase = train_phase(summary)
        checks = {
            "completed_records": int(phase.get("completed_records", -1))
            == records,
            "optimizer_steps": int(phase.get("optimizer_steps", -1))
            == expected_steps,
            "physical_batch": int(
                summary.get("physical_request_batch", -1)
            ) == physical,
            "effective_batch": int(
                summary.get("effective_optimizer_batch", -1)
            ) == physical * micro,
            "perf_minimal_metrics": bool(
                summary.get("perf_minimal_metrics")
            ),
        }

    rows = budget_rows(summary, method)
    checks["transport_rank_count"] = len(rows) == int(cfg["num_chunks"])
    for rank, row in enumerate(rows):
        checks[f"rank{rank}_send_balance"] = int(
            row.get("pending_send_bytes_at_end", -1)
        ) == 0
        checks[f"rank{rank}_recv_balance"] = int(
            row.get("posted_recv_bytes_at_end", -1)
        ) == 0
        checks[f"rank{rank}_send_cap"] = int(
            row.get("peak_pending_send_bytes", -1)
        ) <= int(cfg["max_pending_send_bytes"])
        checks[f"rank{rank}_recv_cap"] = int(
            row.get("peak_posted_recv_bytes", -1)
        ) <= int(cfg["max_posted_recv_bytes"])

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"summary contract failed: {failed}")


def metrics(summary: dict[str, Any], method: str) -> dict[str, Any]:
    if method == "pipedream":
        throughput = float(summary["throughput_per_s"])
        wall_ms = float(summary["wall_ms"])
        optimizer_steps = int(summary["optimizer_steps_per_stage"])
        ranks = summary["by_rank"]
        peak_activation_stash = max(
            int(row["peak_activation_stash"]) for row in ranks
        )
        peak_live_versions = max(
            int(row["peak_live_weight_versions"]) for row in ranks
        )
        peak_weight_stash_bytes = max(
            int(row["peak_weight_stash_bytes"]) for row in ranks
        )
        mean_version_lag = max(
            float(row["mean_backward_version_lag"]) for row in ranks
        )
        max_version_lag = max(
            int(row["max_backward_version_lag"]) for row in ranks
        )
    else:
        phase = train_phase(summary)
        throughput = float(phase["throughput_per_s"])
        wall_ms = float(phase["wall_ms"])
        optimizer_steps = int(phase["optimizer_steps"])
        peak_activation_stash = ""
        peak_live_versions = ""
        peak_weight_stash_bytes = ""
        mean_version_lag = ""
        max_version_lag = ""

    rows = budget_rows(summary, method)
    return {
        "throughput_per_s": throughput,
        "wall_ms": wall_ms,
        "optimizer_steps": optimizer_steps,
        "total_send_bytes": sum(int(row["total_send_bytes"]) for row in rows),
        "total_recv_bytes": sum(int(row["total_recv_bytes"]) for row in rows),
        "peak_pending_send_bytes": max(
            int(row["peak_pending_send_bytes"]) for row in rows
        ),
        "peak_posted_recv_bytes": max(
            int(row["peak_posted_recv_bytes"]) for row in rows
        ),
        "peak_activation_stash": peak_activation_stash,
        "peak_live_weight_versions": peak_live_versions,
        "peak_weight_stash_bytes": peak_weight_stash_bytes,
        "max_stage_mean_version_lag": mean_version_lag,
        "max_backward_version_lag": max_version_lag,
    }


def aggregate(output_root: Path) -> None:
    rows: list[dict[str, Any]] = []
    for metadata_path in sorted(output_root.glob("rep_*/*/run_metadata.json")):
        metadata = read_json(metadata_path)
        summary_path = metadata_path.parent / "summary.json"
        if not summary_path.is_file() or metadata.get("returncode") != 0:
            continue
        method = str(metadata["method"])
        row = {
            "rep": int(metadata["rep"]),
            "order_position": int(metadata["order_position"]),
            "method": method,
            **metrics(read_json(summary_path), method),
        }
        rows.append(row)

    if not rows:
        return
    fieldnames = list(rows[0])
    with (output_root / "runs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows: list[dict[str, Any]] = []
    for method in sorted({str(row["method"]) for row in rows}):
        values = [
            float(row["throughput_per_s"])
            for row in rows
            if row["method"] == method
        ]
        summary_rows.append(
            {
                "method": method,
                "repetitions": len(values),
                "throughput_mean": statistics.mean(values),
                "throughput_stdev": (
                    statistics.stdev(values) if len(values) > 1 else 0.0
                ),
                "throughput_min": min(values),
                "throughput_max": max(values),
            }
        )
    with (output_root / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    write_json(output_root / "summary.json", summary_rows)


def run_one(
    *,
    cfg: dict[str, Any],
    repo_root: Path,
    output_root: Path,
    rep: int,
    order_position: int,
    method: str,
    resume: bool,
    dry_run: bool,
) -> None:
    output_dir = output_root / f"rep_{rep:02d}" / (
        f"{order_position:02d}_{method}"
    )
    summary_path = output_dir / "summary.json"
    if summary_path.is_file() and resume:
        validate_summary(read_json(summary_path), cfg, method)
        print(f"[resume] {summary_path}", flush=True)
        return
    if output_dir.exists() and not dry_run:
        raise FileExistsError(output_dir)

    master_port = 30100 + rep * 10 + order_position
    command = build_command(
        cfg=cfg,
        repo_root=repo_root,
        method=method,
        output_dir=output_dir,
        master_port=master_port,
    )
    print("$", shlex.join(command), flush=True)
    if dry_run:
        return

    output_dir.mkdir(parents=True)
    metadata = {
        "experiment_id": cfg["experiment_id"],
        "rep": rep,
        "order_position": order_position,
        "method": method,
        "command": command,
        "started_epoch_s": time.time(),
    }
    write_json(output_dir / "run_metadata.json", metadata)
    env = os.environ.copy()
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "TRANSFORMERS_VERBOSITY": "error",
            "OMP_NUM_THREADS": "4",
        }
    )
    with (output_dir / "run.log").open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            command,
            cwd=repo_root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    metadata["returncode"] = proc.returncode
    metadata["ended_epoch_s"] = time.time()
    write_json(output_dir / "run_metadata.json", metadata)
    if proc.returncode:
        raise RuntimeError(f"{method} failed; see {output_dir / 'run.log'}")
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    validate_summary(read_json(summary_path), cfg, method)
    print(f"[ok] {summary_path}", flush=True)


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

    repetitions = int(cfg["repetitions"])
    orders = method_orders([str(x) for x in cfg["method_order"]], repetitions)
    manifest = {
        "config": cfg,
        "effective_optimizer_batch": int(cfg["physical_request_batch"])
        * int(cfg["microbatches_per_update"]),
        "optimizer_steps_per_method": int(cfg["train_limit"])
        // (
            int(cfg["physical_request_batch"])
            * int(cfg["microbatches_per_update"])
        ),
        "method_orders": orders,
        "python": sys.executable,
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2), flush=True)
    else:
        write_json(output_root / "experiment_manifest.json", manifest)

    for rep, order in enumerate(orders):
        for position, method in enumerate(order):
            run_one(
                cfg=cfg,
                repo_root=repo_root,
                output_root=output_root,
                rep=rep,
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
