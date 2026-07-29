#!/usr/bin/env python3
"""Run matched BP-free vs exact-BP no-recovery sweeps.

This launcher supports two lightweight experiment families:

1. `grid`: a small stage x outage-window matrix to inspect no-recovery trends.
2. `sample-size`: a fault-free sample-size sweep to find where exact-BP starts
   to reliably outperform BP-free on post-training accuracy.

It intentionally uses one stable model/data setup and writes both raw run roots
and a normalized summary CSV under one output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
BPFREE_MODULE = "sg_exe_trainer.runtime.bpfree.orchestrated_runtime"
F1B_SCRIPT = REPO_ROOT / "src" / "sg_exe_trainer" / "runtime" / "exactbp" / "distributed_runtime.py"

ABSOLUTE_WINDOWS: dict[str, tuple[int, int]] = {
    "early": (64, 192),
    "middle": (192, 320),
    "late": (320, 448),
}
PROPORTIONAL_WINDOW_FRACTIONS: dict[str, tuple[float, float]] = {
    "early": (0.125, 0.375),
    "middle": (0.375, 0.625),
    "late": (0.625, 0.875),
}
GRID_STAGES = [0, 1, 2]


@dataclass(frozen=True)
class OfflineCase:
    name: str
    stage_id: int | None
    start_seq: int | None
    end_seq: int | None


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
        }
    )


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Wrote {path}")


def parse_csv_ints(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one integer")
    return values


def parse_windows(raw: str, *, available: dict[str, tuple[int, int]]) -> list[str]:
    if raw.strip() == "default":
        return list(available.keys())
    names = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [name for name in names if name not in available]
    if unknown:
        raise ValueError(f"unknown windows {unknown}; available={sorted(available)}")
    return names


def quantize_window_boundary(value: float, *, batch_size: int, use_ceil: bool) -> int:
    scaled = value / batch_size
    rounded = math.ceil(scaled) if use_ceil else math.floor(scaled)
    return max(0, int(rounded) * batch_size)


def resolve_windows(*, train_limit: int, batch_size: int, mode: str) -> dict[str, tuple[int, int]]:
    if mode == "absolute":
        return dict(ABSOLUTE_WINDOWS)
    if mode != "proportional":
        raise ValueError(f"unsupported window mode: {mode}")
    windows: dict[str, tuple[int, int]] = {}
    for name, (start_frac, end_frac) in PROPORTIONAL_WINDOW_FRACTIONS.items():
        start_seq = quantize_window_boundary(start_frac * train_limit, batch_size=batch_size, use_ceil=False)
        end_seq = quantize_window_boundary(end_frac * train_limit, batch_size=batch_size, use_ceil=True)
        start_seq = max(0, min(start_seq, train_limit))
        end_seq = max(0, min(end_seq, train_limit))
        if end_seq <= start_seq:
            end_seq = min(train_limit, start_seq + batch_size)
        if end_seq <= start_seq:
            raise ValueError(f"resolved empty window for {name}: start={start_seq}, end={end_seq}")
        windows[name] = (start_seq, end_seq)
    return windows


def default_offline_cases(
    window_names: list[str],
    stage_ids: list[int],
    *,
    windows: dict[str, tuple[int, int]],
) -> list[OfflineCase]:
    cases = [OfflineCase(name="fault_free", stage_id=None, start_seq=None, end_seq=None)]
    for stage_id in stage_ids:
        for window_name in window_names:
            start_seq, end_seq = windows[window_name]
            cases.append(
                OfflineCase(
                    name=f"stage{stage_id}_{window_name}",
                    stage_id=stage_id,
                    start_seq=start_seq,
                    end_seq=end_seq,
                )
            )
    return cases


def phase_by_name_map(summary: dict[str, Any], key: str) -> dict[str, dict[str, Any]]:
    return {str(item.get("phase")): item for item in summary.get(key, [])}


def torchrun_prefix(num_chunks: int) -> list[str]:
    return [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={num_chunks}"]


def common_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    env.setdefault("TQDM_DISABLE", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("TERM", "dumb")
    return env


def run_command(cmd: list[str], *, log_path: Path) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write("$ " + shlex.join(cmd) + "\n")
        log_handle.flush()
        process = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=common_env(),
            check=False,
        )
    elapsed_s = time.perf_counter() - started
    if process.returncode != 0:
        raise RuntimeError(f"command failed with exit code {process.returncode}; see {log_path}")
    return elapsed_s


def bpfree_command(args: argparse.Namespace, output_dir: Path, case: OfflineCase) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        BPFREE_MODULE,
        "--model_name",
        args.model_name,
        "--manifest",
        str(args.train_manifest),
        "--eval_manifest",
        str(args.eval_manifest),
        "--output_dir",
        str(output_dir),
        "--num_chunks",
        str(args.num_chunks),
        "--stage_devices",
        args.stage_devices,
        "--limit",
        str(args.train_limit),
        "--eval_limit",
        str(args.eval_limit),
        "--max_inflight",
        str(args.max_inflight),
        "--scheduler_policy",
        "fifo",
        "--recovery_policy",
        "skip",
        "--max_attempts",
        "1",
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation_steps),
        "--belief_transport_mode",
        args.belief_transport_mode,
        "--trainable_mode",
        "lora",
        "--dtype",
        args.dtype,
        "--optimizer",
        args.optimizer,
        "--lora_rank",
        str(args.lora_rank),
        "--lora_alpha",
        str(args.lora_alpha),
        "--lora_targets",
        args.lora_targets,
        "--lora_init_std",
        str(args.lora_init_std),
        "--seed",
        str(args.seed),
        "--progress_interval",
        str(args.progress_interval),
    ]
    if args.learning_rate is not None:
        cmd.extend(["--learning_rate", str(args.learning_rate)])
    if args.grad_clip is not None:
        cmd.extend(["--grad_clip", str(args.grad_clip)])
    if case.stage_id is not None:
        assert case.start_seq is not None and case.end_seq is not None
        cmd.extend(
            [
                "--offline_stage",
                str(case.stage_id),
                "--offline_start_seq",
                str(case.start_seq),
                "--offline_end_seq",
                str(case.end_seq),
            ]
        )
    return cmd


def one_f1b_command(
    args: argparse.Namespace,
    output_dir: Path,
    case: OfflineCase,
    *,
    num_chunks: int,
    stage_devices: str,
    batch_size: int,
    microbatches: int,
    recovery_policy: str,
) -> list[str]:
    cmd = torchrun_prefix(num_chunks)
    cmd.extend(
        [
            str(F1B_SCRIPT),
            "--model_name",
            args.model_name,
            "--train_manifest",
            str(args.train_manifest),
            "--eval_manifest",
            str(args.eval_manifest),
            "--output_dir",
            str(output_dir),
            "--num_chunks",
            str(num_chunks),
            "--stage_devices",
            stage_devices,
            "--train_limit",
            str(args.train_limit),
            "--eval_limit",
            str(args.eval_limit),
            "--train_epochs",
            "1",
            "--microbatches",
            str(microbatches),
            "--batch_size",
            str(batch_size),
            "--dtype",
            args.dtype,
            "--optimizer",
            args.optimizer,
            "--label_smoothing",
            "0.0",
            "--lora_rank",
            str(args.lora_rank),
            "--lora_alpha",
            str(args.lora_alpha),
            "--lora_targets",
            args.lora_targets,
            "--lora_init_std",
            str(args.lora_init_std),
            "--seed",
            str(args.seed),
            "--progress_interval",
            str(args.progress_interval),
            "--recovery_policy",
            recovery_policy,
            "--grad_clip",
            str(args.grad_clip),
            "--skip_eval_before",
        ]
    )
    if args.learning_rate is not None:
        cmd.extend(["--learning_rate", str(args.learning_rate)])
    if case.stage_id is not None:
        assert case.start_seq is not None and case.end_seq is not None
        cmd.extend(
            [
                "--offline_stage",
                str(case.stage_id),
                "--offline_start_seq",
                str(case.start_seq),
                "--offline_end_seq",
                str(case.end_seq),
            ]
        )
    return cmd


def f1b_command(args: argparse.Namespace, output_dir: Path, case: OfflineCase) -> list[str]:
    return one_f1b_command(
        args,
        output_dir,
        case,
        num_chunks=args.num_chunks,
        stage_devices=args.stage_devices,
        batch_size=args.batch_size,
        microbatches=args.microbatches,
        recovery_policy="strict_skip",
    )


def full_bp_command(args: argparse.Namespace, output_dir: Path, case: OfflineCase) -> list[str]:
    if case.stage_id is not None:
        raise ValueError("Full-BP sample-size scan only supports fault-free runs.")
    return one_f1b_command(
        args,
        output_dir,
        case,
        num_chunks=1,
        stage_devices=args.full_bp_device,
        batch_size=args.batch_size,
        microbatches=1,
        recovery_policy="strict_skip",
    )


def bpfree_train_phase(summary: dict[str, Any]) -> dict[str, Any]:
    return phase_by_name_map(summary, "phase_summaries").get("train", {})


def bpfree_eval_phase(summary: dict[str, Any]) -> dict[str, Any]:
    return phase_by_name_map(summary, "phase_summaries").get("eval", {})


def f1b_train_phase(summary: dict[str, Any]) -> dict[str, Any]:
    return phase_by_name_map(summary, "phases").get("train", {})


def f1b_eval_phase(summary: dict[str, Any]) -> dict[str, Any]:
    return phase_by_name_map(summary, "phases").get("eval_after", {})


def summarize_bpfree(summary: dict[str, Any], *, method: str, case: OfflineCase, elapsed_s: float, output_dir: Path) -> dict[str, Any]:
    train = bpfree_train_phase(summary)
    eval_phase = bpfree_eval_phase(summary)
    retained = train.get("retained_progress", summary.get("retained_progress", {}))
    per_stage = retained.get("per_stage", {})
    return {
        "method": method,
        "case": case.name,
        "stage_id": -1 if case.stage_id is None else int(case.stage_id),
        "window_name": "fault_free" if case.stage_id is None else case.name.split("_", 1)[1],
        "offline_start_seq": "" if case.start_seq is None else int(case.start_seq),
        "offline_end_seq": "" if case.end_seq is None else int(case.end_seq),
        "output_dir": str(output_dir),
        "elapsed_s": elapsed_s,
        "train_limit": int(train.get("records", summary.get("train_limit", 0))),
        "eval_limit": int(eval_phase.get("records", summary.get("eval_limit", 0))),
        "effective_optimizer_batch": int(summary.get("effective_optimizer_batch", 0)),
        "train_completed": int(train.get("completed", 0)),
        "train_failed": int(train.get("failed", 0)),
        "train_choice_accuracy": float(train.get("choice_accuracy", 0.0)),
        "train_avg_loss": float(train.get("avg_loss", 0.0)),
        "train_throughput_per_s": float(train.get("throughput_per_s", 0.0)),
        "eval_choice_accuracy": float(eval_phase.get("choice_accuracy", 0.0)),
        "eval_avg_loss": float(eval_phase.get("avg_loss", 0.0)),
        "eval_completed": int(eval_phase.get("completed", 0)),
        "retained_failed_requests": int(retained.get("failed_requests", 0)),
        "retained_completed_requests": int(retained.get("completed_requests", 0)),
        "retained_stage0_updates_on_failed": int(per_stage.get("0", {}).get("retained_updates_on_failed_requests", 0)),
        "retained_stage1_updates_on_failed": int(per_stage.get("1", {}).get("retained_updates_on_failed_requests", 0)),
        "retained_stage2_updates_on_failed": int(per_stage.get("2", {}).get("retained_updates_on_failed_requests", 0)),
        "skipped_records": 0,
        "skipped_batches": 0,
    }


def summarize_f1b(summary: dict[str, Any], *, method: str, case: OfflineCase, elapsed_s: float, output_dir: Path) -> dict[str, Any]:
    train = f1b_train_phase(summary)
    eval_phase = f1b_eval_phase(summary)
    return {
        "method": method,
        "case": case.name,
        "stage_id": -1 if case.stage_id is None else int(case.stage_id),
        "window_name": "fault_free" if case.stage_id is None else case.name.split("_", 1)[1],
        "offline_start_seq": "" if case.start_seq is None else int(case.start_seq),
        "offline_end_seq": "" if case.end_seq is None else int(case.end_seq),
        "output_dir": str(output_dir),
        "elapsed_s": elapsed_s,
        "train_limit": int(summary.get("train_records", 0)),
        "eval_limit": int(summary.get("eval_records", 0)),
        "effective_optimizer_batch": int(summary.get("effective_optimizer_batch", 0)),
        "train_completed": int(train.get("completed_records", 0)),
        "train_failed": 0,
        "train_choice_accuracy": float(train.get("choice_accuracy", 0.0)),
        "train_avg_loss": float(train.get("avg_loss", 0.0)),
        "train_throughput_per_s": float(train.get("throughput_per_s", 0.0)),
        "eval_choice_accuracy": float(eval_phase.get("choice_accuracy", 0.0)),
        "eval_avg_loss": float(eval_phase.get("avg_loss", 0.0)),
        "eval_completed": int(eval_phase.get("completed", eval_phase.get("choice_count", 0))),
        "retained_failed_requests": 0,
        "retained_completed_requests": 0,
        "retained_stage0_updates_on_failed": 0,
        "retained_stage1_updates_on_failed": 0,
        "retained_stage2_updates_on_failed": 0,
        "skipped_records": int(train.get("skipped_records", 0)),
        "skipped_batches": int(train.get("skipped_batches", 0)),
    }


def run_tag(args: argparse.Namespace) -> str:
    return f"train{args.train_limit}_eval{args.eval_limit}"


def run_bpfree_case(args: argparse.Namespace, root: Path, case: OfflineCase) -> dict[str, Any]:
    output_dir = root / run_tag(args) / f"bpfree_{case.name}_seed{args.seed}"
    summary_path = output_dir / "scheduler_summary.json"
    if summary_path.is_file() and not args.force:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return summarize_bpfree(summary, method="BP-free", case=case, elapsed_s=0.0, output_dir=output_dir)
    elapsed_s = run_command(bpfree_command(args, output_dir, case), log_path=output_dir / "run.log")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return summarize_bpfree(summary, method="BP-free", case=case, elapsed_s=elapsed_s, output_dir=output_dir)


def run_f1b_case(args: argparse.Namespace, root: Path, case: OfflineCase) -> dict[str, Any]:
    output_dir = root / run_tag(args) / f"1f1b_{case.name}_seed{args.seed}"
    summary_path = output_dir / "summary.json"
    if summary_path.is_file() and not args.force:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return summarize_f1b(summary, method="1F1B", case=case, elapsed_s=0.0, output_dir=output_dir)
    elapsed_s = run_command(f1b_command(args, output_dir, case), log_path=output_dir / "run.log")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return summarize_f1b(summary, method="1F1B", case=case, elapsed_s=elapsed_s, output_dir=output_dir)


def run_full_bp_case(args: argparse.Namespace, root: Path, case: OfflineCase) -> dict[str, Any]:
    output_dir = root / run_tag(args) / f"full_bp_{case.name}_seed{args.seed}"
    summary_path = output_dir / "summary.json"
    if summary_path.is_file() and not args.force:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return summarize_f1b(summary, method="Full BP", case=case, elapsed_s=0.0, output_dir=output_dir)
    elapsed_s = run_command(full_bp_command(args, output_dir, case), log_path=output_dir / "run.log")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return summarize_f1b(summary, method="Full BP", case=case, elapsed_s=elapsed_s, output_dir=output_dir)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def add_fault_free_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baselines: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["case"] == "fault_free":
            baselines[str(row["method"])] = row
    enriched: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        baseline = baselines.get(str(row["method"]))
        if baseline is None:
            item["eval_accuracy_delta_vs_fault_free"] = ""
            item["eval_loss_delta_vs_fault_free"] = ""
        else:
            item["eval_accuracy_delta_vs_fault_free"] = float(row["eval_choice_accuracy"]) - float(
                baseline["eval_choice_accuracy"]
            )
            item["eval_loss_delta_vs_fault_free"] = float(row["eval_avg_loss"]) - float(baseline["eval_avg_loss"])
        enriched.append(item)
    return enriched


def method_case_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(row["method"]), str(row["case"])): row for row in rows}


def plot_grid_accuracy(rows: list[dict[str, Any]], path: Path, *, window_names: list[str]) -> None:
    configure_style()
    method_lookup = method_case_lookup(rows)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))
    panels = [
        ("BP-free", "eval_choice_accuracy", "BP-free eval accuracy"),
        ("1F1B", "eval_choice_accuracy", "1F1B eval accuracy"),
        ("gap", "gap", "1F1B - BP-free"),
    ]
    for axis, (method, field, title) in zip(axes, panels):
        matrix = np.zeros((len(GRID_STAGES), len(window_names)), dtype=float)
        for i, stage_id in enumerate(GRID_STAGES):
            for j, window_name in enumerate(window_names):
                case_name = f"stage{stage_id}_{window_name}"
                if method == "gap":
                    f1b = float(method_lookup[("1F1B", case_name)]["eval_choice_accuracy"])
                    bpfree = float(method_lookup[("BP-free", case_name)]["eval_choice_accuracy"])
                    value = f1b - bpfree
                else:
                    value = float(method_lookup[(method, case_name)][field])
                matrix[i, j] = value
        cmap = "viridis" if method != "gap" else "coolwarm"
        image = axis.imshow(matrix, cmap=cmap, aspect="auto")
        axis.set_xticks(np.arange(len(window_names)), window_names)
        axis.set_yticks(np.arange(len(GRID_STAGES)), [f"stage {stage_id}" for stage_id in GRID_STAGES])
        axis.set_title(title)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                axis.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", color="white", fontsize=9)
        plt.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    savefig(path)


def plot_grid_drop(rows: list[dict[str, Any]], path: Path, *, window_names: list[str]) -> None:
    configure_style()
    method_lookup = method_case_lookup(rows)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    for axis, method in zip(axes, ["BP-free", "1F1B"]):
        matrix = np.zeros((len(GRID_STAGES), len(window_names)), dtype=float)
        baseline = float(method_lookup[(method, "fault_free")]["eval_choice_accuracy"])
        for i, stage_id in enumerate(GRID_STAGES):
            for j, window_name in enumerate(window_names):
                case_name = f"stage{stage_id}_{window_name}"
                value = float(method_lookup[(method, case_name)]["eval_choice_accuracy"]) - baseline
                matrix[i, j] = value
        image = axis.imshow(matrix, cmap="coolwarm", aspect="auto")
        axis.set_xticks(np.arange(len(window_names)), window_names)
        axis.set_yticks(np.arange(len(GRID_STAGES)), [f"stage {stage_id}" for stage_id in GRID_STAGES])
        axis.set_title(f"{method} accuracy delta vs fault-free")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                axis.text(j, i, f"{matrix[i, j]:+.3f}", ha="center", va="center", color="white", fontsize=9)
        plt.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    savefig(path)


def plot_sample_size_scan(rows: list[dict[str, Any]], path: Path) -> None:
    configure_style()
    sizes = sorted({int(row["train_limit"]) for row in rows})
    methods = sorted({str(row["method"]) for row in rows})
    palette = {
        "BP-free": ("#2f6fdd", "o"),
        "1F1B": ("#e16b3d", "s"),
        "Full BP": ("#5b9a7b", "^"),
    }
    by_method: dict[str, list[dict[str, Any]]] = {method: [] for method in methods}
    for size in sizes:
        for method in by_method:
            row = next(row for row in rows if int(row["train_limit"]) == size and row["method"] == method)
            by_method[method].append(row)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
    for method in methods:
        color, marker = palette.get(method, ("#666666", "o"))
        axes[0].plot(
            sizes,
            [float(row["eval_choice_accuracy"]) for row in by_method[method]],
            marker=marker,
            color=color,
            label=method,
        )
        axes[1].plot(
            sizes,
            [float(row["eval_avg_loss"]) for row in by_method[method]],
            marker=marker,
            color=color,
            label=method,
        )
    axes[0].set_title("Fault-free eval accuracy")
    axes[0].set_ylabel("Accuracy")
    axes[1].set_title("Fault-free eval loss")
    axes[1].set_ylabel("Loss")
    for axis in axes:
        axis.set_xlabel("Train samples")
        axis.set_xticks(sizes, [str(size) for size in sizes], rotation=20)
        axis.grid(True, alpha=0.22)
    axes[0].legend(frameon=False)
    savefig(path)


def plot_sample_size_gap(rows: list[dict[str, Any]], path: Path, *, bp_method: str) -> None:
    configure_style()
    sizes = sorted({int(row["train_limit"]) for row in rows})
    gaps = []
    for size in sizes:
        bp_row = next(row for row in rows if int(row["train_limit"]) == size and row["method"] == bp_method)
        bpfree_row = next(row for row in rows if int(row["train_limit"]) == size and row["method"] == "BP-free")
        gaps.append(float(bp_row["eval_choice_accuracy"]) - float(bpfree_row["eval_choice_accuracy"]))
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.plot(sizes, gaps, marker="o", color="#444444")
    ax.axhline(0.0, color="#999999", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Train samples")
    ax.set_ylabel(f"{bp_method} - BP-free accuracy")
    ax.set_title("Accuracy gap vs training size")
    ax.set_xticks(sizes, [str(size) for size in sizes], rotation=20)
    ax.grid(True, alpha=0.22)
    for size, gap in zip(sizes, gaps):
        ax.annotate(f"{gap:+.3f}", (size, gap), textcoords="offset points", xytext=(0, 6), ha="center")
    savefig(path)


def run_prefix_exact_bp_controls(
    args: argparse.Namespace,
    *,
    window_names: list[str],
    windows: dict[str, tuple[int, int]],
    main_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    f1b_fault_free = next(row for row in main_rows if row["method"] == "1F1B" and row["case"] == "fault_free")
    rows: list[dict[str, Any]] = []
    for window_name in window_names:
        start_seq, end_seq = windows[window_name]
        run_args = argparse.Namespace(**vars(args))
        run_args.train_limit = start_seq
        prefix = run_f1b_case(
            run_args,
            args.output_root / "prefix_bp_control",
            OfflineCase(name="fault_free", stage_id=None, start_seq=None, end_seq=None),
        )
        strict_rows = [
            row
            for row in main_rows
            if row["method"] == "1F1B" and row["window_name"] == window_name and row["case"] != "fault_free"
        ]
        strict_mean = sum(float(row["eval_choice_accuracy"]) for row in strict_rows) / len(strict_rows) if strict_rows else None
        rows.append(
            {
                **prefix,
                "method": "Exact-BP prefix-only",
                "case": f"prefix_{window_name}",
                "window_name": window_name,
                "full_train_limit": args.train_limit,
                "prefix_train_limit": start_seq,
                "offline_start_seq": start_seq,
                "offline_end_seq": end_seq,
                "control_kind": "early_exit_prefix_only",
                "eval_accuracy_delta_vs_full_train_1f1b": float(prefix["eval_choice_accuracy"])
                - float(f1b_fault_free["eval_choice_accuracy"]),
                "eval_loss_delta_vs_full_train_1f1b": float(prefix["eval_avg_loss"])
                - float(f1b_fault_free["eval_avg_loss"]),
                "strict_skip_accuracy_mean_same_window": "" if strict_mean is None else strict_mean,
                "prefix_minus_strict_skip_mean": "" if strict_mean is None else float(prefix["eval_choice_accuracy"]) - strict_mean,
            }
        )
    return rows


def run_grid(args: argparse.Namespace) -> None:
    windows = resolve_windows(train_limit=args.train_limit, batch_size=args.batch_size, mode=args.window_mode)
    window_names = parse_windows(args.windows, available=windows)
    cases = default_offline_cases(window_names, GRID_STAGES, windows=windows)
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for case in cases:
        rows.append(run_bpfree_case(args, args.output_root, case))
        rows.append(run_f1b_case(args, args.output_root, case))
    rows = add_fault_free_deltas(rows)
    write_csv(args.output_root / "grid_summary.csv", rows)
    plot_grid_accuracy(rows, args.output_root / "figures" / "grid_eval_accuracy.png", window_names=window_names)
    plot_grid_drop(rows, args.output_root / "figures" / "grid_eval_accuracy_drop.png", window_names=window_names)
    if args.include_prefix_bp_control:
        prefix_rows = run_prefix_exact_bp_controls(args, window_names=window_names, windows=windows, main_rows=rows)
        write_csv(args.output_root / "prefix_bp_control.csv", prefix_rows)
    report = [
        "# No-Recovery Grid",
        "",
        f"- Seed: `{args.seed}`",
        f"- Train manifest: `{args.train_manifest}`",
        f"- Train limit: `{args.train_limit}`",
        f"- Eval manifest: `{args.eval_manifest}`",
        f"- Eval limit: `{args.eval_limit}`",
        f"- Window mode: `{args.window_mode}`",
        f"- Effective batch: BP-free `ga={args.gradient_accumulation_steps}`; 1F1B `batch_size={args.batch_size}, microbatches={args.microbatches}`.",
        "",
        "## Resolved windows",
        "",
    ]
    for window_name in window_names:
        start_seq, end_seq = windows[window_name]
        report.append(f"- `{window_name}`: `[{start_seq}, {end_seq})`")
    report.extend(
        [
            "",
        "## Files",
        "",
        "- `grid_summary.csv`",
        "- `figures/grid_eval_accuracy.png`",
        "- `figures/grid_eval_accuracy_drop.png`",
        ]
    )
    if args.include_prefix_bp_control:
        report.append("- `prefix_bp_control.csv`")
    (args.output_root / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_root / 'REPORT.md'}")


def run_sample_size(args: argparse.Namespace) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    fault_free = OfflineCase(name="fault_free", stage_id=None, start_seq=None, end_seq=None)
    bp_runner = run_full_bp_case if args.bp_baseline == "full_bp_1gpu" else run_f1b_case
    bp_label = "Full BP" if args.bp_baseline == "full_bp_1gpu" else "1F1B"
    for train_limit in parse_csv_ints(args.train_limits):
        run_args = argparse.Namespace(**vars(args))
        run_args.train_limit = train_limit
        rows.append(run_bpfree_case(run_args, args.output_root, fault_free))
        rows.append(bp_runner(run_args, args.output_root, fault_free))
    rows = sorted(rows, key=lambda row: (int(row["train_limit"]), str(row["method"])))
    bp_minus_bpfree: dict[int, float] = {}
    for train_limit in sorted({int(row["train_limit"]) for row in rows}):
        f1b = next(row for row in rows if int(row["train_limit"]) == train_limit and row["method"] == bp_label)
        bpfree = next(row for row in rows if int(row["train_limit"]) == train_limit and row["method"] == "BP-free")
        bp_minus_bpfree[train_limit] = float(f1b["eval_choice_accuracy"]) - float(bpfree["eval_choice_accuracy"])
    summary_rows = [
        {
            "train_limit": train_limit,
            "bp_baseline": bp_label,
            "accuracy_gap_bp_minus_bpfree": gap,
        }
        for train_limit, gap in sorted(bp_minus_bpfree.items())
    ]
    write_csv(args.output_root / "sample_size_scan.csv", rows)
    write_csv(args.output_root / "sample_size_gap.csv", summary_rows)
    plot_sample_size_scan(rows, args.output_root / "figures" / "sample_size_scan.png")
    plot_sample_size_gap(rows, args.output_root / "figures" / "sample_size_gap.png", bp_method=bp_label)
    report = [
        "# No-Recovery Sample-Size Scan",
        "",
        f"- Seed: `{args.seed}`",
        f"- BP baseline: `{bp_label}`",
        f"- Train manifest: `{args.train_manifest}`",
        f"- Train limits: `{args.train_limits}`",
        f"- Eval manifest: `{args.eval_manifest}`",
        f"- Eval limit: `{args.eval_limit}`",
        "",
        "## Accuracy gap",
        "",
    ]
    for train_limit, gap in sorted(bp_minus_bpfree.items()):
        report.append(f"- train `{train_limit}`: `{bp_label} - BP-free = {gap:+.4f}`")
    report.extend(
        [
            "",
            "## Files",
            "",
            "- `sample_size_scan.csv`",
            "- `sample_size_gap.csv`",
            "- `figures/sample_size_scan.png`",
            "- `figures/sample_size_gap.png`",
        ]
    )
    (args.output_root / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_root / 'REPORT.md'}")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model_name", default="tinyllama")
    parser.add_argument(
        "--train_manifest",
        type=Path,
        default=REPO_ROOT / "data" / "sft_requests" / "tinyllama_agnews128_label_train10000_formal_v1" / "requests.jsonl",
    )
    parser.add_argument(
        "--eval_manifest",
        type=Path,
        default=REPO_ROOT / "data" / "sft_requests" / "tinyllama_agnews128_label_eval256_seed20260531" / "requests.jsonl",
    )
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--train_limit", type=int, default=1024)
    parser.add_argument("--eval_limit", type=int, default=256)
    parser.add_argument("--num_chunks", type=int, default=3)
    parser.add_argument("--stage_devices", default="cuda:0,cuda:1,cuda:2")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_targets", default="q_proj,v_proj")
    parser.add_argument("--lora_init_std", type=float, default=0.01)
    parser.add_argument("--progress_interval", type=int, default=128)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--microbatches", type=int, default=8)
    parser.add_argument("--max_inflight", type=int, default=8)
    parser.add_argument("--belief_transport_mode", default="terminal")
    parser.add_argument("--full_bp_device", default="cuda:0")
    parser.add_argument("--force", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run no-recovery sweep experiments.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    grid = subparsers.add_parser("grid", help="Run stage x outage-window no-recovery grid.")
    add_common_args(grid)
    grid.add_argument("--windows", default="default", help="Comma-separated outage windows or 'default'.")
    grid.add_argument("--window_mode", default="absolute", choices=["absolute", "proportional"])
    grid.add_argument("--include_prefix_bp_control", action="store_true")

    sample_size = subparsers.add_parser("sample-size", help="Run fault-free sample-size scan.")
    add_common_args(sample_size)
    sample_size.add_argument("--train_limits", default="256,512,1024,2048,4096")
    sample_size.add_argument(
        "--bp_baseline",
        default="full_bp_1gpu",
        choices=["full_bp_1gpu", "1f1b_3gpu"],
        help="Reference BP baseline used to locate the train-size threshold.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "grid":
        run_grid(args)
        return
    if args.mode == "sample-size":
        run_sample_size(args)
        return
    raise ValueError(f"unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
