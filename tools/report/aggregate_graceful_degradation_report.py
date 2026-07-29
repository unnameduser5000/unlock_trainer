#!/usr/bin/env python3
"""Aggregate repeated graceful-degradation and exact-BP recovery-control runs."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from plot_graceful_degradation_report import (
    bpfree_stage_rows,
    configure_style,
    eval_metrics,
    f1b_stage_rows,
    load_json,
    savefig,
)


def parse_seeds(raw: str) -> list[int]:
    seeds = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not seeds:
        raise ValueError("--seeds must not be empty")
    return seeds


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = float(sum(values) / len(values))
    if len(values) == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, float(variance**0.5)


def aggregate(rows: list[dict[str, Any]], keys: list[str], values: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output: list[dict[str, Any]] = []
    for group_key, group_rows in sorted(groups.items()):
        row = {key: value for key, value in zip(keys, group_key)}
        row["runs"] = len(group_rows)
        for value in values:
            mean, std = mean_std([float(item[value]) for item in group_rows])
            row[f"{value}_mean"] = mean
            row[f"{value}_std"] = std
        output.append(row)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_committed(rows: list[dict[str, Any]], path: Path) -> None:
    configure_style()
    stages = sorted({int(row["stage_id"]) for row in rows})
    series = [("BP-free", "fault_free"), ("BP-free", "offline_skip"), ("1F1B", "fault_free"), ("1F1B", "offline_skip")]
    labels = ["BP-free\nnormal", "BP-free\noffline, skip", "1F1B\nnormal", "1F1B\nstrict skip"]
    colors = ["#2f6fdd", "#e16b3d", "#5b9a7b", "#9a6fb0"]
    lookup = {(row["method"], row["case"], row["stage_id"]): row for row in rows}
    x = np.arange(len(stages))
    width = 0.19
    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    for index, ((method, case), label, color) in enumerate(zip(series, labels, colors)):
        selected = [lookup[(method, case, stage)] for stage in stages]
        ax.bar(
            x + (index - 1.5) * width,
            [row["committed_training_records_mean"] for row in selected],
            width,
            yerr=[row["committed_training_records_std"] for row in selected],
            capsize=3,
            label=label,
            color=color,
        )
    ax.set_xticks(x, [f"stage {stage}" for stage in stages])
    ax.set_ylabel("Committed local training records")
    ax.set_title("Training progress retained during a stage-1 outage (mean +/- sd)")
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    savefig(path)


def plot_retained(rows: list[dict[str, Any]], path: Path) -> None:
    configure_style()
    selected = [row for row in rows if row["case"] in {"offline_skip", "recovery_restart"}]
    stages = sorted({int(row["stage_id"]) for row in selected})
    methods = [("BP-free", "offline_skip"), ("1F1B", "offline_skip"), ("1F1B", "recovery_restart")]
    lookup = {(row["method"], row["case"], row["stage_id"]): row for row in selected}
    x = np.arange(len(stages))
    width = 0.26
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    active_methods = [
        item
        for item in methods
        if any(row["method"] == item[0] and row["case"] == item[1] for row in selected)
    ]
    for index, ((method, case), color, label) in enumerate(
        zip(
            active_methods,
            ["#e16b3d", "#5b9a7b", "#2f6fdd"],
            ["BP-free retained local progress", "1F1B strict skip", "1F1B checkpoint-restart"],
        )
    ):
        values = [lookup[(method, case, stage)] for stage in stages]
        mean_key = (
            "retained_on_end_to_end_failed_requests_mean"
            if method == "BP-free"
            else "replayed_after_recovery_mean"
        )
        std_key = (
            "retained_on_end_to_end_failed_requests_std"
            if method == "BP-free"
            else "replayed_after_recovery_std"
        )
        bars = ax.bar(
            x + (index - 1.0) * width,
            [row[mean_key] for row in values],
            width,
            yerr=[row[std_key] for row in values],
            capsize=3,
            label=label,
            color=color,
        )
        for bar, row in zip(bars, values):
            value = row[mean_key]
            ax.annotate(
                f"{value:.0f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                xytext=(0, 3),
                textcoords="offset points",
            )
    ax.set_xticks(x, [f"stage {stage}" for stage in stages])
    ax.set_ylabel("Records retained or replayed after the outage")
    ax.set_title("BP-free retained progress vs exact-BP recovery control (mean +/- sd)")
    ax.legend(frameon=False)
    savefig(path)


def plot_quality(rows: list[dict[str, Any]], path: Path) -> None:
    configure_style()
    methods = ["BP-free", "1F1B"]
    cases = ["fault_free", "offline_skip"]
    lookup = {(row["method"], row["case"]): row for row in rows}
    x = np.arange(len(methods))
    width = 0.33
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.9))
    accuracy_handles = []
    for index, (case, color, label) in enumerate(
        [("fault_free", "#2f6fdd", "fault-free"), ("offline_skip", "#e16b3d", "offline, no recovery")]
    ):
        selected = [lookup[(method, case)] for method in methods]
        accuracy_handles.append(axes[0].bar(
            x + (index - 0.5) * width,
            [row["eval_accuracy_mean"] for row in selected],
            width,
            yerr=[row["eval_accuracy_std"] for row in selected],
            capsize=3,
            label=label,
            color=color,
        ))
        axes[1].bar(
            x + (index - 0.5) * width,
            [row["eval_loss_mean"] for row in selected],
            width,
            yerr=[row["eval_loss_std"] for row in selected],
            capsize=3,
            label=label,
            color=color,
        )
    for axis, title, ylabel in zip(axes, ["Post-training accuracy", "Post-training loss"], ["Accuracy", "Loss"]):
        axis.set_xticks(x, methods)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
    fig.legend(
        accuracy_handles,
        ["fault-free", "offline, no recovery"],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    savefig(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate graceful-degradation reports across seeds.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seeds", default="20260531,20260532,20260533")
    parser.add_argument("--output_dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    output_dir = args.output_dir or args.root / "report"
    stage_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    offline_window: dict[str, Any] = {}
    for seed in seeds:
        bpfree_fault_free = load_json(args.root / f"bpfree_fault_free_seed{seed}" / "scheduler_summary.json")
        bpfree_offline = load_json(args.root / f"bpfree_offline_skip_seed{seed}" / "scheduler_summary.json")
        f1b_fault_free = load_json(args.root / f"1f1b_fault_free_seed{seed}" / "summary.json")
        f1b_offline = load_json(args.root / f"1f1b_offline_skip_seed{seed}" / "summary.json")
        recovery_dir = args.root / f"1f1b_recovery_restart_seed{seed}"
        f1b_recovery = load_json(recovery_dir / "summary.json") if (recovery_dir / "summary.json").is_file() else None
        offline_window = bpfree_offline.get("offline_window", offline_window)
        for row in bpfree_stage_rows(bpfree_fault_free, case="fault_free"):
            row["seed"] = seed
            stage_rows.append(row)
        for row in bpfree_stage_rows(bpfree_offline, case="offline_skip"):
            row["seed"] = seed
            stage_rows.append(row)
        for row in f1b_stage_rows(f1b_fault_free, case="fault_free"):
            row["seed"] = seed
            stage_rows.append(row)
        for row in f1b_stage_rows(f1b_offline, case="offline_skip"):
            row["seed"] = seed
            stage_rows.append(row)
        if f1b_recovery is not None:
            for row in f1b_stage_rows(f1b_recovery, case="recovery_restart"):
                row["seed"] = seed
                stage_rows.append(row)
        for summary, method, case in (
            (bpfree_fault_free, "BP-free", "fault_free"),
            (bpfree_offline, "BP-free", "offline_skip"),
            (f1b_fault_free, "1F1B", "fault_free"),
            (f1b_offline, "1F1B", "offline_skip"),
        ):
            row = eval_metrics(summary, method=method, case=case)
            row["seed"] = seed
            quality_rows.append(row)
        if f1b_recovery is not None:
            row = eval_metrics(f1b_recovery, method="1F1B", case="recovery_restart")
            row["seed"] = seed
            quality_rows.append(row)

    stage_summary = aggregate(
        stage_rows,
        ["method", "case", "stage_id"],
        [
            "committed_training_records",
            "retained_on_end_to_end_failed_requests",
            "replayed_after_recovery",
            "end_to_end_completed",
            "end_to_end_failed",
        ],
    )
    quality_summary = aggregate(quality_rows, ["method", "case"], ["eval_accuracy", "eval_loss"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "stage_progress_per_seed.csv", stage_rows)
    write_csv(output_dir / "stage_progress_summary.csv", stage_summary)
    write_csv(output_dir / "quality_per_seed.csv", quality_rows)
    write_csv(output_dir / "quality_summary.csv", quality_summary)
    figures = output_dir / "figures"
    plot_committed(stage_summary, figures / "committed_training_records_by_stage.png")
    plot_retained(stage_summary, figures / "retained_partial_progress_by_stage.png")
    plot_quality(quality_summary, figures / "post_training_quality.png")
    retained_stage0 = next(
        row["retained_on_end_to_end_failed_requests_mean"]
        for row in stage_summary
        if row["method"] == "BP-free" and row["case"] == "offline_skip" and int(row["stage_id"]) == 0
    )
    report = [
        "# Repeated Strict No-Recovery Graceful-Degradation Report",
        "",
        f"- Seeds: {', '.join(str(seed) for seed in seeds)}.",
        f"- Offline window: stage {offline_window.get('stage_id')} rejects requests in "
        f"`[{offline_window.get('start_seq')}, {offline_window.get('end_seq')})`.",
        "- BP-free uses `recovery_policy=skip`; 1F1B strict skip drops affected batches before scheduling.",
        "- If `1f1b_recovery_restart_seed*` exists, the exact-BP control restores the latest committed batch-boundary checkpoint and replays later committed batches.",
        f"- BP-free stage 0 retained {retained_stage0:.1f} local update records from requests that failed end-to-end.",
        "- This separates retained local progress from exact-BP batch-boundary restart; neither should be read as universal fault tolerance.",
        "",
        "## Files",
        "",
        "- `stage_progress_summary.csv` and `quality_summary.csv` contain mean/std across seeds.",
        "- `figures/committed_training_records_by_stage.png`",
        "- `figures/retained_partial_progress_by_stage.png`",
        "- `figures/post_training_quality.png`",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote {output_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
