#!/usr/bin/env python3
"""Create paper-facing plots from normalized AG News E1/E2 protocol outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHOD_LABELS = {
    "full_bp_1gpu": "Full BP (1 GPU)",
    "1f1b_3gpu": "1F1B (3 GPU)",
    "bpfree_ce_3gpu": "BP-free CE (3 GPU)",
    "bpfree_belief_3gpu": "BP-free belief (3 GPU)",
}
METHOD_COLORS = {
    "full_bp_1gpu": "#1f77b4",
    "1f1b_3gpu": "#ff7f0e",
    "bpfree_ce_3gpu": "#2ca02c",
    "bpfree_belief_3gpu": "#9467bd",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def method_color(method: str) -> str:
    return METHOD_COLORS.get(method, "#555555")


def ordered_methods(rows: list[dict[str, str]]) -> list[str]:
    present = {row["method"] for row in rows if row.get("method")}
    known = [method for method in METHOD_LABELS if method in present]
    return known + sorted(present - set(known))


def configure_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_quality_curve(rows: list[dict[str, str]], figure_dir: Path, metric: str, filename: str, ylabel: str) -> bool:
    groups: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("suite") != "e1_quality":
            continue
        step = to_float(row.get("optimizer_step"))
        value = to_float(row.get(metric))
        if step is not None and value is not None:
            groups[(row["method"], int(step))].append(value)
    if not groups:
        return False

    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    methods = ordered_methods([{"method": method} for method, _ in groups])
    for method in methods:
        points = []
        for (point_method, step), values in groups.items():
            if point_method == method:
                average, deviation = mean_std(values)
                points.append((step, average, deviation, len(values)))
        points.sort()
        x_values = [point[0] for point in points]
        means = [point[1] for point in points]
        deviations = [point[2] for point in points]
        sample_count = max(point[3] for point in points)
        label = f"{method_label(method)} (n={sample_count})"
        color = method_color(method)
        ax.plot(x_values, means, marker="o", markersize=3.6, linewidth=2, color=color, label=label)
        if any(deviations):
            low = [mean - deviation for mean, deviation in zip(means, deviations)]
            high = [mean + deviation for mean, deviation in zip(means, deviations)]
            ax.fill_between(x_values, low, high, color=color, alpha=0.16, linewidth=0)
    ax.set_xlabel("Optimizer updates")
    ax.set_ylabel(ylabel)
    ax.set_title(f"AG News E1 validation {ylabel.lower()}")
    configure_axes(ax)
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.savefig(figure_dir / filename, dpi=220)
    plt.close(fig)
    return True


def save_quality_seed_traces(rows: list[dict[str, str]], figure_dir: Path) -> bool:
    """Show all E1 seed trajectories instead of only the mean and deviation band."""
    seed_groups: dict[tuple[str, str, str], list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        if row.get("suite") != "e1_quality":
            continue
        step = to_float(row.get("optimizer_step"))
        if step is None:
            continue
        for metric in ("choice_accuracy", "choice_loss"):
            value = to_float(row.get(metric))
            if value is not None:
                seed_groups[(metric, row["method"], row["seed"])].append((int(step), value))
    if not seed_groups:
        return False

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.25), constrained_layout=True)
    for axis, metric, ylabel in zip(
        axes,
        ("choice_accuracy", "choice_loss"),
        ("Choice accuracy", "Choice loss"),
    ):
        methods = ordered_methods([{"method": key[1]} for key in seed_groups if key[0] == metric])
        for method in methods:
            by_step: dict[int, list[float]] = defaultdict(list)
            method_seeds = sorted(seed for source_metric, source_method, seed in seed_groups if source_metric == metric and source_method == method)
            for seed in method_seeds:
                points = sorted(seed_groups[(metric, method, seed)])
                if not points:
                    continue
                x_values = [point[0] for point in points]
                values = [point[1] for point in points]
                axis.plot(
                    x_values,
                    values,
                    color=method_color(method),
                    alpha=0.28,
                    linewidth=1.0,
                    marker="o",
                    markersize=2.3,
                )
                for step, value in points:
                    by_step[step].append(value)
            mean_points = sorted((step, statistics.mean(values)) for step, values in by_step.items())
            axis.plot(
                [point[0] for point in mean_points],
                [point[1] for point in mean_points],
                color=method_color(method),
                linewidth=2.5,
                marker="o",
                markersize=3.4,
                label=method_label(method),
            )
        axis.set_xlabel("Optimizer updates")
        axis.set_ylabel(ylabel)
        axis.set_title(f"Validation {ylabel.lower()}: raw seeds (thin) and mean (bold)")
        configure_axes(axis)
        axis.legend(frameon=False, fontsize=8, loc="best")
    fig.savefig(figure_dir / "e1_validation_seed_traces.png", dpi=220)
    plt.close(fig)
    return True


def save_final_accuracy(rows: list[dict[str, str]], figure_dir: Path) -> bool:
    e1_rows = [
        row
        for row in rows
        if row.get("suite") == "e1_quality" and to_float(row.get("final_choice_accuracy")) is not None
    ]
    if not e1_rows:
        return False
    methods = ordered_methods(e1_rows)
    values_by_method: dict[str, list[float]] = defaultdict(list)
    for row in e1_rows:
        values_by_method[row["method"]].append(float(row["final_choice_accuracy"]))

    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    positions = list(range(len(methods)))
    means = [mean_std(values_by_method[method])[0] for method in methods]
    deviations = [mean_std(values_by_method[method])[1] for method in methods]
    ax.bar(
        positions,
        means,
        yerr=deviations,
        capsize=4,
        color=[method_color(method) for method in methods],
        alpha=0.85,
        width=0.66,
    )
    for position, method in zip(positions, methods):
        values = sorted(values_by_method[method])
        offsets = [0.0] if len(values) == 1 else [(-0.15 + 0.30 * index / (len(values) - 1)) for index in range(len(values))]
        ax.scatter([position + offset for offset in offsets], values, color="#1a1a1a", s=20, zorder=3)
    ax.set_xticks(positions, [method_label(method) for method in methods], rotation=16, ha="right")
    ax.set_ylabel("Official test choice accuracy")
    ax.set_title("AG News E1 final test accuracy")
    floor = min(value for method in methods for value in values_by_method[method])
    ax.set_ylim(max(0.0, math.floor((floor - 0.03) * 20) / 20), 1.0)
    configure_axes(ax)
    fig.savefig(figure_dir / "e1_final_test_accuracy.png", dpi=220)
    plt.close(fig)
    return True


def save_stage_memory(rows: list[dict[str, str]], figure_dir: Path) -> bool:
    e1_rows = [
        row
        for row in rows
        if row.get("suite") == "e1_quality" and to_float(row.get("cuda_peak_allocated_bytes")) is not None
    ]
    if not e1_rows:
        return False
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in e1_rows:
        values[(row["method"], row["stage_id"])].append(float(row["cuda_peak_allocated_bytes"]) / (1024**3))
    methods = ordered_methods([{"method": method} for method, _ in values])
    stage_ids = sorted({stage for _, stage in values}, key=int)
    width = 0.72 / max(len(stage_ids), 1)
    fig, ax = plt.subplots(figsize=(7.4, 4.5), constrained_layout=True)
    base = list(range(len(methods)))
    palette = ["#4c78a8", "#f58518", "#54a24b"]
    for stage_index, stage_id in enumerate(stage_ids):
        heights = []
        for method in methods:
            observed = values.get((method, stage_id), [])
            heights.append(statistics.mean(observed) if observed else 0.0)
        offset = (stage_index - (len(stage_ids) - 1) / 2) * width
        ax.bar([item + offset for item in base], heights, width=width, label=f"stage {stage_id}", color=palette[stage_index % len(palette)])
    ax.set_xticks(base, [method_label(method) for method in methods], rotation=16, ha="right")
    ax.set_ylabel("Peak CUDA allocated (GiB)")
    ax.set_title("AG News E1 peak GPU allocation by stage")
    configure_axes(ax)
    ax.legend(frameon=False, title="Model partition")
    fig.savefig(figure_dir / "e1_stage_peak_cuda_allocated.png", dpi=220)
    plt.close(fig)
    return True


def save_e2_sweeps(runs: list[dict[str, str]], peaks: list[dict[str, str]], figure_dir: Path) -> list[str]:
    produced: list[str] = []
    e2_runs = [row for row in runs if row.get("suite") == "e2_system"]
    groups: dict[str, dict[int, list[float]]] = {"1F1B": defaultdict(list), "BP-free": defaultdict(list)}
    for row in e2_runs:
        throughput = to_float(row.get("train_throughput_per_s"))
        if throughput is None:
            continue
        if row["method"].startswith("1f1b_"):
            control = to_float(row.get("microbatches"))
            family = "1F1B"
        elif row["method"].startswith("bpfree_"):
            control = to_float(row.get("max_inflight"))
            family = "BP-free"
        else:
            continue
        if control is not None:
            groups[family][int(control)].append(throughput)
    if any(groups.values()) and any(points for points in groups.values()):
        fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
        for family, color, marker in (("1F1B", "#ff7f0e", "o"), ("BP-free", "#2ca02c", "s")):
            points = sorted(groups[family].items())
            if not points:
                continue
            x_values = [point[0] for point in points]
            means = [mean_std(point[1])[0] for point in points]
            deviations = [mean_std(point[1])[1] for point in points]
            ax.errorbar(x_values, means, yerr=deviations, color=color, marker=marker, capsize=3, linewidth=2, label=family)
        ax.set_xlabel("1F1B microbatches / BP-free inflight window W")
        ax.set_ylabel("Training throughput (requests/s)")
        ax.set_title("AG News E2 concurrency sweep")
        configure_axes(ax)
        ax.legend(frameon=False)
        fig.savefig(figure_dir / "e2_throughput_sweep.png", dpi=220)
        plt.close(fig)
        produced.append("e2_throughput_sweep.png")

    e2_peaks = [row for row in peaks if row.get("suite") == "e2_system"]
    grouped_peaks: dict[str, list[float]] = defaultdict(list)
    for row in e2_peaks:
        allocated = to_float(row.get("cuda_peak_allocated_bytes"))
        if allocated is not None:
            grouped_peaks[row["method"]].append(allocated / (1024**3))
    if grouped_peaks:
        methods = sorted(grouped_peaks, key=lambda name: (not name.startswith("1f1b_"), name))
        fig, ax = plt.subplots(figsize=(8.2, 4.4), constrained_layout=True)
        ax.bar(
            range(len(methods)),
            [statistics.mean(grouped_peaks[method]) for method in methods],
            color=["#ff7f0e" if method.startswith("1f1b_") else "#2ca02c" for method in methods],
        )
        ax.set_xticks(range(len(methods)), methods, rotation=22, ha="right")
        ax.set_ylabel("Mean per-stage peak CUDA allocated (GiB)")
        ax.set_title("AG News E2 peak allocation across concurrency settings")
        configure_axes(ax)
        fig.savefig(figure_dir / "e2_stage_peak_cuda_allocated.png", dpi=220)
        plt.close(fig)
        produced.append("e2_stage_peak_cuda_allocated.png")
    return produced


def save_e2_raw_seed_throughput(runs: list[dict[str, str]], figure_dir: Path) -> bool:
    """Expose raw E2 runs while keeping the CPU-queue families in separate panels."""
    families: dict[str, dict[str, list[tuple[int, float]]]] = {
        "1F1B": defaultdict(list),
        "BP-free CPU queue": defaultdict(list),
    }
    for row in runs:
        if row.get("suite") != "e2_system":
            continue
        throughput = to_float(row.get("train_throughput_per_s"))
        if throughput is None:
            continue
        if row.get("method", "").startswith("1f1b_"):
            control = to_float(row.get("microbatches"))
            family = "1F1B"
        elif row.get("method", "").startswith("bpfree_"):
            control = to_float(row.get("max_inflight"))
            family = "BP-free CPU queue"
        else:
            continue
        if control is not None:
            families[family][row["seed"]].append((int(control), throughput))
    if not any(families.values()):
        return False

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.15), constrained_layout=True, sharey=True)
    colors = ["#1f77b4", "#9467bd", "#8c564b"]
    for axis, (family, by_seed) in zip(axes, families.items()):
        for color, seed in zip(colors, sorted(by_seed)):
            points = sorted(by_seed[seed])
            axis.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                marker="o",
                color=color,
                linewidth=1.8,
                label=f"seed {seed}",
            )
        axis.set_xlabel("Microbatches" if family == "1F1B" else "In-flight window W")
        axis.set_title(f"{family}: raw full-run throughput")
        configure_axes(axis)
        axis.legend(frameon=False, fontsize=8, loc="best")
    axes[0].set_ylabel("Training throughput (requests/s)")
    fig.suptitle("AG News E2 raw seed runs: within-family scheduler sweeps, not a P2P comparison", fontsize=10)
    fig.savefig(figure_dir / "e2_raw_seed_throughput.png", dpi=220)
    plt.close(fig)
    return True


def save_1f1b_microbatch_tradeoff(
    runs: list[dict[str, str]], peaks: list[dict[str, str]], figure_dir: Path
) -> bool:
    """Plot the physical-microbatch trade-off without conflating it with a queue window."""
    run_rows = [
        row
        for row in runs
        if row.get("suite") == "e2_system" and row.get("method", "").startswith("1f1b_")
    ]
    if not run_rows:
        return False
    control_by_method = {
        row["method"]: int(to_float(row.get("microbatches")) or 0)
        for row in run_rows
    }
    batch_size_by_method = {
        row["method"]: int(to_float(row.get("batch_size")) or 0)
        for row in run_rows
    }
    throughput_by_seed: dict[str, dict[int, float]] = defaultdict(dict)
    for row in run_rows:
        throughput = to_float(row.get("train_throughput_per_s"))
        control = control_by_method[row["method"]]
        if throughput is not None and control > 0:
            throughput_by_seed[row["seed"]][control] = throughput
    peak_by_seed: dict[str, dict[int, float]] = defaultdict(dict)
    peak_candidates: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in peaks:
        method = row.get("method", "")
        if row.get("suite") != "e2_system" or method not in control_by_method:
            continue
        allocated = to_float(row.get("cuda_peak_allocated_bytes"))
        if allocated is not None:
            peak_candidates[(row["seed"], control_by_method[method])].append(allocated / (1024**3))
    for (seed, control), values in peak_candidates.items():
        peak_by_seed[seed][control] = max(values)
    controls = sorted({control for values in throughput_by_seed.values() for control in values})
    if not controls:
        return False

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.15), constrained_layout=True)
    colors = ["#1f77b4", "#9467bd", "#8c564b"]
    x_positions = list(range(len(controls)))
    for axis, by_seed, ylabel, title in (
        (axes[0], throughput_by_seed, "Training throughput (requests/s)", "Throughput"),
        (axes[1], peak_by_seed, "Max stage peak CUDA allocated (GiB)", "GPU peak allocation"),
    ):
        per_control: dict[int, list[float]] = defaultdict(list)
        for color, seed in zip(colors, sorted(by_seed)):
            values = by_seed[seed]
            available = [(index, values[control]) for index, control in enumerate(controls) if control in values]
            axis.plot(
                [point[0] for point in available],
                [point[1] for point in available],
                color=color,
                marker="o",
                linewidth=1.6,
                alpha=0.8,
                label=f"seed {seed}",
            )
            for index, value in available:
                per_control[controls[index]].append(value)
        means = [mean_std(per_control[control])[0] for control in controls]
        deviations = [mean_std(per_control[control])[1] for control in controls]
        axis.errorbar(
            x_positions,
            means,
            yerr=deviations,
            color="#111111",
            marker="D",
            markersize=4,
            linewidth=2,
            capsize=3,
            label="mean +/- sd",
        )
        axis.set_xticks(
            x_positions,
            [
                f"mb={control}\nphysical b={batch_size_by_method[next(method for method, value in control_by_method.items() if value == control)] // control}"
                for control in controls
            ],
        )
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        configure_axes(axis)
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("1F1B fixed effective batch=12: microbatch granularity versus GPU efficiency/memory", fontsize=10)
    fig.savefig(figure_dir / "e2_1f1b_microbatch_tradeoff.png", dpi=220)
    plt.close(fig)
    return True


def save_cpu_queue_window_saturation(runs: list[dict[str, str]], figure_dir: Path) -> bool:
    """Show raw CPU-queue saturation separately from the P2P comparison figures."""
    by_seed: dict[str, dict[int, float]] = defaultdict(dict)
    for row in runs:
        if row.get("suite") != "e2_system" or not row.get("method", "").startswith("bpfree_"):
            continue
        window = to_float(row.get("max_inflight"))
        throughput = to_float(row.get("train_throughput_per_s"))
        if window is not None and throughput is not None:
            by_seed[row["seed"]][int(window)] = throughput
    windows = sorted({window for values in by_seed.values() for window in values})
    if not windows:
        return False

    fig, ax = plt.subplots(figsize=(7.0, 4.15), constrained_layout=True)
    colors = ["#1f77b4", "#9467bd", "#8c564b"]
    x_positions = list(range(len(windows)))
    per_window: dict[int, list[float]] = defaultdict(list)
    for color, seed in zip(colors, sorted(by_seed)):
        values = by_seed[seed]
        available = [(index, values[window]) for index, window in enumerate(windows) if window in values]
        ax.plot(
            [point[0] for point in available],
            [point[1] for point in available],
            color=color,
            marker="o",
            linewidth=1.8,
            label=f"seed {seed}",
        )
        for index, value in available:
            per_window[windows[index]].append(value)
    means = [mean_std(per_window[window])[0] for window in windows]
    deviations = [mean_std(per_window[window])[1] for window in windows]
    ax.errorbar(
        x_positions,
        means,
        yerr=deviations,
        color="#111111",
        marker="D",
        markersize=4,
        linewidth=2,
        capsize=3,
        label="mean +/- sd",
    )
    ax.set_xticks(x_positions, [f"W={window}" for window in windows])
    ax.set_ylabel("Training throughput (requests/s)")
    ax.set_title("BP-free CPU-queue FIFO saturation: one worker per GPU stage")
    configure_axes(ax)
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.savefig(figure_dir / "e2_cpu_queue_window_saturation.png", dpi=220)
    plt.close(fig)
    return True


def save_e2_p2p_comparison(
    runs: list[dict[str, str]],
    peaks: list[dict[str, str]],
    figure_dir: Path,
    *,
    labels: list[tuple[str, str, str]],
    title: str,
    filename: str,
) -> bool:
    selected = [
        row
        for row in runs
        if any(row.get("suite") == suite and row.get("method") == method for suite, method, _ in labels)
    ]
    if any(
        len([row for row in selected if row.get("suite") == suite and row.get("method") == method]) != 3
        for suite, method, _ in labels
    ):
        return False
    throughputs: list[tuple[float, float]] = []
    peak_values: list[tuple[float, float]] = []
    for suite, method, _ in labels:
        method_runs = [row for row in selected if row.get("suite") == suite and row.get("method") == method]
        throughput_values = [to_float(row.get("train_throughput_per_s")) for row in method_runs]
        throughputs.append(mean_std([value for value in throughput_values if value is not None]))
        by_seed: dict[str, list[float]] = defaultdict(list)
        for row in peaks:
            if row.get("suite") == suite and row.get("method") == method:
                allocated = to_float(row.get("cuda_peak_allocated_bytes"))
                if allocated is not None:
                    by_seed[row["seed"]].append(allocated / (1024**3))
        per_run_peaks = [max(values) for values in by_seed.values()]
        peak_values.append(mean_std(per_run_peaks))

    fig, axes = plt.subplots(1, 2, figsize=(8.1, 3.9), constrained_layout=True)
    names = [item[2] for item in labels]
    colors = ["#ff7f0e", "#2ca02c"]
    for axis, values, ylabel, axis_title in (
        (axes[0], throughputs, "Training throughput (requests/s)", "Throughput"),
        (axes[1], peak_values, "Max stage peak CUDA allocated (GiB)", "Peak allocation"),
    ):
        axis.bar(range(2), [value[0] for value in values], yerr=[value[1] for value in values], capsize=4, color=colors)
        axis.set_xticks(range(2), names, rotation=10, ha="right")
        axis.set_ylabel(ylabel)
        axis.set_title(axis_title)
        configure_axes(axis)
    fig.suptitle(title, fontsize=12)
    fig.savefig(figure_dir / filename, dpi=220)
    plt.close(fig)
    return True


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def summary_train_wall_seconds(summary: dict[str, Any]) -> float | None:
    wall_ms = to_float(summary.get("wall_ms"))
    if wall_ms is not None:
        return wall_ms / 1000.0
    phases = summary.get("phases")
    if isinstance(phases, list):
        for phase in phases:
            if isinstance(phase, dict) and phase.get("phase") == "train":
                wall_ms = to_float(phase.get("wall_ms"))
                if wall_ms is not None:
                    return wall_ms / 1000.0
    return None


def load_1f1b_progress(seed_dir: Path) -> list[tuple[int, float]]:
    """Reconstruct global batch completion from all three stage clocks."""
    batch_end_ms: dict[int, float] = {}
    train_start_ms: list[float] = []
    for stage_metrics in sorted(seed_dir.glob("rank*_stage_metrics.csv")):
        for row in read_csv(stage_metrics):
            if row.get("phase") != "train":
                continue
            batch_seq = to_float(row.get("batch_seq"))
            start_ms = to_float(row.get("start_epoch_ms"))
            end_ms = to_float(row.get("end_epoch_ms"))
            if batch_seq is None or start_ms is None or end_ms is None:
                continue
            batch = int(batch_seq)
            train_start_ms.append(start_ms)
            batch_end_ms[batch] = max(batch_end_ms.get(batch, float("-inf")), end_ms)
    if not train_start_ms or not batch_end_ms:
        return []

    start_ms = min(train_start_ms)
    timeline = [(batch + 1, (end_ms - start_ms) / 1000.0) for batch, end_ms in sorted(batch_end_ms.items())]
    terminal = timeline[-1][1]
    summary_wall = summary_train_wall_seconds(read_json(seed_dir / "summary.json"))
    if terminal <= 0 or summary_wall is None:
        return []
    scale = summary_wall / terminal
    return [(step, elapsed * scale) for step, elapsed in timeline]


def load_bpfree_progress(seed_dir: Path) -> list[tuple[int, float]]:
    """Aggregate request records into common effective-batch optimizer updates."""
    summary = read_json(seed_dir / "summary.json")
    effective_batch = int(summary.get("effective_optimizer_batch", 0) or 0)
    summary_wall = summary_train_wall_seconds(summary)
    if effective_batch <= 0 or summary_wall is None:
        return []
    rows = read_csv(seed_dir / "train.csv")
    elapsed_ms = [to_float(row.get("elapsed_ms")) for row in rows]
    if any(value is None for value in elapsed_ms):
        return []
    usable = len(elapsed_ms) - len(elapsed_ms) % effective_batch
    if usable == 0:
        return []
    cumulative_seconds: list[float] = []
    elapsed = 0.0
    for value in elapsed_ms[:usable]:
        elapsed += float(value) / 1000.0
        cumulative_seconds.append(elapsed)
    terminal = cumulative_seconds[-1]
    if terminal <= 0:
        return []
    scale = summary_wall / terminal
    return [
        (step, cumulative_seconds[step * effective_batch - 1] * scale)
        for step in range(1, usable // effective_batch + 1)
    ]


def save_e2_p2p_progress(
    output_root: Path,
    report_dir: Path,
    figure_dir: Path,
    *,
    labels: list[tuple[str, str, str]],
    title: str,
    filename: str,
) -> bool:
    """Plot real cumulative wall time and rolling throughput against optimizer updates."""
    raw_root = output_root / "e2_p2p_throughput"
    timelines: dict[str, dict[str, list[tuple[int, float]]]] = defaultdict(dict)
    for _, method, _ in labels:
        method_root = raw_root / method
        for seed_dir in sorted(method_root.glob("seed*")):
            loader = load_1f1b_progress if method.startswith("1f1b_") else load_bpfree_progress
            progress = loader(seed_dir)
            if progress:
                timelines[method][seed_dir.name.removeprefix("seed")] = progress

    if any(len(timelines[method]) != 3 for _, method, _ in labels):
        return False
    update_counts = {
        method: {len(points) for points in timelines[method].values()}
        for _, method, _ in labels
    }
    if any(len(counts) != 1 for counts in update_counts.values()):
        return False
    common_updates = min(next(iter(counts)) for counts in update_counts.values())
    if common_updates < 64:
        return False

    effective_batch = 12
    rolling_window = 32
    curve_rows: list[dict[str, Any]] = []
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.1), constrained_layout=True)
    colors = ["#ff7f0e", "#2ca02c"]
    for method_index, (_, method, label) in enumerate(labels):
        color = colors[method_index]
        seed_points = [timelines[method][seed] for seed in sorted(timelines[method])]
        elapsed_by_update = [
            [points[update - 1][1] for points in seed_points]
            for update in range(1, common_updates + 1)
        ]
        elapsed_mean = [mean_std(values)[0] for values in elapsed_by_update]
        elapsed_std = [mean_std(values)[1] for values in elapsed_by_update]
        updates = list(range(1, common_updates + 1))
        axes[0].plot(updates, elapsed_mean, color=color, linewidth=2, label=label)
        axes[0].fill_between(
            updates,
            [mean - deviation for mean, deviation in zip(elapsed_mean, elapsed_std)],
            [mean + deviation for mean, deviation in zip(elapsed_mean, elapsed_std)],
            color=color,
            alpha=0.16,
        )

        rolling_by_update: dict[int, list[float]] = defaultdict(list)
        for seed, points in zip(sorted(timelines[method]), seed_points):
            elapsed_with_zero = [0.0] + [elapsed for _, elapsed in points]
            for update in range(rolling_window, common_updates + 1):
                duration = elapsed_with_zero[update] - elapsed_with_zero[update - rolling_window]
                throughput = rolling_window * effective_batch / duration if duration > 0 else 0.0
                rolling_by_update[update].append(throughput)
                curve_rows.append(
                    {
                        "method": method,
                        "label": label,
                        "seed": seed,
                        "optimizer_update": update,
                        "records_processed": update * effective_batch,
                        "cumulative_train_wall_s": elapsed_with_zero[update],
                        "rolling_window_updates": rolling_window,
                        "rolling_throughput_per_s": throughput,
                    }
                )
        rolling_updates = sorted(rolling_by_update)
        rolling_mean = [mean_std(rolling_by_update[update])[0] for update in rolling_updates]
        rolling_std = [mean_std(rolling_by_update[update])[1] for update in rolling_updates]
        axes[1].plot(rolling_updates, rolling_mean, color=color, linewidth=2, label=label)
        axes[1].fill_between(
            rolling_updates,
            [mean - deviation for mean, deviation in zip(rolling_mean, rolling_std)],
            [mean + deviation for mean, deviation in zip(rolling_mean, rolling_std)],
            color=color,
            alpha=0.16,
        )

    axes[0].set_title("Cumulative train wall time")
    axes[0].set_xlabel("Optimizer update (effective batch=12)")
    axes[0].set_ylabel("Elapsed training time (s)")
    axes[1].set_title("Rolling training throughput")
    axes[1].set_xlabel("Optimizer update (effective batch=12)")
    axes[1].set_ylabel("Requests/s over 32 updates")
    for axis in axes:
        configure_axes(axis)
        axis.legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle(f"{title} (mean +/- sd, n=3; no activation tracker)", fontsize=11)
    fig.savefig(figure_dir / filename, dpi=220)
    plt.close(fig)

    curve_path = report_dir / f"{Path(filename).stem}_raw.csv"
    with curve_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)
    return True


def save_e2_p2p_activation_stash_by_stage(
    runs: list[dict[str, str]],
    peaks: list[dict[str, str]],
    figure_dir: Path,
    *,
    labels: list[tuple[str, str, str, str]],
    title: str,
    filename: str,
) -> bool:
    """Compare measured autograd stash only when physical/effective batches match."""
    run_rows = {
        (suite, method): [
            row for row in runs if row.get("suite") == suite and row.get("method") == method
        ]
        for suite, method, _, _ in labels
    }
    if any(len(rows) != 3 for rows in run_rows.values()):
        return False

    by_key: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in peaks:
        key = (row.get("suite", ""), row.get("method", ""), row.get("seed", ""), row.get("stage_id", ""))
        by_key[key] = row
    expected_seeds = {
        (suite, method): {row.get("seed", "") for row in method_rows}
        for (suite, method), method_rows in run_rows.items()
    }
    if any(len(seeds) != 3 for seeds in expected_seeds.values()):
        return False
    stage_ids = sorted(
        {
            key[3]
            for key in by_key
            if (key[0], key[1]) in expected_seeds and key[2] in expected_seeds[(key[0], key[1])]
        },
        key=int,
    )
    if not stage_ids:
        return False

    metrics = [
        ("saved_nonleaf_activation_peak_bytes", 1024**2, "Saved non-leaf autograd storage (MiB)"),
        ("cuda_peak_allocated_bytes", 1024**3, "Peak CUDA allocated (GiB)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.1), constrained_layout=True)
    x_positions = list(range(len(stage_ids)))
    width = 0.34
    for axis, (metric, divisor, ylabel) in zip(axes, metrics):
        for method_index, (suite, method, label, color) in enumerate(labels):
            heights: list[float] = []
            deviations: list[float] = []
            for stage_id in stage_ids:
                values = []
                for seed in expected_seeds[(suite, method)]:
                    value = to_float(by_key[(suite, method, seed, stage_id)].get(metric))
                    if value is not None:
                        values.append(value / divisor)
                if len(values) != 3:
                    return False
                mean, deviation = mean_std(values)
                heights.append(mean)
                deviations.append(deviation)
            offset = (method_index - 0.5) * width
            axis.bar(
                [position + offset for position in x_positions],
                heights,
                width=width,
                yerr=deviations,
                capsize=3,
                color=color,
                label=label,
            )
        axis.set_xticks(x_positions, [f"stage {stage_id}" for stage_id in stage_ids])
        axis.set_ylabel(ylabel)
        configure_axes(axis)
    axes[0].set_title("Activation stash measured by autograd hook")
    axes[1].set_title("Total PyTorch allocated peak")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.savefig(figure_dir / filename, dpi=220)
    plt.close(fig)
    return True


def save_e2a_memory_ledger_v0(peaks: list[dict[str, str]], report_dir: Path, figure_dir: Path) -> bool:
    """Plot the current formal E2a half-ledger without over-claiming full attribution."""
    panels = [
        (
            "Physical batch = 1, effective batch = 12",
            [
                ("e2_system", "1f1b_mb12", "S0\n1F1B", "#d97706"),
                ("e2_p2p_matched", "bpfree_p2p_b1_accum12", "S0\nBP-free", "#15803d"),
                ("e2_system", "1f1b_mb12", "S1\n1F1B", "#d97706"),
                ("e2_p2p_matched", "bpfree_p2p_b1_accum12", "S1\nBP-free", "#15803d"),
                ("e2_system", "1f1b_mb12", "S2\n1F1B", "#d97706"),
                ("e2_p2p_matched", "bpfree_p2p_b1_accum12", "S2\nBP-free", "#15803d"),
            ],
            ["0", "0", "1", "1", "2", "2"],
            "b1",
        ),
        (
            "Physical batch = 4, effective batch = 12",
            [
                ("e2_system", "1f1b_mb3", "S0\n1F1B", "#d97706"),
                ("e2_p2p_matched", "bpfree_p2p_b4_accum3", "S0\nBP-free", "#15803d"),
                ("e2_system", "1f1b_mb3", "S1\n1F1B", "#d97706"),
                ("e2_p2p_matched", "bpfree_p2p_b4_accum3", "S1\nBP-free", "#15803d"),
                ("e2_system", "1f1b_mb3", "S2\n1F1B", "#d97706"),
                ("e2_p2p_matched", "bpfree_p2p_b4_accum3", "S2\nBP-free", "#15803d"),
            ],
            ["0", "0", "1", "1", "2", "2"],
            "b4",
        ),
    ]

    by_group: dict[tuple[str, str, str], list[dict[str, float]]] = defaultdict(list)
    for row in peaks:
        suite = row.get("suite", "")
        method = row.get("method", "")
        stage_id = row.get("stage_id", "")
        allocated = to_float(row.get("cuda_peak_allocated_bytes"))
        reserved = to_float(row.get("cuda_peak_reserved_bytes"))
        activation = to_float(row.get("saved_nonleaf_activation_peak_bytes"))
        optimizer = to_float(row.get("optimizer_state_peak_bytes"))
        if allocated is None or reserved is None or optimizer is None:
            continue
        activation = activation or 0.0
        other = max(0.0, allocated - activation - optimizer)
        reserved_slack = max(0.0, reserved - allocated)
        by_group[(suite, method, stage_id)].append(
            {
                "allocated": allocated / (1024**2),
                "reserved": reserved / (1024**2),
                "activation": activation / (1024**2),
                "optimizer": optimizer / (1024**2),
                "other": other / (1024**2),
                "reserved_slack": reserved_slack / (1024**2),
            }
        )

    summary_rows: list[dict[str, Any]] = []
    for _title, method_specs, stage_ids, physical_batch in panels:
        for (suite, method, _label, _edge), stage_id in zip(method_specs, stage_ids):
            rows = by_group.get((suite, method, stage_id), [])
            if len(rows) != 3:
                return False
            aggregates: dict[str, tuple[float, float]] = {}
            for key in ("allocated", "reserved", "activation", "optimizer", "other", "reserved_slack"):
                values = [row[key] for row in rows]
                aggregates[key] = mean_std(values)
            summary_rows.append(
                {
                    "physical_batch": physical_batch,
                    "suite": suite,
                    "method": method,
                    "stage_id": stage_id,
                    "n": len(rows),
                    "peak_allocated_mean_mib": aggregates["allocated"][0],
                    "peak_allocated_std_mib": aggregates["allocated"][1],
                    "peak_reserved_mean_mib": aggregates["reserved"][0],
                    "peak_reserved_std_mib": aggregates["reserved"][1],
                    "saved_activation_mean_mib": aggregates["activation"][0],
                    "saved_activation_std_mib": aggregates["activation"][1],
                    "optimizer_state_mean_mib": aggregates["optimizer"][0],
                    "optimizer_state_std_mib": aggregates["optimizer"][1],
                    "other_allocated_mean_mib": aggregates["other"][0],
                    "other_allocated_std_mib": aggregates["other"][1],
                    "reserved_slack_mean_mib": aggregates["reserved_slack"][0],
                    "reserved_slack_std_mib": aggregates["reserved_slack"][1],
                }
            )

    write_csv(report_dir / "e2a_memory_ledger_v0.csv", summary_rows)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2))
    component_colors = {
        "activation": "#2f80ed",
        "optimizer": "#27ae60",
        "other": "#c7ced8",
        "reserved_slack": "#f4c27a",
    }
    for axis, (title, method_specs, stage_ids, physical_batch) in zip(axes, panels):
        positions = list(range(len(method_specs)))
        activation_heights = []
        optimizer_heights = []
        other_heights = []
        reserved_slack_heights = []
        allocated_means = []
        allocated_stds = []
        edge_colors = []
        tick_labels = []
        for position, ((suite, method, label, edge_color), stage_id) in enumerate(zip(method_specs, stage_ids)):
            row = next(
                item
                for item in summary_rows
                if item["physical_batch"] == physical_batch
                and item["suite"] == suite
                and item["method"] == method
                and item["stage_id"] == stage_id
            )
            activation_heights.append(float(row["saved_activation_mean_mib"]))
            optimizer_heights.append(float(row["optimizer_state_mean_mib"]))
            other_heights.append(float(row["other_allocated_mean_mib"]))
            reserved_slack_heights.append(float(row["reserved_slack_mean_mib"]))
            allocated_means.append(float(row["peak_allocated_mean_mib"]))
            allocated_stds.append(float(row["peak_allocated_std_mib"]))
            edge_colors.append(edge_color)
            tick_labels.append(label)

        axis.bar(
            positions,
            activation_heights,
            color=component_colors["activation"],
            edgecolor=edge_colors,
            linewidth=1.5,
            width=0.76,
            label="saved activation",
        )
        axis.bar(
            positions,
            optimizer_heights,
            bottom=activation_heights,
            color=component_colors["optimizer"],
            edgecolor=edge_colors,
            linewidth=1.5,
            width=0.76,
            label="optimizer state",
        )
        stacked_bottom = [a + b for a, b in zip(activation_heights, optimizer_heights)]
        axis.bar(
            positions,
            other_heights,
            bottom=stacked_bottom,
            color=component_colors["other"],
            edgecolor=edge_colors,
            linewidth=1.5,
            width=0.76,
            label="other allocated",
        )
        allocated_tops = [a + b + c for a, b, c in zip(activation_heights, optimizer_heights, other_heights)]
        axis.bar(
            positions,
            reserved_slack_heights,
            bottom=allocated_tops,
            color=component_colors["reserved_slack"],
            edgecolor=edge_colors,
            linewidth=1.0,
            width=0.76,
            hatch="////",
            alpha=0.45,
            label="reserved slack",
        )
        axis.errorbar(
            positions,
            allocated_means,
            yerr=allocated_stds,
            fmt="none",
            ecolor="#111111",
            elinewidth=1.0,
            capsize=3,
            zorder=5,
        )
        axis.set_xticks(positions, tick_labels)
        axis.set_ylabel("Peak memory (MiB)")
        axis.set_title(title, fontsize=10.5)
        configure_axes(axis)
        axis.set_axisbelow(True)
        for stage_boundary in (1.5, 3.5):
            axis.axvline(stage_boundary, color="#d5d9df", linewidth=0.8, linestyle="--", zorder=0)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=component_colors["activation"], edgecolor="none"),
        plt.Rectangle((0, 0), 1, 1, facecolor=component_colors["optimizer"], edgecolor="none"),
        plt.Rectangle((0, 0), 1, 1, facecolor=component_colors["other"], edgecolor="none"),
        plt.Rectangle((0, 0), 1, 1, facecolor=component_colors["reserved_slack"], edgecolor="none", hatch="////", alpha=0.45),
    ]
    labels = ["saved activation", "optimizer state", "other allocated", "reserved slack"]
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle(
        "AG News E2a memory ledger v0: identifiable peaks plus unresolved allocated remainder",
        fontsize=12,
        y=0.995,
    )
    fig.text(
        0.5,
        0.02,
        "other allocated = peak allocated - saved activation - optimizer state; it still mixes base shard, duplicated local readout, gradients, workspace, and communication/runtime state.",
        ha="center",
        va="bottom",
        fontsize=8.4,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0.02, 0.08, 0.98, 0.92))
    fig.savefig(figure_dir / "e2a_memory_ledger_v0.png", dpi=240)
    plt.close(fig)
    return True


def save_e2_p2p_memory_ledger(peaks: list[dict[str, str]], report_dir: Path, figure_dir: Path) -> bool:
    required_fields = [
        "cuda_peak_allocated_bytes",
        "cuda_peak_reserved_bytes",
        "base_shard_param_bytes",
        "local_readout_param_bytes",
        "input_embedding_param_bytes",
        "gradient_storage_peak_bytes",
        "optimizer_state_peak_bytes",
        "saved_nonleaf_activation_peak_bytes",
        "identified_allocated_peak_bytes",
        "runtime_residual_peak_bytes",
    ]
    relevant = [row for row in peaks if row.get("suite") == "e2_p2p_memory_ledger"]
    if not relevant:
        return False
    if not all(any(row.get(field) not in (None, "") for row in relevant) for field in required_fields):
        return False

    panels = [
        (
            "b1",
            "Physical batch = 1, effective batch = 12",
            [
                ("1f1b_mb12_tracker", 0, "S0\n1F1B", "#d97706"),
                ("bpfree_p2p_b1_accum12_tracker", 0, "S0\nBP-free", "#15803d"),
                ("1f1b_mb12_tracker", 1, "S1\n1F1B", "#d97706"),
                ("bpfree_p2p_b1_accum12_tracker", 1, "S1\nBP-free", "#15803d"),
                ("1f1b_mb12_tracker", 2, "S2\n1F1B", "#d97706"),
                ("bpfree_p2p_b1_accum12_tracker", 2, "S2\nBP-free", "#15803d"),
            ],
        ),
        (
            "b4",
            "Physical batch = 4, effective batch = 12",
            [
                ("1f1b_mb3_tracker", 0, "S0\n1F1B", "#d97706"),
                ("bpfree_p2p_b4_accum3_tracker", 0, "S0\nBP-free", "#15803d"),
                ("1f1b_mb3_tracker", 1, "S1\n1F1B", "#d97706"),
                ("bpfree_p2p_b4_accum3_tracker", 1, "S1\nBP-free", "#15803d"),
                ("1f1b_mb3_tracker", 2, "S2\n1F1B", "#d97706"),
                ("bpfree_p2p_b4_accum3_tracker", 2, "S2\nBP-free", "#15803d"),
            ],
        ),
    ]
    component_colors = {
        "transformer_shard": "#9aa0a6",
        "input_embedding": "#56ccf2",
        "local_readout": "#f2994a",
        "gradient": "#bb6bd9",
        "optimizer": "#27ae60",
        "activation": "#2f80ed",
        "residual": "#d6dde5",
    }

    by_key: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in relevant:
        method = row.get("method", "")
        stage_id = row.get("stage_id", "")
        if method and stage_id:
            by_key[(method, stage_id)].append(row)

    summary_rows: list[dict[str, Any]] = []
    for physical_batch, _title, specs in panels:
        for method, stage_id, _label, _edge_color in specs:
            rows = by_key.get((method, str(stage_id)), [])
            if len(rows) != 3:
                return False

            def metric_values(column: str, *, default: float | None = None) -> list[float]:
                values = []
                for row in rows:
                    value = to_float(row.get(column))
                    if value is None:
                        if default is None:
                            raise ValueError(f"Missing field {column} for {method} stage {stage_id}")
                        value = default
                    values.append(value)
                if any(value is None for value in values):
                    raise ValueError(f"Missing field {column} for {method} stage {stage_id}")
                return [float(value) / (1024**2) for value in values if value is not None]

            peak_allocated_values = metric_values("cuda_peak_allocated_bytes")
            peak_reserved_values = metric_values("cuda_peak_reserved_bytes")
            base_shard_values = metric_values("base_shard_param_bytes")
            input_embedding_values = metric_values("input_embedding_param_bytes")
            transformer_shard_values = [
                max(0.0, base - embed) for base, embed in zip(base_shard_values, input_embedding_values)
            ]
            local_readout_values = metric_values("local_readout_param_bytes")
            gradient_values = metric_values("gradient_storage_peak_bytes")
            optimizer_values = metric_values("optimizer_state_peak_bytes")
            activation_values = metric_values("saved_nonleaf_activation_peak_bytes")
            residual_values = metric_values("runtime_residual_peak_bytes")
            identified_values = metric_values("identified_allocated_peak_bytes")
            output_hidden_values = metric_values("output_hidden_peak_bytes", default=0.0)
            output_log_probs_values = metric_values("output_log_probs_peak_bytes", default=0.0)

            peak_allocated_mean, peak_allocated_std = mean_std(peak_allocated_values)
            peak_reserved_mean, peak_reserved_std = mean_std(peak_reserved_values)
            transformer_shard_mean, transformer_shard_std = mean_std(transformer_shard_values)
            input_embedding_mean, input_embedding_std = mean_std(input_embedding_values)
            local_readout_mean, local_readout_std = mean_std(local_readout_values)
            gradient_mean, gradient_std = mean_std(gradient_values)
            optimizer_mean, optimizer_std = mean_std(optimizer_values)
            activation_mean, activation_std = mean_std(activation_values)
            residual_mean, residual_std = mean_std(residual_values)
            identified_mean, identified_std = mean_std(identified_values)
            output_hidden_mean, output_hidden_std = mean_std(output_hidden_values)
            output_log_probs_mean, output_log_probs_std = mean_std(output_log_probs_values)

            summary_rows.append(
                {
                    "physical_batch": physical_batch,
                    "suite": "e2_p2p_memory_ledger",
                    "method": method,
                    "stage_id": stage_id,
                    "n": len(rows),
                    "peak_allocated_mean_mib": peak_allocated_mean,
                    "peak_allocated_std_mib": peak_allocated_std,
                    "peak_reserved_mean_mib": peak_reserved_mean,
                    "peak_reserved_std_mib": peak_reserved_std,
                    "transformer_shard_mean_mib": transformer_shard_mean,
                    "transformer_shard_std_mib": transformer_shard_std,
                    "input_embedding_mean_mib": input_embedding_mean,
                    "input_embedding_std_mib": input_embedding_std,
                    "local_readout_mean_mib": local_readout_mean,
                    "local_readout_std_mib": local_readout_std,
                    "gradient_storage_mean_mib": gradient_mean,
                    "gradient_storage_std_mib": gradient_std,
                    "optimizer_state_mean_mib": optimizer_mean,
                    "optimizer_state_std_mib": optimizer_std,
                    "saved_activation_mean_mib": activation_mean,
                    "saved_activation_std_mib": activation_std,
                    "identified_allocated_mean_mib": identified_mean,
                    "identified_allocated_std_mib": identified_std,
                    "runtime_residual_mean_mib": residual_mean,
                    "runtime_residual_std_mib": residual_std,
                    "output_hidden_peak_mean_mib": output_hidden_mean,
                    "output_hidden_peak_std_mib": output_hidden_std,
                    "output_log_probs_peak_mean_mib": output_log_probs_mean,
                    "output_log_probs_peak_std_mib": output_log_probs_std,
                }
            )

    write_csv(report_dir / "e2_p2p_memory_ledger.csv", summary_rows)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4), constrained_layout=True, sharey=False)
    for axis, (physical_batch, title, specs) in zip(axes, panels):
        positions = list(range(len(specs)))
        bottoms = [0.0 for _ in specs]
        tick_labels = []
        edge_colors = []
        peak_means = []
        peak_stds = []
        stack_components = [
            ("transformer_shard_mean_mib", "transformer shard", component_colors["transformer_shard"]),
            ("input_embedding_mean_mib", "input embedding", component_colors["input_embedding"]),
            ("local_readout_mean_mib", "local readout", component_colors["local_readout"]),
            ("gradient_storage_mean_mib", "gradient storage", component_colors["gradient"]),
            ("optimizer_state_mean_mib", "optimizer state", component_colors["optimizer"]),
            ("saved_activation_mean_mib", "saved activation", component_colors["activation"]),
            ("runtime_residual_mean_mib", "runtime residual", component_colors["residual"]),
        ]

        by_summary_key = {
            (row["physical_batch"], row["method"], str(row["stage_id"])): row for row in summary_rows
        }
        for method, stage_id, label, edge_color in specs:
            row = by_summary_key.get((physical_batch, method, str(stage_id)))
            if row is None:
                return False
            tick_labels.append(label)
            edge_colors.append(edge_color)
            peak_means.append(float(row["peak_allocated_mean_mib"]))
            peak_stds.append(float(row["peak_allocated_std_mib"]))

        for column, label, color in stack_components:
            heights = []
            for method, stage_id, _label, _edge_color in specs:
                row = by_summary_key[(physical_batch, method, str(stage_id))]
                heights.append(float(row[column]))
            axis.bar(
                positions,
                heights,
                bottom=bottoms,
                width=0.76,
                color=color,
                edgecolor=edge_colors,
                linewidth=1.5,
                label=label,
            )
            bottoms = [bottom + height for bottom, height in zip(bottoms, heights)]

        axis.errorbar(
            positions,
            peak_means,
            yerr=peak_stds,
            fmt="_",
            color="#111111",
            ecolor="#111111",
            elinewidth=1.0,
            capsize=3,
            markersize=14,
            zorder=5,
        )
        for boundary in (1.5, 3.5):
            axis.axvline(boundary, color="#d5d9df", linewidth=0.85, linestyle="--", zorder=0)
        axis.set_xticks(positions, tick_labels)
        axis.set_ylabel("Peak CUDA memory (MiB)")
        axis.set_title(title, fontsize=10.8)
        configure_axes(axis)
        axis.set_axisbelow(True)

    b4_stage0_1f1b = next(
        row for row in summary_rows if row["physical_batch"] == "b4" and row["method"] == "1f1b_mb3_tracker" and row["stage_id"] == 0
    )
    b4_stage0_bpfree = next(
        row
        for row in summary_rows
        if row["physical_batch"] == "b4" and row["method"] == "bpfree_p2p_b4_accum3_tracker" and row["stage_id"] == 0
    )
    activation_delta = float(b4_stage0_1f1b["saved_activation_mean_mib"]) - float(
        b4_stage0_bpfree["saved_activation_mean_mib"]
    )
    readout_delta = float(b4_stage0_bpfree["local_readout_mean_mib"]) - float(
        b4_stage0_1f1b["local_readout_mean_mib"]
    )
    axes[1].annotate(
        (
            f"Stage 0 activation drop: {activation_delta:.1f} MiB\n"
            f"Stage 0 extra local readout: {readout_delta:.1f} MiB"
        ),
        xy=(0.50, 0.84),
        xycoords="axes fraction",
        ha="left",
        va="center",
        fontsize=8.8,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#cbd5e1", "alpha": 0.96},
    )

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="none")
        for color in (
            component_colors["transformer_shard"],
            component_colors["input_embedding"],
            component_colors["local_readout"],
            component_colors["gradient"],
            component_colors["optimizer"],
            component_colors["activation"],
            component_colors["residual"],
        )
    ]
    labels = [
        "transformer shard",
        "input embedding",
        "local readout",
        "gradient storage",
        "optimizer state",
        "saved activation",
        "runtime residual",
    ]
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("AG News E2 matched memory ledger", fontsize=12, y=1.05)
    fig.text(
        0.5,
        -0.02,
        "runtime residual = peak allocated - (resident model params + gradient storage + optimizer state + saved non-leaf activations).",
        ha="center",
        va="top",
        fontsize=8.5,
        color="#4b5563",
    )
    fig.savefig(figure_dir / "e2_p2p_memory_ledger.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    return True


def save_e2a_memory_ledger_partial(report_dir: Path, figure_dir: Path) -> bool:
    rows = read_csv(report_dir / "e2a_memory_ledger_v0.csv")
    if not rows:
        return False

    method_order = [
        ("e2_system", "1f1b_mb12", "b1", 0, "S0\n1F1B", "#d97706"),
        ("e2_p2p_matched", "bpfree_p2p_b1_accum12", "b1", 0, "S0\nBP-free", "#15803d"),
        ("e2_system", "1f1b_mb12", "b1", 1, "S1\n1F1B", "#d97706"),
        ("e2_p2p_matched", "bpfree_p2p_b1_accum12", "b1", 1, "S1\nBP-free", "#15803d"),
        ("e2_system", "1f1b_mb12", "b1", 2, "S2\n1F1B", "#d97706"),
        ("e2_p2p_matched", "bpfree_p2p_b1_accum12", "b1", 2, "S2\nBP-free", "#15803d"),
        ("e2_system", "1f1b_mb3", "b4", 0, "S0\n1F1B", "#d97706"),
        ("e2_p2p_matched", "bpfree_p2p_b4_accum3", "b4", 0, "S0\nBP-free", "#15803d"),
        ("e2_system", "1f1b_mb3", "b4", 1, "S1\n1F1B", "#d97706"),
        ("e2_p2p_matched", "bpfree_p2p_b4_accum3", "b4", 1, "S1\nBP-free", "#15803d"),
        ("e2_system", "1f1b_mb3", "b4", 2, "S2\n1F1B", "#d97706"),
        ("e2_p2p_matched", "bpfree_p2p_b4_accum3", "b4", 2, "S2\nBP-free", "#15803d"),
    ]
    by_key = {
        (
            row["suite"],
            row["method"],
            row["physical_batch"],
            int(row["stage_id"]),
        ): row
        for row in rows
    }

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.4, 5.2),
        constrained_layout=True,
        sharey=False,
    )
    panel_specs = [
        ("b1", "Physical batch = 1, effective batch = 12"),
        ("b4", "Physical batch = 4, effective batch = 12"),
    ]
    component_colors = {
        "activation": "#2f80ed",
        "optimizer": "#27ae60",
        "other": "#c7ced8",
        "slack": "#f4c27a",
    }
    ymax = {"b1": 1040.0, "b4": 1600.0}
    for axis, (physical_batch, title) in zip(axes, panel_specs):
        subset = [spec for spec in method_order if spec[2] == physical_batch]
        positions = list(range(len(subset)))
        tick_labels = []
        edge_colors = []
        activation = []
        optimizer = []
        other = []
        slack = []
        allocated = []
        for suite, method, batch_key, stage_id, label, edge_color in subset:
            row = by_key.get((suite, method, batch_key, stage_id))
            if row is None:
                return False
            tick_labels.append(label)
            edge_colors.append(edge_color)
            activation.append(float(row["saved_activation_mean_mib"]))
            optimizer.append(float(row["optimizer_state_mean_mib"]))
            other.append(float(row["other_allocated_mean_mib"]))
            slack.append(float(row["reserved_slack_mean_mib"]))
            allocated.append(float(row["peak_allocated_mean_mib"]))

        axis.bar(
            positions,
            activation,
            width=0.78,
            color=component_colors["activation"],
            edgecolor=edge_colors,
            linewidth=1.6,
            label="saved activation",
        )
        axis.bar(
            positions,
            optimizer,
            width=0.78,
            bottom=activation,
            color=component_colors["optimizer"],
            edgecolor=edge_colors,
            linewidth=1.6,
            label="optimizer state",
        )
        stack_mid = [a + b for a, b in zip(activation, optimizer)]
        axis.bar(
            positions,
            other,
            width=0.78,
            bottom=stack_mid,
            color=component_colors["other"],
            edgecolor=edge_colors,
            linewidth=1.6,
            label="unresolved allocated remainder",
        )
        stack_top = [a + b + c for a, b, c in zip(activation, optimizer, other)]
        axis.bar(
            positions,
            slack,
            width=0.78,
            bottom=stack_top,
            color=component_colors["slack"],
            edgecolor=edge_colors,
            linewidth=1.0,
            hatch="////",
            alpha=0.45,
            label="reserved slack",
        )
        axis.errorbar(
            positions,
            allocated,
            fmt="_",
            markersize=14,
            color="#111111",
            elinewidth=1.0,
            capsize=0,
            zorder=5,
        )
        for boundary in (1.5, 3.5):
            axis.axvline(boundary, color="#d5d9df", linewidth=0.85, linestyle="--", zorder=0)
        axis.set_xticks(positions, tick_labels)
        axis.set_ylabel("Peak CUDA memory (MiB)")
        axis.set_title(title, fontsize=10.8)
        axis.set_ylim(0, ymax[physical_batch])
        configure_axes(axis)
        axis.set_axisbelow(True)

    b4_stage0_1f1b = by_key[("e2_system", "1f1b_mb3", "b4", 0)]
    b4_stage0_bpfree = by_key[("e2_p2p_matched", "bpfree_p2p_b4_accum3", "b4", 0)]
    activation_drop = (
        float(b4_stage0_1f1b["saved_activation_mean_mib"])
        - float(b4_stage0_bpfree["saved_activation_mean_mib"])
    )
    axes[1].annotate(
        f"Stage 0 saved activation\n{float(b4_stage0_1f1b['saved_activation_mean_mib']):.1f} → {float(b4_stage0_bpfree['saved_activation_mean_mib']):.1f} MiB\n(−{activation_drop:.1f} MiB)",
        xy=(0.52, 0.83),
        xycoords="axes fraction",
        ha="left",
        va="center",
        fontsize=8.8,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#cbd5e1", "alpha": 0.96},
    )

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=component_colors["activation"], edgecolor="none"),
        plt.Rectangle((0, 0), 1, 1, facecolor=component_colors["optimizer"], edgecolor="none"),
        plt.Rectangle((0, 0), 1, 1, facecolor=component_colors["other"], edgecolor="none"),
        plt.Rectangle((0, 0), 1, 1, facecolor=component_colors["slack"], edgecolor="none", hatch="////", alpha=0.45),
    ]
    labels = [
        "saved activation",
        "optimizer state",
        "unresolved allocated remainder",
        "reserved slack",
    ]
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(
        "E2a partial CUDA ledger: activation peak is identified, but most of the peak is not yet fully attributed",
        fontsize=12,
        y=1.03,
    )
    fig.text(
        0.5,
        -0.02,
        "Unresolved allocated remainder = peak allocated - saved activation - optimizer state. It still mixes base shard, local head/readout, gradients, workspace, and communication/runtime state.",
        ha="center",
        va="top",
        fontsize=8.5,
        color="#4b5563",
    )
    fig.savefig(figure_dir / "e2a_memory_ledger_partial.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot normalized AG News E1/E2 protocol reports.")
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--report_dir", type=Path, default=None)
    parser.add_argument("--figure_dir", type=Path, default=None)
    args = parser.parse_args()
    report_dir = args.report_dir or args.output_root / "report"
    figure_dir = args.figure_dir or report_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    curves = read_csv(report_dir / "quality_curve_raw.csv")
    runs = read_csv(report_dir / "normalized_runs.csv")
    peaks = read_csv(report_dir / "stage_peak_metrics.csv")
    produced = []
    if save_quality_curve(curves, figure_dir, "choice_accuracy", "e1_validation_accuracy.png", "Choice accuracy"):
        produced.append("e1_validation_accuracy.png")
    if save_quality_curve(curves, figure_dir, "choice_loss", "e1_validation_loss.png", "Choice loss"):
        produced.append("e1_validation_loss.png")
    if save_quality_seed_traces(curves, figure_dir):
        produced.append("e1_validation_seed_traces.png")
    if save_final_accuracy(runs, figure_dir):
        produced.append("e1_final_test_accuracy.png")
    if save_stage_memory(peaks, figure_dir):
        produced.append("e1_stage_peak_cuda_allocated.png")
    if save_1f1b_microbatch_tradeoff(runs, peaks, figure_dir):
        produced.append("e2_1f1b_microbatch_tradeoff.png")
    if save_cpu_queue_window_saturation(runs, figure_dir):
        produced.append("e2_cpu_queue_window_saturation.png")
    request_throughput_labels = [
        ("e2_p2p_throughput", "1f1b_mb12_no_tracker", "1F1B mb=12"),
        ("e2_p2p_throughput", "bpfree_p2p_b1_accum12_no_tracker", "BP-free P2P b=1"),
    ]
    batched_throughput_labels = [
        ("e2_p2p_throughput", "1f1b_mb3_no_tracker", "1F1B mb=3"),
        ("e2_p2p_throughput", "bpfree_p2p_b4_accum3_no_tracker", "BP-free P2P b=4"),
    ]
    request_memory_labels = [
        ("e2_system", "1f1b_mb12", "1F1B mb=12"),
        ("e2_p2p_matched", "bpfree_p2p_b1_accum12", "BP-free P2P b=1"),
    ]
    batched_memory_labels = [
        ("e2_system", "1f1b_mb3", "1F1B mb=3"),
        ("e2_p2p_matched", "bpfree_p2p_b4_accum3", "BP-free P2P b=4"),
    ]
    if save_e2_p2p_comparison(
        runs,
        peaks,
        figure_dir,
        labels=request_throughput_labels,
        title="AG News E2 P2P throughput: physical batch=1, effective batch=12 (no activation tracker)",
        filename="e2_p2p_request_level_comparison.png",
    ):
        produced.append("e2_p2p_request_level_comparison.png")
    if save_e2_p2p_comparison(
        runs,
        peaks,
        figure_dir,
        labels=batched_throughput_labels,
        title="AG News E2 P2P throughput: physical batch=4, effective batch=12 (no activation tracker)",
        filename="e2_p2p_batched_level_comparison.png",
    ):
        produced.append("e2_p2p_batched_level_comparison.png")
    if save_e2_p2p_progress(
        args.output_root,
        report_dir,
        figure_dir,
        labels=request_throughput_labels,
        title="AG News E2 P2P progress: physical batch=1, effective batch=12",
        filename="e2_p2p_request_progress.png",
    ):
        produced.append("e2_p2p_request_progress.png")
    if save_e2_p2p_progress(
        args.output_root,
        report_dir,
        figure_dir,
        labels=batched_throughput_labels,
        title="AG News E2 P2P progress: physical batch=4, effective batch=12",
        filename="e2_p2p_batched_progress.png",
    ):
        produced.append("e2_p2p_batched_progress.png")
    if save_e2_p2p_activation_stash_by_stage(
        runs,
        peaks,
        figure_dir,
        labels=[(*item, color) for item, color in zip(request_memory_labels, ["#ff7f0e", "#2ca02c"])],
        title="AG News E2 P2P activation: physical batch=1, effective batch=12 (mean +/- sd, n=3)",
        filename="e2_p2p_activation_stash_by_stage.png",
    ):
        produced.append("e2_p2p_activation_stash_by_stage.png")
    if save_e2_p2p_activation_stash_by_stage(
        runs,
        peaks,
        figure_dir,
        labels=[(*item, color) for item, color in zip(batched_memory_labels, ["#ff7f0e", "#2ca02c"])],
        title="AG News E2 P2P activation: physical batch=4, effective batch=12 (mean +/- sd, n=3)",
        filename="e2_p2p_batched_activation_stash_by_stage.png",
    ):
        produced.append("e2_p2p_batched_activation_stash_by_stage.png")
    if save_e2_p2p_memory_ledger(peaks, report_dir, figure_dir):
        produced.append("e2_p2p_memory_ledger.png")
    if save_e2a_memory_ledger_v0(peaks, report_dir, figure_dir):
        produced.append("e2a_memory_ledger_v0.png")
    if save_e2a_memory_ledger_partial(report_dir, figure_dir):
        produced.append("e2a_memory_ledger_partial.png")
    manifest = {"figures": produced, "quality_rows": len(curves), "runs": len(runs), "stage_peak_rows": len(peaks)}
    (figure_dir / "plot_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
