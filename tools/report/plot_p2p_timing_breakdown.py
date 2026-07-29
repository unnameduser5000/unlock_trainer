#!/usr/bin/env python3
"""Turn instrumented E4 P2P stage metrics into an execution-time breakdown.

This deliberately distinguishes timed blocking NCCL calls from the residual
pipeline wait. The latter must not be presented as pure communication latency.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    try:
        return float(value) if value not in ("", None) else None
    except (TypeError, ValueError):
        return None


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def configure_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", alpha=0.24, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def timing_rows(seed_dir: Path, *, method: str, skip_batches: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if method.startswith("1f1b_"):
        metric_paths = sorted(seed_dir.glob("rank*_stage_metrics.csv"))
        for path in metric_paths:
            for row in read_csv(path):
                if row.get("phase") != "train" or int(row.get("batch_seq", -1)) < skip_batches:
                    continue
                forward = number(row, "forward_ms")
                backward = number(row, "backward_ms")
                residual = number(row, "pipeline_unattributed_ms")
                if forward is None or backward is None or residual is None:
                    raise RuntimeError(
                        f"{path} lacks the instrumented 1F1B timing columns. Rerun E4 P2P with the current runner."
                    )
                rows.append(
                    {
                        "seed": seed_dir.name.removeprefix("seed"),
                        "method": method,
                        "stage": int(row["stage_id"]),
                        "forward_ms": forward,
                        "backward_ms": backward,
                        "optimizer_ms": number(row, "optimizer_ms") or 0.0,
                        "transfer_or_wait_ms": residual,
                        "load_ms": number(row, "h2d_ms") or 0.0,
                        "fill_wait_ms": number(row, "pipeline_fill_wait_ms") or 0.0,
                        "interior_wait_ms": number(row, "pipeline_interior_wait_ms") or 0.0,
                        "tail_wait_ms": number(row, "pipeline_tail_wait_ms") or 0.0,
                        "logical_updates": 1.0,
                        "source": "1F1B residual = NCCL rendezvous + pipeline scheduling wait",
                    }
                )
        return rows

    metric_paths = sorted(seed_dir.glob("train.stage*.metrics.csv"))
    for path in metric_paths:
        for row in read_csv(path):
            if row.get("phase") != "train" or int(row.get("batch_seq", -1)) < skip_batches:
                continue
            forward = number(row, "forward_ms")
            backward = number(row, "backward_ms")
            blocking = number(row, "nccl_blocking_ms")
            other = number(row, "unattributed_ms")
            if forward is None or backward is None or blocking is None or other is None:
                raise RuntimeError(
                    f"{path} lacks the instrumented BP-free timing columns. Rerun E4 P2P with the current runner."
                )
            accumulation = number(row, "gradient_accumulation_steps") or 1.0
            rows.append(
                {
                    "seed": seed_dir.name.removeprefix("seed"),
                    "method": method,
                    "stage": int(row["stage_id"]),
                    "forward_ms": forward,
                    "backward_ms": backward,
                    "optimizer_ms": number(row, "optimizer_ms") or 0.0,
                    "transfer_or_wait_ms": blocking,
                    "load_ms": (number(row, "load_hidden_ms") or 0.0) + (number(row, "load_input_ms") or 0.0),
                    "fill_wait_ms": 0.0,
                    "interior_wait_ms": other,
                    "tail_wait_ms": 0.0,
                    # Convert physical-batch measurements to a 12-sample local update.
                    "logical_updates": 1.0 / accumulation,
                    "source": "BP-free transfer = timed blocking NCCL send/recv interval",
                }
            )
    return rows


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_seed_stage: dict[tuple[str, str, int], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in rows:
        key = (str(row["method"]), str(row["seed"]), int(row["stage"]))
        for component in (
            "forward_ms",
            "backward_ms",
            "optimizer_ms",
            "transfer_or_wait_ms",
            "load_ms",
            "fill_wait_ms",
            "interior_wait_ms",
            "tail_wait_ms",
        ):
            by_seed_stage[key][component] += float(row[component])
        by_seed_stage[key]["logical_updates"] += float(row["logical_updates"])

    result: list[dict[str, Any]] = []
    per_method_stage: dict[tuple[str, int], list[dict[str, float]]] = defaultdict(list)
    for (method, _seed, stage), values in by_seed_stage.items():
        updates = values["logical_updates"]
        if updates <= 0:
            continue
        per_method_stage[(method, stage)].append(
            {component: values[component] / updates for component in values if component != "logical_updates"}
        )
    for (method, stage), per_seed in sorted(per_method_stage.items()):
        row: dict[str, Any] = {"method": method, "stage": stage, "seeds": len(per_seed)}
        for component in (
            "forward_ms",
            "backward_ms",
            "optimizer_ms",
            "transfer_or_wait_ms",
            "load_ms",
            "fill_wait_ms",
            "interior_wait_ms",
            "tail_wait_ms",
        ):
            average, deviation = mean_std([values[component] for values in per_seed])
            row[component] = average
            row[f"{component}_sd"] = deviation
        result.append(row)
    return result


def plot_breakdown(rows: list[dict[str, Any]], output: Path, *, left: str, right: str) -> None:
    labels = [(left, "1F1B", "#ff7f0e"), (right, "BP-free", "#2ca02c")]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), constrained_layout=True)
    for axis, (method, label, color) in zip(axes, labels):
        stage_rows = sorted((row for row in rows if row["method"] == method), key=lambda row: row["stage"])
        positions = list(range(len(stage_rows)))
        bottoms = [0.0] * len(stage_rows)
        components = [
            ("load_ms", "Input load", "#4c78a8"),
            ("forward_ms", "Forward", "#72b7b2"),
            ("backward_ms", "Backward", "#eeca3b"),
            ("optimizer_ms", "Optimizer step", "#e45756"),
        ]
        if method.startswith("1f1b_"):
            components.append(("transfer_or_wait_ms", "NCCL + pipeline wait", "#b279a2"))
        else:
            components.extend(
                [
                    ("transfer_or_wait_ms", "Blocking NCCL interval", "#b279a2"),
                    ("interior_wait_ms", "Other runtime wait", "#bab0ac"),
                ]
            )
        for key, component_label, component_color in components:
            values = [float(row[key]) for row in stage_rows]
            if not any(values):
                continue
            axis.bar(positions, values, bottom=bottoms, color=component_color, label=component_label)
            bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
        axis.set_xticks(positions, [f"stage {row['stage']}" for row in stage_rows])
        axis.set_ylabel("Mean ms per 12-sample update")
        axis.set_title(label)
        configure_axes(axis)
    handles: list[Any] = []
    labels_text: list[str] = []
    for axis in axes:
        axis_handles, axis_labels = axis.get_legend_handles_labels()
        for handle, label in zip(axis_handles, axis_labels):
            if label not in labels_text:
                handles.append(handle)
                labels_text.append(label)
    fig.legend(handles, labels_text, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("E2 P2P per-stage execution breakdown (steady state, mean across seeds)")
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_1f1b_wait(rows: list[dict[str, Any]], output: Path, *, method: str) -> None:
    stage_rows = sorted((row for row in rows if row["method"] == method), key=lambda row: row["stage"])
    fig, ax = plt.subplots(figsize=(7.1, 4.2), constrained_layout=True)
    positions = list(range(len(stage_rows)))
    bottoms = [0.0] * len(stage_rows)
    for key, label, color in (
        ("fill_wait_ms", "Fill before first F/B", "#9ecae1"),
        ("interior_wait_ms", "Inter-microbatch wait", "#fdae6b"),
        ("tail_wait_ms", "Tail after final F/B", "#dadaeb"),
    ):
        values = [float(row[key]) for row in stage_rows]
        ax.bar(positions, values, bottom=bottoms, color=color, label=label)
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
    ax.set_xticks(positions, [f"stage {row['stage']}" for row in stage_rows])
    ax.set_ylabel("Mean ms per global update")
    ax.set_title("1F1B fill, interior wait, and tail/drain")
    configure_axes(ax)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot E4 P2P timing breakdown from instrumented metrics.")
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--left_method", default="1f1b_mb3_clean")
    parser.add_argument("--right_method", default="bpfree_p2p_b4_accum3_clean")
    parser.add_argument("--skip_batches", type=int, default=16)
    parser.add_argument("--report_dir", type=Path, default=None)
    args = parser.parse_args()

    report_dir = args.report_dir or args.output_root / "report"
    figure_dir = report_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    raw_root = args.output_root / "e2_p2p_throughput_clean"
    raw_rows: list[dict[str, Any]] = []
    for method in (args.left_method, args.right_method):
        seed_dirs = sorted((raw_root / method).glob("seed*"))
        if len(seed_dirs) != 3:
            raise RuntimeError(f"Expected three seed directories for {method}, found {len(seed_dirs)}")
        for seed_dir in seed_dirs:
            raw_rows.extend(timing_rows(seed_dir, method=method, skip_batches=args.skip_batches))
    summary = aggregate(raw_rows)
    if not summary:
        raise RuntimeError("No instrumented timing rows found.")

    csv_path = report_dir / "e2_p2p_batched_timing_breakdown.csv"
    fields = list(summary[0])
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    plot_breakdown(
        summary,
        figure_dir / "e2_p2p_batched_timing_breakdown.png",
        left=args.left_method,
        right=args.right_method,
    )
    plot_1f1b_wait(summary, figure_dir / "e2_p2p_1f1b_fill_drain.png", method=args.left_method)
    print(
        json.dumps(
            {
                "summary_csv": str(csv_path),
                "breakdown_figure": str(figure_dir / "e2_p2p_batched_timing_breakdown.png"),
                "fill_drain_figure": str(figure_dir / "e2_p2p_1f1b_fill_drain.png"),
                "skip_batches": args.skip_batches,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
