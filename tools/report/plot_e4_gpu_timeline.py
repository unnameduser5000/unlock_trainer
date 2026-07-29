#!/usr/bin/env python3
"""Render one independent, measured CUDA timeline per E4 schedule."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from paper_style import METHOD_LABELS, METHOD_ORDER, configure_paper_style, save_figure


COMPUTE_COLOR = "#3F72B5"
TRANSFER_COLOR = "#43A6A1"
IDLE_COLOR = "#F0F1F2"


def occupancy_bins(
    rows: pd.DataFrame,
    *,
    window_ms: float,
    bin_ms: float,
) -> np.ndarray:
    bins = int(np.ceil(window_ms / bin_ms))
    occupancy = np.zeros(bins, dtype=float)
    for row in rows.itertuples(index=False):
        start = max(0.0, float(row.start_ms))
        end = min(window_ms, float(row.end_ms))
        if end <= start:
            continue
        first = max(0, int(np.floor(start / bin_ms)))
        last = min(bins - 1, int(np.floor(np.nextafter(end, -np.inf) / bin_ms)))
        for index in range(first, last + 1):
            left = index * bin_ms
            right = min(window_ms, left + bin_ms)
            overlap = max(0.0, min(end, right) - max(start, left))
            occupancy[index] += overlap / max(right - left, np.finfo(float).eps)
    return np.clip(occupancy, 0.0, 1.0)


def blend(base: str, strength: np.ndarray) -> np.ndarray:
    color = np.asarray(to_rgb(base), dtype=float)
    background = np.asarray(to_rgb(IDLE_COLOR), dtype=float)
    alpha = np.clip(strength, 0.0, 1.0)[:, None]
    return background + alpha * (color - background)


def timeline_image(
    intervals: pd.DataFrame,
    *,
    window_ms: float,
    bin_ms: float,
) -> np.ndarray:
    bins = int(np.ceil(window_ms / bin_ms))
    band_px = 26
    image = np.empty((3 * band_px, bins, 3), dtype=float)
    image[:] = np.asarray(to_rgb(IDLE_COLOR), dtype=float)
    for stage in range(3):
        stage_rows = intervals[intervals["stage"] == stage]
        compute = occupancy_bins(
            stage_rows[stage_rows["kind"] == "compute"],
            window_ms=window_ms,
            bin_ms=bin_ms,
        )
        transfer = occupancy_bins(
            stage_rows[stage_rows["kind"] == "transfer"],
            window_ms=window_ms,
            bin_ms=bin_ms,
        )
        row0 = stage * band_px
        image[row0 + 5 : row0 + 20, :, :] = blend(COMPUTE_COLOR, compute)
        image[row0 + 20 : row0 + 24, :, :] = blend(TRANSFER_COLOR, transfer)
    return image


def formal_throughput_map(path: Path) -> dict[str, tuple[float, float]]:
    frame = pd.read_csv(path)
    selected = frame[
        (frame["pipeline_stages"] == 3)
        & (frame["regime"] == "throughput_b8_m4")
        & (frame["method"].isin(METHOD_ORDER))
    ]
    if len(selected) != len(METHOD_ORDER):
        raise ValueError(f"expected one three-stage formal row per method in {path}")
    return {
        str(row.method): (
            float(row.mean_throughput_per_s),
            float(row.std_throughput_per_s),
        )
        for row in selected.itertuples(index=False)
    }


def draw_method(
    method: str,
    *,
    intervals: pd.DataFrame,
    metrics: pd.DataFrame,
    throughput: tuple[float, float],
    output_dir: Path,
    bin_ms: float,
) -> None:
    method_intervals = intervals[intervals["method"] == method].copy()
    method_metrics = metrics[metrics["method"] == method].sort_values("stage")
    if len(method_metrics) != 3:
        raise ValueError(f"{method}: expected three stage metric rows")
    window_values = method_metrics["selected_window_ms"].unique()
    if len(window_values) != 1:
        raise ValueError(f"{method}: selected window mismatch")
    window_ms = float(window_values[0])

    image = timeline_image(method_intervals, window_ms=window_ms, bin_ms=bin_ms)
    fig, ax = plt.subplots(figsize=(7.05, 2.7))
    ax.imshow(
        image,
        extent=(0.0, window_ms, -0.5, 2.5),
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        rasterized=True,
    )
    active_ratios = method_metrics.set_index("stage")["total_active_ratio"]
    ax.set_yticks(
        [0, 1, 2],
        [
            f"Stage {stage}   {100.0 * float(active_ratios.loc[stage]):.1f}% active"
            for stage in range(3)
        ],
    )
    ax.set_xlim(0.0, window_ms)
    ax.set_ylim(2.5, -0.5)
    ax.set_xlabel("Time within the measured steady-state interval (ms)")
    mean, std = throughput
    fig.suptitle(
        f"{METHOD_LABELS[method]} GPU activity   |   unprofiled throughput {mean:.1f} +/- {std:.1f} requests/s",
        x=0.145,
        y=0.98,
        ha="left",
    )
    ax.grid(axis="x", color="#FFFFFF", linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.legend(
        handles=[
            Patch(facecolor=COMPUTE_COLOR, label=f"CUDA kernel occupancy ({bin_ms:g} ms bins)"),
            Patch(facecolor=TRANSFER_COLOR, label="Device copy / memset"),
            Patch(facecolor=IDLE_COLOR, edgecolor="#C7C9CC", label="GPU idle"),
        ],
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.58, 0.91),
        columnspacing=1.2,
    )
    fig.subplots_adjust(left=0.145, right=0.99, bottom=0.20, top=0.73)
    save_figure(fig, output_dir, f"e4_gpu_timeline_{method}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intervals", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--formal-throughput", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bin-ms", type=float, default=0.25)
    args = parser.parse_args()
    if args.bin_ms <= 0:
        raise ValueError("--bin-ms must be positive")

    configure_paper_style()
    intervals = pd.read_csv(args.intervals)
    metrics = pd.read_csv(args.metrics)
    throughput = formal_throughput_map(args.formal_throughput)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for method in METHOD_ORDER:
        draw_method(
            method,
            intervals=intervals,
            metrics=metrics,
            throughput=throughput[method],
            output_dir=args.output_dir,
            bin_ms=args.bin_ms,
        )

    source = metrics.copy()
    source["unprofiled_mean_throughput_per_s"] = source["method"].map(
        {method: values[0] for method, values in throughput.items()}
    )
    source["unprofiled_std_throughput_per_s"] = source["method"].map(
        {method: values[1] for method, values in throughput.items()}
    )
    source["render_bin_ms"] = args.bin_ms
    source["selection_rule"] = (
        "centered fixed 1 s window within the common compute-active span of stages 0--2"
    )
    source["valid_for_throughput_comparison"] = False
    source.to_csv(args.output_dir / "e4_gpu_timeline_sources.csv", index=False)
    print(f"Wrote four independent GPU timelines to {args.output_dir}")


if __name__ == "__main__":
    main()
