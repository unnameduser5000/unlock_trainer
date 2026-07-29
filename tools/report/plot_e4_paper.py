#!/usr/bin/env python3
"""Build publication-sized E4 sweep and critical-path figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from paper_style import (
    METHOD_LABELS,
    METHOD_ORDER,
    METHOD_STYLES,
    configure_paper_style,
    finish_axis,
    save_figure,
)


def method_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["method"].isin(METHOD_ORDER)].copy()


def draw_lines(
    ax: plt.Axes,
    frame: pd.DataFrame,
    x_column: str,
    x_values: list[int],
    *,
    errorbars: bool = True,
) -> None:
    for method in METHOD_ORDER:
        rows = frame[frame["method"] == method].set_index(x_column).reindex(x_values)
        values = rows["mean_throughput_per_s"].to_numpy(dtype=float)
        valid = np.isfinite(values)
        if not valid.any():
            continue
        style = METHOD_STYLES[method]
        kwargs: dict[str, object] = {}
        if errorbars and "std_throughput_per_s" in rows:
            errors = rows["std_throughput_per_s"].to_numpy(dtype=float)[valid]
            kwargs.update(yerr=errors, capsize=2, elinewidth=0.8)
        ax.errorbar(
            np.asarray(x_values)[valid],
            values[valid],
            label=METHOD_LABELS[method],
            markerfacecolor="white" if method in {"exactbp_1f1b", "pipedream"} else style["color"],
            markeredgewidth=1.0,
            **style,
            **kwargs,
        )


def save_sweep_figures(input_dir: Path, output_dir: Path) -> None:
    scaling = method_rows(pd.read_csv(input_dir / "e4_1_summary.csv"))
    geometry = method_rows(pd.read_csv(input_dir / "e4_2a_summary.csv"))
    microbatch = method_rows(pd.read_csv(input_dir / "e4_2b_summary.csv"))
    network = pd.read_csv(input_dir / "e4_3_network_summary.csv")

    fig, ax = plt.subplots(figsize=(3.42, 2.35))
    draw_lines(ax, scaling, "pipeline_stages", [2, 3, 4])
    ax.set_xticks([2, 3, 4])
    ax.set_xlabel("Pipeline stages / GPUs")
    ax.set_ylabel("Throughput (requests/s)")
    ax.legend(ncol=2, loc="best")
    finish_axis(ax)
    fig.tight_layout()
    save_figure(fig, output_dir, "e4_stage_scaling")

    fig, ax = plt.subplots(figsize=(3.42, 2.35))
    draw_lines(ax, geometry, "physical_request_batch", [1, 2, 4, 8])
    ax.set_xticks([1, 2, 4, 8], ["1/32", "2/16", "4/8", "8/4"])
    ax.set_xlabel("Physical batch / microbatches (B=32)")
    ax.set_ylabel("Throughput (requests/s)")
    ax.legend(ncol=2, loc="best")
    finish_axis(ax)
    fig.tight_layout()
    save_figure(fig, output_dir, "e4_batch_geometry")

    fig, ax = plt.subplots(figsize=(3.42, 2.35))
    draw_lines(ax, microbatch, "microbatches_per_update", [1, 2, 3, 4, 8, 16, 32])
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32], ["1", "2", "4", "8", "16", "32"])
    ax.set_xlabel("Microbatches per update (b=1)")
    ax.set_ylabel("Throughput (requests/s)")
    ax.legend(ncol=2, loc="lower right")
    finish_axis(ax)
    fig.tight_layout()
    save_figure(fig, output_dir, "e4_microbatch_crossover")

    profiles = ["Local", "Wi-Fi", "Mobile", "Constrained"]
    x = np.arange(len(profiles))
    fig, ax = plt.subplots(figsize=(3.42, 2.35))
    for method in METHOD_ORDER:
        rows = network[network["method"] == method].set_index("profile").reindex(profiles)
        style = METHOD_STYLES[method]
        ax.plot(
            x,
            rows["mean_throughput_per_s"],
            label=METHOD_LABELS[method],
            markerfacecolor="white" if method in {"exactbp_1f1b", "pipedream"} else style["color"],
            markeredgewidth=1.0,
            **style,
        )
    ax.set_yscale("log")
    ax.set_xticks(x, profiles)
    ax.set_xlabel("Sender-side link profile")
    ax.set_ylabel("Throughput (requests/s, log)")
    ax.legend(ncol=2, loc="lower left")
    finish_axis(ax)
    fig.tight_layout()
    save_figure(fig, output_dir, "e4_link_sensitivity")


BREAKDOWN_GROUPS = [
    (
        "Compute",
        [
            "critical_input_h2d_ms",
            "critical_forward_compute_ms",
            "critical_backward_compute_ms",
            "critical_optimizer_ms",
            "critical_weight_stash_ms",
            "critical_gradient_accumulation_ms",
        ],
        "#4C78A8",
    ),
    ("Receive wait", ["critical_transport_recv_wait_ms"], "#E45756"),
    ("Link pacing", ["critical_link_pacing_ms"], "#F2CF5B"),
    (
        "Transfer/runtime",
        [
            "critical_transport_d2h_ms",
            "critical_transport_recv_post_ms",
            "critical_transport_recv_h2d_ms",
            "critical_transport_send_post_runtime_ms",
            "critical_transport_send_wait_ms",
        ],
        "#72B7B2",
    ),
    (
        "Idle/other",
        ["critical_control_ms", "critical_untraced_idle_ms"],
        "#A7A7A7",
    ),
]


def save_breakdown(input_dir: Path, output_dir: Path, profile: str) -> None:
    frame = pd.read_csv(input_dir / "e4_4_case_breakdown.csv")
    frame = frame[
        (frame["regime"] == "throughput_b8_m4")
        & (frame["profile"] == profile)
        & (frame["method"].isin(["bpfree", "pipedream"]))
    ].set_index("method")
    methods = ["bpfree", "pipedream"]
    y = np.arange(2)
    left = np.zeros(2)
    fig, ax = plt.subplots(figsize=(3.42, 2.1))
    for label, columns, color in BREAKDOWN_GROUPS:
        values = np.asarray(
            [sum(float(frame.loc[method, column]) for column in columns) / 1000.0 for method in methods]
        )
        ax.barh(y, values, left=left, color=color, height=0.55, label=label)
        left += values
    for ypos, total in zip(y, left):
        ax.text(total, ypos, f" {total:.2f}s", va="center", fontsize=7)
    ax.set_yticks(y, [METHOD_LABELS[method] for method in methods])
    ax.invert_yaxis()
    ax.set_xlabel("Critical-stage trace span (s)")
    ax.set_xlim(0, max(left) * 1.14)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01), columnspacing=0.8)
    finish_axis(ax, xgrid=True)
    fig.tight_layout()
    save_figure(fig, output_dir, f"e4_critical_path_{profile}_seconds")


WINDOW_COLORS = {2: "#4C78A8", 3: "#F28E2B", 4: "#59A14F"}
ACTION_LABELS = {
    "forward": "F",
    "local head": "H",
    "backward": "B",
    "optimizer": "O",
}


def classify_action(action: str) -> str | None:
    if "WAIT_FINAL" in action:
        return None
    if "RECV_WAIT" in action:
        return "receive stall"
    if "SEND_WAIT" in action:
        return "send stall"
    if "LOCAL_HEAD_LOSS" in action:
        return "local head"
    if "BACKWARD" in action or "GRAD_ACCUM" in action:
        return "backward"
    if "FORWARD" in action or action == "BODY_FORWARD":
        return "forward"
    if "OPTIMIZER" in action or "WEIGHT_SNAPSHOT" in action:
        return "optimizer"
    if action in {"LOAD_STAGE0_HIDDEN", "LOAD_COMMON_INPUTS"}:
        return "transfer"
    if any(token in action for token in ("_H2D", "_D2H", "SEND_POST", "RECV_POST")):
        return "transfer"
    return None


def load_trace(method: str, method_dir: Path) -> tuple[pd.DataFrame, float, float]:
    frames: list[pd.DataFrame] = []
    for stage in range(3):
        frame = pd.read_csv(method_dir / "rep_00" / f"train.stage{stage}.actions.csv")
        key = "window_id" if "window_id" in frame.columns else "batch_seq"
        frame = frame.copy()
        frame["logical_window"] = frame[key].astype(int)
        frame["stage"] = stage
        frame["category"] = frame["action"].map(classify_action)
        frame = frame[frame["category"].notna()]
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    start_action = "LOAD_STAGE0_HIDDEN" if method == "bpfree" else "INPUT_LOAD_H2D"
    end_action = (
        "LOCAL_OPTIMIZER_STEP"
        if method == "bpfree"
        else ("OPTIMIZER_STEP_ASYNC" if method == "pipedream" else "OPTIMIZER_STEP")
    )
    start_anchor = combined[
        (combined["stage"] == 0)
        & (combined["logical_window"] == 2)
        & (combined["mb_id"].astype(int) == 0)
        & (combined["action"] == start_action)
    ]
    end_anchor = combined[
        (combined["stage"] == 2)
        & (combined["logical_window"] == 4)
        & (combined["action"] == end_action)
    ]
    if start_anchor.empty or end_anchor.empty:
        raise ValueError(f"Missing Gantt anchors for {method}: {start_action} -> {end_action}")
    slice_start = float(start_anchor["start_epoch_ms"].min())
    slice_end = float(end_anchor["end_epoch_ms"].max())
    combined = combined[
        (combined["end_epoch_ms"] >= slice_start)
        & (combined["start_epoch_ms"] <= slice_end)
    ].copy()
    return combined, slice_start, slice_end


def save_method_gantt(method: str, frame: pd.DataFrame, t0: float, t1: float, output_dir: Path) -> None:
    frame = frame.sort_values(["start_epoch_ms", "stage"])
    fig, ax = plt.subplots(figsize=(7.08, 2.05))

    # Host-side blocking waits are stalls, not GPU work. Draw them first as a
    # hatched background so they cannot be mistaken for productive busy time.
    stall_specs = {
        "receive stall": ("#F7D6D6", "////", "Receive stall"),
        "send stall": ("#FAEDC3", "\\\\", "Send/link stall"),
    }
    for row in frame[frame["category"].isin(stall_specs)].itertuples(index=False):
        left = max((float(row.start_epoch_ms) - t0) / 1000.0, 0.0)
        right = min((float(row.end_epoch_ms) - t0) / 1000.0, (t1 - t0) / 1000.0)
        if right <= left:
            continue
        color, hatch, _ = stall_specs[str(row.category)]
        ax.barh(
            int(row.stage),
            right - left,
            left=left,
            height=0.68,
            facecolor=color,
            edgecolor="#A35A5A" if row.category == "receive stall" else "#B18B2E",
            hatch=hatch,
            linewidth=0.3,
            zorder=1,
        )

    # H2D/D2H and post runtimes are shown as a thin center strip. They may
    # overlap compute because the transport path is asynchronous.
    for row in frame[frame["category"] == "transfer"].itertuples(index=False):
        left = max((float(row.start_epoch_ms) - t0) / 1000.0, 0.0)
        right = min((float(row.end_epoch_ms) - t0) / 1000.0, (t1 - t0) / 1000.0)
        if right <= left:
            continue
        ax.barh(
            int(row.stage),
            right - left,
            left=left,
            height=0.14,
            color="#66A6A6",
            edgecolor="none",
            zorder=3,
        )

    productive = frame[frame["category"].isin(ACTION_LABELS)]
    for row in productive.itertuples(index=False):
        left = max((float(row.start_epoch_ms) - t0) / 1000.0, 0.0)
        right = min((float(row.end_epoch_ms) - t0) / 1000.0, (t1 - t0) / 1000.0)
        if right <= left:
            continue
        window = int(row.logical_window)
        color = WINDOW_COLORS.get(window, "#B7BBC2")
        ax.barh(
            int(row.stage),
            right - left,
            left=left,
            height=0.48,
            color=color,
            edgecolor="white",
            linewidth=0.25,
            zorder=4,
        )
        if right - left >= 0.018:
            mb = int(row.mb_id)
            label = f"{ACTION_LABELS[str(row.category)]}{window}.{mb}" if mb >= 0 else ACTION_LABELS[str(row.category)]
            ax.text(left + (right - left) / 2, int(row.stage), label, ha="center", va="center", fontsize=5.5, color="white", zorder=5)

    span_s = (t1 - t0) / 1000.0
    ax.set_xlim(0, span_s)
    ax.set_yticks([0, 1, 2], ["S0", "S1", "S2"])
    ax.set_ylim(2.55, -0.55)
    ax.set_xlabel("Time from S0 start of W2/B2 to S2 commit of W4/B4 (s)")
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.5)
    ax.set_axisbelow(True)
    handles = [
        Patch(facecolor=WINDOW_COLORS[2], label="W2/B2"),
        Patch(facecolor=WINDOW_COLORS[3], label="W3/B3"),
        Patch(facecolor=WINDOW_COLORS[4], label="W4/B4"),
        Patch(facecolor="#B7BBC2", label="Other in-flight"),
        Patch(facecolor="#66A6A6", label="Device transfer"),
        Patch(facecolor="#F7D6D6", edgecolor="#A35A5A", hatch="////", label="Receive stall"),
        Patch(facecolor="#FAEDC3", edgecolor="#B18B2E", hatch="\\\\", label="Send/link stall"),
    ]
    ax.legend(handles=handles, ncol=7, loc="lower center", bbox_to_anchor=(0.5, 1.01), columnspacing=0.8)
    fig.tight_layout()
    save_figure(fig, output_dir, f"e4_gantt_{method}")


def save_measured_gantts(trace_root: Path, output_dir: Path) -> None:
    source_rows: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        frame, t0, t1 = load_trace(method, trace_root / method)
        save_method_gantt(method, frame, t0, t1, output_dir)
        source_rows.append(
            {
                "method": method,
                "start_semantics": "S0 mb0 input start for logical window/batch 2",
                "end_semantics": "S2 optimizer commit end for logical window/batch 4",
                "trace_span_ms": t1 - t0,
                "measurement_mode": "per-action CUDA-synchronized diagnostic trace",
                "valid_for_throughput_comparison": False,
                "source": str(trace_root / method / "rep_00"),
            }
        )
    pd.DataFrame(source_rows).to_csv(output_dir / "e4_gantt_sources.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", type=Path, required=True)
    parser.add_argument(
        "--trace-root",
        type=Path,
        help="Deprecated synchronized-action trace input; no Gantt is generated.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_paper_style()
    save_sweep_figures(args.summary_dir, args.output_dir)
    save_breakdown(args.summary_dir, args.output_dir, "local")
    save_breakdown(args.summary_dir, args.output_dir, "constrained")
    print(f"Wrote E4 paper figures to {args.output_dir}")


if __name__ == "__main__":
    main()
