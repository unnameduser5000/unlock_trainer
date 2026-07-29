#!/usr/bin/env python3
"""Generate report-ready figures from mobile BP-free demo runs.

The script intentionally works from checked experiment CSV/log artifacts rather
than a live coordinator, so figures can be regenerated after a context reset or
on the export server with the same command.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_FORMAL_DIR = Path("debug_runs/formal-clean-20260526-215720")
DEFAULT_FAULT_DIR = Path("debug_runs/fault-tolerance-20260527-1516")
DEFAULT_THREE_PHONE_DIR = Path("debug_runs/three-phone-20260528")
DEFAULT_PERF_DIR = DEFAULT_FORMAL_DIR / "train/perf-monitor-20260526-225135"
DEFAULT_OUTPUT_DIR = Path("docs/figures")


@dataclass(frozen=True)
class RunSummary:
    name: str
    path: Path
    rows: int
    success_rows: int
    avg_elapsed_ms: float
    avg_loss: float
    token_accuracy: float


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
        }
    )


def read_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for col in [
        "record_index",
        "dataset_index",
        "valid_labels",
        "processed_stage_id",
        "processed_chunk_idx",
        "elapsed_ms",
        "local_loss",
        "token_correct",
        "token_count",
        "token_accuracy",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["success", "terminal", "eval_only"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().eq("true")
    if "row_idx" not in df.columns:
        df["row_idx"] = np.arange(len(df))
    return df


def successful(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "success" not in df.columns:
        return df
    return df[df["success"]].copy()


def weighted_token_accuracy(df: pd.DataFrame) -> float:
    if df.empty or "token_correct" not in df.columns or "token_count" not in df.columns:
        return math.nan
    denom = pd.to_numeric(df["token_count"], errors="coerce").fillna(0).sum()
    if denom <= 0:
        return math.nan
    num = pd.to_numeric(df["token_correct"], errors="coerce").fillna(0).sum()
    return float(num / denom)


def summarize(name: str, path: Path, df: pd.DataFrame) -> RunSummary:
    ok = successful(df)
    return RunSummary(
        name=name,
        path=path,
        rows=len(df),
        success_rows=len(ok),
        avg_elapsed_ms=float(ok["elapsed_ms"].mean()) if "elapsed_ms" in ok.columns and len(ok) else math.nan,
        avg_loss=float(ok["local_loss"].mean()) if "local_loss" in ok.columns and len(ok) else math.nan,
        token_accuracy=weighted_token_accuracy(ok),
    )


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Wrote {path}")


def rolling(series: pd.Series, window: int = 25) -> pd.Series:
    if len(series) < 3:
        return series
    effective = min(window, max(3, len(series) // 8))
    return series.rolling(effective, min_periods=1).mean()


def plot_training_curves(train_df: pd.DataFrame, out_dir: Path) -> None:
    ok = successful(train_df)
    if ok.empty:
        return
    x = np.arange(len(ok))
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    ax.plot(x, ok["local_loss"], color="#9CA3AF", linewidth=0.9, alpha=0.55, label="per request")
    ax.plot(x, rolling(ok["local_loss"]), color="#2563EB", linewidth=2.0, label="rolling mean")
    ax.set_title("Two-phone training loss over 512 prepared requests")
    ax.set_xlabel("Successful training request index")
    ax.set_ylabel("Terminal local loss")
    ax.legend(frameon=False)
    savefig(out_dir / "training_loss_curve.png")

    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    ax.plot(x, ok["elapsed_ms"] / 1000.0, color="#9CA3AF", linewidth=0.9, alpha=0.55, label="per request")
    ax.plot(x, rolling(ok["elapsed_ms"]) / 1000.0, color="#059669", linewidth=2.0, label="rolling mean")
    ax.set_title("Two-phone end-to-end latency over 512 prepared requests")
    ax.set_xlabel("Successful training request index")
    ax.set_ylabel("Elapsed time (s)")
    ax.legend(frameon=False)
    savefig(out_dir / "training_latency_curve.png")

    if "token_count" in ok.columns and "token_correct" in ok.columns:
        cumulative_correct = ok["token_correct"].fillna(0).cumsum()
        cumulative_tokens = ok["token_count"].fillna(0).cumsum().replace(0, np.nan)
        fig, ax = plt.subplots(figsize=(9.5, 4.4))
        ax.plot(x, cumulative_correct / cumulative_tokens, color="#7C3AED", linewidth=2.0)
        ax.set_title("Cumulative token accuracy during two-phone training")
        ax.set_xlabel("Successful training request index")
        ax.set_ylabel("Cumulative token accuracy")
        ax.set_ylim(bottom=0)
        savefig(out_dir / "training_token_accuracy_curve.png")


def plot_summary_bars(summaries: Iterable[RunSummary], out_dir: Path) -> None:
    rows = [s for s in summaries if s.success_rows > 0]
    if not rows:
        return
    labels = [s.name for s in rows]
    x = np.arange(len(rows))

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0))
    metrics = [
        ("avg_elapsed_ms", "Avg latency (s)", "#059669", 1000.0),
        ("avg_loss", "Avg terminal loss", "#2563EB", 1.0),
        ("token_accuracy", "Token accuracy", "#7C3AED", 1.0),
    ]
    for ax, (attr, ylabel, color, divisor) in zip(axes, metrics):
        values = np.array([getattr(s, attr) for s in rows], dtype=float) / divisor
        ax.bar(x, values, color=color, alpha=0.88)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        for idx, value in enumerate(values):
            if np.isfinite(value):
                fmt = f"{value:.3f}" if value < 10 else f"{value:.1f}"
                ax.text(idx, value, fmt, ha="center", va="bottom", fontsize=8)
    fig.suptitle("Current mobile experiment summary", y=1.03, fontsize=13)
    savefig(out_dir / "experiment_summary_bars.png")


def plot_fault_tolerance(pre_df: pd.DataFrame, post_df: pd.DataFrame, timeline_path: Path, out_dir: Path) -> None:
    pre = successful(pre_df)
    post = successful(post_df)
    if pre.empty and post.empty:
        return

    labels = []
    latency = []
    loss = []
    success_counts = []
    for name, df in [("pre-fault", pre), ("post-fault", post)]:
        if df.empty:
            continue
        labels.append(name)
        latency.append(float(df["elapsed_ms"].mean()) / 1000.0)
        loss.append(float(df["local_loss"].mean()))
        success_counts.append(len(df))

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0))
    x = np.arange(len(labels))
    axes[0].bar(x, latency, color="#EA580C", alpha=0.86)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Avg latency (s)")
    axes[0].set_title("Latency before/after worker restart")
    axes[1].bar(x, loss, color="#2563EB", alpha=0.86)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Avg terminal loss")
    axes[1].set_title("Loss before/after worker restart")
    for ax, values in zip(axes, [latency, loss]):
        for idx, value in enumerate(values):
            ax.text(idx, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle(f"Fault tolerance microbenchmark ({sum(success_counts)}/{sum(success_counts)} completed)", y=1.03)
    savefig(out_dir / "fault_tolerance_summary.png")

    if timeline_path.exists():
        timeline = read_recovery_timeline(timeline_path)
        if not timeline.empty:
            fig, ax = plt.subplots(figsize=(9.5, 3.8))
            ax.step(timeline["seconds"], timeline["liveNodeCount"], where="post", color="#2563EB", linewidth=2.0)
            ax.set_title("Observed coordinator live-node count during recovery window")
            ax.set_xlabel("Seconds since first recovery poll")
            ax.set_ylabel("Live nodes")
            ax.set_ylim(bottom=-0.05, top=max(2.2, timeline["liveNodeCount"].max() + 0.2))
            savefig(out_dir / "fault_recovery_timeline.png")


def read_recovery_timeline(path: Path) -> pd.DataFrame:
    rows = []
    pattern = re.compile(r"liveNodeCount=(\d+).*offlineStageCount=(\d+).*stage0Live=([^,]*)")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        timestamp = line.split(",", 1)[0].strip()
        match = pattern.search(line)
        if not match:
            continue
        rows.append(
            {
                "timestamp": pd.to_datetime(timestamp, errors="coerce"),
                "liveNodeCount": int(match.group(1)),
                "offlineStageCount": int(match.group(2)),
                "stage0Live": match.group(3).strip(),
            }
        )
    df = pd.DataFrame(rows).dropna(subset=["timestamp"])
    if df.empty:
        return df
    df["seconds"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
    return df


def plot_three_phone_partial(eval_df: pd.DataFrame, out_dir: Path) -> None:
    if eval_df.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0), sharex=True)
    x = eval_df["row_idx"]
    success_mask = eval_df["success"] if "success" in eval_df.columns else pd.Series(True, index=eval_df.index)
    ok = eval_df[success_mask]
    fail = eval_df[~success_mask]

    axes[0].plot(ok["row_idx"], ok["elapsed_ms"] / 1000.0, marker="o", color="#059669", linewidth=1.6)
    axes[0].set_ylabel("Latency (s)")
    axes[0].set_title("Latency")
    axes[1].plot(ok["row_idx"], ok["local_loss"], marker="o", color="#2563EB", linewidth=1.6)
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Terminal loss")
    axes[2].plot(ok["row_idx"], ok["token_accuracy"], marker="o", color="#7C3AED", linewidth=1.6)
    axes[2].set_ylabel("Token accuracy")
    axes[2].set_title("Token accuracy")
    for ax in axes:
        ax.set_xlabel("Attempt index")
        if not fail.empty:
            for failed_idx in fail["row_idx"]:
                ax.axvline(failed_idx, color="#DC2626", linewidth=1.4, linestyle="--", alpha=0.8)
    fig.suptitle("Three-phone full-split partial eval (failure marked in red)", y=1.03)
    savefig(out_dir / "three_phone_partial_eval.png")


def read_perf_samples(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    numeric_cols = [
        "elapsed_ms",
        "battery_level",
        "battery_status",
        "battery_temp_c",
        "battery_voltage_mv",
        "battery_current_ua",
        "thermal_status",
        "app_pss_kb",
        "app_private_dirty_kb",
        "proc_rss_kb",
        "proc_vsz_kb",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "elapsed_min" not in df.columns and "elapsed_ms" in df.columns:
        df["elapsed_min"] = df["elapsed_ms"] / 60000.0
    return df


def plot_perf_samples(samples: pd.DataFrame, out_dir: Path) -> None:
    if samples.empty:
        return
    if "serial" not in samples.columns:
        samples["serial"] = "device"
    samples["device_label"] = samples.apply(
        lambda row: f"{row.get('model', 'device')} ({str(row.get('serial', ''))[-4:]})",
        axis=1,
    )

    if "battery_temp_c" in samples.columns:
        fig, ax = plt.subplots(figsize=(9.5, 4.2))
        for label, df in samples.groupby("device_label"):
            ax.plot(df["elapsed_min"], df["battery_temp_c"], label=label, linewidth=1.6)
        ax.set_title("Battery temperature during monitored training window")
        ax.set_xlabel("Elapsed time (min)")
        ax.set_ylabel("Battery temperature (C)")
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
        savefig(out_dir / "battery_temperature_trace.png")

    memory_cols = [col for col in ["app_pss_kb", "proc_rss_kb"] if col in samples.columns]
    if memory_cols:
        fig, ax = plt.subplots(figsize=(9.5, 4.2))
        for label, df in samples.groupby("device_label"):
            for col in memory_cols:
                valid = df.dropna(subset=[col])
                if not valid.empty:
                    ax.plot(valid["elapsed_min"], valid[col] / 1024.0, label=f"{label} {col}", linewidth=1.35)
        ax.set_title("Worker memory footprint during monitored training window")
        ax.set_xlabel("Elapsed time (min)")
        ax.set_ylabel("Memory (MiB)")
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
        savefig(out_dir / "device_memory_trace.png")

    if "battery_current_ua" in samples.columns and samples["battery_current_ua"].notna().any():
        fig, ax = plt.subplots(figsize=(9.5, 4.2))
        for label, df in samples.groupby("device_label"):
            valid = df.dropna(subset=["battery_current_ua"])
            if not valid.empty:
                ax.plot(valid["elapsed_min"], valid["battery_current_ua"] / 1000.0, label=label, linewidth=1.4)
        ax.axhline(0, color="#111827", linewidth=0.8, alpha=0.5)
        ax.set_title("Battery current trace (device-reported, coarse)")
        ax.set_xlabel("Elapsed time (min)")
        ax.set_ylabel("Current (mA; sign convention is device-specific)")
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
        savefig(out_dir / "battery_current_trace.png")


def read_coordinator_export(run_dir: Path | None, file_name: str) -> pd.DataFrame:
    if run_dir is None:
        return pd.DataFrame()
    return read_results(run_dir / file_name)


def plot_coordinator_run_metrics(metrics: pd.DataFrame, out_dir: Path) -> None:
    ok = successful(metrics)
    if ok.empty:
        return
    if "request_index" in ok.columns and ok["request_index"].notna().any():
        x = ok["request_index"]
        xlabel = "Request index"
    else:
        x = np.arange(len(ok))
        xlabel = "Successful request row"

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0))
    axes[0].plot(x, ok["elapsed_ms"] / 1000.0, marker=".", color="#059669", linewidth=1.2)
    axes[0].set_title("Coordinator run latency")
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("Elapsed time (s)")
    if "local_loss" in ok.columns and ok["local_loss"].notna().any():
        axes[1].plot(x, ok["local_loss"], marker=".", color="#2563EB", linewidth=1.2)
    axes[1].set_title("Coordinator run loss")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("Terminal local loss")
    if "token_count" in ok.columns and "token_correct" in ok.columns:
        cumulative_correct = ok["token_correct"].fillna(0).cumsum()
        cumulative_tokens = ok["token_count"].fillna(0).cumsum().replace(0, np.nan)
        axes[2].plot(x, cumulative_correct / cumulative_tokens, color="#7C3AED", linewidth=1.8)
    axes[2].set_title("Coordinator run token accuracy")
    axes[2].set_xlabel(xlabel)
    axes[2].set_ylabel("Cumulative token accuracy")
    fig.suptitle("Coordinator-exported run metrics", y=1.03)
    savefig(out_dir / "coordinator_run_metrics.png")


def plot_stage_timing_breakdown(stage_timings: pd.DataFrame, out_dir: Path) -> None:
    if stage_timings.empty:
        return
    for col in [
        "stage_id",
        "local_ms",
        "input_build_ms",
        "execute_ms",
        "gradients_ms",
        "optimizer_create_ms",
        "optimizer_step_ms",
        "output_convert_ms",
        "forward_ms",
        "total_stage_ms",
    ]:
        if col in stage_timings.columns:
            stage_timings[col] = pd.to_numeric(stage_timings[col], errors="coerce")

    local = stage_timings.dropna(subset=["stage_id", "local_ms"]) if "local_ms" in stage_timings.columns else pd.DataFrame()
    if not local.empty:
        grouped = local.groupby("stage_id", as_index=False).agg(
            local_ms=("local_ms", "mean"),
            execute_ms=("execute_ms", "mean"),
            input_build_ms=("input_build_ms", "mean"),
            optimizer_step_ms=("optimizer_step_ms", "mean"),
            output_convert_ms=("output_convert_ms", "mean"),
        )
        stage_labels = [f"stage {int(s)}" for s in grouped["stage_id"]]
        components = [
            ("input_build_ms", "input build", "#93C5FD"),
            ("execute_ms", "execute", "#2563EB"),
            ("optimizer_step_ms", "optimizer step", "#F97316"),
            ("output_convert_ms", "output convert", "#A78BFA"),
        ]
        fig, ax = plt.subplots(figsize=(9.5, 4.4))
        bottom = np.zeros(len(grouped))
        x = np.arange(len(grouped))
        for col, label, color in components:
            values = grouped[col].fillna(0).to_numpy(dtype=float)
            ax.bar(x, values, bottom=bottom, label=label, color=color, alpha=0.88)
            bottom += values
        ax.scatter(x, grouped["local_ms"], color="#111827", zorder=3, label="reported localMs")
        ax.set_xticks(x)
        ax.set_xticklabels(stage_labels)
        ax.set_ylabel("Average time (ms)")
        ax.set_title("Per-stage local execution timing breakdown")
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
        savefig(out_dir / "stage_local_timing_breakdown.png")

    if "forward_ms" in stage_timings.columns and stage_timings["forward_ms"].notna().any():
        forward = stage_timings.dropna(subset=["stage_id", "forward_ms"])
        grouped = forward.groupby("stage_id", as_index=False).agg(
            forward_ms=("forward_ms", "mean"),
            total_stage_ms=("total_stage_ms", "mean"),
        )
        x = np.arange(len(grouped))
        fig, ax = plt.subplots(figsize=(8.5, 4.0))
        ax.bar(x - 0.18, grouped["forward_ms"], width=0.36, label="forwardMs", color="#DC2626", alpha=0.86)
        if grouped["total_stage_ms"].notna().any():
            ax.bar(x + 0.18, grouped["total_stage_ms"], width=0.36, label="totalStageMs", color="#059669", alpha=0.86)
        ax.set_xticks(x)
        ax.set_xticklabels([f"stage {int(s)}" for s in grouped["stage_id"]])
        ax.set_ylabel("Average time (ms)")
        ax.set_title("Inter-stage forwarding timing")
        ax.legend(frameon=False)
        savefig(out_dir / "stage_forward_timing_breakdown.png")


def plot_architecture(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.set_xlim(0, 10.8)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, title: str, body: str, color: str) -> None:
        rect = plt.Rectangle((x, y), w, h, linewidth=1.5, edgecolor=color, facecolor="#F9FAFB")
        ax.add_patch(rect)
        ax.text(x + 0.18, y + h - 0.35, title, fontsize=11, fontweight="bold", color=color, va="top")
        ax.text(x + 0.18, y + h - 0.86, body, fontsize=9, color="#374151", va="top", linespacing=1.35)

    box(
        0.35,
        3.4,
        2.45,
        1.55,
        "Coordinator",
        "Registration\nScheduling\nRouting / metrics",
        "#111827",
    )
    stage_info = [
        ("Stage 0 phone", "LoRA chunk 0\nlocal F/B + SGD", "#2563EB"),
        ("Stage 1 phone", "LoRA chunk 1\nlocal F/B + SGD", "#059669"),
        ("Stage 2 phone", "LoRA chunk 2\nlocal F/B + SGD", "#7C3AED"),
    ]
    xs = [3.45, 5.95, 8.45]
    for x, (title, body, color) in zip(xs, stage_info):
        box(x, 2.25, 1.95, 1.5, title, body, color)

    def arrow(x1: float, y1: float, x2: float, y2: float, label: str, color: str = "#374151") -> None:
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops={"arrowstyle": "->", "linewidth": 1.6, "color": color},
        )
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.18, label, ha="center", fontsize=8.5, color=color)

    arrow(2.8, 4.1, 3.45, 3.45, "assign route")
    arrow(4.42, 2.25, 5.95, 2.95, "hidden / belief", "#DC2626")
    arrow(6.92, 2.25, 8.45, 2.95, "hidden / belief", "#DC2626")
    arrow(4.42, 3.75, 1.55, 3.4, "heartbeat + events")
    arrow(6.92, 3.75, 1.75, 4.95, "heartbeat + events")
    arrow(9.42, 3.75, 2.05, 4.95, "heartbeat + events")

    ax.text(
        5.0,
        0.9,
        "BP-free boundary: phones do local backward/optimizer, but no cross-stage backward-gradient RPC.",
        ha="center",
        fontsize=10,
        color="#111827",
        fontweight="bold",
    )
    ax.set_title("Mobile BP-free training system architecture", fontsize=13, pad=12)
    savefig(out_dir / "system_architecture.png")


def plot_pipeline_mindmaps(out_dir: Path) -> None:
    def box(
        ax: plt.Axes,
        x: float,
        y: float,
        w: float,
        h: float,
        title: str,
        body: str,
        edge: str,
        face: str = "#F9FAFB",
    ) -> None:
        rect = plt.Rectangle((x, y), w, h, linewidth=1.5, edgecolor=edge, facecolor=face)
        ax.add_patch(rect)
        ax.text(x + 0.16, y + h - 0.25, title, fontsize=10.0, fontweight="bold", color=edge, va="top")
        ax.text(x + 0.16, y + h - 0.68, body, fontsize=8.0, color="#374151", va="top", linespacing=1.18)

    def arrow(
        ax: plt.Axes,
        start: tuple[float, float],
        end: tuple[float, float],
        label: str,
        color: str = "#374151",
        curve: float = 0.0,
        label_offset: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        connectionstyle = "arc3"
        if curve:
            connectionstyle = f"arc3,rad={curve}"
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={
                "arrowstyle": "->",
                "linewidth": 1.55,
                "color": color,
                "connectionstyle": connectionstyle,
            },
        )
        if label:
            mx = (start[0] + end[0]) / 2 + label_offset[0]
            my = (start[1] + end[1]) / 2 + label_offset[1]
            ax.text(mx, my, label, ha="center", va="center", fontsize=8.1, color=color)

    def setup(title: str) -> tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots(figsize=(12.8, 7.4))
        ax.set_xlim(0, 12.8)
        ax.set_ylim(0, 7.4)
        ax.axis("off")
        ax.set_title(title, fontsize=13, pad=14)
        return fig, ax

    fig, ax = setup("Mobile BP-free pipeline mind map")
    box(
        ax,
        4.75,
        5.55,
        3.0,
        1.35,
        "Coordinator control plane",
        "register phones\nplace stages / routes\nleases, retries, metrics",
        "#111827",
        "#F3F4F6",
    )
    box(ax, 0.25, 2.65, 1.95, 1.45, "Prepared SFT request", "tokens / labels\nmask / position ids\nrun id", "#6B7280")
    stage_nodes = [
        (2.75, "Stage 0 phone", "TinyLlama chunk 0\nlocal forward/backward\nlocal LoRA SGD"),
        (5.2, "Stage 1 phone", "TinyLlama chunk 1\nlocal forward/backward\nlocal LoRA SGD"),
        (7.65, "Stage 2 phone", "TinyLlama chunk 2\nlocal forward/backward\nlocal LoRA SGD"),
    ]
    stage_colors = ["#2563EB", "#059669", "#7C3AED"]
    for (x, title, body), color in zip(stage_nodes, stage_colors):
        box(ax, x, 2.65, 1.95, 1.45, title, body, color, "#FFFFFF")
    box(ax, 10.55, 2.65, 1.95, 1.45, "Terminal record", "loss / token acc\nstage timings\npersisted events", "#EA580C")

    arrow(ax, (2.2, 3.42), (2.75, 3.42), "dispatch", label_offset=(0.0, 0.14))
    arrow(ax, (4.7, 3.42), (5.2, 3.42), "hidden/belief", "#DC2626", label_offset=(0.0, 0.16))
    arrow(ax, (7.15, 3.42), (7.65, 3.42), "hidden/belief", "#DC2626", label_offset=(0.0, 0.16))
    arrow(ax, (9.6, 3.42), (10.55, 3.42), "metrics")

    for x in [3.72, 6.17, 8.62]:
        arrow(ax, (6.25, 5.55), (x, 4.1), "", "#111827", curve=0.05)
        arrow(ax, (x, 4.1), (6.2, 5.55), "", "#4B5563", curve=0.08)
    ax.text(6.25, 4.78, "control: stage assignment + heartbeat leases", ha="center", fontsize=8.8, color="#111827")

    box(
        ax,
        0.65,
        0.88,
        2.7,
        1.25,
        "Communication boundary",
        "forward-only tensors across phones\nno cross-stage gradient RPC",
        "#DC2626",
        "#FEF2F2",
    )
    box(
        ax,
        4.15,
        0.88,
        2.4,
        1.25,
        "Mobile scheduler",
        "preferred placement\ndynamic fill\nlease expiration",
        "#0891B2",
        "#ECFEFF",
    )
    box(
        ax,
        7.35,
        0.88,
        2.5,
        1.25,
        "Fault evidence",
        "node lost / rejoin\nretry new requests\nobservable recovery",
        "#EA580C",
        "#FFF7ED",
    )
    box(
        ax,
        10.45,
        0.88,
        2.05,
        1.25,
        "Record keeping",
        "SQLite run tables\nCSV export\nreport figures",
        "#4F46E5",
        "#EEF2FF",
    )
    arrow(ax, (6.25, 2.65), (2.0, 2.13), "", "#DC2626", curve=-0.12)
    arrow(ax, (5.25, 5.55), (5.35, 2.13), "", "#0891B2", curve=0.08)
    arrow(ax, (8.5, 2.65), (8.6, 2.13), "", "#EA580C")
    arrow(ax, (11.55, 2.65), (11.48, 2.13), "", "#4F46E5")
    ax.text(
        6.4,
        0.28,
        "Key claim: each shard trains locally; phones exchange hidden/belief signals, so failure handling and scheduling live in the mobile runtime.",
        ha="center",
        fontsize=9.5,
        color="#111827",
        fontweight="bold",
    )
    savefig(out_dir / "bpfree_pipeline_mindmap.png")

    fig, ax = setup("Conventional BP pipeline mind map")
    box(
        ax,
        4.75,
        5.55,
        3.0,
        1.35,
        "1F1B pipeline scheduler",
        "microbatch forward wave\nbackward wave after loss\nstage order is tightly coupled",
        "#111827",
        "#F3F4F6",
    )
    box(ax, 0.25, 2.65, 1.95, 1.45, "Microbatches", "input batches\nactivation stash\nloss target", "#6B7280")
    bp_stage_nodes = [
        (2.75, "Stage 0 worker", "layers 0..k\nforward activations\nwaits for grad"),
        (5.2, "Stage 1 worker", "middle layers\nstores activations\nsends grad left"),
        (7.65, "Stage 2 worker", "last layers + loss\nstarts backward\noptimizer after grad"),
    ]
    for (x, title, body), color in zip(bp_stage_nodes, stage_colors):
        box(ax, x, 2.65, 1.95, 1.45, title, body, color, "#FFFFFF")
    box(ax, 10.55, 2.65, 1.95, 1.45, "Loss + update", "global loss\nfull backprop chain\nstep after dependencies", "#EA580C")

    arrow(ax, (2.2, 3.78), (2.75, 3.78), "F")
    arrow(ax, (4.7, 3.78), (5.2, 3.78), "activation", "#DC2626", label_offset=(0.0, 0.18))
    arrow(ax, (7.15, 3.78), (7.65, 3.78), "activation", "#DC2626", label_offset=(0.0, 0.18))
    arrow(ax, (9.6, 3.78), (10.55, 3.78), "logits/loss")
    arrow(ax, (10.55, 3.08), (9.6, 3.08), "loss grad", "#7F1D1D", label_offset=(0.0, -0.16))
    arrow(ax, (7.65, 3.08), (7.15, 3.08), "grad", "#7F1D1D", label_offset=(0.0, -0.16))
    arrow(ax, (5.2, 3.08), (4.7, 3.08), "grad", "#7F1D1D", label_offset=(0.0, -0.16))
    arrow(ax, (2.75, 3.08), (2.2, 3.08), "input grad", "#7F1D1D", label_offset=(0.0, -0.16))

    for x in [3.72, 6.17, 8.62]:
        arrow(ax, (6.25, 5.55), (x, 4.1), "", "#111827", curve=0.05)
    ax.text(6.25, 4.78, "scheduler enforces 1F1B microbatch order", ha="center", fontsize=8.8, color="#111827")

    box(
        ax,
        0.65,
        0.88,
        2.7,
        1.25,
        "Cross-stage dependency",
        "backward gradients cross devices\nearlier stages cannot step early",
        "#7F1D1D",
        "#FEF2F2",
    )
    box(
        ax,
        4.15,
        0.88,
        2.4,
        1.25,
        "Memory pressure",
        "activation stash\nbubble scheduling\noptimizer state per shard",
        "#0891B2",
        "#ECFEFF",
    )
    box(
        ax,
        7.35,
        0.88,
        2.5,
        1.25,
        "Failure behavior",
        "one dead stage blocks\nthe gradient chain\nin-flight microbatches fail",
        "#EA580C",
        "#FFF7ED",
    )
    box(
        ax,
        10.45,
        0.88,
        2.05,
        1.25,
        "Comparison role",
        "baseline mental model\nnot mobile-aware by itself",
        "#4F46E5",
        "#EEF2FF",
    )
    arrow(ax, (6.25, 2.65), (2.0, 2.13), "", "#7F1D1D", curve=-0.12)
    arrow(ax, (5.9, 2.65), (5.35, 2.13), "", "#0891B2")
    arrow(ax, (8.5, 2.65), (8.6, 2.13), "", "#EA580C")
    arrow(ax, (11.55, 2.65), (11.48, 2.13), "", "#4F46E5")
    ax.text(
        6.4,
        0.28,
        "Key contrast: BP pipeline communicates both activations and gradients; mobile BP-free keeps inter-phone traffic forward-only.",
        ha="center",
        fontsize=9.5,
        color="#111827",
        fontweight="bold",
    )
    savefig(out_dir / "bp_pipeline_mindmap.png")


def write_summary_csv(summaries: list[RunSummary], out_dir: Path) -> None:
    rows = [
        {
            "name": s.name,
            "path": str(s.path),
            "rows": s.rows,
            "success_rows": s.success_rows,
            "avg_elapsed_ms": s.avg_elapsed_ms,
            "avg_loss": s.avg_loss,
            "token_accuracy": s.token_accuracy,
        }
        for s in summaries
    ]
    out = out_dir / "figure_metric_summary.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal_dir", type=Path, default=DEFAULT_FORMAL_DIR)
    parser.add_argument("--fault_dir", type=Path, default=DEFAULT_FAULT_DIR)
    parser.add_argument("--three_phone_dir", type=Path, default=DEFAULT_THREE_PHONE_DIR)
    parser.add_argument("--perf_dir", type=Path, default=DEFAULT_PERF_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--coordinator_run_dir",
        type=Path,
        default=None,
        help="Optional directory containing coordinator-exported metrics.csv and stage-timings.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    eval_before = read_results(args.formal_dir / "eval-before/results.csv")
    train = read_results(args.formal_dir / "train-combined/results.csv")
    ft_pre = read_results(args.fault_dir / "pre-fault/results.csv")
    ft_post = read_results(args.fault_dir / "post-fault/results.csv")
    three_eval = read_results(args.three_phone_dir / "eval128/results.csv")
    three_smoke = read_results(args.three_phone_dir / "smoke/results.csv")
    perf_samples = read_perf_samples(args.perf_dir / "samples.csv")
    coordinator_metrics = read_coordinator_export(args.coordinator_run_dir, "metrics.csv")
    coordinator_stage_timings = read_coordinator_export(args.coordinator_run_dir, "stage-timings.csv")

    summaries = [
        summarize("eval-before", args.formal_dir / "eval-before/results.csv", eval_before),
        summarize("train512", args.formal_dir / "train-combined/results.csv", train),
        summarize("fault pre", args.fault_dir / "pre-fault/results.csv", ft_pre),
        summarize("fault post", args.fault_dir / "post-fault/results.csv", ft_post),
        summarize("3-phone smoke", args.three_phone_dir / "smoke/results.csv", three_smoke),
        summarize("3-phone partial", args.three_phone_dir / "eval128/results.csv", three_eval),
    ]

    plot_architecture(args.output_dir)
    plot_pipeline_mindmaps(args.output_dir)
    plot_training_curves(train, args.output_dir)
    plot_summary_bars(summaries, args.output_dir)
    plot_fault_tolerance(ft_pre, ft_post, args.fault_dir / "recovery_timeline.csv", args.output_dir)
    plot_three_phone_partial(three_eval, args.output_dir)
    plot_perf_samples(perf_samples, args.output_dir)
    plot_coordinator_run_metrics(coordinator_metrics, args.output_dir)
    plot_stage_timing_breakdown(coordinator_stage_timings, args.output_dir)
    write_summary_csv(summaries, args.output_dir)


if __name__ == "__main__":
    main()
