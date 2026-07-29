#!/usr/bin/env python3
"""Generate figures and a Markdown report for scheduler scenario runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Wrote {path}")


def policy_rank(name: str) -> tuple[int, int, str]:
    if name == "all_updates":
        return (0, 0, name)
    if name.startswith("stride"):
        return (1, parse_suffix_int(name, 999), name)
    if name.startswith("queue"):
        return (2, parse_suffix_int(name, 999), name)
    return (9, 999, name)


def parse_suffix_int(name: str, default: int) -> int:
    digits = "".join(ch for ch in name if ch.isdigit())
    return int(digits) if digits else default


def ordered_policies(values: list[str]) -> list[str]:
    return sorted(set(values), key=policy_rank)


def numeric_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def read_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs_path = root / "scheduler_scenario_runs.csv"
    summary_path = root / "scheduler_scenario_summary.csv"
    if not runs_path.is_file():
        raise FileNotFoundError(f"Missing {runs_path}")
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing {summary_path}")
    runs = pd.read_csv(runs_path)
    summary = pd.read_csv(summary_path)
    numeric_cols(
        runs,
        [
            "seed",
            "train_throughput_per_s",
            "train_choice_accuracy",
            "train_avg_loss",
            "eval_throughput_per_s",
            "eval_choice_accuracy",
            "eval_avg_loss",
            "target_stage_train_tasks",
            "target_stage_updates",
            "target_stage_avg_execute_ms",
            "target_stage_avg_queue_ms",
            "target_stage_peak_alloc_mib",
            "target_stage_peak_reserved_mib",
            "duplicate_update_events",
            "timeout_events",
        ],
    )
    numeric_cols(summary, [col for col in summary.columns if col.endswith("_mean") or col.endswith("_std")])
    return runs, summary


def read_1f1b_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    specs = [
        ("sample_budget", "1f1b", root / "1f1b_baseline"),
        ("update_budget", "1f1b_update", root / "1f1b_update_matched"),
    ]
    run_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    for protocol, policy_name, baseline_root in specs:
        runs_path = baseline_root / "1f1b_runs.csv"
        summary_path = baseline_root / "1f1b_summary.csv"
        if not runs_path.is_file() or not summary_path.is_file():
            continue
        runs = pd.read_csv(runs_path)
        summary = pd.read_csv(summary_path)
        runs["protocol"] = protocol
        summary["protocol"] = protocol
        runs["policy"] = policy_name
        summary["policy"] = policy_name
        numeric_cols(
            runs,
            [col for col in runs.columns if col not in {"runner", "policy", "output_dir", "dtype", "optimizer", "protocol"}],
        )
        numeric_cols(summary, [col for col in summary.columns if col.endswith("_mean") or col.endswith("_std") or col == "runs"])
        run_frames.append(runs)
        summary_frames.append(summary)
    if not run_frames:
        return pd.DataFrame(), pd.DataFrame()
    return pd.concat(run_frames, ignore_index=True), pd.concat(summary_frames, ignore_index=True)


def resolve_run_dir(path_value: Any, root: Path) -> Path:
    path = Path(str(path_value))
    candidate = root / path.name
    if (candidate / "scheduler_summary.json").is_file():
        return candidate
    if path.is_absolute() and (path / "scheduler_summary.json").is_file():
        return path
    if (path / "scheduler_summary.json").is_file():
        return path
    if candidate.exists():
        return candidate
    if path.exists():
        return path
    return path


def resolve_existing_dir(path_value: Any, root: Path) -> Path:
    path = Path(str(path_value))
    candidates = [
        path,
        root / path.name,
        root / path,
        Path.cwd() / path,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return path


def load_stage_worker_metrics(root: Path, runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, run in runs.iterrows():
        run_dir = resolve_run_dir(run["output_dir"], root)
        summary_path = run_dir / "scheduler_summary.json"
        if not summary_path.is_file():
            continue
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        for worker in data.get("gpu_metrics_by_worker", []):
            row = {
                "policy": run["policy"],
                "seed": run["seed"],
                "run_dir": str(run_dir),
            }
            row.update(worker)
            rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        numeric_cols(
            df,
            [
                "seed",
                "stage_id",
                "tasks",
                "train_tasks",
                "updates_applied",
                "avg_execute_ms",
                "avg_scheduler_queue_ms",
                "avg_worker_queue_ms",
                "avg_optimizer_ms",
                "avg_stage_total_ms",
                "max_cuda_peak_memory_allocated_mib",
                "max_cuda_peak_memory_reserved_mib",
                "max_autograd_saved_cuda_peak_mib",
                "max_autograd_saved_cuda_nonleaf_peak_mib",
                "max_autograd_saved_cuda_leaf_peak_mib",
                "max_autograd_saved_cuda_unique_peak_mib",
                "max_autograd_saved_cuda_nonleaf_unique_peak_mib",
                "max_autograd_saved_cuda_leaf_unique_peak_mib",
                "max_autograd_saved_cuda_live_final_mib",
                "max_autograd_saved_cuda_unique_live_final_mib",
            ],
        )
    return df


def load_1f1b_stage_metrics(root: Path, f1b_runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if f1b_runs.empty:
        return pd.DataFrame()
    for _, run in f1b_runs.iterrows():
        run_dir = resolve_existing_dir(run["output_dir"], root)
        metrics_path = run_dir / "stage_metrics.csv"
        if not metrics_path.is_file():
            continue
        metrics = pd.read_csv(metrics_path)
        if "phase" in metrics.columns:
            metrics = metrics[metrics["phase"] == "train"].copy()
        numeric_cols(
            metrics,
            [
                "stage_id",
                "rank",
                "records",
                "cuda_peak_memory_allocated",
                "cuda_peak_memory_reserved",
                "autograd_saved_cuda_bytes_peak",
                "autograd_saved_cuda_nonleaf_bytes_peak",
                "autograd_saved_cuda_leaf_bytes_peak",
                "autograd_saved_cuda_unique_bytes_peak",
                "autograd_saved_cuda_nonleaf_unique_bytes_peak",
                "autograd_saved_cuda_leaf_unique_bytes_peak",
                "autograd_saved_cuda_bytes_live_final",
                "autograd_saved_cuda_unique_bytes_live_final",
            ],
        )
        stage_col = "stage_id" if "stage_id" in metrics.columns else "rank"
        for stage_id, stage_rows in metrics.groupby(stage_col):
            peak_alloc = stage_rows["cuda_peak_memory_allocated"].max()
            peak_reserved = stage_rows["cuda_peak_memory_reserved"].max()
            device = stage_rows["device"].dropna().astype(str).iloc[0] if "device" in stage_rows and not stage_rows.empty else ""
            rows.append(
                {
                    "policy": run.get("policy", "1f1b"),
                    "protocol": run.get("protocol", "sample_budget"),
                    "seed": run.get("seed", np.nan),
                    "run_dir": str(run_dir),
                    "worker_id": f"rank{int(stage_id)}",
                    "stage_id": int(stage_id),
                    "device": device,
                    "tasks": len(stage_rows),
                    "train_tasks": len(stage_rows),
                    "updates_applied": len(stage_rows),
                    "max_cuda_peak_memory_allocated_mib": peak_alloc / (1024.0 * 1024.0),
                    "max_cuda_peak_memory_reserved_mib": peak_reserved / (1024.0 * 1024.0),
                    "max_autograd_saved_cuda_peak_mib": (
                        stage_rows["autograd_saved_cuda_bytes_peak"].max() / (1024.0 * 1024.0)
                        if "autograd_saved_cuda_bytes_peak" in stage_rows
                        else np.nan
                    ),
                    "max_autograd_saved_cuda_nonleaf_peak_mib": (
                        stage_rows["autograd_saved_cuda_nonleaf_bytes_peak"].max() / (1024.0 * 1024.0)
                        if "autograd_saved_cuda_nonleaf_bytes_peak" in stage_rows
                        else np.nan
                    ),
                    "max_autograd_saved_cuda_leaf_peak_mib": (
                        stage_rows["autograd_saved_cuda_leaf_bytes_peak"].max() / (1024.0 * 1024.0)
                        if "autograd_saved_cuda_leaf_bytes_peak" in stage_rows
                        else np.nan
                    ),
                    "max_autograd_saved_cuda_unique_peak_mib": (
                        stage_rows["autograd_saved_cuda_unique_bytes_peak"].max() / (1024.0 * 1024.0)
                        if "autograd_saved_cuda_unique_bytes_peak" in stage_rows
                        else np.nan
                    ),
                    "max_autograd_saved_cuda_nonleaf_unique_peak_mib": (
                        stage_rows["autograd_saved_cuda_nonleaf_unique_bytes_peak"].max() / (1024.0 * 1024.0)
                        if "autograd_saved_cuda_nonleaf_unique_bytes_peak" in stage_rows
                        else np.nan
                    ),
                    "max_autograd_saved_cuda_leaf_unique_peak_mib": (
                        stage_rows["autograd_saved_cuda_leaf_unique_bytes_peak"].max() / (1024.0 * 1024.0)
                        if "autograd_saved_cuda_leaf_unique_bytes_peak" in stage_rows
                        else np.nan
                    ),
                    "max_autograd_saved_cuda_live_final_mib": (
                        stage_rows["autograd_saved_cuda_bytes_live_final"].max() / (1024.0 * 1024.0)
                        if "autograd_saved_cuda_bytes_live_final" in stage_rows
                        else np.nan
                    ),
                    "max_autograd_saved_cuda_unique_live_final_mib": (
                        stage_rows["autograd_saved_cuda_unique_bytes_live_final"].max() / (1024.0 * 1024.0)
                        if "autograd_saved_cuda_unique_bytes_live_final" in stage_rows
                        else np.nan
                    ),
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        numeric_cols(
            df,
            [
                "seed",
                "stage_id",
                "tasks",
                "train_tasks",
                "updates_applied",
                "max_cuda_peak_memory_allocated_mib",
                "max_cuda_peak_memory_reserved_mib",
                "max_autograd_saved_cuda_peak_mib",
                "max_autograd_saved_cuda_nonleaf_peak_mib",
                "max_autograd_saved_cuda_leaf_peak_mib",
                "max_autograd_saved_cuda_unique_peak_mib",
                "max_autograd_saved_cuda_nonleaf_unique_peak_mib",
                "max_autograd_saved_cuda_leaf_unique_peak_mib",
                "max_autograd_saved_cuda_live_final_mib",
                "max_autograd_saved_cuda_unique_live_final_mib",
            ],
        )
    return df


def parse_decision_counts(runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, run in runs.iterrows():
        raw = run.get("target_stage_decisions_json", "{}")
        try:
            decisions = json.loads(raw)
        except json.JSONDecodeError:
            decisions = {}
        total = sum(int(value) for value in decisions.values())
        for decision, count in decisions.items():
            rows.append(
                {
                    "policy": run["policy"],
                    "seed": run["seed"],
                    "decision": decision or "blank",
                    "count": int(count),
                    "fraction": int(count) / total if total else 0.0,
                }
            )
    return pd.DataFrame(rows)


def plot_summary_core(summary: pd.DataFrame, policies: list[str], out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.0))
    specs = [
        ("train_throughput_per_s", "Train throughput (req/s)", "#059669"),
        ("eval_choice_accuracy", "Eval accuracy", "#7C3AED"),
        ("eval_avg_loss", "Eval loss", "#2563EB"),
        ("target_stage_updates", "Target-stage updates", "#EA580C"),
    ]
    x = np.arange(len(policies))
    for ax, (field, title, color) in zip(axes.ravel(), specs):
        rows = summary.set_index("policy").reindex(policies)
        means = rows[f"{field}_mean"].to_numpy(dtype=float)
        stds = rows.get(f"{field}_std", pd.Series(0.0, index=rows.index)).to_numpy(dtype=float)
        ax.bar(x, means, yerr=stds, color=color, alpha=0.86, capsize=3)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(policies, rotation=22, ha="right")
        for idx, value in enumerate(means):
            if math.isfinite(value):
                fmt = f"{value:.3f}" if abs(value) < 10 else f"{value:.1f}"
                ax.text(idx, value, fmt, ha="center", va="bottom", fontsize=8)
    fig.suptitle("Scheduler policy summary", y=1.02, fontsize=13)
    savefig(out_dir / "summary_core_metrics.png")


def plot_tradeoff(summary: pd.DataFrame, policies: list[str], out_dir: Path) -> None:
    rows = summary.set_index("policy").reindex(policies).reset_index()
    baseline_updates = float(rows.loc[rows["policy"] == "all_updates", "target_stage_updates_mean"].iloc[0])
    update_fraction = rows["target_stage_updates_mean"] / baseline_updates if baseline_updates else 1.0
    sizes = 120 + 520 * update_fraction.fillna(0)

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    scatter = ax.scatter(
        rows["train_throughput_per_s_mean"],
        rows["eval_choice_accuracy_mean"],
        s=sizes,
        c=update_fraction,
        cmap="viridis",
        alpha=0.86,
        edgecolors="#111827",
        linewidths=0.7,
    )
    ax.errorbar(
        rows["train_throughput_per_s_mean"],
        rows["eval_choice_accuracy_mean"],
        xerr=rows["train_throughput_per_s_std"],
        yerr=rows["eval_choice_accuracy_std"],
        fmt="none",
        ecolor="#6B7280",
        alpha=0.7,
        capsize=3,
    )
    for _, row in rows.iterrows():
        ax.annotate(row["policy"], (row["train_throughput_per_s_mean"], row["eval_choice_accuracy_mean"]), xytext=(5, 5), textcoords="offset points")
    ax.set_title("Throughput / quality tradeoff")
    ax.set_xlabel("Train throughput (req/s)")
    ax.set_ylabel("Eval accuracy")
    colorbar = plt.colorbar(scatter, ax=ax)
    colorbar.set_label("Target-stage update fraction")
    savefig(out_dir / "throughput_quality_tradeoff.png")


def plot_loss_tradeoff(summary: pd.DataFrame, policies: list[str], out_dir: Path) -> None:
    rows = summary.set_index("policy").reindex(policies).reset_index()
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    ax.errorbar(
        rows["train_throughput_per_s_mean"],
        rows["eval_avg_loss_mean"],
        xerr=rows["train_throughput_per_s_std"],
        yerr=rows["eval_avg_loss_std"],
        fmt="o",
        color="#2563EB",
        ecolor="#6B7280",
        alpha=0.86,
        capsize=3,
    )
    for _, row in rows.iterrows():
        ax.annotate(row["policy"], (row["train_throughput_per_s_mean"], row["eval_avg_loss_mean"]), xytext=(5, 5), textcoords="offset points")
    ax.set_title("Throughput / eval-loss tradeoff")
    ax.set_xlabel("Train throughput (req/s)")
    ax.set_ylabel("Eval loss (lower is better)")
    savefig(out_dir / "throughput_eval_loss_tradeoff.png")


def plot_queue_execute(summary: pd.DataFrame, policies: list[str], out_dir: Path) -> None:
    rows = summary.set_index("policy").reindex(policies)
    x = np.arange(len(policies))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(x - width / 2, rows["target_stage_avg_queue_ms_mean"], width, label="scheduler queue", color="#F97316", alpha=0.86)
    ax.bar(x + width / 2, rows["target_stage_avg_execute_ms_mean"], width, label="execute", color="#2563EB", alpha=0.86)
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=22, ha="right")
    ax.set_ylabel("Average time (ms)")
    ax.set_title("Target-stage queue and execute time")
    ax.legend(frameon=False)
    savefig(out_dir / "target_stage_queue_execute.png")


def plot_eval_by_seed(runs: pd.DataFrame, policies: list[str], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    for policy in policies:
        df = runs[runs["policy"] == policy].sort_values("seed")
        if df.empty:
            continue
        ax.plot(df["seed"].astype(str), df["eval_choice_accuracy"], marker="o", linewidth=1.5, label=policy)
    ax.set_title("Eval accuracy by seed")
    ax.set_xlabel("Seed")
    ax.set_ylabel("Eval accuracy")
    ax.legend(frameon=False, ncol=3)
    savefig(out_dir / "eval_accuracy_by_seed.png")


def plot_decision_mix(decisions: pd.DataFrame, policies: list[str], out_dir: Path) -> None:
    if decisions.empty:
        return
    grouped = decisions.groupby(["policy", "decision"], as_index=False)["count"].sum()
    pivot = grouped.pivot(index="policy", columns="decision", values="count").fillna(0.0).reindex(policies).fillna(0.0)
    fractions = pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    bottom = np.zeros(len(fractions))
    colors = plt.get_cmap("tab20").colors
    for idx, decision in enumerate(fractions.columns):
        values = fractions[decision].to_numpy(dtype=float)
        ax.bar(np.arange(len(fractions)), values, bottom=bottom, label=decision, color=colors[idx % len(colors)], alpha=0.88)
        bottom += values
    ax.set_xticks(np.arange(len(fractions)))
    ax.set_xticklabels(fractions.index.tolist(), rotation=22, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Fraction of target-stage train-phase tasks")
    ax.set_title("Target-stage update decision mix")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    savefig(out_dir / "target_stage_update_decision_mix.png")


def plot_stage_timing(stage_metrics: pd.DataFrame, policies: list[str], out_dir: Path) -> None:
    if stage_metrics.empty:
        return
    grouped = (
        stage_metrics.groupby(["policy", "stage_id"], as_index=False)
        .agg(
            execute_ms=("avg_execute_ms", "mean"),
            queue_ms=("avg_scheduler_queue_ms", "mean"),
            worker_queue_ms=("avg_worker_queue_ms", "mean"),
            optimizer_ms=("avg_optimizer_ms", "mean"),
        )
    )
    stages = sorted(int(stage) for stage in grouped["stage_id"].dropna().unique())
    fig, axes = plt.subplots(1, len(stages), figsize=(4.8 * len(stages), 4.6), sharey=True)
    if len(stages) == 1:
        axes = [axes]
    for ax, stage in zip(axes, stages):
        rows = grouped[grouped["stage_id"] == stage].set_index("policy").reindex(policies).fillna(0.0)
        x = np.arange(len(policies))
        ax.bar(x, rows["queue_ms"], label="scheduler queue", color="#F97316", alpha=0.84)
        ax.bar(x, rows["worker_queue_ms"], bottom=rows["queue_ms"], label="worker queue", color="#A78BFA", alpha=0.84)
        ax.bar(
            x,
            rows["execute_ms"],
            bottom=rows["queue_ms"] + rows["worker_queue_ms"],
            label="execute",
            color="#2563EB",
            alpha=0.84,
        )
        ax.set_title(f"Stage {stage}")
        ax.set_xticks(x)
        ax.set_xticklabels(policies, rotation=28, ha="right")
    axes[0].set_ylabel("Average time (ms)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle("Per-stage scheduler / worker / execute timing", y=1.13, fontsize=13)
    savefig(out_dir / "stage_timing_breakdown.png")


def plot_memory(stage_metrics: pd.DataFrame, policies: list[str], out_dir: Path) -> None:
    if stage_metrics.empty:
        return
    grouped = (
        stage_metrics.groupby(["policy", "stage_id"], as_index=False)
        .agg(peak_alloc=("max_cuda_peak_memory_allocated_mib", "max"), peak_reserved=("max_cuda_peak_memory_reserved_mib", "max"))
    )
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    for stage_id in sorted(grouped["stage_id"].dropna().unique()):
        rows = grouped[grouped["stage_id"] == stage_id].set_index("policy").reindex(policies)
        ax.plot(policies, rows["peak_alloc"], marker="o", linewidth=1.6, label=f"stage {int(stage_id)} alloc")
    ax.set_xticks(np.arange(len(policies)))
    ax.set_xticklabels(policies, rotation=22, ha="right")
    ax.set_ylabel("Peak allocated memory (MiB)")
    ax.set_title("CUDA peak allocation by stage")
    ax.legend(frameon=False, ncol=3)
    savefig(out_dir / "cuda_peak_memory_by_stage.png")


def memory_comparison_rows(stage_metrics: pd.DataFrame, f1b_stage_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    if not stage_metrics.empty:
        bpfree = stage_metrics[stage_metrics["policy"] == "all_updates"].copy()
        if not bpfree.empty:
            bpfree["protocol"] = "sample_budget"
            bpfree["family"] = "BP-free"
            bpfree["memory_label"] = "BP-free all_updates"
            rows.append(bpfree)
    if not f1b_stage_metrics.empty:
        f1b = f1b_stage_metrics[f1b_stage_metrics["policy"] == "1f1b"].copy()
        if not f1b.empty:
            f1b["family"] = "Full backward"
            f1b["memory_label"] = "1F1B"
            rows.append(f1b)
    if not rows:
        return pd.DataFrame()
    combined = pd.concat(rows, ignore_index=True)
    grouped = (
        combined.groupby(["memory_label", "family", "protocol", "stage_id"], as_index=False)
        .agg(
            peak_alloc_mean=("max_cuda_peak_memory_allocated_mib", "mean"),
            peak_alloc_std=("max_cuda_peak_memory_allocated_mib", "std"),
            peak_reserved_mean=("max_cuda_peak_memory_reserved_mib", "mean"),
            peak_reserved_std=("max_cuda_peak_memory_reserved_mib", "std"),
            runs=("seed", "nunique"),
        )
        .fillna({"peak_alloc_std": 0.0, "peak_reserved_std": 0.0})
    )
    return grouped


def plot_1f1b_memory_comparison(memory_compare: pd.DataFrame, out_dir: Path) -> None:
    if memory_compare.empty:
        return
    labels = [label for label in ["BP-free all_updates", "1F1B"] if label in set(memory_compare["memory_label"])]
    stages = sorted(memory_compare["stage_id"].dropna().astype(int).unique())
    if not labels or not stages:
        return
    x = np.arange(len(stages))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    colors = {"BP-free all_updates": "#059669", "1F1B": "#111827"}
    offsets = np.linspace(-width / 2, width / 2, num=len(labels))
    for label, offset in zip(labels, offsets):
        rows = memory_compare[memory_compare["memory_label"] == label].set_index("stage_id").reindex(stages)
        means = rows["peak_alloc_mean"].to_numpy(dtype=float)
        stds = rows["peak_alloc_std"].to_numpy(dtype=float)
        ax.bar(x + offset, means, yerr=stds, width=width, color=colors.get(label, "#2563EB"), alpha=0.86, capsize=3, label=label)
        for idx, value in enumerate(means):
            if math.isfinite(value):
                ax.text(x[idx] + offset, value, f"{value:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"stage {stage}" for stage in stages])
    ax.set_ylabel("Peak allocated memory (MiB)")
    ax.set_title("BP-free vs 1F1B CUDA peak allocation")
    ax.legend(frameon=False)
    savefig(out_dir / "bpfree_vs_1f1b_memory.png")


def activation_comparison_rows(stage_metrics: pd.DataFrame, f1b_stage_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    required = "max_autograd_saved_cuda_nonleaf_unique_peak_mib"
    if not stage_metrics.empty and required in stage_metrics.columns:
        bpfree = stage_metrics[stage_metrics["policy"] == "all_updates"].copy()
        bpfree = bpfree[pd.to_numeric(bpfree[required], errors="coerce").notna()]
        if not bpfree.empty:
            bpfree["protocol"] = "sample_budget"
            bpfree["family"] = "BP-free"
            bpfree["memory_label"] = "BP-free all_updates"
            rows.append(bpfree)
    if not f1b_stage_metrics.empty and required in f1b_stage_metrics.columns:
        f1b = f1b_stage_metrics[f1b_stage_metrics["policy"] == "1f1b"].copy()
        f1b = f1b[pd.to_numeric(f1b[required], errors="coerce").notna()]
        if not f1b.empty:
            f1b["family"] = "Full backward"
            f1b["memory_label"] = "1F1B"
            rows.append(f1b)
    if not rows:
        return pd.DataFrame()
    combined = pd.concat(rows, ignore_index=True)
    grouped = (
        combined.groupby(["memory_label", "family", "protocol", "stage_id"], as_index=False)
        .agg(
            autograd_cuda_nonleaf_unique_peak_mean=(required, "mean"),
            autograd_cuda_nonleaf_unique_peak_std=(required, "std"),
            autograd_cuda_unique_peak_mean=("max_autograd_saved_cuda_unique_peak_mib", "mean"),
            autograd_cuda_unique_peak_std=("max_autograd_saved_cuda_unique_peak_mib", "std"),
            runs=("seed", "nunique"),
        )
        .fillna({"autograd_cuda_nonleaf_unique_peak_std": 0.0, "autograd_cuda_unique_peak_std": 0.0})
    )
    return grouped


def plot_activation_memory_comparison(activation_compare: pd.DataFrame, out_dir: Path) -> None:
    if activation_compare.empty:
        return
    labels = [label for label in ["BP-free all_updates", "1F1B"] if label in set(activation_compare["memory_label"])]
    stages = sorted(activation_compare["stage_id"].dropna().astype(int).unique())
    if not labels or not stages:
        return
    x = np.arange(len(stages))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    colors = {"BP-free all_updates": "#059669", "1F1B": "#111827"}
    offsets = np.linspace(-width / 2, width / 2, num=len(labels))
    for label, offset in zip(labels, offsets):
        rows = activation_compare[activation_compare["memory_label"] == label].set_index("stage_id").reindex(stages)
        means = rows["autograd_cuda_nonleaf_unique_peak_mean"].to_numpy(dtype=float)
        stds = rows["autograd_cuda_nonleaf_unique_peak_std"].to_numpy(dtype=float)
        ax.bar(x + offset, means, yerr=stds, width=width, color=colors.get(label, "#2563EB"), alpha=0.86, capsize=3, label=label)
        for idx, value in enumerate(means):
            if math.isfinite(value):
                ax.text(x[idx] + offset, value, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"stage {stage}" for stage in stages])
    ax.set_ylabel("Autograd saved CUDA non-leaf unique peak (MiB)")
    ax.set_title("Measured activation stash: BP-free vs 1F1B")
    ax.legend(frameon=False)
    savefig(out_dir / "bpfree_vs_1f1b_activation_stash.png")


def rolling(series: pd.Series, window: int = 35) -> pd.Series:
    if len(series) < 3:
        return series
    effective = min(window, max(3, len(series) // 10))
    return series.rolling(effective, min_periods=1).mean()


def plot_train_loss_curves(root: Path, runs: pd.DataFrame, policies: list[str], out_dir: Path) -> None:
    seed = sorted(runs["seed"].dropna().astype(int).unique())[0]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    any_curve = False
    for policy in policies:
        selected = runs[(runs["policy"] == policy) & (runs["seed"] == seed)]
        if selected.empty:
            continue
        run_dir = resolve_run_dir(selected.iloc[0]["output_dir"], root)
        results_path = run_dir / "scheduler_results.csv"
        if not results_path.is_file():
            continue
        df = pd.read_csv(results_path)
        if "phase" in df.columns:
            df = df[df["phase"] == "train"]
        if df.empty or "loss" not in df.columns:
            continue
        df["loss"] = pd.to_numeric(df["loss"], errors="coerce")
        df = df.dropna(subset=["loss"]).sort_values("seq")
        if df.empty:
            continue
        ax.plot(df["seq"], rolling(df["loss"]), linewidth=1.6, label=policy)
        any_curve = True
    if not any_curve:
        plt.close(fig)
        return
    ax.set_title(f"Train loss rolling mean (seed {seed})")
    ax.set_xlabel("Request sequence")
    ax.set_ylabel("Loss")
    ax.legend(frameon=False, ncol=3)
    savefig(out_dir / "train_loss_curves_seed_first.png")


def scheduler_sample_stats(runs: pd.DataFrame) -> pd.DataFrame:
    if runs.empty or "train_completed" not in runs.columns:
        return pd.DataFrame()
    grouped = (
        runs.groupby("policy", as_index=False)
        .agg(
            train_samples_seen_mean=("train_completed", "mean"),
            train_samples_seen_std=("train_completed", "std"),
            train_wall_ms_mean=("train_throughput_per_s", lambda values: np.nan),
        )
    )
    for index, row in grouped.iterrows():
        policy_runs = runs[runs["policy"] == row["policy"]]
        wall_values = []
        for _, policy_run in policy_runs.iterrows():
            throughput = float(policy_run.get("train_throughput_per_s", 0.0) or 0.0)
            samples = float(policy_run.get("train_completed", 0.0) or 0.0)
            if throughput > 0:
                wall_values.append(samples / throughput * 1000.0)
        grouped.loc[index, "train_wall_ms_mean"] = sum(wall_values) / len(wall_values) if wall_values else np.nan
        if len(wall_values) > 1:
            mean = grouped.loc[index, "train_wall_ms_mean"]
            grouped.loc[index, "train_wall_ms_std"] = (sum((value - mean) ** 2 for value in wall_values) / len(wall_values)) ** 0.5
        else:
            grouped.loc[index, "train_wall_ms_std"] = 0.0
    grouped["train_samples_seen_std"] = grouped["train_samples_seen_std"].fillna(0.0)
    return grouped


def comparison_rows(summary: pd.DataFrame, f1b_summary: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    selected = ["all_updates", "queue0", "queue1", "queue2"]
    rows: list[dict[str, Any]] = []
    sample_stats = scheduler_sample_stats(runs).set_index("policy") if not runs.empty else pd.DataFrame()
    for _, row in summary[summary["policy"].isin(selected)].iterrows():
        stats = sample_stats.loc[row["policy"]] if not sample_stats.empty and row["policy"] in sample_stats.index else {}
        rows.append(
            {
                "policy": row["policy"],
                "family": "BP-free",
                "protocol": "sample_budget",
                "train_throughput_per_s_mean": row["train_throughput_per_s_mean"],
                "train_throughput_per_s_std": row.get("train_throughput_per_s_std", 0.0),
                "train_wall_ms_mean": stats.get("train_wall_ms_mean", np.nan) if hasattr(stats, "get") else np.nan,
                "train_wall_ms_std": stats.get("train_wall_ms_std", 0.0) if hasattr(stats, "get") else 0.0,
                "train_samples_seen_mean": stats.get("train_samples_seen_mean", np.nan) if hasattr(stats, "get") else np.nan,
                "train_samples_seen_std": stats.get("train_samples_seen_std", 0.0) if hasattr(stats, "get") else 0.0,
                "eval_choice_accuracy_mean": row["eval_choice_accuracy_mean"],
                "eval_choice_accuracy_std": row.get("eval_choice_accuracy_std", 0.0),
                "eval_avg_loss_mean": row["eval_avg_loss_mean"],
                "eval_avg_loss_std": row.get("eval_avg_loss_std", 0.0),
                "train_update_units_mean": row.get("target_stage_updates_mean", np.nan),
                "train_update_units_std": row.get("target_stage_updates_std", 0.0),
                "train_update_unit_name": "target-stage local updates",
            }
        )
    for _, row in f1b_summary.iterrows():
        rows.append(
            {
                "policy": row.get("policy", "1f1b"),
                "family": "Full backward",
                "protocol": row.get("protocol", "sample_budget"),
                "train_throughput_per_s_mean": row["train_throughput_per_s_mean"],
                "train_throughput_per_s_std": row.get("train_throughput_per_s_std", 0.0),
                "train_wall_ms_mean": row.get("train_wall_ms_mean", np.nan),
                "train_wall_ms_std": row.get("train_wall_ms_std", 0.0),
                "train_samples_seen_mean": row.get("train_records_mean", np.nan),
                "train_samples_seen_std": row.get("train_records_std", 0.0),
                "eval_choice_accuracy_mean": row["eval_choice_accuracy_mean"],
                "eval_choice_accuracy_std": row.get("eval_choice_accuracy_std", 0.0),
                "eval_avg_loss_mean": row["eval_avg_loss_mean"],
                "eval_avg_loss_std": row.get("eval_avg_loss_std", 0.0),
                "train_update_units_mean": row.get("train_batches_mean", np.nan),
                "train_update_units_std": row.get("train_batches_std", 0.0),
                "train_update_unit_name": "pipeline optimizer steps",
            }
        )
    return pd.DataFrame(rows)


def plot_1f1b_comparison(compare: pd.DataFrame, out_dir: Path) -> None:
    sample_compare = compare[compare.get("protocol", "sample_budget") == "sample_budget"] if "protocol" in compare.columns else compare
    if sample_compare.empty or "1f1b" not in set(sample_compare["policy"]):
        return
    order = [name for name in ["all_updates", "queue0", "queue1", "queue2", "1f1b"] if name in set(sample_compare["policy"])]
    rows = sample_compare.set_index("policy").reindex(order)
    colors = ["#059669" if family == "BP-free" else "#111827" for family in rows["family"]]
    x = np.arange(len(rows))

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.2))
    specs = [
        ("train_throughput_per_s", "Train throughput (samples/s)", True),
        ("eval_choice_accuracy", "Eval accuracy", True),
        ("eval_avg_loss", "Eval loss", False),
        ("train_update_units", "Train update units", True),
    ]
    for ax, (field, title, annotate) in zip(axes.ravel(), specs):
        means = rows[f"{field}_mean"].to_numpy(dtype=float)
        stds = rows.get(f"{field}_std", pd.Series(0.0, index=rows.index)).to_numpy(dtype=float)
        ax.bar(x, means, yerr=stds, color=colors, alpha=0.86, capsize=3)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=22, ha="right")
        if annotate:
            for idx, value in enumerate(means):
                if math.isfinite(value):
                    fmt = f"{value:.3f}" if abs(value) < 10 else f"{value:.1f}"
                    ax.text(idx, value, fmt, ha="center", va="bottom", fontsize=8)
    fig.suptitle("BP-free scheduler policies vs full-backward 1F1B", y=1.02, fontsize=13)
    savefig(out_dir / "bpfree_vs_1f1b_core.png")

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    for _, row in sample_compare.iterrows():
        marker = "D" if row["policy"] == "1f1b" else "o"
        color = "#111827" if row["policy"] == "1f1b" else "#059669"
        ax.errorbar(
            row["train_throughput_per_s_mean"],
            row["eval_choice_accuracy_mean"],
            xerr=row.get("train_throughput_per_s_std", 0.0),
            yerr=row.get("eval_choice_accuracy_std", 0.0),
            fmt=marker,
            markersize=8,
            color=color,
            ecolor="#6B7280",
            capsize=3,
        )
        ax.annotate(
            row["policy"],
            (row["train_throughput_per_s_mean"], row["eval_choice_accuracy_mean"]),
            xytext=(6, 5),
            textcoords="offset points",
        )
    ax.set_title("Train throughput / eval quality with 1F1B baseline")
    ax.set_xlabel("Train throughput (samples/s)")
    ax.set_ylabel("Eval accuracy")
    savefig(out_dir / "bpfree_vs_1f1b_tradeoff.png")

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    for _, row in sample_compare.iterrows():
        marker = "D" if row["policy"] == "1f1b" else "o"
        color = "#111827" if row["policy"] == "1f1b" else "#2563EB"
        ax.errorbar(
            row["train_throughput_per_s_mean"],
            row["eval_avg_loss_mean"],
            xerr=row.get("train_throughput_per_s_std", 0.0),
            yerr=row.get("eval_avg_loss_std", 0.0),
            fmt=marker,
            markersize=8,
            color=color,
            ecolor="#6B7280",
            capsize=3,
        )
        ax.annotate(
            row["policy"],
            (row["train_throughput_per_s_mean"], row["eval_avg_loss_mean"]),
            xytext=(6, 5),
            textcoords="offset points",
        )
    ax.set_title("Train throughput / eval loss with 1F1B baseline")
    ax.set_xlabel("Train throughput (samples/s)")
    ax.set_ylabel("Eval loss (lower is better)")
    savefig(out_dir / "bpfree_vs_1f1b_loss_tradeoff.png")


def plot_update_budget_comparison(compare: pd.DataFrame, out_dir: Path) -> None:
    if compare.empty or "protocol" not in compare.columns:
        return
    all_updates = compare[(compare["policy"] == "all_updates") & (compare["protocol"] == "sample_budget")]
    f1b_update = compare[(compare["policy"] == "1f1b_update") & (compare["protocol"] == "update_budget")]
    if all_updates.empty or f1b_update.empty:
        return
    rows = pd.concat([all_updates.iloc[[0]], f1b_update.iloc[[0]]], ignore_index=True)
    labels = ["BP-free all_updates", "1F1B update-matched"]
    colors = ["#059669", "#111827"]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.2))
    specs = [
        ("train_update_units", "Optimizer/update units", True),
        ("train_samples_seen", "Training samples consumed", True),
        ("train_wall_ms", "Train wall time (ms)", True),
        ("eval_avg_loss", "Eval loss", False),
    ]
    for ax, (field, title, annotate) in zip(axes.ravel(), specs):
        means = rows[f"{field}_mean"].to_numpy(dtype=float)
        stds = rows.get(f"{field}_std", pd.Series(0.0, index=rows.index)).to_numpy(dtype=float)
        ax.bar(x, means, yerr=stds, color=colors, alpha=0.86, capsize=3)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        if annotate:
            for idx, value in enumerate(means):
                if math.isfinite(value):
                    ax.text(idx, value, f"{value:.0f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Update-budget comparison", y=1.02, fontsize=13)
    savefig(out_dir / "bpfree_vs_1f1b_update_budget.png")


def write_stage_metrics_csv(stage_metrics: pd.DataFrame, out_dir: Path) -> None:
    if stage_metrics.empty:
        return
    path = out_dir / "stage_worker_metrics.csv"
    stage_metrics.to_csv(path, index=False)
    print(f"Wrote {path}")


def write_decision_counts_csv(decisions: pd.DataFrame, out_dir: Path) -> None:
    if decisions.empty:
        return
    path = out_dir / "target_stage_decision_counts.csv"
    decisions.to_csv(path, index=False)
    print(f"Wrote {path}")


def write_1f1b_comparison_csv(compare: pd.DataFrame, root: Path) -> None:
    if compare.empty:
        return
    path = root / "bpfree_vs_1f1b_comparison.csv"
    compare.to_csv(path, index=False)
    print(f"Wrote {path}")


def write_memory_comparison_csv(memory_compare: pd.DataFrame, root: Path) -> None:
    if memory_compare.empty:
        return
    path = root / "bpfree_vs_1f1b_memory.csv"
    memory_compare.to_csv(path, index=False)
    print(f"Wrote {path}")


def write_activation_comparison_csv(activation_compare: pd.DataFrame, root: Path) -> None:
    if activation_compare.empty:
        return
    path = root / "bpfree_vs_1f1b_activation_stash.csv"
    activation_compare.to_csv(path, index=False)
    print(f"Wrote {path}")


def select_compare_row(compare: pd.DataFrame, policy: str, protocol: str | None = None) -> pd.Series | None:
    rows = compare[compare["policy"] == policy]
    if protocol is not None and "protocol" in rows.columns:
        rows = rows[rows["protocol"] == protocol]
    if rows.empty:
        return None
    return rows.iloc[0]


def formal_protocol_rows(compare: pd.DataFrame) -> pd.DataFrame:
    if compare.empty:
        return pd.DataFrame()
    pairs = [
        (
            "sample_budget",
            "Same train/eval request count; optimizer/update count is reported, not matched.",
            select_compare_row(compare, "all_updates", "sample_budget"),
            select_compare_row(compare, "1f1b", "sample_budget"),
        ),
        (
            "update_budget",
            "Same nominal update count; consumed samples and wall time are reported, not matched.",
            select_compare_row(compare, "all_updates", "sample_budget"),
            select_compare_row(compare, "1f1b_update", "update_budget"),
        ),
    ]
    rows: list[dict[str, Any]] = []
    for protocol, definition, bpfree, f1b in pairs:
        if bpfree is None or f1b is None:
            continue
        bpfree_tput = float(bpfree.get("train_throughput_per_s_mean", np.nan))
        f1b_tput = float(f1b.get("train_throughput_per_s_mean", np.nan))
        bpfree_wall = float(bpfree.get("train_wall_ms_mean", np.nan))
        f1b_wall = float(f1b.get("train_wall_ms_mean", np.nan))
        bpfree_samples = float(bpfree.get("train_samples_seen_mean", np.nan))
        f1b_samples = float(f1b.get("train_samples_seen_mean", np.nan))
        rows.append(
            {
                "protocol": protocol,
                "definition": definition,
                "bpfree_policy": bpfree["policy"],
                "full_backward_policy": f1b["policy"],
                "bpfree_train_samples": bpfree_samples,
                "full_backward_train_samples": f1b_samples,
                "sample_ratio_full_backward_over_bpfree": f1b_samples / bpfree_samples if bpfree_samples else np.nan,
                "bpfree_update_units": float(bpfree.get("train_update_units_mean", np.nan)),
                "full_backward_update_units": float(f1b.get("train_update_units_mean", np.nan)),
                "bpfree_update_unit_name": bpfree.get("train_update_unit_name", ""),
                "full_backward_update_unit_name": f1b.get("train_update_unit_name", ""),
                "bpfree_train_wall_s": bpfree_wall / 1000.0,
                "full_backward_train_wall_s": f1b_wall / 1000.0,
                "wall_ratio_full_backward_over_bpfree": f1b_wall / bpfree_wall if bpfree_wall else np.nan,
                "bpfree_train_throughput_per_s": bpfree_tput,
                "full_backward_train_throughput_per_s": f1b_tput,
                "throughput_ratio_full_backward_over_bpfree": f1b_tput / bpfree_tput if bpfree_tput else np.nan,
                "bpfree_eval_choice_accuracy": float(bpfree.get("eval_choice_accuracy_mean", np.nan)),
                "full_backward_eval_choice_accuracy": float(f1b.get("eval_choice_accuracy_mean", np.nan)),
                "accuracy_delta_full_backward_minus_bpfree": float(f1b.get("eval_choice_accuracy_mean", np.nan))
                - float(bpfree.get("eval_choice_accuracy_mean", np.nan)),
                "bpfree_eval_loss": float(bpfree.get("eval_avg_loss_mean", np.nan)),
                "full_backward_eval_loss": float(f1b.get("eval_avg_loss_mean", np.nan)),
                "loss_delta_full_backward_minus_bpfree": float(f1b.get("eval_avg_loss_mean", np.nan))
                - float(bpfree.get("eval_avg_loss_mean", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def write_formal_protocol_csv(protocol_compare: pd.DataFrame, root: Path) -> None:
    if protocol_compare.empty:
        return
    path = root / "formal_protocol_comparison.csv"
    protocol_compare.to_csv(path, index=False)
    print(f"Wrote {path}")


def format_number(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    if abs(number) >= 100:
        return f"{number:.1f}"
    return f"{number:.{digits}f}"


def formal_protocol_table(protocol_compare: pd.DataFrame, has_choice_metric: bool) -> str:
    if protocol_compare.empty:
        return ""
    metric_header = "eval acc delta" if has_choice_metric else "eval loss delta"
    headers = [
        "protocol",
        "samples",
        "updates",
        "wall time",
        "throughput",
        metric_header,
    ]
    lines = ["|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, row in protocol_compare.iterrows():
        sample_cell = (
            f"{format_number(row['bpfree_train_samples'], 1)} vs "
            f"{format_number(row['full_backward_train_samples'], 1)}"
        )
        update_cell = (
            f"{format_number(row['bpfree_update_units'], 1)} local vs "
            f"{format_number(row['full_backward_update_units'], 1)} pipe"
        )
        wall_cell = (
            f"{format_number(row['bpfree_train_wall_s'], 1)}s vs "
            f"{format_number(row['full_backward_train_wall_s'], 1)}s"
        )
        tput_cell = (
            f"{format_number(row['bpfree_train_throughput_per_s'], 3)} vs "
            f"{format_number(row['full_backward_train_throughput_per_s'], 3)} samples/s"
        )
        if has_choice_metric:
            metric_cell = format_number(row["accuracy_delta_full_backward_minus_bpfree"], 4)
        else:
            metric_cell = format_number(row["loss_delta_full_backward_minus_bpfree"], 4)
        lines.append(
            "|"
            + "|".join(
                [
                    str(row["protocol"]),
                    sample_cell,
                    update_cell,
                    wall_cell,
                    tput_cell,
                    metric_cell,
                ]
            )
            + "|"
        )
    return "\n".join(lines)


def markdown_table(summary: pd.DataFrame, policies: list[str]) -> str:
    rows = summary.set_index("policy").reindex(policies).reset_index()
    headers = ["policy", "train req/s", "eval acc", "eval loss", "stage updates", "stage queue ms"]
    lines = ["|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, row in rows.iterrows():
        lines.append(
            "|"
            + "|".join(
                [
                    str(row["policy"]),
                    f"{row['train_throughput_per_s_mean']:.3f} +/- {row['train_throughput_per_s_std']:.3f}",
                    f"{row['eval_choice_accuracy_mean']:.4f} +/- {row['eval_choice_accuracy_std']:.4f}",
                    f"{row['eval_avg_loss_mean']:.4f} +/- {row['eval_avg_loss_std']:.4f}",
                    f"{row['target_stage_updates_mean']:.1f} +/- {row['target_stage_updates_std']:.1f}",
                    f"{row['target_stage_avg_queue_ms_mean']:.1f} +/- {row['target_stage_avg_queue_ms_std']:.1f}",
                ]
            )
            + "|"
        )
    return "\n".join(lines)


def write_report(
    root: Path,
    out_dir: Path,
    runs: pd.DataFrame,
    summary: pd.DataFrame,
    policies: list[str],
    compare: pd.DataFrame,
    protocol_compare: pd.DataFrame,
) -> None:
    baseline = summary[summary["policy"] == "all_updates"]
    best_tput = summary.loc[summary["train_throughput_per_s_mean"].idxmax()]
    has_choice_metric = float(summary["eval_choice_accuracy_mean"].max()) > 0
    best_acc = summary.loc[summary["eval_choice_accuracy_mean"].idxmax()] if has_choice_metric else None
    best_loss = summary.loc[summary["eval_avg_loss_mean"].idxmin()]
    lines = [
        "# Scheduler Scenario Report",
        "",
        f"Root: `{root}`",
        f"Runs: {len(runs)}",
        f"Policies: {', '.join(policies)}",
        "",
        "## Summary",
        "",
        markdown_table(summary, policies),
        "",
        f"Best train throughput: `{best_tput['policy']}` at {best_tput['train_throughput_per_s_mean']:.3f} req/s.",
    ]
    if best_acc is not None:
        lines.append(f"Best eval accuracy: `{best_acc['policy']}` at {best_acc['eval_choice_accuracy_mean']:.4f}.")
    else:
        lines.append(
            f"No choice-label metric is available for this scenario; compare eval loss instead. "
            f"Best eval loss: `{best_loss['policy']}` at {best_loss['eval_avg_loss_mean']:.4f}."
        )
    if not baseline.empty:
        base = baseline.iloc[0]
        gain = best_tput["train_throughput_per_s_mean"] / base["train_throughput_per_s_mean"] - 1.0
        if has_choice_metric:
            acc_delta = best_tput["eval_choice_accuracy_mean"] - base["eval_choice_accuracy_mean"]
            lines.append(
                f"Throughput winner vs all_updates: {gain * 100:.1f}% throughput change, "
                f"{acc_delta:+.4f} eval-accuracy change."
            )
        else:
            loss_delta = best_tput["eval_avg_loss_mean"] - base["eval_avg_loss_mean"]
            lines.append(
                f"Throughput winner vs all_updates: {gain * 100:.1f}% throughput change, "
                f"{loss_delta:+.4f} eval-loss change."
            )
    if not compare.empty and "1f1b" in set(compare["policy"]):
        rows = compare.set_index("policy")
        f1b = rows.loc["1f1b"]
        all_updates = rows.loc["all_updates"] if "all_updates" in rows.index else None
        queue2 = rows.loc["queue2"] if "queue2" in rows.index else None
        lines.extend(["", "## 1F1B Baseline", ""])
        if all_updates is not None:
            lines.append(
                "Sample-matched 1F1B uses full backward with "
                f"{f1b['train_update_units_mean']:.0f} pipeline optimizer steps "
                f"(batch size 8), while BP-free all_updates applies "
                f"{all_updates['train_update_units_mean']:.0f} target-stage local updates."
            )
            if has_choice_metric:
                lines.append(
                    f"Against all_updates, 1F1B train throughput changes from "
                    f"{all_updates['train_throughput_per_s_mean']:.3f} to "
                    f"{f1b['train_throughput_per_s_mean']:.3f} samples/s, and eval accuracy changes from "
                    f"{all_updates['eval_choice_accuracy_mean']:.4f} to {f1b['eval_choice_accuracy_mean']:.4f}."
                )
            else:
                lines.append(
                    f"Against all_updates, 1F1B train throughput changes from "
                    f"{all_updates['train_throughput_per_s_mean']:.3f} to "
                    f"{f1b['train_throughput_per_s_mean']:.3f} samples/s, and eval loss changes from "
                    f"{all_updates['eval_avg_loss_mean']:.4f} to {f1b['eval_avg_loss_mean']:.4f}."
                )
        if queue2 is not None:
            if has_choice_metric:
                lines.append(
                    f"Against queue2, 1F1B eval accuracy is {f1b['eval_choice_accuracy_mean']:.4f} "
                    f"vs {queue2['eval_choice_accuracy_mean']:.4f}, with train throughput "
                    f"{f1b['train_throughput_per_s_mean']:.3f} vs {queue2['train_throughput_per_s_mean']:.3f} samples/s."
                )
            else:
                lines.append(
                    f"Against queue2, 1F1B eval loss is {f1b['eval_avg_loss_mean']:.4f} "
                    f"vs {queue2['eval_avg_loss_mean']:.4f}, with train throughput "
                    f"{f1b['train_throughput_per_s_mean']:.3f} vs {queue2['train_throughput_per_s_mean']:.3f} samples/s."
                )
        if "1f1b_update" in rows.index:
            f1b_update = rows.loc["1f1b_update"]
            lines.extend(["", "## Formal Protocols", ""])
            if not protocol_compare.empty:
                lines.extend(
                    [
                        "Primary comparisons are reported as matched-budget protocols; queue/stride policies remain ablations.",
                        "",
                        formal_protocol_table(protocol_compare, has_choice_metric),
                        "",
                    ]
                )
            lines.append(
                "Sample-budget comparison keeps the same training/eval request count and answers throughput-per-sample "
                "and quality-per-sample questions."
            )
            lines.append(
                "Update-budget comparison matches optimizer/update units: BP-free all_updates uses "
                f"{all_updates['train_update_units_mean']:.0f} target-stage local updates, while update-matched "
                f"1F1B uses {f1b_update['train_update_units_mean']:.0f} pipeline optimizer steps."
            )
            lines.append(
                f"Update-matched 1F1B consumes {f1b_update['train_samples_seen_mean']:.0f} training samples "
                f"and takes {f1b_update['train_wall_ms_mean'] / 1000.0:.1f}s train wall time; "
                f"BP-free all_updates consumes {all_updates['train_samples_seen_mean']:.0f} samples "
                f"and takes {all_updates['train_wall_ms_mean'] / 1000.0:.1f}s."
            )
            if has_choice_metric:
                lines.append(
                    f"Update-budget eval accuracy: 1F1B {f1b_update['eval_choice_accuracy_mean']:.4f} vs "
                    f"BP-free all_updates {all_updates['eval_choice_accuracy_mean']:.4f}."
                )
            else:
                lines.append(
                    f"Update-budget eval loss: 1F1B {f1b_update['eval_avg_loss_mean']:.4f} vs "
                    f"BP-free all_updates {all_updates['eval_avg_loss_mean']:.4f}."
                )
        else:
            lines.append(
                "Caveat: this is a sample-budget comparison. A separate update-budget comparison would repeat "
                "1F1B for more epochs so it receives a similar number of optimizer steps."
            )
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "![summary](figures/summary_core_metrics.png)",
            "",
            "![bpfree-1f1b](figures/bpfree_vs_1f1b_core.png)",
            "",
            "![bpfree-1f1b-tradeoff](figures/bpfree_vs_1f1b_tradeoff.png)",
            "",
            "![bpfree-1f1b-loss](figures/bpfree_vs_1f1b_loss_tradeoff.png)",
            "",
            "![bpfree-1f1b-memory](figures/bpfree_vs_1f1b_memory.png)",
            "",
            "![tradeoff](figures/throughput_quality_tradeoff.png)",
            "",
            "![loss-tradeoff](figures/throughput_eval_loss_tradeoff.png)",
            "",
            "![queue-execute](figures/target_stage_queue_execute.png)",
            "",
            "![stage-timing](figures/stage_timing_breakdown.png)",
            "",
            "![decision-mix](figures/target_stage_update_decision_mix.png)",
            "",
            "![seed-accuracy](figures/eval_accuracy_by_seed.png)",
            "",
            "![loss-curves](figures/train_loss_curves_seed_first.png)",
            "",
            "![memory](figures/cuda_peak_memory_by_stage.png)",
            "",
        ]
    )
    if (out_dir / "bpfree_vs_1f1b_activation_stash.png").is_file():
        lines.extend(
            [
                "## Activation Stash Figure",
                "",
                "![activation-stash](figures/bpfree_vs_1f1b_activation_stash.png)",
                "",
            ]
        )
    if (out_dir / "bpfree_vs_1f1b_update_budget.png").is_file():
        lines.extend(
            [
                "## Update-Budget Figure",
                "",
                "![update-budget](figures/bpfree_vs_1f1b_update_budget.png)",
                "",
            ]
        )
    schedule_figures = [
        ("all-updates schedule", "schedule_gantt_all_updates_seed20260531.png"),
        ("queue0 schedule", "schedule_gantt_queue0_seed20260531.png"),
        ("queue2 schedule", "schedule_gantt_queue2_seed20260531.png"),
        ("stride3 schedule", "schedule_gantt_stride3_seed20260531.png"),
        ("busy/bubble by policy", "device_busy_bubble_by_policy.png"),
        ("bubble fraction by policy", "device_bubble_fraction_by_policy.png"),
    ]
    existing_schedule_figures = [(label, name) for label, name in schedule_figures if (out_dir / name).is_file()]
    if existing_schedule_figures:
        lines.extend(["## Per-Device Schedule", ""])
        for label, name in existing_schedule_figures:
            lines.extend([f"![{label}](figures/{name})", ""])
    path = root / "REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Scenario output root.")
    parser.add_argument("--output_dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    root = args.root
    out_dir = args.output_dir or root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    runs, summary = read_inputs(root)
    f1b_runs, f1b_summary = read_1f1b_inputs(root)
    policies = ordered_policies(summary["policy"].astype(str).tolist())
    stage_metrics = load_stage_worker_metrics(root, runs)
    f1b_stage_metrics = load_1f1b_stage_metrics(root, f1b_runs)
    memory_compare = memory_comparison_rows(stage_metrics, f1b_stage_metrics)
    activation_compare = activation_comparison_rows(stage_metrics, f1b_stage_metrics)
    decisions = parse_decision_counts(runs)
    compare = comparison_rows(summary, f1b_summary, runs)
    protocol_compare = formal_protocol_rows(compare)

    write_stage_metrics_csv(stage_metrics, root)
    write_decision_counts_csv(decisions, root)
    write_1f1b_comparison_csv(compare, root)
    write_formal_protocol_csv(protocol_compare, root)
    write_memory_comparison_csv(memory_compare, root)
    write_activation_comparison_csv(activation_compare, root)
    plot_summary_core(summary, policies, out_dir)
    plot_1f1b_comparison(compare, out_dir)
    plot_update_budget_comparison(compare, out_dir)
    plot_1f1b_memory_comparison(memory_compare, out_dir)
    plot_activation_memory_comparison(activation_compare, out_dir)
    plot_tradeoff(summary, policies, out_dir)
    plot_loss_tradeoff(summary, policies, out_dir)
    plot_queue_execute(summary, policies, out_dir)
    plot_eval_by_seed(runs, policies, out_dir)
    plot_decision_mix(decisions, policies, out_dir)
    plot_stage_timing(stage_metrics, policies, out_dir)
    plot_memory(stage_metrics, policies, out_dir)
    plot_train_loss_curves(root, runs, policies, out_dir)
    write_report(root, out_dir, runs, summary, policies, compare, protocol_compare)


if __name__ == "__main__":
    main()
