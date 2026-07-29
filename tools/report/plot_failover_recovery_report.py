#!/usr/bin/env python3
"""Report real-GPU failover recovery from scheduler artifacts.

This script intentionally uses scheduler timestamps instead of inferred bubble
fractions. It compares a wait-for-rejoin policy with active stage migration
from the cached upstream boundary to a warm replica.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class RunData:
    policy: str
    seed: int
    run_dir: Path
    summary: dict[str, Any]
    metrics: list[dict[str, Any]]
    event: dict[str, Any]


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
            "axes.grid": True,
            "grid.alpha": 0.22,
        }
    )


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Wrote {path}")


def numeric(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    value = row.get(key, "")
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def integer(row: dict[str, Any], key: str, default: int = -1) -> int:
    value = numeric(row, key, float(default))
    return int(value) if np.isfinite(value) else default


def truthy(row: dict[str, Any], key: str) -> bool:
    return str(row.get(key, "")).strip().lower() in {"1", "true", "yes"}


def load_run(run_dir: Path, policy: str, seed: int) -> RunData:
    summary_path = run_dir / "scheduler_summary.json"
    metrics_path = run_dir / "scheduler_stage_metrics.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        metrics = list(csv.DictReader(handle))
    migration_events = summary.get("migration", {}).get("events", [])
    rejoin_events = summary.get("rejoin", {}).get("events", [])
    events = migration_events or rejoin_events
    if not events:
        raise RuntimeError(f"No failover or rejoin event recorded in {summary_path}")
    return RunData(policy=policy, seed=seed, run_dir=run_dir, summary=summary, metrics=metrics, event=events[0])


def find_runs(root: Path, policies: list[str]) -> list[RunData]:
    runs: list[RunData] = []
    for policy in policies:
        for run_dir in sorted(root.glob(f"{policy}_seed*")):
            match = re.search(r"_seed(\d+)$", run_dir.name)
            if match is None:
                continue
            if (run_dir / "scheduler_summary.json").is_file() and (run_dir / "scheduler_stage_metrics.csv").is_file():
                runs.append(load_run(run_dir, policy=policy, seed=int(match.group(1))))
    if not runs:
        raise RuntimeError(f"No completed failover runs found below {root}")
    return runs


def train_metrics(run: RunData) -> list[dict[str, Any]]:
    rows = [row for row in run.metrics if row.get("phase") == "train"]
    return rows if rows else list(run.metrics)


def is_failure(row: dict[str, Any]) -> bool:
    return bool(str(row.get("failure", "")).strip())


def event_rows(run: RunData) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    rows = train_metrics(run)
    seq = int(run.event["seq"])
    stage = int(run.event["stage_id"])
    failed = [
        row
        for row in rows
        if integer(row, "seq") == seq and integer(row, "stage_id") == stage and is_failure(row)
    ]
    if not failed:
        raise RuntimeError(f"Could not find failed stage event for {run.run_dir}")
    failure = min(failed, key=lambda row: numeric(row, "worker_end_epoch_ms"))
    retries = [
        row
        for row in rows
        if integer(row, "seq") == seq
        and integer(row, "stage_id") == stage
        and integer(row, "attempt") > integer(failure, "attempt")
        and not is_failure(row)
    ]
    if not retries:
        raise RuntimeError(f"Could not find successful retry for {run.run_dir}")
    retry = min(retries, key=lambda row: numeric(row, "worker_end_epoch_ms"))
    final_stage = max(integer(row, "stage_id") for row in rows)
    finals = [
        row
        for row in rows
        if integer(row, "seq") == seq and integer(row, "stage_id") == final_stage and not is_failure(row)
    ]
    if not finals:
        raise RuntimeError(f"Could not find end-to-end completion for {run.run_dir}")
    final = min(finals, key=lambda row: numeric(row, "worker_end_epoch_ms"))
    return failure, retry, final


def checkpoint_restore_row(run: RunData) -> dict[str, Any] | None:
    restored = [
        row
        for row in train_metrics(run)
        if numeric(row, "checkpoint_restore_bytes", 0.0) > 0
    ]
    if not restored:
        return None
    return min(restored, key=lambda row: numeric(row, "worker_start_epoch_ms"))


def train_summary(run: RunData) -> dict[str, Any]:
    for phase in run.summary.get("phase_summaries", []):
        if phase.get("phase") == "train":
            return phase
    return run.summary


def per_seed_rows(runs: list[RunData]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        failure, retry, final = event_rows(run)
        failure_end = numeric(failure, "worker_end_epoch_ms")
        retry_end = numeric(retry, "worker_end_epoch_ms")
        final_end = numeric(final, "worker_end_epoch_ms")
        retries_stage0 = sum(
            1
            for metric in train_metrics(run)
            if integer(metric, "seq") == int(run.event["seq"])
            and integer(metric, "stage_id") == 0
            and not is_failure(metric)
        )
        phase = train_summary(run)
        restored = checkpoint_restore_row(run)
        checkpoint_captures = sum(
            1 for metric in train_metrics(run) if numeric(metric, "checkpoint_captured_bytes", 0.0) > 0
        )
        checkpoint_interval = run.summary.get("migration", {}).get("checkpoint_interval")
        if checkpoint_interval is None:
            checkpoint_interval = 1 if checkpoint_captures else 0
        rows.append(
            {
                "policy": run.policy,
                "seed": run.seed,
                "failure_point": str(run.summary.get("failure_point", "")),
                "failure_seq": int(run.event["seq"]),
                "failure_stage": int(run.event["stage_id"]),
                "gradient_accumulation_steps": int(run.summary.get("gradient_accumulation_steps", 1)),
                "stage_recovery_ms": retry_end - failure_end,
                "request_completion_recovery_ms": final_end - failure_end,
                "checkpoint_restore_bytes": (
                    numeric(restored, "checkpoint_restore_bytes", 0.0) if restored is not None else 0.0
                ),
                "checkpoint_restore_ms": (
                    numeric(restored, "checkpoint_restore_ms", 0.0) if restored is not None else 0.0
                ),
                "checkpoint_captures": checkpoint_captures,
                "checkpoint_captured_bytes_total": sum(
                    numeric(metric, "checkpoint_captured_bytes", 0.0) for metric in train_metrics(run)
                ),
                "checkpoint_interval": int(checkpoint_interval),
                "catchup_updates": int(run.event.get("catchup_updates", 0)),
                "catchup_input_bytes": float(run.event.get("catchup_input_bytes", 0.0)),
                "stage0_executions_for_failed_request": retries_stage0,
                "train_completed": int(phase.get("completed", 0)),
                "train_failed": int(phase.get("failed", 0)),
                "train_wall_ms": float(phase.get("wall_ms", 0.0)),
                "train_throughput_per_s": float(phase.get("throughput_per_s", 0.0)),
                "eval_accuracy": float(run.summary.get("choice_accuracy", 0.0)),
                "eval_loss": float(run.summary.get("avg_loss", 0.0)),
            }
        )
    return rows


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = float(np.mean(values))
    return mean, float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    numeric_keys = [
        "stage_recovery_ms",
        "request_completion_recovery_ms",
        "checkpoint_restore_bytes",
        "checkpoint_restore_ms",
        "checkpoint_captures",
        "checkpoint_captured_bytes_total",
        "checkpoint_interval",
        "catchup_updates",
        "catchup_input_bytes",
        "stage0_executions_for_failed_request",
        "train_completed",
        "train_failed",
        "train_wall_ms",
        "train_throughput_per_s",
        "eval_accuracy",
        "eval_loss",
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["policy"])].append(row)
    output: list[dict[str, Any]] = []
    for policy, group in sorted(grouped.items()):
        result: dict[str, Any] = {"policy": policy, "runs": len(group)}
        for key in numeric_keys:
            mean, std = mean_std([float(row[key]) for row in group])
            result[f"{key}_mean"] = mean
            result[f"{key}_std"] = std
        output.append(result)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def display_policy(policy: str) -> str:
    return {
        "wait_for_rejoin": "wait for rejoin",
        "migrate_from_boundary": "boundary handoff (K=1)",
        "handoff_every_step": "boundary handoff (K=1)",
        "handoff_journal_k8": "journal handoff (K=8)",
    }.get(policy, policy.replace("_", " "))


def policy_color(policy: str) -> str:
    if policy == "wait_for_rejoin":
        return "#6B7280"
    if policy in {"migrate_from_boundary", "handoff_every_step"}:
        return "#E16B3D"
    if policy == "handoff_journal_k8":
        return "#2563EB"
    return "#7C3AED"


def short_policy_label(policy: str) -> str:
    return {
        "wait_for_rejoin": "Wait\n2 s",
        "migrate_from_boundary": "Handoff\nK=1",
        "handoff_every_step": "Handoff\nK=1",
        "handoff_journal_k8": "Journal\nK=8",
    }.get(policy, policy.replace("_", "\n"))


def plot_recovery_metrics(rows: list[dict[str, Any]], path: Path) -> None:
    configure_style()
    policies = [row["policy"] for row in rows]
    labels = [short_policy_label(policy) for policy in policies]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))
    metrics = [
        ("stage_recovery_ms", "Stage-1 recovery", "Milliseconds from failure to stage-1 success"),
        ("request_completion_recovery_ms", "End-to-end request recovery", "Milliseconds from failure to final stage"),
    ]
    for axis, (key, title, ylabel) in zip(axes, metrics):
        values = [row[f"{key}_mean"] for row in rows]
        errors = [row[f"{key}_std"] for row in rows]
        colors = [policy_color(str(row["policy"])) for row in rows]
        bars = axis.bar(x, values, yerr=errors, capsize=3, color=colors)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.0f} ms", ha="center", va="bottom", fontsize=9)
        axis.set_xticks(x, labels)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
    savefig(path)


def plot_steady_state_tradeoff(rows: list[dict[str, Any]], path: Path) -> None:
    configure_style()
    labels = [short_policy_label(str(row["policy"])) for row in rows]
    colors = [policy_color(str(row["policy"])) for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.1))
    panels = [
        ("train_throughput_per_s", "Train throughput", "Requests / second"),
        ("eval_accuracy", "Post-training accuracy", "Accuracy"),
        ("eval_loss", "Post-training loss", "Loss"),
    ]
    for axis, (key, title, ylabel) in zip(axes, panels):
        values = [row[f"{key}_mean"] for row in rows]
        errors = [row[f"{key}_std"] for row in rows]
        bars = axis.bar(x, values, yerr=errors, capsize=3, color=colors)
        for index, (bar, value) in enumerate(zip(bars, values)):
            if key == "train_throughput_per_s":
                checkpoint_mib = rows[index]["checkpoint_captured_bytes_total_mean"] / (1024.0 * 1024.0)
                catchups = rows[index]["catchup_updates_mean"]
                label = (
                    f"{value:.2f}/s\n{checkpoint_mib:.0f} MiB checkpoint traffic"
                    f"\n{catchups:.0f} catch-up updates"
                )
            else:
                label = f"{value:.4f}"
            axis.text(bar.get_x() + bar.get_width() / 2, value, label, ha="center", va="bottom", fontsize=7.5)
        axis.set_xticks(x, labels)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
    savefig(path)


def plot_cumulative_completions(runs: list[RunData], path: Path) -> None:
    configure_style()
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    for policy in sorted({run.policy for run in runs}):
        run = min((item for item in runs if item.policy == policy), key=lambda item: item.seed)
        failure, _, _ = event_rows(run)
        failure_end = numeric(failure, "worker_end_epoch_ms")
        final_stage = max(integer(row, "stage_id") for row in train_metrics(run))
        completion_times = sorted(
            numeric(row, "worker_end_epoch_ms")
            for row in train_metrics(run)
            if integer(row, "stage_id") == final_stage and not is_failure(row)
        )
        relative_s = [(time_ms - failure_end) / 1000.0 for time_ms in completion_times if time_ms >= failure_end]
        completed_after = list(range(1, len(relative_s) + 1))
        relative_s = [0.0, *relative_s]
        completed_after = [0, *completed_after]
        ax.step(
            relative_s,
            completed_after,
            where="post",
            label=f"{display_policy(policy)} (seed {run.seed})",
            color=policy_color(policy),
        )
    ax.axvline(0, color="#DC2626", linestyle="--", linewidth=1.2, label="stage-1 failure")
    ax.set_xlim(0.0, 6.0)
    visible_counts = []
    for line in ax.lines:
        xdata = np.asarray(line.get_xdata(), dtype=float)
        ydata = np.asarray(line.get_ydata(), dtype=float)
        visible_counts.extend(ydata[(xdata >= 0.0) & (xdata <= 6.0)])
    if visible_counts:
        ax.set_ylim(0.0, max(5.0, float(max(visible_counts)) + 4.0))
    ax.set_xlabel("Seconds relative to stage-1 failure")
    ax.set_ylabel("End-to-end requests completed after failure")
    ax.set_title("Actual completion recovery after the same stage-1 failure")
    ax.legend(frameon=False, loc="upper left")
    savefig(path)


def worker_labels(run: RunData) -> list[tuple[str, str]]:
    specs = sorted(run.summary.get("workers", []), key=lambda item: (item["stage_id"], item["worker_id"]))
    return [(str(item["worker_id"]), f"{item['worker_id']} / stage {item['stage_id']} / {item['device']}") for item in specs]


def plot_timeline_windows(runs: list[RunData], path: Path) -> None:
    configure_style()
    policies = sorted({run.policy for run in runs})
    fig, axes = plt.subplots(len(policies), 1, figsize=(12.2, 3.6 * len(policies)), sharex=True)
    if len(policies) == 1:
        axes = [axes]
    colors = {"normal": "#2563EB", "recovery": "#E16B3D", "failure": "#DC2626"}
    for axis, policy in zip(axes, policies):
        run = min((item for item in runs if item.policy == policy), key=lambda item: item.seed)
        failure, retry, _ = event_rows(run)
        failure_end = numeric(failure, "worker_end_epoch_ms")
        window_start = failure_end - 450.0
        window_end = failure_end + 3500.0
        labels = worker_labels(run)
        y_pos = {worker_id: index for index, (worker_id, _) in enumerate(labels)}
        for row in train_metrics(run):
            worker_id = str(row.get("worker_id", ""))
            if worker_id not in y_pos:
                continue
            start = numeric(row, "worker_start_epoch_ms")
            end = numeric(row, "worker_end_epoch_ms")
            if not np.isfinite(start) or not np.isfinite(end) or end < window_start or start > window_end:
                continue
            left = max(start, window_start) - failure_end
            width = max(1.0, min(end, window_end) - max(start, window_start))
            if is_failure(row):
                color = colors["failure"]
            elif row.get("mode") == "catchup":
                color = "#7C3AED"
            elif integer(row, "attempt") > 0:
                color = colors["recovery"]
            else:
                color = colors["normal"]
            axis.broken_barh([(left / 1000.0, width / 1000.0)], (y_pos[worker_id] - 0.32, 0.64), facecolors=color, edgecolors="#111827", linewidth=0.3)
        primary_worker = str(failure.get("worker_id"))
        if primary_worker in y_pos:
            unavailable_end = numeric(retry, "worker_start_epoch_ms") if policy == "wait_for_rejoin" else window_end
            axis.broken_barh(
                [(0.0, max(0.0, unavailable_end - failure_end) / 1000.0)],
                (y_pos[primary_worker] - 0.40, 0.80),
                facecolors="#FEE2E2",
                edgecolors="#DC2626",
                linewidth=0.5,
                hatch="//",
                alpha=0.8,
            )
        if policy != "wait_for_rejoin":
            restored = checkpoint_restore_row(run)
            restore_ms = numeric(restored, "checkpoint_restore_ms", 0.0) if restored is not None else 0.0
            restore_start = numeric(restored, "worker_start_epoch_ms") if restored is not None else 0.0
            restore_worker = str(restored.get("worker_id")) if restored is not None else ""
            if restore_worker in y_pos and restore_ms > 0:
                axis.broken_barh(
                    [((restore_start - failure_end) / 1000.0, restore_ms / 1000.0)],
                    (y_pos[restore_worker] - 0.40, 0.80),
                    facecolors="#F59E0B",
                    edgecolors="#92400E",
                    linewidth=0.4,
                    alpha=0.95,
                )
        axis.axvline(0, color="#991B1B", linestyle="--", linewidth=1.0)
        axis.set_yticks(list(y_pos.values()), [label for _, label in labels])
        axis.set_ylabel("GPU worker")
        axis.set_title(f"{display_policy(policy)}: measured schedule around the failure (seed {run.seed})")
        axis.set_xlim((window_start - failure_end) / 1000.0, (window_end - failure_end) / 1000.0)
        axis.grid(axis="y", visible=False)
    axes[-1].set_xlabel("Seconds relative to stage-1 failure")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors["normal"], label="normal stage task"),
        plt.Rectangle((0, 0), 1, 1, color=colors["recovery"], label="recovered boundary task"),
        plt.Rectangle((0, 0), 1, 1, color="#7C3AED", label="journal catch-up update"),
        plt.Rectangle((0, 0), 1, 1, color="#F59E0B", label="checkpoint restore"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#FEE2E2", edgecolor="#DC2626", hatch="//", label="primary unavailable"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    savefig(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--policies", default="wait_for_rejoin,migrate_from_boundary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policies = [item.strip() for item in args.policies.split(",") if item.strip()]
    runs = find_runs(args.root, policies)
    output_dir = args.output_dir or args.root / "report"
    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed = per_seed_rows(runs)
    summary = aggregate_rows(per_seed)
    write_csv(output_dir / "recovery_per_seed.csv", per_seed)
    write_csv(output_dir / "recovery_summary.csv", summary)
    figures = output_dir / "figures"
    plot_timeline_windows(runs, figures / "failover_gpu_timeline.png")
    plot_cumulative_completions(runs, figures / "cumulative_completion_recovery.png")
    plot_recovery_metrics(summary, figures / "recovery_time_breakdown.png")
    plot_steady_state_tradeoff(summary, figures / "steady_state_cost_and_quality.png")
    failure_points = sorted({str(run.summary.get("failure_point", "")) for run in runs if run.summary.get("failure_point")})
    report = [
        "# Real-GPU Failover Recovery Report",
        "",
        "- Both policies retry the same failed stage task from the cached upstream boundary.",
        "- `wait_for_rejoin` holds the pipeline until the original stage worker returns.",
        "- `migrate_from_boundary` restores LoRA parameters plus optimizer state on a warm replica and promotes it.",
        (
            "- In `after_update` runs, `wait_for_rejoin` replays the failed stage in forward-only mode after rejoin, "
            "so the lost downstream boundary is regenerated without a second optimizer step."
            if "after_update" in failure_points
            else "- In `before_execute` runs, the failed stage has not applied a local optimizer step yet."
        ),
        "- Timings are derived from worker start/end timestamps; completion counts come from final-stage task completions.",
        "",
        "## Files",
        "",
        "- `recovery_per_seed.csv` and `recovery_summary.csv`",
        "- `figures/failover_gpu_timeline.png`",
        "- `figures/cumulative_completion_recovery.png`",
        "- `figures/recovery_time_breakdown.png`",
        "- `figures/steady_state_cost_and_quality.png`",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote {output_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
