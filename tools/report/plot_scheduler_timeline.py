#!/usr/bin/env python3
"""Plot per-device scheduler timelines and bubble time from stage metrics."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TIMESTAMP_COLS = [
    "queue_enter_epoch_ms",
    "dispatch_epoch_ms",
    "worker_start_epoch_ms",
    "worker_end_epoch_ms",
]


@dataclass(frozen=True)
class RunTimeline:
    policy: str
    seed: int
    run_dir: Path
    metrics: pd.DataFrame


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


def infer_policy_seed(run_dir: Path) -> tuple[str, int]:
    match = re.match(r"(.+)_seed(\d+)$", run_dir.name)
    if not match:
        return run_dir.name, -1
    return match.group(1), int(match.group(2))


def read_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in [
        "seq",
        "stage_id",
        "attempt",
        "scheduler_queue_ms",
        "worker_queue_ms",
        "execute_ms",
        "optimizer_ms",
        "stage_total_ms",
        *TIMESTAMP_COLS,
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["train", "update_applied"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().eq("true")
    return df


def find_runs(root: Path, policies: set[str], seed: int | None) -> list[RunTimeline]:
    runs: list[RunTimeline] = []
    for run_dir in sorted(root.glob("*_seed*")):
        if not run_dir.is_dir():
            continue
        policy, parsed_seed = infer_policy_seed(run_dir)
        if policies and policy not in policies:
            continue
        if seed is not None and parsed_seed != seed:
            continue
        metrics_path = run_dir / "scheduler_stage_metrics.csv"
        if not metrics_path.is_file():
            continue
        metrics = read_metrics(metrics_path)
        if not all(col in metrics.columns for col in TIMESTAMP_COLS):
            continue
        metrics = metrics.dropna(subset=["worker_start_epoch_ms", "worker_end_epoch_ms"])
        metrics = metrics[metrics["worker_end_epoch_ms"] > metrics["worker_start_epoch_ms"]]
        if metrics.empty:
            continue
        runs.append(RunTimeline(policy=policy, seed=parsed_seed, run_dir=run_dir, metrics=metrics))
    return runs


def train_phase(df: pd.DataFrame) -> pd.DataFrame:
    if "phase" not in df.columns:
        return df.copy()
    train = df[df["phase"] == "train"].copy()
    return train if not train.empty else df.copy()


def worker_order(df: pd.DataFrame) -> list[str]:
    rows = (
        df[["worker_id", "stage_id", "device"]]
        .drop_duplicates()
        .sort_values(["stage_id", "worker_id"])
    )
    return rows["worker_id"].astype(str).tolist()


def worker_label(row: pd.Series) -> str:
    return f"{row['worker_id']} / stage {int(row['stage_id'])} / {row['device']}"


def plot_run_gantt(run: RunTimeline, out_dir: Path, *, max_seq: int) -> None:
    df = train_phase(run.metrics)
    if max_seq > 0:
        df = df[df["seq"] < max_seq]
    if df.empty:
        return
    base = float(df["queue_enter_epoch_ms"].min())
    order = worker_order(df)
    labels = {}
    for _, row in df[["worker_id", "stage_id", "device"]].drop_duplicates().iterrows():
        labels[str(row["worker_id"])] = worker_label(row)
    y_pos = {worker_id: idx for idx, worker_id in enumerate(order)}
    colors = {
        "update": "#2563EB",
        "forward": "#9CA3AF",
        "queue": "#F97316",
        "bubble": "#FEE2E2",
    }

    fig, ax = plt.subplots(figsize=(13.5, max(4.8, 1.15 * len(order) + 1.6)))
    for worker_id in order:
        worker_rows = df[df["worker_id"].astype(str) == worker_id].sort_values("worker_start_epoch_ms")
        y = y_pos[worker_id]
        previous_end = None
        for _, row in worker_rows.iterrows():
            queue_start = (float(row["queue_enter_epoch_ms"]) - base) / 1000.0
            dispatch = (float(row["dispatch_epoch_ms"]) - base) / 1000.0
            start = (float(row["worker_start_epoch_ms"]) - base) / 1000.0
            end = (float(row["worker_end_epoch_ms"]) - base) / 1000.0
            if previous_end is not None and start > previous_end:
                ax.broken_barh(
                    [(previous_end, start - previous_end)],
                    (y - 0.32, 0.18),
                    facecolors=colors["bubble"],
                    edgecolors="none",
                    alpha=0.9,
                )
            if dispatch > queue_start:
                ax.broken_barh(
                    [(queue_start, dispatch - queue_start)],
                    (y - 0.08, 0.16),
                    facecolors=colors["queue"],
                    edgecolors="none",
                    alpha=0.35,
                )
            color = colors["update"] if bool(row.get("update_applied", False)) else colors["forward"]
            ax.broken_barh(
                [(start, max(0.001, end - start))],
                (y - 0.30, 0.60),
                facecolors=color,
                edgecolors="#111827",
                linewidth=0.35,
                alpha=0.86,
            )
            if max_seq <= 32 and (end - start) > 0.018:
                ax.text(
                    start + (end - start) / 2,
                    y,
                    str(int(row["seq"])),
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color="#111827",
                )
            previous_end = max(previous_end or end, end)

    ax.set_yticks([y_pos[worker_id] for worker_id in order])
    ax.set_yticklabels([labels.get(worker_id, worker_id) for worker_id in order])
    ax.set_xlabel("Seconds from first queued task")
    title_suffix = f"first {max_seq} train requests" if max_seq > 0 else "full train phase"
    ax.set_title(f"{run.policy} seed {run.seed}: per-device schedule ({title_suffix})")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors["update"], label="local update"),
        plt.Rectangle((0, 0), 1, 1, color=colors["forward"], label="forward-only"),
        plt.Rectangle((0, 0), 1, 1, color=colors["queue"], alpha=0.35, label="scheduler queue wait"),
        plt.Rectangle((0, 0), 1, 1, color=colors["bubble"], label="device bubble"),
    ]
    ax.legend(handles=handles, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    savefig(out_dir / f"schedule_gantt_{run.policy}_seed{run.seed}.png")


def bubble_summary(run: RunTimeline) -> pd.DataFrame:
    df = train_phase(run.metrics).copy()
    rows: list[dict[str, object]] = []
    for worker_id, worker_rows in df.groupby("worker_id"):
        worker_rows = worker_rows.sort_values("worker_start_epoch_ms")
        start = float(worker_rows["worker_start_epoch_ms"].min())
        end = float(worker_rows["worker_end_epoch_ms"].max())
        busy_ms = float((worker_rows["worker_end_epoch_ms"] - worker_rows["worker_start_epoch_ms"]).sum())
        span_ms = max(0.0, end - start)
        bubble_ms = max(0.0, span_ms - busy_ms)
        first = worker_rows.iloc[0]
        rows.append(
            {
                "policy": run.policy,
                "seed": run.seed,
                "worker_id": worker_id,
                "stage_id": int(first["stage_id"]),
                "device": first["device"],
                "tasks": len(worker_rows),
                "updates": int(worker_rows["update_applied"].sum()) if "update_applied" in worker_rows else 0,
                "busy_ms": busy_ms,
                "bubble_ms": bubble_ms,
                "span_ms": span_ms,
                "busy_fraction": busy_ms / span_ms if span_ms > 0 else 0.0,
                "bubble_fraction": bubble_ms / span_ms if span_ms > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def plot_bubble_comparison(summary: pd.DataFrame, out_dir: Path) -> None:
    if summary.empty:
        return
    summary = summary.sort_values(["policy", "stage_id"])
    labels = [f"{row.policy}\nstage {int(row.stage_id)}" for row in summary.itertuples()]
    x = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(max(9.5, 0.72 * len(summary)), 4.8))
    busy_s = summary["busy_ms"].to_numpy(dtype=float) / 1000.0
    bubble_s = summary["bubble_ms"].to_numpy(dtype=float) / 1000.0
    ax.bar(x, busy_s, color="#2563EB", alpha=0.84, label="busy")
    ax.bar(x, bubble_s, bottom=busy_s, color="#FEE2E2", edgecolor="#DC2626", alpha=0.9, label="bubble")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Train phase span (s)")
    ax.set_title("Per-device busy time and bubbles")
    ax.legend(frameon=False)
    savefig(out_dir / "device_busy_bubble_by_policy.png")

    pivot = summary.pivot(index="policy", columns="stage_id", values="bubble_fraction")
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    pivot.plot(kind="bar", ax=ax, width=0.75)
    ax.set_ylabel("Bubble fraction")
    ax.set_title("Bubble fraction by policy and stage")
    ax.legend(title="stage", frameon=False)
    savefig(out_dir / "device_bubble_fraction_by_policy.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Directory containing *_seed*/scheduler_stage_metrics.csv.")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--policies", default="", help="Comma-separated policies to include.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_seq", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    policies = {item.strip() for item in args.policies.split(",") if item.strip()}
    out_dir = args.output_dir or args.root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = find_runs(args.root, policies=policies, seed=args.seed)
    if not runs:
        raise RuntimeError(
            f"No timestamped scheduler_stage_metrics.csv files found under {args.root}."
        )

    summaries: list[pd.DataFrame] = []
    for run in runs:
        plot_run_gantt(run, out_dir, max_seq=args.max_seq)
        summaries.append(bubble_summary(run))
    combined = pd.concat(summaries, ignore_index=True)
    combined_path = args.root / "device_bubble_summary.csv"
    combined.to_csv(combined_path, index=False)
    print(f"Wrote {combined_path}")
    plot_bubble_comparison(combined, out_dir)


if __name__ == "__main__":
    main()
