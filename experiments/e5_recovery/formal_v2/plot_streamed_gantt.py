#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _stream_metrics(summary: dict[str, Any], rank: int, phase: str) -> list[dict[str, Any]]:
    rank_summary = next(item for item in summary["rank_summaries"] if int(item["rank"]) == rank)
    return [item for item in rank_summary["metrics"] if item["phase"] == phase]


def build_timeline(summary: dict[str, Any]) -> list[dict[str, Any]]:
    if summary["protocol"].get("catchup_policy") != "window_streamed":
        raise ValueError("Gantt input must use window_streamed catch-up")
    stage1 = _stream_metrics(summary, 1, "catchup_stage1_streamed")
    stage2 = _stream_metrics(summary, 2, "catchup_stage2_streamed")
    if not stage1 or len(stage1) != len(stage2):
        raise ValueError("streamed per-window metrics are incomplete")

    origin = min(
        int(item["ready_wait_start_monotonic_ns"])
        for item in stage1 + stage2
    )
    rows: list[dict[str, Any]] = []

    def append(stage: int, window: int, action: str, start_ns: int, end_ns: int) -> None:
        rows.append(
            {
                "stage": stage,
                "window_id": window,
                "action": action,
                "start_ms": (start_ns - origin) / 1_000_000.0,
                "end_ms": (end_ns - origin) / 1_000_000.0,
                "duration_ms": (end_ns - start_ns) / 1_000_000.0,
            }
        )

    for item in stage1:
        append(
            1,
            int(item["window_id"]),
            "compute_and_commit",
            int(item["compute_start_monotonic_ns"]),
            int(item["compute_end_monotonic_ns"]),
        )
    for item in stage2:
        wait_start = int(item["ready_wait_start_monotonic_ns"])
        compute_start = int(item["compute_start_monotonic_ns"])
        if compute_start > wait_start:
            append(2, int(item["window_id"]), "wait_for_commit", wait_start, compute_start)
        append(
            2,
            int(item["window_id"]),
            "compute_and_commit",
            compute_start,
            int(item["compute_end_monotonic_ns"]),
        )
    return sorted(rows, key=lambda row: (row["start_ms"], row["stage"], row["window_id"]))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(path: Path, rows: list[dict[str, Any]]) -> None:
    colors = {"wait_for_commit": "#A8ADB4", "compute_and_commit": "#2F6B9A"}
    fig, ax = plt.subplots(figsize=(11, 3.4))
    for row in rows:
        y = 1 if row["stage"] == 1 else 0
        ax.broken_barh(
            [(row["start_ms"], row["duration_ms"])],
            (y - 0.28, 0.56),
            facecolors=colors[row["action"]],
            edgecolors="white",
            linewidth=0.8,
        )
        if row["action"] == "compute_and_commit":
            ax.text(
                row["start_ms"] + row["duration_ms"] / 2,
                y,
                f"W{row['window_id']}",
                ha="center",
                va="center",
                color="white",
                fontsize=8,
            )
    ax.set_yticks([0, 1], labels=["stage 2", "stage 1"])
    ax.set_xlabel("Time since streamed catch-up start (ms)")
    ax.set_title("BP-free window-streamed catch-up")
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors["compute_and_commit"]),
        plt.Rectangle((0, 0), 1, 1, color=colors["wait_for_commit"]),
    ]
    ax.legend(
        handles,
        ["compute + durable commit", "wait for upstream commit"],
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot an E5 streamed catch-up Gantt")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    rows = build_timeline(summary)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "streamed_timeline.json").write_text(
        json.dumps(rows, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(args.output_dir / "streamed_timeline.csv", rows)
    _plot(args.output_dir / "streamed_gantt.png", rows)


if __name__ == "__main__":
    main()
