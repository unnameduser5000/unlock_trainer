#!/usr/bin/env python3
"""Report graceful-degradation and exact-BP recovery-control experiments.

The four input directories must contain the following summaries:

* BP-free, fault-free: scheduler_summary.json
* BP-free, stage offline + skip: scheduler_summary.json
* 1F1B, fault-free: summary.json
* 1F1B, exact-BP control: summary.json

This tool intentionally separates retained local training work from end-to-end
completion. A retained BP-free update is not reported as a recovered request,
and an exact-BP checkpoint-restart control is not reported as stage-local
retention.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def bpfree_stage_rows(summary: dict[str, Any], *, case: str) -> list[dict[str, Any]]:
    progress = summary.get("retained_progress", {})
    stages = progress.get("per_stage", {})
    rows: list[dict[str, Any]] = []
    for stage_raw, values in sorted(stages.items(), key=lambda item: int(item[0])):
        rows.append(
            {
                "method": "BP-free",
                "case": case,
                "stage_id": int(stage_raw),
                "committed_training_records": int(values.get("update_events", 0)),
                "retained_on_end_to_end_failed_requests": int(
                    values.get("retained_updates_on_failed_requests", 0)
                ),
                "end_to_end_completed": int(progress.get("completed_requests", 0)),
                "end_to_end_failed": int(progress.get("failed_requests", 0)),
            }
        )
    return rows


def f1b_phase(summary: dict[str, Any], name: str) -> dict[str, Any]:
    phases = {str(item.get("phase")): item for item in summary.get("phases", [])}
    if name not in phases:
        raise ValueError(f"1F1B summary has no {name!r} phase")
    return phases[name]


def f1b_stage_rows(summary: dict[str, Any], *, case: str) -> list[dict[str, Any]]:
    train = f1b_phase(summary, "train")
    committed = int(train.get("completed_records", train.get("rows", 0)))
    skipped = int(train.get("skipped_records", 0))
    num_chunks = int(summary.get("num_chunks", 0))
    recovery = summary.get("recovery_baseline", {})
    replayed_records = 0
    if case == "recovery_restart":
        rank_summaries = summary.get("recovery_rank_summaries", [])
        if rank_summaries:
            replayed_records = int(rank_summaries[0].get("replayed_records", 0))
    return [
        {
            "method": "1F1B",
            "case": case,
            "stage_id": stage_id,
            "committed_training_records": committed,
            "retained_on_end_to_end_failed_requests": 0,
            "end_to_end_completed": committed,
            "end_to_end_failed": skipped,
            "replayed_after_recovery": replayed_records,
            "recovery_policy": recovery.get("policy", "strict_skip"),
        }
        for stage_id in range(num_chunks)
    ]


def eval_metrics(summary: dict[str, Any], *, method: str, case: str) -> dict[str, Any]:
    if method == "BP-free":
        phases = {str(item.get("phase")): item for item in summary.get("phase_summaries", [])}
        eval_phase = phases.get("eval")
    else:
        eval_phase = f1b_phase(summary, "eval_after")
    if eval_phase is None:
        raise ValueError(f"{method} run {case} has no post-training eval phase")
    return {
        "method": method,
        "case": case,
        "eval_accuracy": float(eval_phase.get("choice_accuracy", 0.0)),
        "eval_loss": float(eval_phase.get("avg_loss", 0.0)),
        "eval_records": int(eval_phase.get("records", eval_phase.get("rows", 0))),
    }


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


def plot_committed(rows: list[dict[str, Any]], path: Path) -> None:
    configure_style()
    stages = sorted({int(row["stage_id"]) for row in rows})
    series = [("BP-free", "fault_free"), ("BP-free", "offline_skip"), ("1F1B", "fault_free"), ("1F1B", "offline_skip")]
    labels = ["BP-free\nnormal", "BP-free\noffline, skip", "1F1B\nnormal", "1F1B\nstrict skip"]
    lookup = {(row["method"], row["case"], row["stage_id"]): row for row in rows}
    x = np.arange(len(stages))
    width = 0.19
    colors = ["#2f6fdd", "#e16b3d", "#5b9a7b", "#9a6fb0"]
    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    for index, ((method, case), label, color) in enumerate(zip(series, labels, colors)):
        values = [lookup[(method, case, stage)]["committed_training_records"] for stage in stages]
        ax.bar(x + (index - 1.5) * width, values, width, label=label, color=color)
    ax.set_xticks(x, [f"stage {stage}" for stage in stages])
    ax.set_ylabel("Committed local training records")
    ax.set_title("Training progress retained during a stage-1 outage")
    ax.legend(ncol=2, frameon=False)
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
    for index, ((method, case), color, label) in enumerate(
        zip(
            methods,
            ["#e16b3d", "#5b9a7b", "#2f6fdd"],
            ["BP-free retained local progress", "1F1B strict skip", "1F1B checkpoint-restart"],
        )
    ):
        key = "retained_on_end_to_end_failed_requests" if method == "BP-free" else "replayed_after_recovery"
        values = [lookup[(method, case, stage)][key] for stage in stages]
        ax.bar(x + (index - 1.0) * width, values, width, label=label, color=color)
    ax.set_xticks(x, [f"stage {stage}" for stage in stages])
    ax.set_ylabel("Records retained or replayed after the outage")
    ax.set_title("BP-free retained progress vs exact-BP recovery control")
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
    for index, (case, color, label) in enumerate(
        [("fault_free", "#2f6fdd", "fault-free"), ("offline_skip", "#e16b3d", "offline, no recovery")]
    ):
        axes[0].bar(
            x + (index - 0.5) * width,
            [lookup[(method, case)]["eval_accuracy"] for method in methods],
            width,
            label=label,
            color=color,
        )
        axes[1].bar(
            x + (index - 0.5) * width,
            [lookup[(method, case)]["eval_loss"] for method in methods],
            width,
            label=label,
            color=color,
        )
    for axis, title, ylabel in zip(axes, ["Post-training accuracy", "Post-training loss"], ["Accuracy", "Loss"]):
        axis.set_xticks(x, methods)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
    axes[0].legend(frameon=False)
    savefig(path)


def write_report(
    *,
    output_dir: Path,
    stage_rows: list[dict[str, Any]],
    quality_rows: list[dict[str, Any]],
    offline_window: dict[str, Any],
) -> None:
    bp_rows = [row for row in stage_rows if row["method"] == "BP-free" and row["case"] == "offline_skip"]
    retained_total = sum(int(row["retained_on_end_to_end_failed_requests"]) for row in bp_rows)
    failed = bp_rows[0]["end_to_end_failed"] if bp_rows else 0
    lines = [
        "# Strict No-Recovery Graceful-Degradation Report",
        "",
        "## Setup",
        "",
        f"- Offline window: stage {offline_window.get('stage_id')} rejects train requests in "
        f"`[{offline_window.get('start_seq')}, {offline_window.get('end_seq')})`.",
        "- BP-free uses `recovery_policy=skip`: no retry, no replay, and no substitute worker.",
        "- 1F1B strict skip uses batch drop before pipeline admission whenever a batch overlaps the same outage window.",
        "- The recovery-aware exact-BP control drops the interrupted batch, restores the latest committed batch-boundary checkpoint, and replays later committed batches before continuing.",
        "- A retained BP-free update is not an end-to-end completed request.",
        "",
        "## Result",
        "",
        f"- BP-free retained {retained_total} local update events on {failed} requests that still failed end-to-end.",
        "- Strict 1F1B retains zero committed training records from its dropped batches by construction.",
        "- The exact-BP recovery control can recover committed progress only at batch boundaries; it does not retain stage-local partial work.",
        "- The quality plot is descriptive. Compare each method to its own fault-free run, not BP-free directly to 1F1B as an optimization-equivalence claim.",
        "",
        "## Figures",
        "",
        "- `figures/committed_training_records_by_stage.png`",
        "- `figures/retained_partial_progress_by_stage.png`",
        "- `figures/post_training_quality.png`",
        "",
        "## Raw Tables",
        "",
        "- `stage_progress.csv`",
        "- `quality.csv`",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate graceful-degradation comparison figures.")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--bpfree_fault_free", type=Path, required=True)
    parser.add_argument("--bpfree_offline_skip", type=Path, required=True)
    parser.add_argument("--f1b_fault_free", type=Path, required=True)
    parser.add_argument("--f1b_offline_skip", type=Path, required=True)
    parser.add_argument("--f1b_recovery_restart", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bpfree_fault_free = load_json(args.bpfree_fault_free / "scheduler_summary.json")
    bpfree_offline = load_json(args.bpfree_offline_skip / "scheduler_summary.json")
    f1b_fault_free = load_json(args.f1b_fault_free / "summary.json")
    f1b_offline = load_json(args.f1b_offline_skip / "summary.json")
    f1b_recovery = (
        load_json(args.f1b_recovery_restart / "summary.json")
        if args.f1b_recovery_restart is not None
        else None
    )

    stage_rows = (
        bpfree_stage_rows(bpfree_fault_free, case="fault_free")
        + bpfree_stage_rows(bpfree_offline, case="offline_skip")
        + f1b_stage_rows(f1b_fault_free, case="fault_free")
        + f1b_stage_rows(f1b_offline, case="offline_skip")
        + (f1b_stage_rows(f1b_recovery, case="recovery_restart") if f1b_recovery is not None else [])
    )
    quality_rows = [
        eval_metrics(bpfree_fault_free, method="BP-free", case="fault_free"),
        eval_metrics(bpfree_offline, method="BP-free", case="offline_skip"),
        eval_metrics(f1b_fault_free, method="1F1B", case="fault_free"),
        eval_metrics(f1b_offline, method="1F1B", case="offline_skip"),
    ]
    if f1b_recovery is not None:
        quality_rows.append(eval_metrics(f1b_recovery, method="1F1B", case="recovery_restart"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "stage_progress.csv", stage_rows)
    write_csv(args.output_dir / "quality.csv", quality_rows)
    figures = args.output_dir / "figures"
    plot_committed(stage_rows, figures / "committed_training_records_by_stage.png")
    plot_retained(stage_rows, figures / "retained_partial_progress_by_stage.png")
    plot_quality(quality_rows, figures / "post_training_quality.png")
    write_report(
        output_dir=args.output_dir,
        stage_rows=stage_rows,
        quality_rows=quality_rows,
        offline_window=bpfree_offline.get("offline_window", {}),
    )
    print(f"Wrote {args.output_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
