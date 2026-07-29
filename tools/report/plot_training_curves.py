#!/usr/bin/env python3
"""Plot comparable BP-free and 1F1B training curves.

The BP-free scheduler emits one result row per request, while the 1F1B runner
emits one train row per pipeline batch.  This script normalizes the x-axis to
"training records seen" so the two curves are readable on the same figure.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


BYTES_PER_MIB = 1024 * 1024


@dataclass
class Curve:
    label: str
    family: str
    x: list[float]
    y: list[float]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def to_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def rolling(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    out: list[float] = []
    running = 0.0
    queue: list[float] = []
    for value in values:
        queue.append(value)
        running += value
        if len(queue) > window:
            running -= queue.pop(0)
        out.append(running / len(queue))
    return out


def parse_run_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"Expected LABEL=PATH, got {spec!r}")
    label, raw_path = spec.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError(f"Empty label in {spec!r}")
    return label, Path(raw_path)


def load_bpfree_curve(label: str, run_dir: Path) -> Curve:
    rows = read_csv(run_dir / "scheduler_results.csv")
    x: list[float] = []
    y: list[float] = []
    for row in rows:
        if row.get("phase", "train") != "train":
            continue
        loss = row.get("loss")
        if loss in (None, ""):
            continue
        seq = to_float(row.get("seq"))
        x.append(seq + 1)
        y.append(to_float(loss))
    return Curve(label=label, family="BP-free", x=x, y=y)


def load_1f1b_curve(label: str, run_dir: Path) -> Curve:
    rows = read_csv(run_dir / "train_batches.csv")
    if not rows:
        rows = [
            row
            for row in read_csv(run_dir / "stage_metrics.csv")
            if row.get("phase") == "train" and row.get("avg_loss") not in (None, "")
        ]
    x: list[float] = []
    y: list[float] = []
    seen = 0.0
    for row in rows:
        records = to_float(row.get("records"), 1.0)
        seen += records
        loss = row.get("avg_loss")
        if loss in (None, ""):
            continue
        x.append(seen)
        y.append(to_float(loss))
    return Curve(label=label, family="1F1B", x=x, y=y)


def load_bpfree_stage_curves(label: str, run_dir: Path) -> list[Curve]:
    rows = read_csv(run_dir / "scheduler_stage_metrics.csv")
    by_stage: dict[str, tuple[list[float], list[float]]] = {}
    for row in rows:
        if row.get("phase", "train") != "train":
            continue
        if row.get("local_loss") in (None, ""):
            continue
        stage = row.get("stage_id", "?")
        x, y = by_stage.setdefault(stage, ([], []))
        x.append(to_float(row.get("seq")) + 1)
        y.append(to_float(row.get("local_loss")))
    curves = []
    for stage, (x, y) in sorted(by_stage.items(), key=lambda item: item[0]):
        curves.append(Curve(label=f"{label} stage {stage}", family="BP-free local", x=x, y=y))
    return curves


def summarize_bpfree(label: str, run_dir: Path, default_learning_rate: str) -> dict[str, Any]:
    summary = read_json(run_dir / "scheduler_summary.json")
    rows = read_csv(run_dir / "scheduler_stage_metrics.csv")
    train_rows = [row for row in rows if row.get("phase", "train") == "train"]
    by_stage: dict[str, list[dict[str, str]]] = {}
    for row in train_rows:
        by_stage.setdefault(row.get("stage_id", "?"), []).append(row)
    stage_peaks = {}
    for stage, stage_rows in by_stage.items():
        peak = max((to_float(row.get("cuda_peak_memory_allocated")) for row in stage_rows), default=0.0)
        stage_peaks[f"stage{stage}_cuda_peak_mib"] = peak / BYTES_PER_MIB
    first = train_rows[0] if train_rows else {}
    eval_choice_count = _phase_value_any(summary, ["eval", "eval_after"], "choice_count", summary.get("choice_count", ""))
    eval_choice_accuracy = (
        ""
        if to_float(eval_choice_count) <= 0
        else _phase_value_any(summary, ["eval", "eval_after"], "choice_accuracy", summary.get("choice_accuracy", ""))
    )
    grad_accum = int(to_float(summary.get("gradient_accumulation_steps"), 1.0))
    batch_description = "1 request/stage task"
    if grad_accum > 1:
        batch_description = f"1 request task; optimizer batch {grad_accum}"
    return {
        "label": label,
        "family": "BP-free",
        "path": str(run_dir),
        "trainable_mode": summary.get("trainable_mode", first.get("trainable_mode", "")),
        "train_records": _phase_value(summary, "train", "records", summary.get("records", "")),
        "eval_records": _phase_value_any(summary, ["eval", "eval_after"], "records", ""),
        "max_inflight": summary.get("max_inflight", ""),
        "batch_size": batch_description,
        "microbatches": "",
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": _learning_rate(summary, default_learning_rate),
        "train_throughput_per_s": _phase_value(summary, "train", "throughput_per_s", summary.get("throughput_per_s", "")),
        "train_avg_loss": _phase_value(summary, "train", "avg_loss", ""),
        "eval_choice_accuracy": eval_choice_accuracy,
        "eval_avg_loss": _phase_value_any(summary, ["eval", "eval_after"], "avg_loss", summary.get("avg_loss", "")),
        **stage_peaks,
    }


def summarize_1f1b(label: str, run_dir: Path, default_learning_rate: str) -> dict[str, Any]:
    summary = read_json(run_dir / "summary.json")
    rows = read_csv(run_dir / "stage_metrics.csv")
    train_rows = [row for row in rows if row.get("phase", "train") == "train"]
    by_stage: dict[str, list[dict[str, str]]] = {}
    for row in train_rows:
        by_stage.setdefault(row.get("stage_id", "?"), []).append(row)
    stage_peaks = {}
    for stage, stage_rows in by_stage.items():
        peak = max((to_float(row.get("cuda_peak_memory_allocated")) for row in stage_rows), default=0.0)
        stage_peaks[f"stage{stage}_cuda_peak_mib"] = peak / BYTES_PER_MIB
    eval_choice_count = _phase_value_any(
        summary,
        ["eval", "eval_after"],
        "choice_count",
        summary.get("choice_count", ""),
    )
    eval_choice_accuracy = (
        ""
        if to_float(eval_choice_count) <= 0
        else _phase_value_any(summary, ["eval", "eval_after"], "choice_accuracy", summary.get("choice_accuracy", ""))
    )
    return {
        "label": label,
        "family": "1F1B",
        "path": str(run_dir),
        "trainable_mode": summary.get("trainable_mode", ""),
        "train_records": summary.get("train_records", ""),
        "eval_records": summary.get("eval_records", ""),
        "max_inflight": "",
        "batch_size": summary.get("batch_size", ""),
        "microbatches": summary.get("microbatches", ""),
        "gradient_accumulation_steps": "",
        "learning_rate": _learning_rate(summary, default_learning_rate),
        "train_throughput_per_s": _phase_value(summary, "train", "throughput_per_s", summary.get("throughput_per_s", "")),
        "train_avg_loss": _phase_value(summary, "train", "avg_loss", summary.get("avg_loss", "")),
        "eval_choice_accuracy": eval_choice_accuracy,
        "eval_avg_loss": _phase_value_any(summary, ["eval", "eval_after"], "avg_loss", summary.get("avg_loss", "")),
        **stage_peaks,
    }


def _phase_value(summary: dict[str, Any], phase: str, key: str, default: Any = "") -> Any:
    for item in summary.get("phase_summaries", []) + summary.get("phases", []):
        if item.get("phase") == phase:
            return item.get(key, default)
    return default


def _phase_value_any(summary: dict[str, Any], phases: list[str], key: str, default: Any = "") -> Any:
    for phase in phases:
        value = _phase_value(summary, phase, key, None)
        if value is not None:
            return value
    return default


def _learning_rate(summary: dict[str, Any], default_learning_rate: str) -> Any:
    value = summary.get("learning_rate", "")
    if value not in (None, ""):
        return value
    return default_learning_rate or "manifest"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_curves(path: Path, title: str, ylabel: str, curves: list[Curve], *, window: int) -> None:
    plt.figure(figsize=(10, 5.8))
    for curve in curves:
        if not curve.x or not curve.y:
            continue
        plt.plot(curve.x, rolling(curve.y, window), label=f"{curve.label} ({curve.family})", linewidth=2)
    plt.title(title)
    plt.xlabel("Training records seen")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_summary(path: Path, summaries: list[dict[str, Any]]) -> None:
    labels = [row["label"] for row in summaries]
    throughput = [to_float(row.get("train_throughput_per_s")) for row in summaries]
    acc = [to_float(row.get("eval_choice_accuracy"), float("nan")) for row in summaries]
    peak = []
    for row in summaries:
        stage_peaks = [to_float(value) for key, value in row.items() if key.endswith("_cuda_peak_mib")]
        peak.append(max(stage_peaks) if stage_peaks else 0.0)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    metrics = [
        ("Train throughput (records/s)", throughput),
        ("Eval accuracy", acc),
        ("Max CUDA peak allocated (MiB)", peak),
    ]
    colors = ["#4078c0", "#2da44e", "#bf8700"]
    for ax, (title, values), color in zip(axes, metrics, colors):
        ax.bar(labels, values, color=color, alpha=0.82)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fmt_metric(value: Any, digits: int = 4) -> str:
    if value in (None, ""):
        return ""
    try:
        text = f"{float(value):.{digits}f}"
        return text.rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return str(value)


def max_cuda_peak_mib(row: dict[str, Any]) -> float:
    stage_peaks = [to_float(value) for key, value in row.items() if key.endswith("_cuda_peak_mib")]
    return max(stage_peaks) if stage_peaks else 0.0


def write_markdown(path: Path, summaries: list[dict[str, Any]], dataset_note: str, curve_window: int) -> None:
    lines = [
        "# Training Curves Report",
        "",
        "## Dataset and Run Settings",
        "",
        dataset_note.strip() or "Dataset note was not provided.",
        "",
        f"Loss curves use a rolling window of {curve_window} records/batches for readability.",
        "",
        "## Runs",
        "",
        "|label|family|trainable|train records|eval records|batch|microbatches|grad accum|max inflight|lr|train records/s|train loss|eval loss|eval acc|max CUDA peak MiB|",
        "|---|---|---|---:|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "|{label}|{family}|{trainable_mode}|{train_records}|{eval_records}|{batch_size}|{microbatches}|"
            "{gradient_accumulation_steps}|{max_inflight}|{learning_rate}|{train_throughput}|"
            "{train_loss}|{eval_loss}|{eval_acc}|{peak}|".format(
                **row,
                train_throughput=fmt_metric(row.get("train_throughput_per_s"), 2),
                train_loss=fmt_metric(row.get("train_avg_loss"), 4),
                eval_loss=fmt_metric(row.get("eval_avg_loss"), 4),
                eval_acc=fmt_metric(row.get("eval_choice_accuracy"), 4),
                peak=fmt_metric(max_cuda_peak_mib(row), 1),
            )
        )
    if any(row.get("eval_choice_accuracy") in (None, "") for row in summaries):
        lines.extend(
            [
                "",
                "Blank eval accuracy means the run did not emit end-to-end choices for that protocol.",
            ]
        )
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "![final loss](training_final_loss_curves.png)",
            "",
            "![BP-free local stage loss](bpfree_stage_local_loss_curves.png)",
            "",
            "![summary](training_summary_bars.png)",
        ]
    )
    memory_dir = path.parent / "memory"
    memory_figures = [
        ("Trainable state and activation stash", "memory/trainable_state_vs_activation.png"),
        ("Saved tensor split", "memory/activation_saved_unique_breakdown.png"),
        ("CUDA peak breakdown", "memory/activation_cuda_peak_breakdown.png"),
        ("BP-free payloads", "memory/activation_bpfree_payloads.png"),
    ]
    existing_memory_figures = [
        (title, rel_path)
        for title, rel_path in memory_figures
        if (memory_dir / Path(rel_path).name).exists()
    ]
    if existing_memory_figures:
        lines.extend(["", "## Memory Figures"])
        for title, rel_path in existing_memory_figures:
            lines.extend(["", f"![{title}]({rel_path})"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bpfree", action="append", default=[], type=parse_run_spec, metavar="LABEL=PATH")
    parser.add_argument("--onef1b", action="append", default=[], type=parse_run_spec, metavar="LABEL=PATH")
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--dataset_note", default="")
    parser.add_argument("--default_learning_rate", default="")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    final_curves: list[Curve] = []
    stage_curves: list[Curve] = []
    summaries: list[dict[str, Any]] = []

    for label, path in args.bpfree:
        final_curves.append(load_bpfree_curve(label, path))
        stage_curves.extend(load_bpfree_stage_curves(label, path))
        summaries.append(summarize_bpfree(label, path, args.default_learning_rate))

    for label, path in args.onef1b:
        final_curves.append(load_1f1b_curve(label, path))
        summaries.append(summarize_1f1b(label, path, args.default_learning_rate))

    plot_curves(
        args.output_dir / "training_final_loss_curves.png",
        "Comparable train loss curves",
        "Final-stage / pipeline batch loss",
        final_curves,
        window=args.window,
    )
    plot_curves(
        args.output_dir / "bpfree_stage_local_loss_curves.png",
        "BP-free local objective loss by stage",
        "Local stage loss",
        stage_curves,
        window=args.window,
    )
    plot_summary(args.output_dir / "training_summary_bars.png", summaries)
    write_csv(args.output_dir / "training_run_summary.csv", summaries)
    write_markdown(args.output_dir / "REPORT.md", summaries, args.dataset_note, args.window)


if __name__ == "__main__":
    main()
