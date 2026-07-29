#!/usr/bin/env python3
"""Build the canonical E4.4 timeline, stage ledger, and headline table."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


METHODS = ("bpfree", "exactbp_1f1b")
METHOD_LABELS = {"bpfree": "BP-free", "exactbp_1f1b": "Exact BP (1F1B)"}
CATEGORY_ORDER = (
    "input",
    "forward",
    "recv_wait",
    "communication",
    "backward",
    "optimizer",
)
CATEGORY_LABELS = {
    "input": "Input preparation",
    "forward": "Forward + loss",
    "recv_wait": "Receive wait",
    "communication": "Transfer/runtime",
    "backward": "Backward",
    "optimizer": "Optimizer",
    "idle_other": "Idle / other",
}
COLORS = {
    "input": "#8172B2",
    "forward": "#4C78A8",
    "recv_wait": "#E45756",
    "communication": "#54A24B",
    "backward": "#F58518",
    "optimizer": "#B279A2",
    "idle_other": "#D9D9D9",
}


@dataclass(frozen=True)
class Event:
    stage_id: int
    window_id: int
    category: str
    start_ms: float
    end_ms: float

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_int(raw: str | None, default: int = -1) -> int:
    try:
        return int(raw) if raw not in (None, "") else default
    except ValueError:
        return default


def parse_float(raw: str | None, default: float = math.nan) -> float:
    try:
        return float(raw) if raw not in (None, "") else default
    except ValueError:
        return default


def window_id(row: dict[str, str]) -> int:
    for key in ("window_id", "batch_seq", "global_batch_seq", "seq_start"):
        value = parse_int(row.get(key))
        if value >= 0:
            return value
    return -1


def timeline_category(action: str) -> str | None:
    upper = action.upper()
    if upper in {"BEGIN_WINDOW", "PHASE_STEP_WINDOW", "MAINTAIN_RECV_INFLIGHT"}:
        return None
    if "RECV_WAIT" in upper:
        return "recv_wait"
    if "OPTIMIZER" in upper or "ZERO_GRAD" in upper or "GRAD_CLIP" in upper:
        return "optimizer"
    if "BACKWARD" in upper or "_BWD" in upper or upper.startswith("BWD"):
        return "backward"
    if (
        "BODY_FORWARD" in upper
        or "LOCAL_FORWARD" in upper
        or "LOCAL_HEAD_LOSS" in upper
        or "FWD_COMPUTE" in upper
    ):
        return "forward"
    if "INPUT" in upper or "LOAD_STAGE0_HIDDEN" in upper or "LOAD_COMMON_INPUTS" in upper:
        return "input"
    if any(token in upper for token in ("D2H", "RECV_POST", "RECV_H2D", "SEND_POST", "SEND_WAIT")):
        return "communication"
    return None


def load_timeline_events(run_dir: Path, start_window: int, duration_ms: float) -> list[Event]:
    events: list[Event] = []
    for path in sorted(run_dir.glob("train.stage*.actions.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                category = timeline_category(row.get("action", ""))
                current_window = window_id(row)
                start_ms = parse_float(row.get("start_epoch_ms"))
                end_ms = parse_float(row.get("end_epoch_ms"))
                if (
                    category is None
                    or not math.isfinite(start_ms)
                    or not math.isfinite(end_ms)
                    or end_ms <= start_ms
                ):
                    continue
                events.append(
                    Event(
                        stage_id=parse_int(row.get("stage_id")),
                        window_id=current_window,
                        category=category,
                        start_ms=start_ms,
                        end_ms=end_ms,
                    )
                )
    if not events:
        raise ValueError(f"no timeline events under {run_dir}")

    start_candidates = [
        event.start_ms
        for event in events
        if event.window_id == start_window
        and event.stage_id == 0
        and event.category == "input"
    ]
    if not start_candidates:
        raise ValueError(f"cannot locate stage-0 W{start_window} under {run_dir}")
    interval_start = min(start_candidates)
    interval_end = interval_start + duration_ms
    return [
        Event(
            stage_id=event.stage_id,
            window_id=event.window_id,
            category=event.category,
            start_ms=max(interval_start, event.start_ms),
            end_ms=min(interval_end, event.end_ms),
        )
        for event in events
        if event.start_ms < interval_end and event.end_ms > interval_start
    ]


def plot_timeline(
    *,
    raw_root: Path,
    output_dir: Path,
    rep: int,
    start_window: int,
    duration_ms: float,
    dpi: int,
) -> None:
    method_events = {
        method: load_timeline_events(
            raw_root / "local" / "throughput_b8_m4" / method / f"rep_{rep:02d}",
            start_window,
            duration_ms,
        )
        for method in METHODS
    }
    common_xlim = (0.0, duration_ms)
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 6.2), sharex=True)

    legend_handles: dict[str, Any] = {}
    for ax, method in zip(axes, METHODS):
        events = method_events[method]
        origin = min(event.start_ms for event in events)
        for category in CATEGORY_ORDER:
            subset = [event for event in events if event.category == category]
            if not subset:
                continue
            bars = ax.barh(
                [event.stage_id for event in subset],
                [event.duration_ms for event in subset],
                left=[event.start_ms - origin for event in subset],
                height=0.64,
                color=COLORS[category],
                edgecolor="none",
                label=CATEGORY_LABELS[category],
            )
            legend_handles.setdefault(category, bars[0])

        stage_labels = []
        for stage_id in (0, 1, 2):
            local_windows = sorted(
                {
                    event.window_id
                    for event in events
                    if event.stage_id == stage_id
                    and event.category != "communication"
                }
            )
            if local_windows:
                stage_labels.append(
                    f"Stage {stage_id}  (local W{local_windows[0]}-W{local_windows[-1]})"
                )
            else:
                stage_labels.append(f"Stage {stage_id}")
        ax.set_yticks((0, 1, 2), stage_labels)
        ax.invert_yaxis()
        ax.set_xlim(*common_xlim)
        ax.set_title(METHOD_LABELS[method], loc="left", pad=8)
        ax.grid(axis="x", linewidth=0.4, alpha=0.5)

    axes[-1].set_xlabel("Elapsed time within fixed wall-clock slice (ms)")
    fig.suptitle(
        f"E4.4 fixed wall-clock execution slices (b=8, m=4, B=32; "
        f"representative stable rep {rep}; {duration_ms:.0f} ms)"
    )
    fig.legend(
        [legend_handles[item] for item in CATEGORY_ORDER if item in legend_handles],
        [CATEGORY_LABELS[item] for item in CATEGORY_ORDER if item in legend_handles],
        loc="lower center",
        ncol=6,
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.95))
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "e4_4_steady_timeline"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def grouped_stage_rows(stage_csv: Path) -> list[dict[str, Any]]:
    raw_rows: list[dict[str, Any]] = []
    with stage_csv.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            if raw["profile"] != "local" or raw["regime"] != "throughput_b8_m4":
                continue
            span = float(raw["trace_span_ms"])
            windows = int(raw["window_count"])
            communication = sum(
                float(raw[key])
                for key in (
                    "transport_d2h",
                    "transport_recv_post",
                    "transport_recv_h2d",
                    "transport_send_post_runtime",
                    "link_pacing",
                    "transport_send_wait",
                )
            )
            buckets = {
                "input": float(raw["input_h2d"]),
                "forward": float(raw["forward_compute"]),
                "recv_wait": float(raw["transport_recv_wait"]),
                "communication": communication,
                "backward": float(raw["backward_compute"]),
                "optimizer": float(raw["optimizer"]),
                "idle_other": float(raw["control"]) + float(raw["untraced_idle"]),
            }
            total = sum(buckets.values())
            row: dict[str, Any] = {
                "method": raw["method"],
                "stage_id": int(raw["stage_id"]),
                "rep": int(raw["rep"]),
                "trace_windows": windows,
                "trace_span_ms_per_window": span / windows,
                "throughput_records_per_s": float(raw["throughput_per_s"]),
            }
            for category, value in buckets.items():
                row[f"{category}_ms_per_window"] = value / windows
                row[f"{category}_percent"] = 100.0 * value / total if total else 0.0
            raw_rows.append(row)

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        grouped[(row["method"], row["stage_id"])].append(row)

    rows: list[dict[str, Any]] = []
    for (method, stage_id), samples in grouped.items():
        samples.sort(key=lambda row: row["rep"])
        row = {
            "method": method,
            "stage_id": stage_id,
            "repetitions": len(samples),
            "trace_windows_per_rep": samples[0]["trace_windows"],
            "trace_span_ms_per_window": statistics.mean(
                item["trace_span_ms_per_window"] for item in samples
            ),
            "trace_span_ms_per_window_std": statistics.stdev(
                item["trace_span_ms_per_window"] for item in samples
            ) if len(samples) > 1 else 0.0,
            "throughput_records_per_s": statistics.mean(
                item["throughput_records_per_s"] for item in samples
            ),
            "throughput_records_per_s_std": statistics.stdev(
                item["throughput_records_per_s"] for item in samples
            ) if len(samples) > 1 else 0.0,
        }
        for category in (*CATEGORY_ORDER, "idle_other"):
            ms_values = [item[f"{category}_ms_per_window"] for item in samples]
            percent_values = [item[f"{category}_percent"] for item in samples]
            row[f"{category}_ms_per_window"] = statistics.mean(ms_values)
            row[f"{category}_ms_per_window_std"] = (
                statistics.stdev(ms_values) if len(ms_values) > 1 else 0.0
            )
            row[f"{category}_percent"] = statistics.mean(percent_values)
            row[f"{category}_percent_std"] = (
                statistics.stdev(percent_values) if len(percent_values) > 1 else 0.0
            )
        rows.append(row)
    rows.sort(key=lambda row: (METHODS.index(row["method"]), row["stage_id"]))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_stage_breakdown(rows: list[dict[str, Any]], output_dir: Path, dpi: int) -> None:
    labels = [f"{METHOD_LABELS[row['method']]} S{row['stage_id']}" for row in rows]
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    left = [0.0] * len(rows)
    categories = (*CATEGORY_ORDER, "idle_other")
    for category in categories:
        values = [row[f"{category}_percent"] for row in rows]
        bars = ax.barh(
            labels,
            values,
            left=left,
            color=COLORS[category],
            edgecolor="white",
            linewidth=0.4,
            label=CATEGORY_LABELS[category],
        )
        for bar, value in zip(bars, values):
            if value >= 7.0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
        left = [current + value for current, value in zip(left, values)]
    ax.set_xlim(0.0, 100.0)
    ax.invert_yaxis()
    ax.set_xlabel("Share of per-stage traced wall span (%)")
    ax.set_title("E4.4 late-window per-stage time decomposition (b=8, m=4, B=32)")
    ax.grid(axis="x", linewidth=0.4, alpha=0.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False)
    fig.tight_layout()
    stem = output_dir / "e4_4_stage_time_breakdown"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def train_phase(summary: dict[str, Any]) -> dict[str, Any]:
    return next(item for item in summary["phases"] if item["phase"] == "train")


def headline_numbers(raw_root: Path, analysis_dir: Path) -> dict[str, Any]:
    samples: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    for method in METHODS:
        method_root = raw_root / "local" / "throughput_b8_m4" / method
        for rep_dir in sorted(method_root.glob("rep_*")):
            rep = int(rep_dir.name.removeprefix("rep_"))
            phase = train_phase(read_json(rep_dir / "summary.json"))
            samples[method][rep] = {
                "throughput_records_per_s": float(phase["throughput_per_s"]),
                "wall_ms": float(phase["wall_ms"]),
            }
    paired_reps = sorted(set(samples["bpfree"]) & set(samples["exactbp_1f1b"]))
    if not paired_reps:
        raise ValueError("no paired E4.4 repetitions")
    ratios = {
        rep: samples["bpfree"][rep]["throughput_records_per_s"]
        / samples["exactbp_1f1b"][rep]["throughput_records_per_s"]
        for rep in paired_reps
    }
    median_ratio = statistics.median(ratios.values())
    steady = read_json(analysis_dir / "steady_state_report.json")
    stable_runs = {
        (item["method"], int(item["rep"]))
        for item in steady["runs"]
        if item["stable"]
    }
    stable_paired_reps = [
        rep
        for rep in paired_reps
        if ("bpfree", rep) in stable_runs
        and ("exactbp_1f1b", rep) in stable_runs
    ]
    if not stable_paired_reps:
        raise ValueError("no paired repetition passed the steady-state checks")
    stable_median_ratio = statistics.median(ratios[rep] for rep in stable_paired_reps)
    representative_rep = min(
        stable_paired_reps,
        key=lambda rep: (abs(ratios[rep] - stable_median_ratio), rep),
    )
    periods = {
        method: float(steady["method_summary"][method]["median_period_ms_per_window"])
        for method in METHODS
    }

    method_summary = {}
    for method in METHODS:
        throughput_values = [
            samples[method][rep]["throughput_records_per_s"] for rep in paired_reps
        ]
        wall_values = [samples[method][rep]["wall_ms"] for rep in paired_reps]
        method_summary[method] = {
            "repetitions": len(paired_reps),
            "throughput_records_per_s": throughput_values,
            "throughput_mean": statistics.mean(throughput_values),
            "throughput_std": statistics.stdev(throughput_values)
            if len(throughput_values) > 1 else 0.0,
            "wall_ms": wall_values,
            "wall_mean_ms": statistics.mean(wall_values),
            "wall_std_ms": statistics.stdev(wall_values)
            if len(wall_values) > 1 else 0.0,
            "median_steady_window_period_ms": periods[method],
        }
    return {
        "geometry": {
            "physical_request_batch": 8,
            "microbatches_per_update": 4,
            "effective_optimizer_batch": 32,
        },
        "methods": method_summary,
        "paired_throughput_ratios": ratios,
        "median_bpfree_throughput_over_exactbp_1f1b": median_ratio,
        "stable_paired_repetitions": stable_paired_reps,
        "representative_rep": representative_rep,
        "exactbp_period_over_bpfree": periods["exactbp_1f1b"] / periods["bpfree"],
        "scope": "diagnostic repetitions with selective synchronized tracing",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-window", type=int, required=True)
    parser.add_argument("--duration-ms", type=float, default=1600.0)
    parser.add_argument("--dpi", type=int, default=240)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    numbers = headline_numbers(args.raw_root, args.analysis_dir)
    plot_timeline(
        raw_root=args.raw_root,
        output_dir=args.output_dir,
        rep=int(numbers["representative_rep"]),
        start_window=args.start_window,
        duration_ms=args.duration_ms,
        dpi=args.dpi,
    )
    rows = grouped_stage_rows(args.analysis_dir / "stage_breakdown.csv")
    write_csv(args.output_dir / "e4_4_stage_time_breakdown.csv", rows)
    plot_stage_breakdown(rows, args.output_dir, args.dpi)
    (args.output_dir / "e4_4_headline_numbers.json").write_text(
        json.dumps(numbers, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
