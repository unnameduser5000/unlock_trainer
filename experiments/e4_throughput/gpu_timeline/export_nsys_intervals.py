#!/usr/bin/env python3
"""Export compact, auditable GPU activity intervals from E4 Nsight traces."""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path
from typing import Iterable


METHODS = ("bpfree", "exactbp_gpipe", "exactbp_1f1b", "pipedream")
ACTIVITY_TABLES = {
    "compute": ("CUPTI_ACTIVITY_KIND_KERNEL",),
    "transfer": (
        "CUPTI_ACTIVITY_KIND_MEMCPY",
        "CUPTI_ACTIVITY_KIND_MEMSET",
    ),
}


def merge_intervals(
    intervals: Iterable[tuple[int, int]], *, max_gap_ns: int = 0
) -> list[tuple[int, int]]:
    ordered = sorted((int(start), int(end)) for start, end in intervals if end > start)
    merged: list[tuple[int, int]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1] + max_gap_ns:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def clipped(
    intervals: Iterable[tuple[int, int]], start_ns: int, end_ns: int
) -> list[tuple[int, int]]:
    return [
        (max(start_ns, start), min(end_ns, end))
        for start, end in intervals
        if end > start_ns and start < end_ns
    ]


def duration_ns(intervals: Iterable[tuple[int, int]]) -> int:
    return sum(end - start for start, end in merge_intervals(intervals))


def read_activity(
    connection: sqlite3.Connection,
    *,
    tables: Iterable[str],
    device_id: int,
    start_ns: int,
    end_ns: int,
) -> list[tuple[int, int]]:
    rows: list[tuple[int, int]] = []
    for table in tables:
        rows.extend(
            (int(start), int(end))
            for start, end in connection.execute(
                f"SELECT start, end FROM {table} "
                "WHERE deviceId = ? AND end > ? AND start < ?",
                (device_id, start_ns, end_ns),
            )
        )
    return clipped(rows, start_ns, end_ns)


def latest_activity_end(connection: sqlite3.Connection) -> int:
    values: list[int] = []
    for tables in ACTIVITY_TABLES.values():
        for table in tables:
            value = connection.execute(
                f"SELECT MAX(end) FROM {table} WHERE deviceId IN (0, 1, 2)"
            ).fetchone()[0]
            if value is not None:
                values.append(int(value))
    if not values:
        raise ValueError("trace has no CUDA activity on devices 0--2")
    return max(values)


def choose_midpoint_window(
    compute_by_stage: dict[int, list[tuple[int, int]]],
    *,
    training_start_ns: int,
    training_end_ns: int,
    requested_window_ns: int,
) -> tuple[int, int, int, int]:
    firsts = [min(start for start, _ in compute_by_stage[stage]) for stage in range(3)]
    lasts = [max(end for _, end in compute_by_stage[stage]) for stage in range(3)]
    common_start_ns = max(firsts)
    common_end_ns = min(lasts)
    if common_end_ns <= common_start_ns:
        raise ValueError("the three stages have no common active training interval")

    common_span_ns = common_end_ns - common_start_ns
    selected_span_ns = min(requested_window_ns, common_span_ns)
    selected_start_ns = common_start_ns + (common_span_ns - selected_span_ns) // 2
    selected_end_ns = selected_start_ns + selected_span_ns
    if not (training_start_ns <= selected_start_ns < selected_end_ns <= training_end_ns):
        raise ValueError("selected interval escaped the measured training wall interval")
    return selected_start_ns, selected_end_ns, common_start_ns, common_end_ns


def export_method(
    method_dir: Path,
    *,
    method: str,
    requested_window_ms: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    metadata_path = method_dir / "timeline_metadata.json"
    sqlite_path = method_dir / "cuda_timeline.sqlite"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    training = metadata["training"]
    training_wall_ns = int(round(float(training["wall_ms"]) * 1_000_000.0))

    connection = sqlite3.connect(sqlite_path)
    try:
        training_end_ns = latest_activity_end(connection)
        training_start_ns = training_end_ns - training_wall_ns
        compute_by_stage = {
            stage: read_activity(
                connection,
                tables=ACTIVITY_TABLES["compute"],
                device_id=stage,
                start_ns=training_start_ns,
                end_ns=training_end_ns,
            )
            for stage in range(3)
        }
        if any(not compute_by_stage[stage] for stage in range(3)):
            raise ValueError(f"{method}: missing compute activity on one or more stages")
        selected_start_ns, selected_end_ns, common_start_ns, common_end_ns = (
            choose_midpoint_window(
                compute_by_stage,
                training_start_ns=training_start_ns,
                training_end_ns=training_end_ns,
                requested_window_ns=int(round(requested_window_ms * 1_000_000.0)),
            )
        )

        interval_rows: list[dict[str, object]] = []
        metric_rows: list[dict[str, object]] = []
        selected_span_ns = selected_end_ns - selected_start_ns
        for stage in range(3):
            selected_by_kind: dict[str, list[tuple[int, int]]] = {}
            for kind, tables in ACTIVITY_TABLES.items():
                raw = read_activity(
                    connection,
                    tables=tables,
                    device_id=stage,
                    start_ns=selected_start_ns,
                    end_ns=selected_end_ns,
                )
                selected_by_kind[kind] = raw
                # A 0.02 ms visual merge removes only sub-pixel gaps. Metrics below
                # are computed from the exact, unfilled union.
                for start_ns, end_ns in merge_intervals(raw, max_gap_ns=20_000):
                    interval_rows.append(
                        {
                            "method": method,
                            "stage": stage,
                            "kind": kind,
                            "start_ms": (start_ns - selected_start_ns) / 1_000_000.0,
                            "end_ms": (end_ns - selected_start_ns) / 1_000_000.0,
                            "duration_ms": (end_ns - start_ns) / 1_000_000.0,
                        }
                    )

            compute_ns = duration_ns(selected_by_kind["compute"])
            transfer_ns = duration_ns(selected_by_kind["transfer"])
            active_ns = duration_ns(
                [*selected_by_kind["compute"], *selected_by_kind["transfer"]]
            )
            metric_rows.append(
                {
                    "method": method,
                    "stage": stage,
                    "selected_window_ms": selected_span_ns / 1_000_000.0,
                    "compute_busy_ms": compute_ns / 1_000_000.0,
                    "transfer_active_ms": transfer_ns / 1_000_000.0,
                    "total_active_ms": active_ns / 1_000_000.0,
                    "gpu_idle_ms": (selected_span_ns - active_ns) / 1_000_000.0,
                    "compute_busy_ratio": compute_ns / selected_span_ns,
                    "total_active_ratio": active_ns / selected_span_ns,
                    "profiled_wall_ms": float(training["wall_ms"]),
                    "profiled_throughput_per_s": float(training["throughput_per_s"]),
                }
            )
    finally:
        connection.close()

    source = {
        "method": method,
        "sqlite": str(sqlite_path),
        "timeline_metadata": str(metadata_path),
        "selection_rule": "centered fixed window within the common compute-active span of stages 0--2",
        "requested_window_ms": requested_window_ms,
        "selected_window_ms": (selected_end_ns - selected_start_ns) / 1_000_000.0,
        "selected_offset_from_training_start_ms": (
            selected_start_ns - training_start_ns
        )
        / 1_000_000.0,
        "common_active_start_offset_ms": (common_start_ns - training_start_ns)
        / 1_000_000.0,
        "common_active_end_offset_ms": (common_end_ns - training_start_ns)
        / 1_000_000.0,
        "profiled_training": training,
        "action_trace_enabled": False,
        "sync_action_trace": False,
    }
    return interval_rows, metric_rows, source


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/e4_throughput/raw/e4_gpu_timeline_nsys_v1"),
    )
    parser.add_argument("--window-ms", type=float, default=1000.0)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.window_ms <= 0:
        raise ValueError("--window-ms must be positive")

    output_dir = args.output_dir or args.root / "analysis"
    all_intervals: list[dict[str, object]] = []
    all_metrics: list[dict[str, object]] = []
    sources: list[dict[str, object]] = []
    for method in METHODS:
        intervals, metrics, source = export_method(
            args.root / method,
            method=method,
            requested_window_ms=args.window_ms,
        )
        all_intervals.extend(intervals)
        all_metrics.extend(metrics)
        sources.append(source)

    write_csv(output_dir / "gpu_timeline_intervals.csv", all_intervals)
    write_csv(output_dir / "gpu_timeline_metrics.csv", all_metrics)
    (output_dir / "gpu_timeline_sources.json").write_text(
        json.dumps(sources, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
