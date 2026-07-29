#!/usr/bin/env python3
"""Export and plot worker telemetry for a prepared experiment time window."""

from __future__ import annotations

import argparse
import csv
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class TelemetryRow:
    observed_epoch_ms: int
    device_id: str
    node_id: int
    stage_id: int
    battery_level: float
    is_charging: bool
    power_source: str
    battery_temp_c: float | None
    battery_voltage_mv: int | None
    battery_current_ua: int | None
    thermal_status: str
    app_pss_kb: int | None
    app_private_dirty_kb: int | None
    runtime_used_memory_kb: int | None
    worker_state: str


@dataclass(frozen=True)
class StageMemoryPoint:
    time_epoch_ms: int
    device_id: str
    stage_id: int
    pss_peak_mb: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("coordinator/coordinator/data/coordinator.db"))
    parser.add_argument("--results_csv", type=Path, required=True)
    parser.add_argument("--stage_memory_csv", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--pad_ms", type=int, default=60_000)
    return parser.parse_args()


def run_window(results_csv: Path, pad_ms: int) -> tuple[int, int]:
    starts: list[int] = []
    ends: list[int] = []
    with results_csv.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            starts.append(int(float(row["submitted_epoch_ms"])))
            ends.append(int(float(row["completed_epoch_ms"])))
    if not starts or not ends:
        raise ValueError(f"No timing rows in {results_csv}")
    return min(starts) - pad_ms, max(ends) + pad_ms


def read_result_midpoints(results_csv: Path) -> dict[int, int]:
    midpoints: dict[int, int] = {}
    with results_csv.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            submission_seq = int(row["submission_seq"])
            submitted_ms = int(float(row["submitted_epoch_ms"]))
            completed_ms = int(float(row["completed_epoch_ms"]))
            midpoints[submission_seq] = submitted_ms + ((completed_ms - submitted_ms) // 2)
    return midpoints


def read_rows(db_path: Path, start_ms: int, end_ms: int) -> list[TelemetryRow]:
    query = """
        SELECT
          observed_epoch_ms,
          device_id,
          node_id,
          stage_id,
          battery_level,
          is_charging,
          power_source,
          battery_temp_c,
          battery_voltage_mv,
          battery_current_ua,
          thermal_status,
          app_pss_kb,
          app_private_dirty_kb,
          runtime_used_memory_kb,
          worker_state
        FROM worker_telemetry
        WHERE observed_epoch_ms BETWEEN ? AND ?
        ORDER BY observed_epoch_ms ASC, device_id ASC
    """
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(query, (start_ms, end_ms)).fetchall()
    return [
        TelemetryRow(
            observed_epoch_ms=int(row[0]),
            device_id=str(row[1]),
            node_id=int(row[2]),
            stage_id=int(row[3]),
            battery_level=float(row[4]),
            is_charging=bool(row[5]),
            power_source=str(row[6]),
            battery_temp_c=as_float(row[7]),
            battery_voltage_mv=as_int(row[8]),
            battery_current_ua=as_int(row[9]),
            thermal_status=str(row[10]),
            app_pss_kb=as_int(row[11]),
            app_private_dirty_kb=as_int(row[12]),
            runtime_used_memory_kb=as_int(row[13]),
            worker_state=str(row[14]),
        )
        for row in rows
    ]


def as_float(value: object) -> float | None:
    return None if value is None else float(value)


def as_int(value: object) -> int | None:
    return None if value is None else int(value)


def write_csv(path: Path, rows: list[TelemetryRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "observed_epoch_ms",
                "device_id",
                "node_id",
                "stage_id",
                "battery_level",
                "is_charging",
                "power_source",
                "battery_temp_c",
                "battery_voltage_mv",
                "battery_current_ua",
                "thermal_status",
                "app_pss_kb",
                "app_private_dirty_kb",
                "runtime_used_memory_kb",
                "worker_state",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.observed_epoch_ms,
                    row.device_id,
                    row.node_id,
                    row.stage_id,
                    row.battery_level,
                    row.is_charging,
                    row.power_source,
                    row.battery_temp_c if row.battery_temp_c is not None else "",
                    row.battery_voltage_mv if row.battery_voltage_mv is not None else "",
                    row.battery_current_ua if row.battery_current_ua is not None else "",
                    row.thermal_status,
                    row.app_pss_kb if row.app_pss_kb is not None else "",
                    row.app_private_dirty_kb if row.app_private_dirty_kb is not None else "",
                    row.runtime_used_memory_kb if row.runtime_used_memory_kb is not None else "",
                    row.worker_state,
                ]
            )


def write_summary(path: Path, rows: list[TelemetryRow], start_ms: int, end_ms: int) -> None:
    by_device: dict[str, list[TelemetryRow]] = defaultdict(list)
    for row in rows:
        by_device[row.device_id].append(row)

    lines = [
        f"window_start_epoch_ms={start_ms}",
        f"window_end_epoch_ms={end_ms}",
        f"window_duration_min={(end_ms - start_ms) / 60000.0:.2f}",
        f"telemetry_rows={len(rows)}",
        "",
    ]
    for device_id, device_rows in sorted(by_device.items()):
        device_rows = sorted(device_rows, key=lambda item: item.observed_epoch_ms)
        levels = [row.battery_level for row in device_rows if row.battery_level >= 0]
        temps = [row.battery_temp_c for row in device_rows if row.battery_temp_c is not None]
        currents = [row.battery_current_ua for row in device_rows if row.battery_current_ua is not None]
        pss = [row.app_pss_kb for row in device_rows if row.app_pss_kb is not None]
        lines.extend(
            [
                f"device={device_id}",
                f"  samples={len(device_rows)}",
                f"  stage_ids={','.join(str(stage) for stage in sorted({row.stage_id for row in device_rows}))}",
                f"  battery_first={levels[0]:.1f} battery_last={levels[-1]:.1f} battery_min={min(levels):.1f} battery_max={max(levels):.1f}" if levels else "  battery=missing",
                f"  temp_min_c={min(temps):.1f} temp_avg_c={mean(temps):.1f} temp_max_c={max(temps):.1f}" if temps else "  temp=missing",
                f"  current_min_ma={min(currents) / 1000.0:.1f} current_avg_ma={mean(currents) / 1000.0:.1f} current_max_ma={max(currents) / 1000.0:.1f}" if currents else "  current=missing",
                f"  app_pss_min_mb={min(pss) / 1024.0:.1f} app_pss_avg_mb={mean(pss) / 1024.0:.1f} app_pss_max_mb={max(pss) / 1024.0:.1f}" if pss else "  app_pss=missing",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def read_stage_memory_points(
    path: Path | None,
    result_midpoints: dict[int, int],
) -> list[StageMemoryPoint]:
    if path is None or not path.exists():
        return []
    points: list[StageMemoryPoint] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            submission_seq_text = row.get("submission_seq", "").strip()
            device_id = row.get("device_id", "").strip()
            stage_id_text = row.get("stage_id", "").strip()
            pss_peak_text = row.get("pss_peak_kb", "").strip()
            if not submission_seq_text or not device_id or not stage_id_text or not pss_peak_text:
                continue
            submission_seq = int(submission_seq_text)
            time_epoch_ms = result_midpoints.get(submission_seq)
            if time_epoch_ms is None:
                continue
            points.append(
                StageMemoryPoint(
                    time_epoch_ms=time_epoch_ms,
                    device_id=device_id,
                    stage_id=int(stage_id_text),
                    pss_peak_mb=int(pss_peak_text) / 1024.0,
                )
            )
    return points


def plot(path: Path, rows: list[TelemetryRow], stage_memory_points: list[StageMemoryPoint]) -> None:
    if not rows:
        return
    base_ms = min(row.observed_epoch_ms for row in rows)
    by_device: dict[str, list[TelemetryRow]] = defaultdict(list)
    for row in rows:
        by_device[row.device_id].append(row)

    fig, axes = plt.subplots(4, 1, figsize=(11.5, 9.0), sharex=True)
    pss_axis = axes[-1]
    specs = [
        ("battery_level", "Battery (%)", lambda row: row.battery_level),
        ("battery_temp_c", "Battery temp (C)", lambda row: row.battery_temp_c),
        ("battery_current_ua", "Battery current (mA)", lambda row: None if row.battery_current_ua is None else row.battery_current_ua / 1000.0),
        ("app_pss_kb", "App PSS (MB)", lambda row: None if row.app_pss_kb is None else row.app_pss_kb / 1024.0),
    ]

    for ax, (_, ylabel, getter) in zip(axes, specs):
        for device_id, device_rows in sorted(by_device.items()):
            device_rows = sorted(device_rows, key=lambda item: item.observed_epoch_ms)
            xs: list[float] = []
            ys: list[float] = []
            for row in device_rows:
                value = getter(row)
                if value is None:
                    continue
                xs.append((row.observed_epoch_ms - base_ms) / 60000.0)
                ys.append(float(value))
            if xs:
                stage_ids = ",".join(str(stage) for stage in sorted({row.stage_id for row in device_rows}))
                line = ax.plot(xs, ys, marker=".", linewidth=1.2, markersize=3, label=f"{device_id} stage {stage_ids}")[0]
                if ax is pss_axis:
                    device_points = [
                        point
                        for point in stage_memory_points
                        if point.device_id == device_id and point.stage_id in {row.stage_id for row in device_rows}
                    ]
                    if device_points:
                        point_xs = [(point.time_epoch_ms - base_ms) / 60000.0 for point in device_points]
                        point_ys = [point.pss_peak_mb for point in device_points]
                        ax.scatter(
                            point_xs,
                            point_ys,
                            color=line.get_color(),
                            marker="x",
                            s=18,
                            linewidths=0.8,
                            alpha=0.65,
                            label=f"{device_id} stage peak samples",
                        )
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("Minutes from first telemetry sample")
    if stage_memory_points:
        pss_axis.text(
            0.01,
            0.03,
            "solid: heartbeat PSS samples; x: per-request stage-local PSS peaks",
            transform=pss_axis.transAxes,
            fontsize=8,
            va="bottom",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
    fig.suptitle("Worker telemetry with stage-local peak memory")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start_ms, end_ms = run_window(args.results_csv, args.pad_ms)
    result_midpoints = read_result_midpoints(args.results_csv)
    rows = read_rows(args.db, start_ms, end_ms)
    stage_memory_points = read_stage_memory_points(args.stage_memory_csv, result_midpoints)
    write_csv(args.output_dir / "worker-telemetry-final.csv", rows)
    write_summary(args.output_dir / "worker-telemetry-summary.txt", rows, start_ms, end_ms)
    plot(args.output_dir / "worker-telemetry-timeseries.png", rows, stage_memory_points)
    print(f"rows={len(rows)}")
    print(f"wrote={args.output_dir / 'worker-telemetry-final.csv'}")
    print(f"wrote={args.output_dir / 'worker-telemetry-summary.txt'}")
    print(f"wrote={args.output_dir / 'worker-telemetry-timeseries.png'}")


if __name__ == "__main__":
    main()
