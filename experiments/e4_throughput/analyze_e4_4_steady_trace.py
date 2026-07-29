#!/usr/bin/env python3
"""Analyze canonical E4.4 steady-state traces without creating figure clutter."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

OPTIMIZER_ACTIONS = {"LOCAL_OPTIMIZER_STEP", "OPTIMIZER_STEP"}
METHODS = ("bpfree", "exactbp_1f1b")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


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


def linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("linear_fit requires at least two paired values")
    x_mean = statistics.mean(xs)
    y_mean = statistics.mean(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom <= 0:
        raise ValueError("linear_fit has zero x variance")
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denom
    intercept = y_mean - slope * x_mean
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 if ss_tot <= 0 else max(0.0, 1.0 - ss_res / ss_tot)
    return slope, intercept, r2


def optimizer_completions(run_dir: Path) -> dict[int, dict[int, float]]:
    by_stage: dict[int, dict[int, float]] = defaultdict(dict)
    paths = sorted(run_dir.glob("train.stage*.actions.csv"))
    if not paths:
        raise FileNotFoundError(f"no action traces under {run_dir}")
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("action") not in OPTIMIZER_ACTIONS:
                    continue
                stage = parse_int(row.get("stage_id"))
                window = window_id(row)
                end_ms = parse_float(row.get("end_epoch_ms"))
                if stage < 0 or window < 0 or not math.isfinite(end_ms):
                    continue
                by_stage[stage][window] = max(
                    by_stage[stage].get(window, end_ms), end_ms
                )
    if not by_stage:
        raise ValueError(f"no optimizer completion events under {run_dir}")
    return dict(by_stage)


def analyze_method(
    *,
    method: str,
    rep: int,
    run_dir: Path,
    trace_start: int,
    trace_end: int,
    trim: int,
    period_spread_limit: float,
    lag_drift_limit: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    completions = optimizer_completions(run_dir)
    stages = sorted(completions)
    if stages != list(range(len(stages))):
        raise ValueError(f"{method}: unexpected stages {stages}")

    common_windows = set(range(trace_start, trace_end))
    for stage in stages:
        common_windows &= set(completions[stage])
    ordered = sorted(common_windows)
    if trim:
        ordered = ordered[trim:-trim]
    if len(ordered) < 8:
        raise ValueError(
            f"{method}: only {len(ordered)} common analyzed windows after trim"
        )

    origin = min(completions[stage][ordered[0]] for stage in stages)
    completion_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    periods: list[float] = []
    max_abs_lag_drift = 0.0

    for stage in stages:
        xs = [float(window) for window in ordered]
        ys = [completions[stage][window] - origin for window in ordered]
        period_slope, _, period_r2 = linear_fit(xs, ys)
        periods.append(period_slope)

        if stage == 0:
            lag_values = [0.0 for _ in ordered]
            lag_slope = 0.0
            lag_r2 = 1.0
        else:
            lag_values = [
                completions[stage][window] - completions[0][window]
                for window in ordered
            ]
            lag_slope, _, lag_r2 = linear_fit(xs, lag_values)
            max_abs_lag_drift = max(max_abs_lag_drift, abs(lag_slope))

        stage_rows.append(
            {
                "method": method,
                "rep": rep,
                "stage_id": stage,
                "analyzed_window_start": ordered[0],
                "analyzed_window_end_exclusive": ordered[-1] + 1,
                "window_count": len(ordered),
                "period_ms_per_window": period_slope,
                "period_fit_r2": period_r2,
                "mean_same_window_lag_ms": statistics.mean(lag_values),
                "p95_same_window_lag_ms": sorted(lag_values)[
                    min(len(lag_values) - 1, math.ceil(0.95 * len(lag_values)) - 1)
                ],
                "lag_drift_ms_per_window": lag_slope,
                "lag_fit_r2": lag_r2,
            }
        )

        for window, completion_ms, lag_ms in zip(ordered, ys, lag_values):
            completion_rows.append(
                {
                    "method": method,
                    "rep": rep,
                    "stage_id": stage,
                    "window_id": window,
                    "optimizer_end_ms_from_origin": completion_ms,
                    "same_window_lag_vs_stage0_ms": lag_ms,
                }
            )

    median_period = statistics.median(periods)
    period_spread = (
        (max(periods) - min(periods)) / median_period
        if median_period > 0 else math.inf
    )
    lag_drift_fraction = (
        max_abs_lag_drift / median_period
        if median_period > 0 else math.inf
    )
    stable = (
        period_spread <= period_spread_limit
        and lag_drift_fraction <= lag_drift_limit
    )

    center = ordered[len(ordered) // 2]
    recommended_start = max(ordered[0], center - 2)
    report = {
        "method": method,
        "rep": rep,
        "stable": stable,
        "analyzed_windows": [ordered[0], ordered[-1] + 1],
        "window_count": len(ordered),
        "median_period_ms_per_window": median_period,
        "relative_stage_period_spread": period_spread,
        "max_abs_lag_drift_ms_per_window": max_abs_lag_drift,
        "relative_lag_drift": lag_drift_fraction,
        "recommended_schedule_start_window": recommended_start,
        "recommended_schedule_window_count": 4,
    }
    return completion_rows, stage_rows, report


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trace-start", type=int, required=True)
    parser.add_argument("--trace-end", type=int, required=True)
    parser.add_argument("--trim", type=int, default=2)
    parser.add_argument("--period-spread-limit", type=float, default=0.05)
    parser.add_argument("--lag-drift-limit", type=float, default=0.05)
    args = parser.parse_args()

    all_completions: list[dict[str, Any]] = []
    all_stages: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []

    for method in METHODS:
        method_root = args.root / "local" / "throughput_b8_m4" / method
        rep_dirs = sorted(method_root.glob("rep_*"))
        if not rep_dirs:
            raise FileNotFoundError(f"no repetitions under {method_root}")
        for run_dir in rep_dirs:
            rep = int(run_dir.name.removeprefix("rep_"))
            completions, stages, report = analyze_method(
                method=method,
                rep=rep,
                run_dir=run_dir,
                trace_start=args.trace_start,
                trace_end=args.trace_end,
                trim=args.trim,
                period_spread_limit=args.period_spread_limit,
                lag_drift_limit=args.lag_drift_limit,
            )
            all_completions.extend(completions)
            all_stages.extend(stages)
            reports.append(report)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "window_completion.csv", all_completions)
    write_csv(args.output_dir / "lag_summary.csv", all_stages)
    overall = {
        "trace_window_range": [args.trace_start, args.trace_end],
        "trimmed_windows_per_side": args.trim,
        "runs": reports,
        "method_summary": {
            method: {
                "repetitions": sum(item["method"] == method for item in reports),
                "stable_repetitions": sum(
                    item["method"] == method and item["stable"] for item in reports
                ),
                "median_period_ms_per_window": statistics.median(
                    item["median_period_ms_per_window"]
                    for item in reports
                    if item["method"] == method
                ),
            }
            for method in METHODS
        },
        "all_methods_stable": all(item["stable"] for item in reports),
        "note": (
            "Stable means stage completion periods agree and same-window lag "
            "does not drift beyond the configured relative thresholds."
        ),
    }
    write_json(args.output_dir / "steady_state_report.json", overall)

    print("===== E4.4 steady-state lag diagnosis =====")
    for report in reports:
        status = "STABLE" if report["stable"] else "DRIFTING"
        print(
            f"{report['method']} rep_{report['rep']:02d}: {status}; "
            f"period={report['median_period_ms_per_window']:.3f} ms/window; "
            f"period_spread={100.0 * report['relative_stage_period_spread']:.2f}%; "
            f"lag_drift={report['max_abs_lag_drift_ms_per_window']:.3f} ms/window"
        )
    print(args.output_dir / "steady_state_report.json")


if __name__ == "__main__":
    main()
