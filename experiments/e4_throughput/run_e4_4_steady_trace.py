#!/usr/bin/env python3
"""Run the single canonical E4.4 long trace and diagnose stage-lag stability."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

DEFAULT_CONFIG = "experiments/e4_throughput/configs/e4_4_steady_trace.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    cfg = json.loads(config_path.read_text(encoding="utf-8"))

    command = [
        sys.executable,
        "experiments/e4_throughput/run_e4_4_overhead_decomposition.py",
        "--config",
        str(config_path),
        "--profiles",
        "local",
        "--regimes",
        "throughput_b8_m4",
        "--repetitions",
        str(cfg["repetitions"]),
        "--skip-analysis",
    ]
    if args.dry_run:
        command.append("--dry-run")
    elif args.overwrite:
        command.append("--overwrite")
    else:
        command.append("--resume")

    subprocess.run(command, cwd=repo_root, check=True)
    if args.dry_run:
        return

    subprocess.run(
        [
            sys.executable,
            "experiments/e4_throughput/analyze_e4_4_steady_trace.py",
            "--root",
            str(repo_root / cfg["output_root"]),
            "--output-dir",
            str(repo_root / cfg["analysis_output"]),
            "--trace-start",
            str(cfg["action_trace_start_window"]),
            "--trace-end",
            str(cfg["action_trace_end_window"]),
            "--trim",
            str(cfg["analysis_trim_windows"]),
            "--period-spread-limit",
            str(cfg["stability_period_spread_fraction"]),
            "--lag-drift-limit",
            str(cfg["stability_lag_drift_fraction"]),
        ],
        cwd=repo_root,
        check=True,
    )
    analysis_dir = repo_root / cfg["analysis_output"]
    subprocess.run(
        [
            sys.executable,
            "experiments/e4_throughput/analyze_e4_4_traces.py",
            "--root",
            str(repo_root / cfg["output_root"]),
            "--output-dir",
            str(analysis_dir),
        ],
        cwd=repo_root,
        check=True,
    )
    steady_report = json.loads(
        (analysis_dir / "steady_state_report.json").read_text(encoding="utf-8")
    )
    starts = {
        int(item["recommended_schedule_start_window"])
        for item in steady_report["runs"]
    }
    if len(starts) != 1:
        raise ValueError(f"methods recommend different schedule starts: {sorted(starts)}")
    subprocess.run(
        [
            sys.executable,
            "experiments/e4_throughput/build_e4_4_paper_artifacts.py",
            "--raw-root",
            str(repo_root / cfg["output_root"]),
            "--analysis-dir",
            str(analysis_dir),
            "--output-dir",
            str(repo_root / cfg["figure_output"]),
            "--start-window",
            str(starts.pop()),
            "--duration-ms",
            str(cfg.get("timeline_duration_ms", 1600.0)),
        ],
        cwd=repo_root,
        check=True,
    )
    print("\nE4.4 canonical steady trace completed successfully.")


if __name__ == "__main__":
    main()
