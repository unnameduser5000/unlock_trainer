#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from sg_exe_trainer.runtime.recovery.durable_io import atomic_write_json


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_policy_comparison(
    *,
    drain: dict[str, Any],
    streamed: dict[str, Any],
) -> dict[str, Any]:
    if drain["shared_setup"] != streamed["shared_setup"]:
        raise ValueError("drain-first and window-streamed setups differ")
    if drain["model"] != streamed["model"]:
        raise ValueError("drain-first and window-streamed model environments differ")
    exact_drain = drain["rows"][0]
    exact_streamed = streamed["rows"][0]
    for field in (
        "method",
        "rejoin_to_terminal_target_ms_values",
        "checkpoint_file_bytes",
        "boundary_file_bytes",
    ):
        if exact_drain[field] != exact_streamed[field]:
            raise ValueError(f"Exact-BP reference differs between reports: {field}")

    drain_row = drain["rows"][1]
    streamed_row = streamed["rows"][1]
    drain_ms = float(drain_row["rejoin_to_terminal_target_ms_mean"])
    streamed_ms = float(streamed_row["rejoin_to_terminal_target_ms_mean"])
    exact_ms = float(exact_drain["rejoin_to_terminal_target_ms_mean"])
    return {
        "comparison": "E5 formal-v2 catch-up policy comparison",
        "shared_setup": drain["shared_setup"],
        "model": drain["model"],
        "repeats": drain["repeats"],
        "rows": [exact_drain, drain_row, streamed_row],
        "derived": {
            "streamed_latency_reduction_vs_drain_first": 1.0 - streamed_ms / drain_ms,
            "streamed_recovery_time_over_exactbp": streamed_ms / exact_ms,
            "drain_first_recovery_time_over_exactbp": drain_ms / exact_ms,
        },
        "interpretation": (
            "Window-level streaming removes most of the stage-major drain bubble, "
            "but the current BP-free local objectives and durable boundary path still "
            "leave recovery slower than the overlapped Exact-BP 1F1B baseline."
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "method",
        "repeats",
        "stage0_commits_at_rejoin",
        "stage1_commits_at_rejoin",
        "stage2_commits_at_rejoin",
        "analytical_stage_windows_remaining",
        "rejoin_to_terminal_target_ms_mean",
        "rejoin_to_terminal_target_ms_std",
        "checkpoint_file_bytes",
        "boundary_file_bytes",
        "total_durable_file_bytes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# E5 catch-up policy comparison",
        "",
        "| Method | Commits at rejoin (s0/s1/s2) | Remaining stage-windows | Recovery (ms, mean +/- std) | Durable state (MiB) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        commits = "/".join(str(row[f"stage{stage}_commits_at_rejoin"]) for stage in range(3))
        lines.append(
            f"| {row['method']} | {commits} | "
            f"{row['analytical_stage_windows_remaining']} | "
            f"{row['rejoin_to_terminal_target_ms_mean']:.3f} +/- "
            f"{row['rejoin_to_terminal_target_ms_std']:.3f} | "
            f"{row['total_durable_file_bytes'] / 2**20:.3f} |"
        )
    lines.extend(["", payload["interpretation"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine E5 catch-up policy reports")
    parser.add_argument("--drain-report", type=Path, required=True)
    parser.add_argument("--streamed-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    payload = build_policy_comparison(
        drain=_load(args.drain_report),
        streamed=_load(args.streamed_report),
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    atomic_write_json(args.output_dir / "policy_comparison.json", payload)
    _write_csv(args.output_dir / "policy_comparison.csv", payload["rows"])
    _write_markdown(args.output_dir / "policy_comparison.md", payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
