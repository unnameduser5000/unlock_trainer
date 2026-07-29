#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

from .build_comparison import _load, build_comparison
from sg_exe_trainer.runtime.recovery.durable_io import atomic_write_json


def _aggregate_method(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    invariant_fields = (
        "method",
        "stage0_commits_at_rejoin",
        "stage1_commits_at_rejoin",
        "stage2_commits_at_rejoin",
        "recovery_state",
        "backlog_or_outbox_windows",
        "analytical_stage_windows_remaining",
        "checkpoint_file_bytes",
        "boundary_file_bytes",
        "total_durable_file_bytes",
    )
    for row in rows[1:]:
        mismatches = {
            field: (first[field], row[field])
            for field in invariant_fields
            if row[field] != first[field]
        }
        if mismatches:
            raise ValueError(f"repeat invariant mismatch for {first['method']}: {mismatches}")

    timings = [float(row["rejoin_to_terminal_target_ms"]) for row in rows]
    peaks = list(zip(*(row["peak_cuda_allocated_bytes_by_stage"] for row in rows)))
    return {
        **{field: first[field] for field in invariant_fields},
        "repeats": len(rows),
        "rejoin_to_terminal_target_ms_values": timings,
        "rejoin_to_terminal_target_ms_mean": statistics.mean(timings),
        "rejoin_to_terminal_target_ms_std": statistics.stdev(timings) if len(timings) > 1 else 0.0,
        "peak_cuda_allocated_bytes_mean_by_stage": [
            statistics.mean(int(value) for value in stage_values)
            for stage_values in peaks
        ],
    }


def build_repeated(
    *,
    bpfree_summaries: list[dict[str, Any]],
    exactbp_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(bpfree_summaries) != len(exactbp_summaries) or not bpfree_summaries:
        raise ValueError("BP-free and Exact-BP need the same non-zero repeat count")
    pairs = [
        build_comparison(bpfree=bpfree, exactbp=exactbp)
        for bpfree, exactbp in zip(bpfree_summaries, exactbp_summaries)
    ]
    first = pairs[0]
    for pair in pairs[1:]:
        if pair["shared_setup"] != first["shared_setup"] or pair["model"] != first["model"]:
            raise ValueError("repeat setup or environment differs")

    exact = _aggregate_method([pair["rows"][0] for pair in pairs])
    bpfree = _aggregate_method([pair["rows"][1] for pair in pairs])
    return {
        "comparison": first["comparison"],
        "shared_setup": first["shared_setup"],
        "model": first["model"],
        "repeats": len(pairs),
        "rows": [exact, bpfree],
        "derived": {
            "bpfree_mean_recovery_time_over_exactbp": (
                bpfree["rejoin_to_terminal_target_ms_mean"]
                / exact["rejoin_to_terminal_target_ms_mean"]
            ),
            "bpfree_durable_bytes_over_exactbp": (
                bpfree["total_durable_file_bytes"] / exact["total_durable_file_bytes"]
            ),
            "bpfree_stage_window_reduction_fraction": 1.0
            - (
                bpfree["analytical_stage_windows_remaining"]
                / exact["analytical_stage_windows_remaining"]
            ),
        },
        "repeat_rows": [pair["rows"] for pair in pairs],
        "interpretation": first["interpretation"],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "method",
        "repeats",
        "stage0_commits_at_rejoin",
        "stage1_commits_at_rejoin",
        "stage2_commits_at_rejoin",
        "recovery_state",
        "analytical_stage_windows_remaining",
        "rejoin_to_terminal_target_ms_mean",
        "rejoin_to_terminal_target_ms_std",
        "checkpoint_file_bytes",
        "boundary_file_bytes",
        "total_durable_file_bytes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    setup = payload["shared_setup"]
    lines = [
        "# E5 formal-v2 repeated paired result",
        "",
        (
            f"TinyLlama, 3 stages, stage-1 outage, b={setup['physical_batch_size']}, "
            f"M={setup['microbatches_per_window']}, B={setup['effective_batch_size']}, "
            f"{setup['outage_windows']} outage windows, n={payload['repeats']}."
        ),
        "",
        "| Method | Commits at rejoin (s0/s1/s2) | Remaining stage-windows | Rejoin to terminal (ms, mean +/- std) | Total durable (MiB) |",
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
    parser = argparse.ArgumentParser(description="Aggregate repeated E5 formal-v2 pairs")
    parser.add_argument("--bpfree-summary", type=Path, action="append", required=True)
    parser.add_argument("--exactbp-summary", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    payload = build_repeated(
        bpfree_summaries=[_load(path) for path in args.bpfree_summary],
        exactbp_summaries=[_load(path) for path in args.exactbp_summary],
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    atomic_write_json(args.output_dir / "comparison_repeated.json", payload)
    _write_csv(args.output_dir / "comparison_repeated.csv", payload["rows"])
    _write_markdown(args.output_dir / "comparison_repeated.md", payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
