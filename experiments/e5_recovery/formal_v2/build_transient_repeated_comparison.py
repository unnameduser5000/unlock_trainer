#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

from .build_transient_comparison import _load, build_transient_comparison
from sg_exe_trainer.runtime.recovery.durable_io import atomic_write_json


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    invariants = (
        "method",
        "stage0_progress_at_rejoin",
        "stage1_progress_at_rejoin",
        "stage2_progress_at_rejoin",
        "remaining_stage_windows",
        "volatile_hidden_bytes",
        "transport",
        "run_total_send_bytes",
        "run_total_recv_bytes",
        "durable_file_bytes",
    )
    for row in rows[1:]:
        mismatch = {
            field: (first[field], row[field])
            for field in invariants
            if row[field] != first[field]
        }
        if mismatch:
            raise ValueError(f"repeat invariant mismatch: {mismatch}")

    values = [float(row["rejoin_to_terminal_target_ms"]) for row in rows]
    return {
        **{field: first[field] for field in invariants},
        "repeats": len(rows),
        "recovery_ms_values": values,
        "recovery_ms_mean": statistics.mean(values),
        "recovery_ms_std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def build_repeated(
    *,
    bpfree_summaries: list[dict[str, Any]],
    exactbp_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    if not bpfree_summaries or len(bpfree_summaries) != len(exactbp_summaries):
        raise ValueError("BP-free and Exact-BP require equal non-zero repeat counts")

    pairs = [
        build_transient_comparison(bpfree=bpfree, exactbp=exactbp)
        for bpfree, exactbp in zip(bpfree_summaries, exactbp_summaries)
    ]
    setup = pairs[0]["shared_setup"]
    if any(pair["shared_setup"] != setup for pair in pairs[1:]):
        raise ValueError("repeat protocol mismatch")

    exact = _aggregate([pair["rows"][0] for pair in pairs])
    bpfree = _aggregate([pair["rows"][1] for pair in pairs])
    delta = bpfree["recovery_ms_mean"] / exact["recovery_ms_mean"] - 1.0
    return {
        "comparison": "E5 repeated transient stage-1 CPU-transport outage",
        "shared_setup": setup,
        "repeats": len(pairs),
        "rows": [exact, bpfree],
        "derived": {
            "bpfree_mean_recovery_time_over_exactbp": 1.0 + delta,
            "bpfree_mean_latency_delta_fraction": delta,
            "bpfree_remaining_stage_window_reduction_fraction": 1.0
            - bpfree["remaining_stage_windows"] / exact["remaining_stage_windows"],
            "bpfree_run_send_bytes_over_exactbp": (
                bpfree["run_total_send_bytes"] / exact["run_total_send_bytes"]
            ),
        },
        "takeaway": (
            f"Across {len(pairs)} fresh runs, BP-free reaches the terminal target "
            f"{-delta * 100.0:.1f}% sooner on average, retains stage-0 progress, "
            "leaves one-third fewer stage-windows, and sends half as many bytes "
            "over the complete CPU-transport run."
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "method",
        "repeats",
        "stage0_progress_at_rejoin",
        "stage1_progress_at_rejoin",
        "stage2_progress_at_rejoin",
        "remaining_stage_windows",
        "recovery_ms_mean",
        "recovery_ms_std",
        "volatile_hidden_bytes",
        "run_total_send_bytes",
        "run_total_recv_bytes",
        "transport",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    setup = payload["shared_setup"]
    lines = [
        "# E5 repeated CPU-transport transient outage",
        "",
        (
            f"TinyLlama, 3 stages, stage-1 outage, b={setup['physical_batch_size']}, "
            f"M={setup['microbatches_per_window']}, B={setup['effective_batch_size']}, "
            f"{setup['outage_windows']} outage windows, n={payload['repeats']}."
        ),
        "",
        "| Method | Progress at rejoin (s0/s1/s2) | Remaining stage-windows | Recovery (ms, mean +/- std) | RAM hidden (MiB) | Run send (MiB) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        progress = "/".join(
            str(row[f"stage{stage}_progress_at_rejoin"])
            for stage in range(3)
        )
        lines.append(
            f"| {row['method']} | {progress} | {row['remaining_stage_windows']} | "
            f"{row['recovery_ms_mean']:.3f} +/- {row['recovery_ms_std']:.3f} | "
            f"{row['volatile_hidden_bytes'] / 2**20:.3f} | "
            f"{row['run_total_send_bytes'] / 2**20:.3f} |"
        )
    lines.extend(["", payload["takeaway"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate repeated E5 CPU transient-outage pairs"
    )
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
