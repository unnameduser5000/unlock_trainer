#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .build_comparison import _peak_by_stage, _shared_setup
from sg_exe_trainer.runtime.recovery.durable_io import atomic_write_json


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _transport_bytes(summary: dict[str, Any], field: str) -> int:
    return sum(
        int(rank.get("transport_summary", {}).get(field, 0))
        for rank in summary.get("rank_summaries", [])
    )


def build_transient_comparison(
    *,
    bpfree: dict[str, Any],
    exactbp: dict[str, Any],
) -> dict[str, Any]:
    setup = _shared_setup(bpfree, exactbp)
    if bpfree.get("recovery_state_mode") != "volatile":
        raise ValueError("BP-free input is not a volatile recovery run")
    if exactbp.get("recovery_state_mode") != "volatile":
        raise ValueError("Exact-BP input is not a volatile recovery run")
    bpfree_transport = str(bpfree.get("transport", "unspecified"))
    exactbp_transport = str(exactbp.get("transport", "unspecified"))

    outage_windows = int(setup["outage_windows"])
    num_stages = int(setup["num_stages"])
    failure_stage = int(setup["failure_stage"])
    bpfree_rejoin = bpfree.get("progress_at_rejoin", {}).get(
        "local_optimizer_windows",
        bpfree["stage_commit_counts_at_rejoin"],
    )
    exact_rejoin = exactbp.get("progress_at_rejoin", {}).get(
        "local_optimizer_windows",
        exactbp["backlog_at_rejoin"]["stage_commit_counts"],
    )
    rows = [
        {
            "method": "Exact BP / 1F1B volatile",
            "stage0_progress_at_rejoin": int(exact_rejoin["0"]),
            "stage1_progress_at_rejoin": int(exact_rejoin["1"]),
            "stage2_progress_at_rejoin": int(exact_rejoin["2"]),
            "remaining_stage_windows": outage_windows * num_stages,
            "rejoin_to_terminal_target_ms": float(
                exactbp["recovery_timing"]["rejoin_to_terminal_target_ms"]
            ),
            "volatile_hidden_bytes": 0,
            "transport": exactbp_transport,
            "run_total_send_bytes": _transport_bytes(exactbp, "total_send_bytes"),
            "run_total_recv_bytes": _transport_bytes(exactbp, "total_recv_bytes"),
            "durable_file_bytes": int(exactbp["durable_state"]["total_file_bytes"]),
            "peak_cuda_allocated_bytes_by_stage": _peak_by_stage(exactbp),
        },
        {
            "method": "BP-free volatile replay",
            "stage0_progress_at_rejoin": int(bpfree_rejoin["0"]),
            "stage1_progress_at_rejoin": int(bpfree_rejoin["1"]),
            "stage2_progress_at_rejoin": int(bpfree_rejoin["2"]),
            "remaining_stage_windows": outage_windows * (num_stages - failure_stage),
            "rejoin_to_terminal_target_ms": float(
                bpfree["recovery_timing"]["rejoin_to_terminal_target_ms"]
            ),
            "volatile_hidden_bytes": int(
                bpfree["volatile_state_at_rejoin"]["hidden_tensor_bytes"]
            ),
            "transport": bpfree_transport,
            "run_total_send_bytes": _transport_bytes(bpfree, "total_send_bytes"),
            "run_total_recv_bytes": _transport_bytes(bpfree, "total_recv_bytes"),
            "durable_file_bytes": int(bpfree["durable_state"]["total_file_bytes"]),
            "peak_cuda_allocated_bytes_by_stage": _peak_by_stage(bpfree),
        },
    ]
    exact_row, bpfree_row = rows
    latency_delta = (
        bpfree_row["rejoin_to_terminal_target_ms"]
        / exact_row["rejoin_to_terminal_target_ms"]
        - 1.0
    )
    if latency_delta < 0:
        takeaway = (
            f"On this {bpfree_transport} point, BP-free reaches the terminal "
            f"target {-latency_delta * 100.0:.1f}% sooner than Exact BP. It "
            "retains stage-0 progress, leaves one-third fewer stage-windows after "
            "rejoin, and sends no hidden-gradient messages."
        )
    else:
        takeaway = (
            f"On this {bpfree_transport} point, BP-free reaches the terminal "
            f"target {latency_delta * 100.0:.1f}% later than Exact BP despite "
            "retaining stage-0 progress and leaving one-third fewer stage-windows."
        )
    return {
        "comparison": "E5 transient stage-1 service outage, volatile state",
        "status": "single-seed stop-condition diagnostic",
        "shared_setup": setup,
        "rows": rows,
        "derived": {
            "bpfree_recovery_time_over_exactbp": (
                bpfree_row["rejoin_to_terminal_target_ms"]
                / exact_row["rejoin_to_terminal_target_ms"]
            ),
            "bpfree_latency_delta_fraction": latency_delta,
            "bpfree_remaining_stage_window_reduction_fraction": 1.0
            - (
                bpfree_row["remaining_stage_windows"]
                / exact_row["remaining_stage_windows"]
            ),
        },
        "takeaway": takeaway,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat_rows = []
    for row in rows:
        flat = dict(row)
        flat["peak_cuda_allocated_bytes_by_stage"] = json.dumps(
            flat["peak_cuda_allocated_bytes_by_stage"]
        )
        flat_rows.append(flat)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    setup = payload["shared_setup"]
    lines = [
        "# E5 transient-outage diagnostic",
        "",
        (
            f"One seed, b={setup['physical_batch_size']}, "
            f"M={setup['microbatches_per_window']}, "
            f"{setup['outage_windows']} outage windows. No checkpoint or boundary "
            "file is written in either method."
        ),
        "",
        "| Method | Progress at rejoin (s0/s1/s2) | Remaining stage-windows | Recovery (ms) | RAM hidden (MiB) | Run send (MiB) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        progress = "/".join(
            str(row[f"stage{stage}_progress_at_rejoin"])
            for stage in range(3)
        )
        lines.append(
            f"| {row['method']} | {progress} | "
            f"{row['remaining_stage_windows']} | "
            f"{row['rejoin_to_terminal_target_ms']:.3f} | "
            f"{row['volatile_hidden_bytes'] / 2**20:.3f} | "
            f"{row['run_total_send_bytes'] / 2**20:.3f} |"
        )
    lines.extend(["", payload["takeaway"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the E5 volatile comparison")
    parser.add_argument("--bpfree-summary", type=Path, required=True)
    parser.add_argument("--exactbp-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    payload = build_transient_comparison(
        bpfree=_load(args.bpfree_summary),
        exactbp=_load(args.exactbp_summary),
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    atomic_write_json(args.output_dir / "comparison.json", payload)
    _write_csv(args.output_dir / "comparison.csv", payload["rows"])
    _write_markdown(args.output_dir / "comparison.md", payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
