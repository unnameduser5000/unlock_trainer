#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from sg_exe_trainer.runtime.recovery.durable_io import atomic_write_json


SHARED_PROTOCOL_FIELDS = (
    "num_stages",
    "failure_stage",
    "prelude_windows",
    "outage_windows",
    "resumed_windows",
    "physical_batch_size",
    "microbatches_per_window",
    "effective_batch_size",
    "max_pending_windows",
    "failure_model",
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _shared_setup(bpfree: dict[str, Any], exactbp: dict[str, Any]) -> dict[str, Any]:
    left = bpfree["protocol"]
    right = exactbp["protocol"]
    mismatches = {
        field: {"bpfree": left.get(field), "exactbp": right.get(field)}
        for field in SHARED_PROTOCOL_FIELDS
        if left.get(field) != right.get(field)
    }
    for field in ("resolved_model", "dtype", "learning_rate", "seed"):
        if bpfree.get(field) != exactbp.get(field):
            mismatches[field] = {
                "bpfree": bpfree.get(field),
                "exactbp": exactbp.get(field),
            }
    if bpfree.get("training_config") != exactbp.get("training_config"):
        mismatches["training_config"] = {
            "bpfree": bpfree.get("training_config"),
            "exactbp": exactbp.get("training_config"),
        }
    if mismatches:
        raise ValueError(f"comparison inputs are not aligned: {mismatches}")
    return {field: left[field] for field in SHARED_PROTOCOL_FIELDS}


def _peak_by_stage(summary: dict[str, Any]) -> list[int]:
    return [
        int(item["peak_cuda_allocated_bytes"])
        for item in sorted(summary["rank_summaries"], key=lambda row: int(row["rank"]))
    ]


def build_comparison(
    *,
    bpfree: dict[str, Any],
    exactbp: dict[str, Any],
) -> dict[str, Any]:
    setup = _shared_setup(bpfree, exactbp)
    outage_windows = int(setup["outage_windows"])
    num_stages = int(setup["num_stages"])
    failure_stage = int(setup["failure_stage"])
    catchup_policy = str(bpfree["protocol"].get("catchup_policy", "drain_first"))

    bpfree_rejoin = bpfree["stage_commit_counts_at_rejoin"]
    exact_rejoin = exactbp["backlog_at_rejoin"]["stage_commit_counts"]
    rows = [
        {
            "method": "Exact BP / 1F1B",
            "stage0_commits_at_rejoin": int(exact_rejoin["0"]),
            "stage1_commits_at_rejoin": int(exact_rejoin["1"]),
            "stage2_commits_at_rejoin": int(exact_rejoin["2"]),
            "recovery_state": "raw request backlog",
            "backlog_or_outbox_windows": int(exactbp["backlog_at_rejoin"]["windows"]),
            "analytical_stage_windows_remaining": outage_windows * num_stages,
            "rejoin_to_terminal_target_ms": float(
                exactbp["recovery_timing"]["rejoin_to_terminal_target_ms"]
            ),
            "checkpoint_file_bytes": int(
                exactbp["durable_state"]["checkpoint_file_bytes"]
            ),
            "boundary_file_bytes": int(exactbp["durable_state"]["boundary_file_bytes"]),
            "total_durable_file_bytes": int(exactbp["durable_state"]["total_file_bytes"]),
            "peak_cuda_allocated_bytes_by_stage": _peak_by_stage(exactbp),
        },
        {
            "method": f"BP-free / {catchup_policy.replace('_', '-')}",
            "stage0_commits_at_rejoin": int(bpfree_rejoin["0"]),
            "stage1_commits_at_rejoin": int(bpfree_rejoin["1"]),
            "stage2_commits_at_rejoin": int(bpfree_rejoin["2"]),
            "recovery_state": "versioned hidden outbox",
            "backlog_or_outbox_windows": len(
                bpfree["pending_outbox_windows_at_rejoin"]["stage0_to_stage1"]
            ),
            "analytical_stage_windows_remaining": (
                outage_windows * (num_stages - failure_stage)
            ),
            "rejoin_to_terminal_target_ms": float(
                bpfree["recovery_timing"]["rejoin_to_terminal_target_ms"]
            ),
            "checkpoint_file_bytes": int(
                bpfree["durable_state"]["checkpoint_file_bytes"]
            ),
            "boundary_file_bytes": int(bpfree["durable_state"]["boundary_file_bytes"]),
            "total_durable_file_bytes": int(bpfree["durable_state"]["total_file_bytes"]),
            "peak_cuda_allocated_bytes_by_stage": _peak_by_stage(bpfree),
        },
    ]
    exact_row, bpfree_row = rows
    interpretation = (
        "BP-free preserves prefix optimizer progress. Window-streamed catch-up "
        "allows stage 2 to consume each committed stage-1 window while stage 1 "
        "continues with the next window; the measured row reports whether that "
        "overlap converts the smaller remaining stage-window count into latency."
        if catchup_policy == "window_streamed"
        else (
            "BP-free preserves prefix optimizer progress, but the current stage-major "
            "drain-first catch-up serializes stages 1 and 2. Exact BP replays more "
            "stage-window work while overlapping all stages with Schedule1F1B. This "
            "point demonstrates a progress/storage tradeoff, not a BP-free recovery "
            "latency win."
        )
    )
    return {
        "comparison": "E5 formal-v2 stage-1 controlled service outage",
        "shared_setup": setup,
        "model": {
            "resolved_model": bpfree["resolved_model"],
            "dtype": bpfree["dtype"],
            "learning_rate": bpfree["learning_rate"],
            "seed": bpfree["seed"],
            "training_config": bpfree["training_config"],
            "environment": bpfree["environment"],
        },
        "rows": rows,
        "derived": {
            "bpfree_recovery_time_over_exactbp": (
                bpfree_row["rejoin_to_terminal_target_ms"]
                / exact_row["rejoin_to_terminal_target_ms"]
            ),
            "bpfree_durable_bytes_over_exactbp": (
                bpfree_row["total_durable_file_bytes"]
                / exact_row["total_durable_file_bytes"]
            ),
            "bpfree_stage_window_reduction_fraction": 1.0
            - (
                bpfree_row["analytical_stage_windows_remaining"]
                / exact_row["analytical_stage_windows_remaining"]
            ),
        },
        "interpretation": interpretation,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    rows = payload["rows"]
    lines = [
        "# E5 formal-v2 paired result",
        "",
        (
            f"TinyLlama, 3 stages, failure stage 1, b={setup['physical_batch_size']}, "
            f"M={setup['microbatches_per_window']}, B={setup['effective_batch_size']}, "
            f"{setup['outage_windows']} outage windows."
        ),
        "",
        "| Method | Commits at rejoin (s0/s1/s2) | Recovery state | Remaining stage-windows | Rejoin to terminal (ms) | Checkpoint (MiB) | Boundary (MiB) |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        commits = "/".join(
            str(row[f"stage{stage}_commits_at_rejoin"])
            for stage in range(3)
        )
        lines.append(
            f"| {row['method']} | {commits} | {row['recovery_state']} | "
            f"{row['analytical_stage_windows_remaining']} | "
            f"{row['rejoin_to_terminal_target_ms']:.3f} | "
            f"{row['checkpoint_file_bytes'] / 2**20:.3f} | "
            f"{row['boundary_file_bytes'] / 2**20:.3f} |"
        )
    lines.extend(["", payload["interpretation"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the paired E5 formal-v2 table")
    parser.add_argument("--bpfree-summary", type=Path, required=True)
    parser.add_argument("--exactbp-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    payload = build_comparison(
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
