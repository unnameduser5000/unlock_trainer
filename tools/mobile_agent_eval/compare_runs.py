#!/usr/bin/env python3
"""Compare multiple mobile-agent eval run directories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_metrics(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    quality = data["quality"]
    runtime = data["runtime"]
    return {
        "run": run_dir.name,
        "rows": quality["rows"],
        "route_accuracy": quality["route_accuracy"],
        "tool_selection_accuracy": quality["tool_selection_accuracy"],
        "parameter_exact_match": quality["parameter_exact_match"],
        "multi_action_exact_match": quality["multi_action_exact_match"],
        "no_tool_rejection_rate": quality["no_tool_rejection_rate"],
        "cloud_route_recall": quality["cloud_route_recall"],
        "clarify_recall": quality["clarify_recall"],
        "failure_count": quality["failure_count"],
        "load_ms": runtime["load_ms"],
        "ttft_ms_mean": runtime["ttft_ms_mean"],
        "ttft_ms_p95": runtime["ttft_ms_p95"],
        "e2e_ms_mean": runtime["e2e_ms_mean"],
        "e2e_ms_p95": runtime["e2e_ms_p95"],
        "tokens_per_s_mean": runtime["tokens_per_s_mean"],
        "cuda_peak_alloc_mb_max": runtime["cuda_peak_alloc_mb_max"],
        "parse_error_rate": runtime["parse_error_rate"],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, int):
        return str(value)
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = [
        "run",
        "rows",
        "route",
        "tool",
        "param",
        "ttft mean",
        "e2e mean",
        "cuda MB",
        "load ms",
    ]
    lines = [
        "# Mobile Agent Eval Comparison",
        "",
        "|" + "|".join(headers) + "|",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append(
            "|"
            + "|".join(
                [
                    str(row["run"]),
                    str(row["rows"]),
                    fmt(row["route_accuracy"]),
                    fmt(row["tool_selection_accuracy"]),
                    fmt(row["parameter_exact_match"]),
                    fmt(row["ttft_ms_mean"], 1),
                    fmt(row["e2e_ms_mean"], 1),
                    fmt(row["cuda_peak_alloc_mb_max"], 1),
                    fmt(row["load_ms"], 1),
                ]
            )
            + "|"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    rows = [read_metrics(path) for path in args.run_dirs]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "comparison.csv", rows)
    write_markdown(args.output_dir / "REPORT.md", rows)
    print(f"Wrote {args.output_dir / 'comparison.csv'}")
    print(f"Wrote {args.output_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
