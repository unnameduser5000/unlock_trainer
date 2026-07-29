#!/usr/bin/env python3
"""Summarize quality and runtime metrics for one mobile-agent model run."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from evaluate_predictions import evaluate, read_jsonl


NUMERIC_FIELDS = [
    "ttft_ms",
    "e2e_ms",
    "output_tokens",
    "tokens_per_s",
    "process_rss_mb",
    "cuda_peak_alloc_mb",
]


def percentile(values: list[float], q: float) -> float:
    clean = sorted(value for value in values if value == value)
    if not clean:
        return 0.0
    if len(clean) == 1:
        return clean[0]
    pos = (len(clean) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(clean) - 1)
    frac = pos - lower
    return clean[lower] * (1.0 - frac) + clean[upper] * frac


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def summarize_runtime(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"rows": len(rows)}
    for field in NUMERIC_FIELDS:
        values = [as_float(row.get(field)) for row in rows]
        out[f"{field}_mean"] = mean(values) if values else 0.0
        out[f"{field}_p50"] = percentile(values, 0.50)
        out[f"{field}_p95"] = percentile(values, 0.95)
        out[f"{field}_max"] = max(values) if values else 0.0
    out["load_ms"] = max((as_float(row.get("load_ms_first_row")) for row in rows), default=0.0)
    out["parse_error_count"] = sum(1 for row in rows if str(row.get("parse_error", "")).strip())
    out["parse_error_rate"] = out["parse_error_count"] / len(rows) if rows else 0.0
    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        categories[str(row.get("category", "unknown"))].append(row)
    out["category_runtime"] = {
        category: {
            "rows": len(category_rows),
            "ttft_ms_mean": mean([as_float(row.get("ttft_ms")) for row in category_rows]),
            "e2e_ms_mean": mean([as_float(row.get("e2e_ms")) for row in category_rows]),
            "parse_error_rate": sum(1 for row in category_rows if str(row.get("parse_error", "")).strip()) / len(category_rows),
        }
        for category, category_rows in sorted(categories.items())
        if category_rows
    }
    return out


def write_summary_csv(path: Path, runtime_summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for field in NUMERIC_FIELDS:
        rows.append(
            {
                "metric": field,
                "mean": runtime_summary.get(f"{field}_mean", 0.0),
                "p50": runtime_summary.get(f"{field}_p50", 0.0),
                "p95": runtime_summary.get(f"{field}_p95", 0.0),
                "max": runtime_summary.get(f"{field}_max", 0.0),
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "mean", "p50", "p95", "max"])
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, quality: dict[str, Any], runtime: dict[str, Any]) -> None:
    lines = [
        "# Mobile Agent Tool Eval Run",
        "",
        "## Quality",
        "",
        f"- Rows: {quality['rows']}",
        f"- Route accuracy: {quality['route_accuracy']:.4f}",
        f"- Tool selection accuracy: {quality['tool_selection_accuracy']:.4f}",
        f"- Parameter exact match: {quality['parameter_exact_match']:.4f}",
        f"- Multi-action exact match: {quality['multi_action_exact_match']:.4f}",
        f"- No-tool rejection rate: {quality['no_tool_rejection_rate']:.4f}",
        f"- Cloud route recall: {quality['cloud_route_recall']:.4f}",
        f"- Clarify recall: {quality['clarify_recall']:.4f}",
        f"- Failure count: {quality['failure_count']}",
        "",
        "## Runtime",
        "",
        f"- Load time: {runtime['load_ms']:.1f} ms",
        f"- TTFT mean / p95: {runtime['ttft_ms_mean']:.1f} / {runtime['ttft_ms_p95']:.1f} ms",
        f"- E2E mean / p95: {runtime['e2e_ms_mean']:.1f} / {runtime['e2e_ms_p95']:.1f} ms",
        f"- Output tokens/s mean: {runtime['tokens_per_s_mean']:.2f}",
        f"- Process RSS max: {runtime['process_rss_mb_max']:.1f} MiB",
        f"- CUDA peak alloc max: {runtime['cuda_peak_alloc_mb_max']:.1f} MiB",
        f"- Parse error rate: {runtime['parse_error_rate']:.4f}",
        "",
        "## Category Route Accuracy",
        "",
    ]
    for category, value in quality["category_route_accuracy"].items():
        lines.append(f"- {category}: {value:.4f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--predicted_only", action="store_true", help="Score only gold rows whose ids appear in --pred.")
    args = parser.parse_args()

    quality = evaluate(read_jsonl(args.gold), read_jsonl(args.pred), predicted_only=args.predicted_only)
    runtime_summary = summarize_runtime(read_jsonl(args.runtime))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps({"quality": quality, "runtime": runtime_summary}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_summary_csv(args.output_dir / "runtime_summary.csv", runtime_summary)
    write_report(args.output_dir / "REPORT.md", quality, runtime_summary)
    print(f"Wrote {args.output_dir / 'metrics.json'}")
    print(f"Wrote {args.output_dir / 'runtime_summary.csv'}")
    print(f"Wrote {args.output_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
