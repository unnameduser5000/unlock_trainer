#!/usr/bin/env python3
"""Aggregate E4.4 synchronized action traces into comparable overhead buckets."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

WRAPPER_ACTIONS = {
    "PHASE_STEP_WINDOW",
    "BEGIN_WINDOW",
    "MAINTAIN_RECV_INFLIGHT",
}

CATEGORY_ORDER = (
    "input_h2d",
    "forward_compute",
    "backward_compute",
    "optimizer",
    "weight_stash",
    "gradient_accumulation",
    "transport_d2h",
    "transport_recv_post",
    "transport_recv_wait",
    "transport_recv_h2d",
    "transport_send_post_runtime",
    "link_pacing",
    "transport_send_wait",
    "control",
    "untraced_idle",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def train_phase(summary: dict[str, Any]) -> dict[str, Any]:
    for phase in summary.get("phases", []):
        if isinstance(phase, dict) and phase.get("phase") == "train":
            return phase
    raise ValueError(f"summary has no train phase: {summary.get('runner')}")


def is_pipedream(summary: dict[str, Any]) -> bool:
    return str(summary.get("runner", "")).startswith(
        "pipedream-continuous-1f1b"
    )


def training_metrics(summary: dict[str, Any]) -> dict[str, float | int]:
    if is_pipedream(summary):
        return {
            "completed_records": int(summary["completed_records"]),
            "optimizer_steps": int(summary["optimizer_steps_per_stage"]),
            "wall_ms": float(summary["wall_ms"]),
            "throughput_per_s": float(summary["throughput_per_s"]),
        }
    phase = train_phase(summary)
    return {
        "completed_records": int(phase["completed_records"]),
        "optimizer_steps": int(phase["optimizer_steps"]),
        "wall_ms": float(phase["wall_ms"]),
        "throughput_per_s": float(phase["throughput_per_s"]),
    }


def category_for(action: str) -> str:
    if action in {"LOAD_STAGE0_HIDDEN", "LOAD_COMMON_INPUTS", "INPUT_LOAD_H2D"}:
        return "input_h2d"
    if action.endswith("_RECV_H2D"):
        return "transport_recv_h2d"
    if action.endswith("_RECV_POST_CPU"):
        return "transport_recv_post"
    if action.endswith("_RECV_WAIT_CPU"):
        return "transport_recv_wait"
    if action.endswith("_D2H"):
        return "transport_d2h"
    if action.endswith("_SEND_POST_CPU"):
        return "transport_send_post_observed"
    if "SEND_WAIT" in action:
        return "transport_send_wait"
    if action in {
        "BODY_FORWARD",
        "LOCAL_HEAD_LOSS",
        "FWD_COMPUTE_INCLUDES_LOCAL_HEAD",
        "LOCAL_FORWARD_EXACT",
        "LOCAL_FORWARD_PIPEDREAM",
    }:
        return "forward_compute"
    if action in {
        "LOCAL_BACKWARD",
        "LOCAL_BACKWARD_EXACT_LOSS",
        "LOCAL_BACKWARD_EXACT_GRAD",
        "LOCAL_BACKWARD_PIPEDREAM_LOSS",
        "LOCAL_BACKWARD_PIPEDREAM_GRAD",
    }:
        return "backward_compute"
    if action in {
        "LOCAL_OPTIMIZER_STEP",
        "LOCAL_ZERO_GRAD_AFTER_STEP",
        "BEGIN_WINDOW_ZERO_GRAD",
        "ZERO_GRAD",
        "GRAD_CLIP",
        "OPTIMIZER_STEP",
        "OPTIMIZER_STEP_ASYNC",
    }:
        return "optimizer"
    if action == "WEIGHT_SNAPSHOT_CLONE":
        return "weight_stash"
    if action == "WEIGHT_GRAD_ACCUM":
        return "gradient_accumulation"
    return "control"


def union_duration(intervals: Iterable[tuple[float, float]]) -> float:
    ordered = sorted((float(a), float(b)) for a, b in intervals if float(b) >= float(a))
    if not ordered:
        return 0.0
    total = 0.0
    start, end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def parse_trace(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            action = str(raw.get("action", ""))
            if not action or action in WRAPPER_ACTIONS:
                continue
            window_raw = raw.get("window_id", raw.get("batch_seq", ""))
            if window_raw in (None, ""):
                window_raw = raw.get("batch_seq", 0)
            rows.append(
                {
                    "stage_id": int(raw["stage_id"]),
                    "window_id": int(window_raw),
                    "action": action,
                    "category": category_for(action),
                    "start_ms": float(raw["start_epoch_ms"]),
                    "end_ms": float(raw["end_epoch_ms"]),
                    "duration_ms": float(raw["duration_ms"]),
                }
            )
    return rows


def metadata_from_path(run_dir: Path, root: Path) -> tuple[str, str, str, int]:
    rel = run_dir.relative_to(root)
    parts = rel.parts
    if len(parts) < 4:
        raise ValueError(f"unexpected E4.4 run directory: {run_dir}")
    profile, regime, method, rep_name = parts[:4]
    rep = int(rep_name.removeprefix("rep_"))
    return profile, regime, method, rep


def link_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    if is_pipedream(summary):
        rows = [
            row.get("transport_budget")
            for row in summary.get("by_rank", [])
        ]
        return [row for row in rows if isinstance(row, dict)]
    rows = summary.get("transport_budget_by_rank")
    if not isinstance(rows, list):
        rows = train_phase(summary).get("transport_budget_by_rank")
    return rows if isinstance(rows, list) else []


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze(root: Path, output_dir: Path) -> None:
    action_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []

    for summary_path in sorted(root.glob("*/*/*/rep_*/summary.json")):
        run_dir = summary_path.parent
        profile, regime, method, rep = metadata_from_path(run_dir, root)
        summary = read_json(summary_path)
        metrics = training_metrics(summary)
        trace_paths = sorted(run_dir.glob("train.stage*.actions.csv"))
        if not trace_paths:
            raise FileNotFoundError(f"no action traces beside {summary_path}")

        all_events: list[dict[str, Any]] = []
        for trace_path in trace_paths:
            all_events.extend(parse_trace(trace_path))

        link_by_rank: dict[int, float] = {}
        for rank, row in enumerate(link_rows(summary)):
            if not isinstance(row, dict):
                continue
            link = row.get("link_emulation", {})
            if isinstance(link, dict):
                link_by_rank[rank] = float(link.get("injected_delay_ms", 0.0))

        per_stage: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for event in all_events:
            per_stage[event["stage_id"]].append(event)

        current_stage_rows: list[dict[str, Any]] = []
        for stage_id, events in sorted(per_stage.items()):
            start_ms = min(event["start_ms"] for event in events)
            end_ms = max(event["end_ms"] for event in events)
            span_ms = max(0.0, end_ms - start_ms)
            union_ms = union_duration((event["start_ms"], event["end_ms"]) for event in events)
            category_totals: dict[str, float] = defaultdict(float)
            action_totals: dict[tuple[str, str], list[float]] = defaultdict(list)
            for event in events:
                category_totals[event["category"]] += event["duration_ms"]
                action_totals[(event["category"], event["action"])].append(event["duration_ms"])

            observed_send_post = category_totals.pop("transport_send_post_observed", 0.0)
            pacing_ms = min(max(0.0, link_by_rank.get(stage_id, 0.0)), observed_send_post)
            category_totals["link_pacing"] = pacing_ms
            category_totals["transport_send_post_runtime"] = max(0.0, observed_send_post - pacing_ms)
            category_totals["untraced_idle"] = max(0.0, span_ms - union_ms)

            # Window-gap bounds exclude receive preposts and deferred send waits.
            # BP-free can prepost a future window early and drain old sends only at
            # phase end; including those actions would make window bounds overlap
            # for bookkeeping reasons rather than reflecting the compute/update
            # boundary. Exact-BP drain is still represented because OPTIMIZER_STEP
            # occurs after its per-window send drain.
            boundary_events = [
                event for event in events
                if event["category"] not in {
                    "transport_recv_post",
                    "transport_send_wait",
                    "control",
                }
            ]
            windows: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for event in boundary_events:
                windows[event["window_id"]].append(event)
            bounds = []
            for window_id, window_events in sorted(windows.items()):
                bounds.append(
                    (
                        window_id,
                        min(event["start_ms"] for event in window_events),
                        max(event["end_ms"] for event in window_events),
                    )
                )
            gaps = [max(0.0, bounds[i + 1][1] - bounds[i][2]) for i in range(len(bounds) - 1)]
            gap_row = {
                "profile": profile,
                "regime": regime,
                "method": method,
                "rep": rep,
                "stage_id": stage_id,
                "window_count": len(bounds),
                "gap_count": len(gaps),
                "mean_window_gap_ms": statistics.mean(gaps) if gaps else 0.0,
                "median_window_gap_ms": statistics.median(gaps) if gaps else 0.0,
                "p95_window_gap_ms": percentile(gaps, 0.95),
                "max_window_gap_ms": max(gaps) if gaps else 0.0,
            }
            gap_rows.append(gap_row)

            stage_row: dict[str, Any] = {
                "profile": profile,
                "regime": regime,
                "method": method,
                "rep": rep,
                "stage_id": stage_id,
                "trace_span_ms": span_ms,
                "traced_union_ms": union_ms,
                "wall_ms": float(metrics["wall_ms"]),
                "throughput_per_s": float(metrics["throughput_per_s"]),
                **{category: category_totals.get(category, 0.0) for category in CATEGORY_ORDER},
                **{key: value for key, value in gap_row.items() if key.endswith("window_gap_ms") or key == "window_count"},
            }
            stage_rows.append(stage_row)
            current_stage_rows.append(stage_row)

            for (category, action), durations in sorted(action_totals.items()):
                action_rows.append(
                    {
                        "profile": profile,
                        "regime": regime,
                        "method": method,
                        "rep": rep,
                        "stage_id": stage_id,
                        "category": category,
                        "action": action,
                        "count": len(durations),
                        "total_ms": sum(durations),
                        "mean_ms": statistics.mean(durations),
                        "p95_ms": percentile(durations, 0.95),
                    }
                )

        critical = max(current_stage_rows, key=lambda row: float(row["trace_span_ms"]))
        case_row: dict[str, Any] = {
            "profile": profile,
            "regime": regime,
            "method": method,
            "rep": rep,
            "pipeline_schedule": (
                "pipedream" if is_pipedream(summary)
                else summary.get("pipeline_schedule", "bpfree")
            ),
            "completed_records": int(metrics["completed_records"]),
            "optimizer_steps": int(metrics["optimizer_steps"]),
            "wall_ms": float(metrics["wall_ms"]),
            "throughput_per_s": float(metrics["throughput_per_s"]),
            "critical_stage_id": int(critical["stage_id"]),
            "critical_trace_span_ms": float(critical["trace_span_ms"]),
            "critical_mean_window_gap_ms": float(critical["mean_window_gap_ms"]),
            "critical_p95_window_gap_ms": float(critical["p95_window_gap_ms"]),
        }
        for category in CATEGORY_ORDER:
            value = float(critical.get(category, 0.0))
            case_row[f"critical_{category}_ms"] = value
            case_row[f"critical_{category}_percent"] = (
                100.0 * value / float(critical["trace_span_ms"])
                if float(critical["trace_span_ms"]) > 0 else 0.0
            )
        case_rows.append(case_row)

    if not case_rows:
        raise FileNotFoundError(f"no completed E4.4 runs under {root}")

    action_fields = [
        "profile", "regime", "method", "rep", "stage_id", "category", "action",
        "count", "total_ms", "mean_ms", "p95_ms",
    ]
    stage_fields = [
        "profile", "regime", "method", "rep", "stage_id", "trace_span_ms",
        "traced_union_ms", "wall_ms", "throughput_per_s", *CATEGORY_ORDER,
        "window_count", "mean_window_gap_ms", "median_window_gap_ms",
        "p95_window_gap_ms", "max_window_gap_ms",
    ]
    case_fields = [
        "profile", "regime", "method", "rep", "pipeline_schedule",
        "completed_records", "optimizer_steps", "wall_ms", "throughput_per_s",
        "critical_stage_id", "critical_trace_span_ms",
        "critical_mean_window_gap_ms", "critical_p95_window_gap_ms",
        *[f"critical_{category}_{suffix}" for category in CATEGORY_ORDER for suffix in ("ms", "percent")],
    ]
    gap_fields = [
        "profile", "regime", "method", "rep", "stage_id", "window_count", "gap_count",
        "mean_window_gap_ms", "median_window_gap_ms", "p95_window_gap_ms", "max_window_gap_ms",
    ]

    write_csv(output_dir / "action_summary.csv", action_fields, action_rows)
    write_csv(output_dir / "stage_breakdown.csv", stage_fields, stage_rows)
    write_csv(output_dir / "case_breakdown.csv", case_fields, case_rows)
    write_csv(output_dir / "window_gap_summary.csv", gap_fields, gap_rows)

    print("\n===== E4.4 critical-stage overhead decomposition =====")
    for row in case_rows:
        print(
            f"{row['profile']} {row['regime']} {row['method']}: "
            f"stage={row['critical_stage_id']} wall={row['wall_ms']:.1f}ms "
            f"fwd={row['critical_forward_compute_percent']:.1f}% "
            f"bwd={row['critical_backward_compute_percent']:.1f}% "
            f"recv_wait={row['critical_transport_recv_wait_percent']:.1f}% "
            f"pacing={row['critical_link_pacing_percent']:.1f}% "
            f"idle={row['critical_untraced_idle_percent']:.1f}% "
            f"gap_p95={row['critical_p95_window_gap_ms']:.3f}ms"
        )
    print(f"Wrote {output_dir / 'case_breakdown.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.root.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
