#!/usr/bin/env python3
"""Evaluate mobile tool-routing JSONL predictions against the seed set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def normalize_call(call: dict[str, Any]) -> dict[str, Any]:
    return {
        "tool": str(call.get("tool", "")),
        "args": call.get("args") or {},
    }


def normalize_expected(row: dict[str, Any]) -> dict[str, Any]:
    expected = row.get("expected") or {}
    return {
        "route": expected.get("route", ""),
        "calls": [normalize_call(call) for call in expected.get("calls") or []],
    }


def normalize_prediction(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id", ""),
        "route": row.get("route", ""),
        "calls": [normalize_call(call) for call in row.get("calls") or []],
    }


def tool_sequence(calls: list[dict[str, Any]]) -> list[str]:
    return [call["tool"] for call in calls]


def args_exact(expected: list[dict[str, Any]], predicted: list[dict[str, Any]]) -> bool:
    if len(expected) != len(predicted):
        return False
    for exp, pred in zip(expected, predicted):
        if exp["tool"] != pred["tool"]:
            return False
        if exp["args"] != pred["args"]:
            return False
    return True


def safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate(gold_rows: list[dict[str, Any]], pred_rows: list[dict[str, Any]], *, predicted_only: bool = False) -> dict[str, Any]:
    pred_by_id = {str(row.get("id", "")): normalize_prediction(row) for row in pred_rows}
    if predicted_only:
        gold_rows = [row for row in gold_rows if str(row.get("id", "")) in pred_by_id]
    totals = {
        "rows": len(gold_rows),
        "missing_predictions": 0,
        "route_correct": 0,
        "local_rows": 0,
        "tool_sequence_exact": 0,
        "parameter_exact": 0,
        "multi_action_rows": 0,
        "multi_action_exact": 0,
        "no_tool_rows": 0,
        "no_tool_reject_correct": 0,
        "cloud_rows": 0,
        "cloud_route_correct": 0,
        "clarify_rows": 0,
        "clarify_correct": 0,
    }

    category_totals: dict[str, dict[str, int]] = {}
    failures: list[dict[str, Any]] = []
    for gold in gold_rows:
        category = str(gold.get("category", "unknown"))
        category_stats = category_totals.setdefault(category, {"rows": 0, "route_correct": 0})
        category_stats["rows"] += 1

        expected = normalize_expected(gold)
        pred = pred_by_id.get(str(gold["id"]))
        if pred is None:
            totals["missing_predictions"] += 1
            failures.append({"id": gold["id"], "reason": "missing_prediction", "text": gold.get("text", "")})
            continue

        route_ok = expected["route"] == pred["route"]
        if route_ok:
            totals["route_correct"] += 1
            category_stats["route_correct"] += 1

        if expected["route"] == "local_tool":
            totals["local_rows"] += 1
            tool_ok = tool_sequence(expected["calls"]) == tool_sequence(pred["calls"])
            param_ok = args_exact(expected["calls"], pred["calls"])
            if tool_ok:
                totals["tool_sequence_exact"] += 1
            if param_ok:
                totals["parameter_exact"] += 1
            if len(expected["calls"]) > 1:
                totals["multi_action_rows"] += 1
                if param_ok:
                    totals["multi_action_exact"] += 1
            if not param_ok:
                failures.append(
                    {
                        "id": gold["id"],
                        "reason": "local_tool_or_param_mismatch",
                        "text": gold.get("text", ""),
                        "expected": expected,
                        "predicted": pred,
                    }
                )
        elif expected["route"] == "no_tool":
            totals["no_tool_rows"] += 1
            if pred["route"] == "no_tool":
                totals["no_tool_reject_correct"] += 1
            else:
                failures.append({"id": gold["id"], "reason": "no_tool_false_accept", "text": gold.get("text", ""), "predicted": pred})
        elif expected["route"] == "cloud":
            totals["cloud_rows"] += 1
            if pred["route"] == "cloud":
                totals["cloud_route_correct"] += 1
            else:
                failures.append({"id": gold["id"], "reason": "cloud_route_missed", "text": gold.get("text", ""), "predicted": pred})
        elif expected["route"] == "clarify":
            totals["clarify_rows"] += 1
            if pred["route"] == "clarify":
                totals["clarify_correct"] += 1
            else:
                failures.append({"id": gold["id"], "reason": "clarify_missed", "text": gold.get("text", ""), "predicted": pred})

    metrics = {
        "rows": totals["rows"],
        "missing_predictions": totals["missing_predictions"],
        "route_accuracy": safe_rate(totals["route_correct"], totals["rows"]),
        "tool_selection_accuracy": safe_rate(totals["tool_sequence_exact"], totals["local_rows"]),
        "parameter_exact_match": safe_rate(totals["parameter_exact"], totals["local_rows"]),
        "multi_action_exact_match": safe_rate(totals["multi_action_exact"], totals["multi_action_rows"]),
        "no_tool_rejection_rate": safe_rate(totals["no_tool_reject_correct"], totals["no_tool_rows"]),
        "cloud_route_recall": safe_rate(totals["cloud_route_correct"], totals["cloud_rows"]),
        "clarify_recall": safe_rate(totals["clarify_correct"], totals["clarify_rows"]),
        "category_route_accuracy": {
            category: safe_rate(stats["route_correct"], stats["rows"])
            for category, stats in sorted(category_totals.items())
        },
        "failure_count": len(failures),
        "failures_preview": failures[:25],
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--predicted_only", action="store_true", help="Score only gold rows whose ids appear in --pred.")
    args = parser.parse_args()

    metrics = evaluate(read_jsonl(args.gold), read_jsonl(args.pred), predicted_only=args.predicted_only)
    text = json.dumps(metrics, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")
    print(text)


if __name__ == "__main__":
    main()
