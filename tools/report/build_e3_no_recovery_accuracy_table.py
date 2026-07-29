#!/usr/bin/env python3
"""Build a compact E3 no-recovery accuracy table.

Input:
  debug_runs/graceful_degradation_train512_eval256_3seeds/report/quality_per_seed.csv

Output:
  report/no_recovery_accuracy_summary.csv

This script is intentionally report-only. It does not rerun training and does
not modify any experiment result files except the derived summary table.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def as_float(row: dict[str, str], *names: str) -> float:
    for name in names:
        value = row.get(name, "")
        if value not in {"", None}:
            return float(value)
    raise KeyError(f"None of {names} present in row: {row}")


def as_int(row: dict[str, str], *names: str) -> int:
    for name in names:
        value = row.get(name, "")
        if value not in {"", None}:
            return int(float(value))
    raise KeyError(f"None of {names} present in row: {row}")


def norm_method(raw: str) -> str:
    x = raw.strip().lower().replace("_", "-").replace(" ", "-")
    if "bpfree" in x or "bp-free" in x:
        return "bpfree"
    if "1f1b" in x or "f1b" in x:
        return "1f1b"
    return x


def norm_case(raw: str) -> str:
    x = raw.strip().lower().replace("_", "-").replace(" ", "-")
    if "fault" in x and "free" in x:
        return "fault_free"
    if "offline" in x or "skip" in x:
        return "offline_skip"
    return x


def sample_std(vals: list[float]) -> float:
    if len(vals) <= 1:
        return 0.0
    return stdev(vals)


def fmt(x: float) -> str:
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return ""
    return f"{x:.10g}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("debug_runs/graceful_degradation_train512_eval256_3seeds"),
    )
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report_dir = args.root / "report"
    input_path = args.input or report_dir / "quality_per_seed.csv"
    output_path = args.output or report_dir / "no_recovery_accuracy_summary.csv"

    if not input_path.exists():
        raise FileNotFoundError(f"Missing input CSV: {input_path}")

    rows = read_csv(input_path)
    if not rows:
        raise RuntimeError(f"No rows in {input_path}")

    by_method_case_seed: dict[tuple[str, str, int], dict[str, Any]] = {}

    for row in rows:
        method = norm_method(row.get("method", row.get("runner", "")))
        case = norm_case(row.get("case", row.get("policy", "")))
        seed = as_int(row, "seed")
        acc = as_float(row, "eval_accuracy", "choice_accuracy", "accuracy")
        loss = None
        try:
            loss = as_float(row, "eval_loss", "avg_loss", "loss")
        except Exception:
            pass

        by_method_case_seed[(method, case, seed)] = {
            "method": method,
            "case": case,
            "seed": seed,
            "eval_accuracy": acc,
            "eval_loss": loss,
        }

    methods = sorted({key[0] for key in by_method_case_seed})
    out: list[dict[str, Any]] = []

    for method in methods:
        ff = {
            seed: item
            for (m, case, seed), item in by_method_case_seed.items()
            if m == method and case == "fault_free"
        }
        off = {
            seed: item
            for (m, case, seed), item in by_method_case_seed.items()
            if m == method and case == "offline_skip"
        }

        ff_acc = [item["eval_accuracy"] for _, item in sorted(ff.items())]
        off_acc = [item["eval_accuracy"] for _, item in sorted(off.items())]

        paired_seeds = sorted(set(ff) & set(off))
        deltas = [
            off[seed]["eval_accuracy"] - ff[seed]["eval_accuracy"]
            for seed in paired_seeds
        ]

        row = {
            "method": method,
            "fault_free_n": len(ff_acc),
            "offline_skip_n": len(off_acc),
            "paired_n": len(deltas),
            "fault_free_eval_accuracy_mean": fmt(mean(ff_acc)) if ff_acc else "",
            "fault_free_eval_accuracy_std": fmt(sample_std(ff_acc)) if ff_acc else "",
            "offline_skip_eval_accuracy_mean": fmt(mean(off_acc)) if off_acc else "",
            "offline_skip_eval_accuracy_std": fmt(sample_std(off_acc)) if off_acc else "",
            "offline_minus_fault_free_eval_accuracy_mean": fmt(mean(deltas)) if deltas else "",
            "offline_minus_fault_free_eval_accuracy_std": fmt(sample_std(deltas)) if deltas else "",
            "paired_seeds": ",".join(str(seed) for seed in paired_seeds),
        }
        out.append(row)

    write_csv(output_path, out)

    print(f"Wrote {output_path}")
    for row in out:
        print(row)


if __name__ == "__main__":
    main()
