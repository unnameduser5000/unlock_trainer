#!/usr/bin/env python3
"""Normalize E1/E2 raw outputs into paper-facing CSV tables.

The individual runners intentionally retain their native detailed logs. This
script produces a compact, shared schema without discarding those raw files.

For E2 throughput, this script is intentionally conservative: full-run
throughput is always reported when present, while steady-state/fill/drain fields
are populated only when a runner summary or timeline has enough information.
The shared fill/drain field names are method-aligned, not mechanism-identical:
1F1B uses pipeline fill/drain semantics, while BP-free uses update-window
start/tail semantics. Missing phase fields mean the run cannot support a final
steady-state throughput claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    return float(value)


def optional_float(value: Any) -> float | str:
    if value in (None, ""):
        return ""
    return float(value)


def max_optional(rows: list[dict[str, str]], column: str) -> float | str:
    values = [to_float(row[column]) for row in rows if row.get(column) not in (None, "")]
    return max(values) if values else ""


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def phase(summary: dict[str, Any], names: list[str]) -> dict[str, Any]:
    for item in summary.get("phases", []) + summary.get("phase_summaries", []):
        if item.get("phase") in names:
            return item
    return {}


def run_dirs(root: Path) -> list[tuple[str, str, int, Path]]:
    found: list[tuple[str, str, int, Path]] = []
    for suite_dir in sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith("e")):
        for method_dir in sorted(path for path in suite_dir.iterdir() if path.is_dir()):
            for seed_dir in sorted(path for path in method_dir.iterdir() if path.is_dir() and path.name.startswith("seed")):
                try:
                    seed = int(seed_dir.name.removeprefix("seed"))
                except ValueError:
                    continue
                if (seed_dir / "summary.json").is_file() or (seed_dir / "scheduler_summary.json").is_file():
                    found.append((suite_dir.name, method_dir.name, seed, seed_dir))
    return found


def native_summary(run_dir: Path) -> tuple[str, dict[str, Any]]:
    scheduler = run_dir / "scheduler_summary.json"
    if scheduler.is_file():
        return "bpfree", read_json(scheduler)
    summary = read_json(run_dir / "summary.json")
    if summary.get("runner") == "torch.distributed-sendrecv":
        return "bpfree_p2p", summary
    return "full_bp", summary


def timeline_phase_metrics(run_dir: Path, summary: dict[str, Any], family: str, train: dict[str, Any]) -> dict[str, Any]:
    """Return phase-split throughput fields when defensible.

    Existing historical runs often have only full-run wall time. For those, the
    steady-state fields stay blank so the registry/audit can prevent accidental
    promotion. If future runners write explicit phase metrics into summary.json,
    they are passed through here.
    """

    explicit = summary.get("pipeline_phase_metrics") or train.get("pipeline_phase_metrics")
    if isinstance(explicit, dict):
        return {
            "full_run_throughput_per_s": optional_float(
                explicit.get("full_run_throughput_per_s", train.get("throughput_per_s", ""))
            ),
            "steady_state_throughput_per_s": optional_float(explicit.get("steady_state_throughput_per_s")),
            "warmup_or_fill_ms": optional_float(explicit.get("warmup_or_fill_ms")),
            "drain_ms": optional_float(explicit.get("drain_ms")),
            "fill_drain_overhead_ms": optional_float(explicit.get("fill_drain_overhead_ms")),
            "phase_metrics_status": explicit.get("status", "explicit"),
            "phase_metrics_source": explicit.get("source", "summary.pipeline_phase_metrics"),
            "phase_semantics": explicit.get("phase_semantics", summary.get("pipeline_phase_semantics", "")),
            "phase_alignment_note": explicit.get(
                "phase_alignment_note",
                summary.get("pipeline_phase_alignment_note", ""),
            ),
        }

    full_run = train.get("throughput_per_s", "")
    status = "missing_phase_split"
    source = ""

    if family == "full_bp" and (run_dir / "timeline_events.csv").is_file():
        # 1F1B timeline events are available, but historical summaries do not
        # yet define a canonical trim rule. Keep steady-state blank until the
        # clean E2 protocol defines and writes the policy.
        status = "timeline_available_no_trim_policy"
        source = str(run_dir / "timeline_events.csv")
    elif family == "bpfree_p2p" and list(run_dir.glob("train.stage*.metrics.csv")):
        status = "stage_metrics_available_no_phase_policy"
        source = "train.stage*.metrics.csv"
    elif family == "bpfree":
        status = "scheduler_runtime_full_run_only"
        source = "scheduler_summary.json"

    return {
        "full_run_throughput_per_s": optional_float(full_run),
        "steady_state_throughput_per_s": "",
        "warmup_or_fill_ms": "",
        "drain_ms": "",
        "fill_drain_overhead_ms": "",
        "phase_metrics_status": status,
        "phase_metrics_source": source,
        "phase_semantics": summary.get("pipeline_phase_semantics", ""),
        "phase_alignment_note": summary.get("pipeline_phase_alignment_note", ""),
    }


def summary_row(suite: str, method: str, seed: int, run_dir: Path, family: str, summary: dict[str, Any]) -> dict[str, Any]:
    train = phase(summary, ["train"])
    final_eval = phase(summary, ["eval_after", "eval"])
    lora = summary.get("lora", {})
    fingerprint: Any = lora.get("initialization_fingerprint", "")
    if not fingerprint:
        values = lora.get("initialization_fingerprints", [])
        fingerprint = ";".join(values) if isinstance(values, list) else values
    phase_metrics = timeline_phase_metrics(run_dir, summary, family, train)
    return {
        "suite": suite,
        "method": method,
        "family": family,
        "seed": seed,
        "run_dir": str(run_dir),
        "train_samples": train.get("completed_records", train.get("completed", train.get("rows", ""))),
        "optimizer_steps": train.get("optimizer_steps", ""),
        "train_wall_ms": train.get("wall_ms", ""),
        "train_throughput_per_s": train.get("throughput_per_s", ""),
        "full_run_throughput_per_s": phase_metrics["full_run_throughput_per_s"],
        "steady_state_throughput_per_s": phase_metrics["steady_state_throughput_per_s"],
        "warmup_or_fill_ms": phase_metrics["warmup_or_fill_ms"],
        "drain_ms": phase_metrics["drain_ms"],
        "fill_drain_overhead_ms": phase_metrics["fill_drain_overhead_ms"],
        "phase_metrics_status": phase_metrics["phase_metrics_status"],
        "phase_metrics_source": phase_metrics["phase_metrics_source"],
        "phase_semantics": phase_metrics["phase_semantics"],
        "phase_alignment_note": phase_metrics["phase_alignment_note"],
        "train_loss": train.get("avg_loss", ""),
        "final_eval_records": final_eval.get("completed_records", final_eval.get("completed", final_eval.get("records", ""))),
        "final_choice_accuracy": final_eval.get("choice_accuracy", ""),
        "final_choice_loss": final_eval.get("avg_loss", ""),
        "final_eval_wall_ms": final_eval.get("wall_ms", ""),
        "dtype": summary.get("dtype", ""),
        "transport": summary.get("transport", ""),
        "learning_rate": summary.get("learning_rate", ""),
        "batch_size": summary.get("batch_size", ""),
        "physical_request_batch": summary.get("physical_request_batch", ""),
        "effective_optimizer_batch": summary.get("effective_optimizer_batch", ""),
        "microbatches": summary.get("microbatches", ""),
        "max_inflight": summary.get("max_inflight", ""),
        "gradient_accumulation_steps": summary.get("gradient_accumulation_steps", ""),
        "belief_transport_mode": summary.get("belief_transport_mode", ""),
        "belief_alpha": summary.get("alpha", ""),
        "activation_tracking_enabled": summary.get("activation_tracking_enabled", ""),
        "gc_interval_batches": summary.get("gc_interval_batches", ""),
        "lora_init_seed": lora.get("init_seed", ""),
        "lora_initialization_fingerprint": fingerprint,
        "validation_curve_csv": summary.get("validation_curve_csv", ""),
    }


def stage_peak_rows(suite: str, method: str, seed: int, run_dir: Path, family: str) -> list[dict[str, Any]]:
    by_stage: dict[str, list[dict[str, str]]] = {}
    if family == "bpfree":
        sources = [run_dir / "scheduler_stage_metrics.csv"]
    elif family == "bpfree_p2p":
        sources = sorted(run_dir.glob("train.stage*.metrics.csv"))
    else:
        sources = [run_dir / "stage_metrics.csv"]
    for source in sources:
        for row in read_csv(source):
            phase_name = row.get("phase", "")
            if not (phase_name == "train" or phase_name.startswith("train_to_")):
                continue
            by_stage.setdefault(row.get("stage_id", ""), []).append(row)
    out: list[dict[str, Any]] = []
    for stage_id, rows in sorted(by_stage.items()):
        out.append(
            {
                "suite": suite,
                "method": method,
                "family": family,
                "seed": seed,
                "stage_id": stage_id,
                "rows": len(rows),
                "cuda_peak_allocated_bytes": max_optional(rows, "cuda_peak_memory_allocated"),
                "cuda_peak_reserved_bytes": max_optional(rows, "cuda_peak_memory_reserved"),
                "local_param_bytes": max_optional(rows, "local_param_bytes"),
                "local_trainable_param_bytes": max_optional(rows, "local_trainable_param_bytes"),
                "resident_model_param_bytes": max_optional(rows, "resident_model_param_bytes"),
                "resident_frozen_param_bytes": max_optional(rows, "resident_frozen_param_bytes"),
                "base_shard_param_bytes": max_optional(rows, "base_shard_param_bytes"),
                "base_shard_trainable_param_bytes": max_optional(rows, "base_shard_trainable_param_bytes"),
                "local_readout_param_bytes": max_optional(rows, "local_readout_param_bytes"),
                "local_readout_trainable_param_bytes": max_optional(rows, "local_readout_trainable_param_bytes"),
                "input_embedding_param_bytes": max_optional(rows, "input_embedding_param_bytes"),
                "input_embedding_trainable_param_bytes": max_optional(
                    rows,
                    "input_embedding_trainable_param_bytes",
                ),
                "saved_nonleaf_activation_peak_bytes": max_optional(
                    rows, "autograd_saved_cuda_nonleaf_unique_bytes_peak"
                ),
                "gradient_storage_peak_bytes": max_optional(rows, "gradient_storage_bytes"),
                "optimizer_state_peak_bytes": max_optional(rows, "optimizer_state_bytes"),
                "identified_allocated_peak_bytes": max_optional(rows, "identified_allocated_bytes"),
                "runtime_residual_peak_bytes": max_optional(rows, "runtime_residual_bytes"),
                "output_hidden_peak_bytes": max_optional(rows, "output_hidden_bytes"),
                "output_log_probs_peak_bytes": max_optional(rows, "output_log_probs_bytes"),
                "mean_step_or_execute_ms": statistics.mean(
                    to_float(row.get("execute_ms", row.get("step_ms", 0.0))) for row in rows
                ),
            }
        )
    return out


def quality_rows(suite: str, method: str, seed: int, run_dir: Path) -> list[dict[str, Any]]:
    out = []
    for row in read_csv(run_dir / "validation_curve.csv"):
        out.append(
            {
                "suite": suite,
                "method": method,
                "seed": seed,
                "optimizer_step": int(float(row["optimizer_step"])),
                "train_samples_seen": int(float(row["train_samples_seen"])),
                "choice_accuracy": to_float(row.get("choice_accuracy")),
                "choice_loss": to_float(row.get("avg_loss")),
                "validation_records": int(float(row.get("validation_records") or 0)),
            }
        )
    return out


def aggregate(rows: list[dict[str, Any]], keys: list[str], metrics: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    out: list[dict[str, Any]] = []
    for group_key, group_rows in sorted(groups.items()):
        row = dict(zip(keys, group_key))
        row["runs"] = len(group_rows)
        for metric in metrics:
            values = [to_float(item.get(metric)) for item in group_rows if item.get(metric) not in (None, "")]
            if values:
                row[f"{metric}_mean"] = statistics.mean(values)
                row[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate formal AG News E1/E2 logs.")
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--report_dir", type=Path, default=None)
    args = parser.parse_args()
    report_dir = args.report_dir or args.output_root / "report"

    runs: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    stage_peaks: list[dict[str, Any]] = []
    for suite, method, seed, run_dir in run_dirs(args.output_root):
        family, summary = native_summary(run_dir)
        runs.append(summary_row(suite, method, seed, run_dir, family, summary))
        curves.extend(quality_rows(suite, method, seed, run_dir))
        stage_peaks.extend(stage_peak_rows(suite, method, seed, run_dir, family))

    write_csv(report_dir / "normalized_runs.csv", runs)
    write_csv(report_dir / "quality_curve_raw.csv", curves)
    write_csv(
        report_dir / "quality_curve_summary.csv",
        aggregate(curves, ["suite", "method", "optimizer_step", "train_samples_seen"], ["choice_accuracy", "choice_loss"]),
    )
    write_csv(report_dir / "stage_peak_metrics.csv", stage_peaks)
    write_csv(
        report_dir / "method_summary.csv",
        aggregate(
            runs,
            ["suite", "method", "family"],
            [
                "final_choice_accuracy",
                "final_choice_loss",
                "train_throughput_per_s",
                "train_wall_ms",
                "full_run_throughput_per_s",
                "steady_state_throughput_per_s",
                "warmup_or_fill_ms",
                "drain_ms",
                "fill_drain_overhead_ms",
            ],
        ),
    )
    print(json.dumps({"runs": len(runs), "quality_points": len(curves), "report_dir": str(report_dir)}, indent=2))


if __name__ == "__main__":
    main()
