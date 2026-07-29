#!/usr/bin/env python3
"""Build E2 GPU-memory tables, figures, and a numeric audit."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np


MIB = 1024.0 * 1024.0
METHOD_ORDER = ("bpfree", "exactbp_1f1b", "exactbp_gpipe", "pipedream")
METHOD_LABELS = {
    "bpfree": "BP-free",
    "exactbp_1f1b": "Sync 1F1B",
    "exactbp_gpipe": "GPipe",
    "pipedream": "PipeDream",
}
METHOD_COLORS = {
    "bpfree": "#2F855A",
    "exactbp_1f1b": "#D97706",
    "exactbp_gpipe": "#2563A6",
    "pipedream": "#B83250",
}
GEOMETRY_ORDER = ("b1_m32", "b2_m16", "b4_m8", "b8_m4")
BYTE_FIELDS = (
    "resident_model_param_bytes",
    "resident_frozen_param_bytes",
    "base_shard_param_bytes",
    "base_shard_trainable_param_bytes",
    "local_readout_param_bytes",
    "local_readout_trainable_param_bytes",
    "input_embedding_param_bytes",
    "input_embedding_trainable_param_bytes",
    "baseline_cuda_allocated_bytes",
    "baseline_cuda_reserved_bytes",
    "pre_schedule_cuda_allocated_bytes",
    "peak_cuda_allocated_bytes",
    "peak_cuda_reserved_bytes",
    "peak_runtime_delta_bytes",
    "gradient_storage_bytes",
    "optimizer_state_bytes",
    "saved_nonleaf_unique_peak_bytes",
    "saved_leaf_unique_peak_bytes",
    "peak_activation_cache_bytes",
    "peak_weight_stash_bytes",
    "peak_master_gradient_bytes",
    "peak_snapshot_gradient_bytes",
    "host_peak_pending_send_bytes",
    "host_peak_posted_recv_bytes",
    "output_hidden_payload_bytes",
    "output_log_probs_payload_bytes",
)


def number(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def mean_std(values: Iterable[float]) -> tuple[float, float]:
    items = list(values)
    if not items:
        return 0.0, 0.0
    return (
        statistics.mean(items),
        statistics.stdev(items) if len(items) > 1 else 0.0,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = {key for row in rows for key in row}
    preferred = [
        "protocol_version",
        "geometry",
        "physical_batch_size",
        "microbatches",
        "effective_batch",
        "train_windows",
        "method",
        "rep",
        "stage_id",
        "device",
    ]
    ordered = [field for field in preferred if field in fields]
    ordered.extend(sorted(field for field in fields if field not in ordered))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def base_row(
    metadata: dict[str, Any],
    stage_id: int,
    device: str,
) -> dict[str, Any]:
    geometry = metadata["geometry"]
    return {
        "protocol_version": metadata["protocol_version"],
        "geometry": geometry["id"],
        "physical_batch_size": int(geometry["physical_batch_size"]),
        "microbatches": int(geometry["microbatches"]),
        "effective_batch": int(metadata["effective_batch"]),
        "train_windows": int(metadata["train_windows"]),
        "method": metadata["method"],
        "rep": int(metadata["rep"]),
        "stage_id": stage_id,
        "device": device,
    }


def stage_count(metadata: dict[str, Any]) -> int:
    devices = [
        item.strip()
        for item in str(metadata.get("stage_devices", "")).split(",")
        if item.strip()
    ]
    return len(devices) if devices else 3


def transport_fields(transport: dict[str, Any]) -> dict[str, int]:
    return {
        "host_peak_pending_send_bytes": number(
            transport.get("peak_pending_send_bytes")
        ),
        "host_peak_posted_recv_bytes": number(
            transport.get("peak_posted_recv_bytes")
        ),
    }


def ledger_fields(ledger: dict[str, Any]) -> dict[str, int]:
    return {
        key: number(ledger.get(key))
        for key in BYTE_FIELDS
        if key in ledger
    }


def exact_rows(
    run_dir: Path,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    output = []
    for stage in range(stage_count(metadata)):
        summary = json.loads(
            (run_dir / f"rank{stage}.summary.json").read_text(encoding="utf-8")
        )
        aggregate = summary["train"]["memory_profile"]["aggregate"]
        row = base_row(metadata, stage, f"cuda:{stage}")
        row.update(ledger_fields(summary["memory_ledger"]))
        row.update(transport_fields(summary["transport_budget"]))
        row.update(
            {
                "baseline_cuda_allocated_bytes": number(
                    aggregate.get("baseline_cuda_allocated_bytes")
                ),
                "baseline_cuda_reserved_bytes": number(
                    aggregate.get("baseline_cuda_reserved_bytes")
                ),
                "pre_schedule_cuda_allocated_bytes": number(
                    aggregate.get("pre_schedule_cuda_allocated_bytes")
                ),
                "peak_cuda_allocated_bytes": number(
                    aggregate.get("peak_cuda_allocated_bytes")
                ),
                "peak_cuda_reserved_bytes": number(
                    aggregate.get("peak_cuda_reserved_bytes")
                ),
                "peak_runtime_delta_bytes": number(
                    aggregate.get("peak_runtime_delta_bytes")
                ),
                "gradient_storage_bytes": number(
                    aggregate.get("gradient_storage_bytes")
                ),
                "optimizer_state_bytes": number(
                    aggregate.get("optimizer_state_bytes")
                ),
                "saved_nonleaf_unique_peak_bytes": number(
                    aggregate.get(
                        "autograd_saved_cuda_nonleaf_unique_bytes_peak"
                    )
                ),
                "saved_leaf_unique_peak_bytes": number(
                    aggregate.get("autograd_saved_cuda_leaf_unique_bytes_peak")
                ),
                "peak_activation_cache_entries": number(
                    aggregate.get("peak_activation_cache_entries")
                ),
                "peak_activation_cache_bytes": number(
                    aggregate.get("peak_activation_cache_bytes")
                ),
                "peak_weight_stash_bytes": 0,
                "peak_live_weight_versions": 0,
                "peak_master_gradient_bytes": number(
                    aggregate.get("gradient_storage_bytes")
                ),
                "peak_snapshot_gradient_bytes": 0,
                "output_hidden_payload_bytes": 0,
                "output_log_probs_payload_bytes": 0,
            }
        )
        output.append(row)
    return output


def pipedream_rows(
    run_dir: Path,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    output = []
    for stage in range(stage_count(metadata)):
        summary = json.loads(
            (run_dir / f"rank{stage}.summary.json").read_text(encoding="utf-8")
        )
        activation = summary.get("activation_memory", {})
        row = base_row(
            metadata,
            stage,
            str(summary.get("device", f"cuda:{stage}")),
        )
        row.update(ledger_fields(summary["memory_ledger"]))
        row.update(transport_fields(summary["transport_budget"]))
        row.update(
            {
                "baseline_cuda_allocated_bytes": number(
                    summary.get("baseline_cuda_allocated_bytes")
                ),
                "baseline_cuda_reserved_bytes": number(
                    summary.get("baseline_cuda_reserved_bytes")
                ),
                "pre_schedule_cuda_allocated_bytes": 0,
                "peak_cuda_allocated_bytes": number(
                    summary.get("peak_cuda_allocated_bytes")
                ),
                "peak_cuda_reserved_bytes": number(
                    summary.get("peak_cuda_reserved_bytes")
                ),
                "peak_runtime_delta_bytes": number(
                    summary.get("peak_runtime_delta_bytes")
                ),
                "gradient_storage_bytes": number(
                    summary.get("gradient_storage_bytes")
                ),
                "optimizer_state_bytes": number(
                    summary.get("optimizer_state_bytes")
                ),
                "saved_nonleaf_unique_peak_bytes": number(
                    activation.get(
                        "autograd_saved_cuda_nonleaf_unique_bytes_peak"
                    )
                ),
                "saved_leaf_unique_peak_bytes": number(
                    activation.get(
                        "autograd_saved_cuda_leaf_unique_bytes_peak"
                    )
                ),
                "peak_activation_cache_entries": number(
                    summary.get("peak_activation_cache_entries")
                ),
                "peak_activation_cache_bytes": number(
                    summary.get("peak_activation_cache_bytes")
                ),
                "peak_weight_stash_bytes": number(
                    summary.get("peak_weight_stash_bytes")
                ),
                "peak_live_weight_versions": number(
                    summary.get("peak_live_weight_versions")
                ),
                "peak_master_gradient_bytes": number(
                    summary.get("peak_master_gradient_bytes")
                ),
                "peak_snapshot_gradient_bytes": number(
                    summary.get("peak_snapshot_gradient_bytes")
                ),
                "output_hidden_payload_bytes": 0,
                "output_log_probs_payload_bytes": 0,
            }
        )
        output.append(row)
    return output


def max_csv(rows: list[dict[str, str]], key: str) -> int:
    return max((number(row.get(key)) for row in rows), default=0)


def bpfree_rows(
    run_dir: Path,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    summary = json.loads(
        (run_dir / "summary.json").read_text(encoding="utf-8")
    )
    train = next(item for item in summary["phases"] if item["phase"] == "train")
    transports = train["transport_budget_by_rank"]
    output = []
    for stage in range(stage_count(metadata)):
        metrics_path = run_dir / f"train.stage{stage}.metrics.csv"
        with metrics_path.open(newline="", encoding="utf-8") as handle:
            metrics = list(csv.DictReader(handle))
        if not metrics:
            raise ValueError(f"empty BP-free memory metrics: {metrics_path}")
        row = base_row(metadata, stage, str(metrics[0]["device"]))
        for key in (
            "resident_model_param_bytes",
            "resident_frozen_param_bytes",
            "base_shard_param_bytes",
            "base_shard_trainable_param_bytes",
            "local_readout_param_bytes",
            "local_readout_trainable_param_bytes",
            "input_embedding_param_bytes",
            "input_embedding_trainable_param_bytes",
        ):
            row[key] = max_csv(metrics, key)
        row.update(transport_fields(transports[stage]))
        row.update(
            {
                "baseline_cuda_allocated_bytes": 0,
                "baseline_cuda_reserved_bytes": 0,
                "pre_schedule_cuda_allocated_bytes": 0,
                "peak_cuda_allocated_bytes": max_csv(
                    metrics, "cuda_peak_memory_allocated"
                ),
                "peak_cuda_reserved_bytes": max_csv(
                    metrics, "cuda_peak_memory_reserved"
                ),
                "peak_runtime_delta_bytes": 0,
                "gradient_storage_bytes": max_csv(
                    metrics, "gradient_storage_bytes"
                ),
                "optimizer_state_bytes": max_csv(
                    metrics, "optimizer_state_bytes"
                ),
                "saved_nonleaf_unique_peak_bytes": max_csv(
                    metrics,
                    "autograd_saved_cuda_nonleaf_unique_bytes_peak",
                ),
                "saved_leaf_unique_peak_bytes": max_csv(
                    metrics,
                    "autograd_saved_cuda_leaf_unique_bytes_peak",
                ),
                "peak_activation_cache_entries": 1,
                "peak_activation_cache_bytes": 0,
                "peak_weight_stash_bytes": 0,
                "peak_live_weight_versions": 0,
                "peak_master_gradient_bytes": max_csv(
                    metrics, "gradient_storage_bytes"
                ),
                "peak_snapshot_gradient_bytes": 0,
                "output_hidden_payload_bytes": max_csv(
                    metrics, "output_hidden_bytes"
                ),
                "output_log_probs_payload_bytes": max_csv(
                    metrics, "output_log_probs_bytes"
                ),
            }
        )
        output.append(row)
    return output


def discover(raw_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metadata_path in sorted(
        raw_root.glob("*/*/rep_*/run_metadata.json")
    ):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") != "complete":
            continue
        run_dir = metadata_path.parent
        method = metadata["method"]
        if method in {"exactbp_1f1b", "exactbp_gpipe"}:
            rows.extend(exact_rows(run_dir, metadata))
        elif method == "pipedream":
            rows.extend(pipedream_rows(run_dir, metadata))
        elif method == "bpfree":
            rows.extend(bpfree_rows(run_dir, metadata))
        else:
            raise ValueError(f"unknown E2 method in {metadata_path}: {method}")
    return rows


def add_mib_columns(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        for key in BYTE_FIELDS:
            row[f"{key[:-6]}_mib"] = number(row.get(key)) / MIB


def max_stage_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (str(row["geometry"]), str(row["method"]), int(row["rep"]))
        ].append(row)
    output = []
    for (geometry, method, rep), group in sorted(grouped.items()):
        peak = max(group, key=lambda item: item["peak_cuda_allocated_bytes"])
        output.append(
            {
                "protocol_version": peak["protocol_version"],
                "geometry": geometry,
                "method": method,
                "rep": rep,
                "physical_batch_size": peak["physical_batch_size"],
                "microbatches": peak["microbatches"],
                "effective_batch": peak["effective_batch"],
                "train_windows": peak["train_windows"],
                "max_stage_id": peak["stage_id"],
                "max_stage_device": peak["device"],
                "max_stage_peak_cuda_allocated_bytes": peak[
                    "peak_cuda_allocated_bytes"
                ],
                "max_stage_peak_cuda_allocated_mib": (
                    peak["peak_cuda_allocated_bytes"] / MIB
                ),
            }
        )
    return output


def component_rows(
    rows: list[dict[str, Any]],
    primary_geometry: str,
) -> list[dict[str, Any]]:
    fields = {
        "resident_model_mib": "resident_model_param_bytes",
        "runtime_delta_mib": "peak_runtime_delta_bytes",
        "saved_nonleaf_mib": "saved_nonleaf_unique_peak_bytes",
        "activation_cache_mib": "peak_activation_cache_bytes",
        "optimizer_state_mib": "optimizer_state_bytes",
        "weight_stash_mib": "peak_weight_stash_bytes",
        "master_gradient_mib": "peak_master_gradient_bytes",
        "snapshot_gradient_mib": "peak_snapshot_gradient_bytes",
        "cuda_peak_mib": "peak_cuda_allocated_bytes",
    }
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["geometry"] == primary_geometry:
            grouped[(str(row["method"]), int(row["stage_id"]))].append(row)
    output = []
    for method in METHOD_ORDER:
        for stage in range(3):
            group = grouped.get((method, stage), [])
            if not group:
                continue
            result: dict[str, Any] = {
                "geometry": primary_geometry,
                "method": method,
                "stage_id": stage,
                "peak_semantics": "independent_non_additive_peaks",
            }
            for output_key, source_key in fields.items():
                result[output_key] = statistics.mean(
                    number(item.get(source_key)) / MIB for item in group
                )
            result["host_transport_mib"] = statistics.mean(
                (
                    number(item.get("host_peak_pending_send_bytes"))
                    + number(item.get("host_peak_posted_recv_bytes"))
                )
                / MIB
                for item in group
            )
            output.append(result)
    return output


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_geometry(
    max_rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in max_rows:
        grouped[(row["geometry"], row["method"])].append(
            float(row["max_stage_peak_cuda_allocated_mib"])
        )
    x = np.arange(len(GEOMETRY_ORDER), dtype=float)
    width = 0.19
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    for index, method in enumerate(METHOD_ORDER):
        stats = [
            mean_std(grouped.get((geometry, method), []))
            for geometry in GEOMETRY_ORDER
        ]
        ax.bar(
            x + (index - 1.5) * width,
            [item[0] for item in stats],
            width,
            yerr=[item[1] for item in stats],
            capsize=2,
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
        )
    ax.set_xticks(x, ["(1,32)", "(2,16)", "(4,8)", "(8,4)"])
    ax.set_xlabel("Physical batch and microbatches (b, m), B=32")
    ax.set_ylabel("Maximum stage CUDA peak (MiB)")
    ax.grid(axis="y", color="#D9DDE3", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(ncol=4, frameon=False, fontsize=8)
    save_figure(fig, output_dir / "gpu_memory_by_geometry")


def plot_stage_peaks(
    rows: list[dict[str, Any]],
    primary_geometry: str,
    output_dir: Path,
) -> None:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        if row["geometry"] == primary_geometry:
            grouped[(row["method"], int(row["stage_id"]))].append(
                number(row["peak_cuda_allocated_bytes"]) / MIB
            )
    x = np.arange(3, dtype=float)
    width = 0.19
    fig, ax = plt.subplots(figsize=(6.6, 3.7))
    for index, method in enumerate(METHOD_ORDER):
        stats = [
            mean_std(grouped.get((method, stage), []))
            for stage in range(3)
        ]
        ax.bar(
            x + (index - 1.5) * width,
            [item[0] for item in stats],
            width,
            yerr=[item[1] for item in stats],
            capsize=2,
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
        )
    ax.set_xticks(x, ["Stage 0", "Stage 1", "Stage 2"])
    ax.set_ylabel("CUDA peak allocated (MiB)")
    ax.grid(axis="y", color="#D9DDE3", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(ncol=4, frameon=False, fontsize=8)
    save_figure(fig, output_dir / f"{primary_geometry}_stage_peaks")


def plot_components(
    rows: list[dict[str, Any]],
    primary_geometry: str,
    output_dir: Path,
) -> None:
    value_keys = (
        "resident_model_mib",
        "runtime_delta_mib",
        "saved_nonleaf_mib",
        "activation_cache_mib",
        "optimizer_state_mib",
        "weight_stash_mib",
        "master_gradient_mib",
        "snapshot_gradient_mib",
        "host_transport_mib",
        "cuda_peak_mib",
    )
    labels = (
        "Resident\nmodel",
        "Runtime\ndelta",
        "Saved\nnon-leaf",
        "Activation\ncache",
        "Optimizer\nstate",
        "Weight\nstash",
        "Master\ngrad",
        "Snapshot\ngrad",
        "Host\ntransport",
        "CUDA\npeak",
    )
    matrix = np.array(
        [[float(row[key]) for key in value_keys] for row in rows],
        dtype=float,
    )
    ylabels = [
        f"{METHOD_LABELS[row['method']]} S{row['stage_id']}" for row in rows
    ]
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(labels)), labels, fontsize=7)
    ax.set_yticks(np.arange(len(ylabels)), ylabels, fontsize=7)
    ax.set_title("Independent measured peaks (MiB; columns are not additive)")
    colorbar = fig.colorbar(image, ax=ax, pad=0.015)
    colorbar.set_label("MiB")
    save_figure(fig, output_dir / f"{primary_geometry}_component_peaks")


def numeric_audit(
    max_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in max_rows:
        grouped[(row["geometry"], row["method"])].append(
            float(row["max_stage_peak_cuda_allocated_mib"])
        )
    output: dict[str, Any] = {}
    for geometry in GEOMETRY_ORDER:
        output[geometry] = {}
        for method in METHOD_ORDER:
            values = grouped.get((geometry, method), [])
            if not values:
                continue
            mean, std = mean_std(values)
            output[geometry][method] = {
                "repetitions": len(values),
                "mean_mib": mean,
                "std_mib": std,
                "min_mib": min(values),
                "max_mib": max(values),
            }
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--primary-geometry", default="b8_m4")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = discover(args.raw_root)
    if not rows:
        raise RuntimeError(f"no completed E2 runs under {args.raw_root}")
    unknown = sorted({row["method"] for row in rows} - set(METHOD_ORDER))
    if unknown:
        raise ValueError(f"unexpected E2 methods: {unknown}")

    add_mib_columns(rows)
    max_rows = max_stage_rows(rows)
    components = component_rows(rows, args.primary_geometry)
    tables = args.output_dir / "tables"
    figures = args.output_dir / "figures"
    write_csv(tables / "memory_ledger.csv", rows)
    write_csv(tables / "max_stage_memory.csv", max_rows)
    write_csv(tables / "component_peak_matrix.csv", components)
    plot_geometry(max_rows, figures)
    plot_stage_peaks(rows, args.primary_geometry, figures)
    plot_components(components, args.primary_geometry, figures)
    write_json(
        args.output_dir / "report_audit.json",
        {
            "raw_root": str(args.raw_root.resolve()),
            "primary_geometry": args.primary_geometry,
            "stage_rows": len(rows),
            "run_rows": len(max_rows),
            "component_rows": len(components),
            "method_labels": METHOD_LABELS,
            "component_semantics": (
                "Each component is an independently measured peak; component "
                "columns must not be summed to reconstruct CUDA peak."
            ),
            "max_stage_cuda_peak": numeric_audit(max_rows),
        },
    )
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
