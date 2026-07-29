#!/usr/bin/env python3
"""Build an auditable registry for formal and exploratory experiment runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


FIELDS = [
    "suite",
    "method",
    "seed",
    "status",
    "runner",
    "transport",
    "transport_evidence",
    "claim_scope",
    "stage_devices",
    "physical_request_batch",
    "effective_batch",
    "microbatches",
    "gradient_accumulation_steps",
    "max_inflight",
    "belief_transport_mode",
    "learning_rate",
    "dtype",
    "lora_init_seed",
    "lora_initialization_fingerprint",
    "summary_path",
    "command_path",
    "metrics_source",
    "metadata_quality",
]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def command_text(run_dir: Path) -> str:
    path = run_dir / "command.txt"
    return path.read_text(encoding="utf-8").strip() if path.is_file() else ""


def infer_transport(summary: dict[str, Any], command: str) -> tuple[str, str]:
    if summary.get("transport"):
        return str(summary["transport"]), "summary.explicit"
    num_chunks = summary.get("num_chunks")
    if isinstance(num_chunks, (int, float)) and int(num_chunks) <= 1:
        return "single-gpu-local", "summary.inferred"
    if (
        "orchestrated_runtime" in command
        or "run_bpfree_scheduler_lab.py" in command
        or summary.get("runner") in {"scheduler_lab", "bpfree-orchestrated-runtime-v1"}
    ):
        return "cpu-mp-queue", "runner.inferred"
    if (
        "sg_exe_trainer.runtime.bpfree.gpu_runner" in command
        or "sg_exe_trainer/runtime/bpfree/gpu_runner.py" in command
    ):
        return "nccl-sendrecv", "command.inferred"
    if "sg_exe_trainer/runtime/exactbp/distributed_runtime.py" in command:
        return "nccl-pipeline", "command.inferred"
    return "", "missing"


def status_from_suite_csv(suite_dir: Path, method: str, seed: str) -> str:
    """Read the formal protocol's live status file when the summary is not written yet."""
    for status_path in suite_dir.parent.glob("*_run_status.csv"):
        with status_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if (
                    row.get("suite") == suite_dir.name
                    and row.get("method") == method
                    and row.get("seed") == seed
                    and row.get("status")
                ):
                    return str(row["status"])
    return "planned"


def claim_scope(suite: str, transport: str) -> str:
    if suite == "e1_quality":
        return "algorithm-quality"
    if suite == "e2_p2p":
        return "p2p-request-transport"
    if suite == "e2_p2p_matched":
        return "p2p-matched-throughput-memory"
    if suite == "e2_p2p_throughput":
        return "p2p-throughput-no-instrumentation"
    if suite == "e2_system" and transport == "cpu-mp-queue":
        return "queue-scheduler-saturation"
    if suite.startswith("e2"):
        return "system-throughput"
    return "exploratory"


def physical_request_batch(summary: dict[str, Any], transport: str) -> int | str:
    explicit = summary.get("physical_request_batch")
    if isinstance(explicit, (int, float)):
        return int(explicit)
    if transport in {"cpu-mp-queue", "nccl-sendrecv"}:
        return 1
    batch_size = summary.get("batch_size")
    microbatches = summary.get("microbatches")
    if isinstance(batch_size, (int, float)) and isinstance(microbatches, (int, float)) and microbatches:
        return int(batch_size // microbatches)
    return ""


def metrics_source(run_dir: Path, transport: str) -> str:
    if transport == "cpu-mp-queue":
        return str(run_dir / "scheduler_stage_metrics.csv")
    if transport == "nccl-sendrecv":
        return str(run_dir / "train.stage*.metrics.csv")
    return str(run_dir / "stage_metrics.csv")


def row_for_run(suite_dir: Path, method: str, seed_dir: Path) -> dict[str, Any]:
    suite = suite_dir.name
    seed = seed_dir.name.removeprefix("seed")
    scheduler_summary = seed_dir / "scheduler_summary.json"
    summary_path = scheduler_summary if scheduler_summary.is_file() else seed_dir / "summary.json"
    status = "completed" if summary_path.is_file() else status_from_suite_csv(suite_dir, method, seed)
    if not summary_path.is_file() and (seed_dir / "exit_code.txt").is_file():
        status = "failed"
    summary = read_json(summary_path) if summary_path.is_file() else {}
    command = command_text(seed_dir)
    transport, transport_evidence = infer_transport(summary, command)
    lora = summary.get("lora", {}) if isinstance(summary.get("lora", {}), dict) else {}
    batch_size = summary.get("batch_size", "")
    accumulation = summary.get("gradient_accumulation_steps", "")
    explicit_effective_batch = summary.get("effective_optimizer_batch")
    effective_batch: int | str = ""
    if isinstance(explicit_effective_batch, (int, float)):
        effective_batch = int(explicit_effective_batch)
    elif isinstance(batch_size, (int, float)) and isinstance(accumulation, (int, float)):
        effective_batch = int(batch_size * accumulation)
    elif isinstance(batch_size, (int, float)):
        effective_batch = int(batch_size)
    metadata_quality = "explicit" if transport_evidence == "summary.explicit" else "inferred_legacy"
    if not transport:
        metadata_quality = "incomplete"
    return {
        "suite": suite,
        "method": method,
        "seed": seed,
        "status": status,
        "runner": summary.get("runner", ""),
        "transport": transport,
        "transport_evidence": transport_evidence,
        "claim_scope": claim_scope(suite, transport),
        "stage_devices": ";".join(summary.get("stage_devices", [])) if isinstance(summary.get("stage_devices"), list) else "",
        "physical_request_batch": physical_request_batch(summary, transport),
        "effective_batch": effective_batch,
        "microbatches": summary.get("microbatches", ""),
        "gradient_accumulation_steps": accumulation,
        "max_inflight": summary.get("max_inflight", ""),
        "belief_transport_mode": summary.get("belief_transport_mode", ""),
        "learning_rate": summary.get("learning_rate", ""),
        "dtype": summary.get("dtype", ""),
        "lora_init_seed": lora.get("init_seed", ""),
        "lora_initialization_fingerprint": lora.get("initialization_fingerprint", "")
        or ";".join(lora.get("initialization_fingerprints", [])),
        "summary_path": str(summary_path) if summary_path.is_file() else "",
        "command_path": str(seed_dir / "command.txt") if (seed_dir / "command.txt").is_file() else "",
        "metrics_source": metrics_source(seed_dir, transport),
        "metadata_quality": metadata_quality,
    }


def collect(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for suite_dir in sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith("e")):
        for method_dir in sorted(path for path in suite_dir.iterdir() if path.is_dir()):
            for seed_dir in sorted(path for path in method_dir.iterdir() if path.is_dir() and path.name.startswith("seed")):
                rows.append(row_for_run(suite_dir, method_dir.name, seed_dir))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    completed = [row for row in rows if row["status"] == "completed"]
    transport_counts = Counter(row["transport"] or "missing" for row in completed)
    quality_counts = Counter(row["metadata_quality"] for row in rows)
    lines = [
        "# Experiment Provenance Audit",
        "",
        f"- Runs discovered: {len(rows)}",
        f"- Completed runs: {len(completed)}",
        f"- Transport counts: {dict(sorted(transport_counts.items()))}",
        f"- Metadata quality: {dict(sorted(quality_counts.items()))}",
        "",
        "| Suite | Method | Seeds | Transport | Claim scope | Metadata |",
        "|---|---|---:|---|---|---|",
    ]
    groups: dict[tuple[str, str, str, str, str], int] = {}
    for row in rows:
        key = (row["suite"], row["method"], row["transport"], row["claim_scope"], row["metadata_quality"])
        groups[key] = groups.get(key, 0) + 1
    for (suite, method, transport, scope, quality), count in sorted(groups.items()):
        lines.append(f"| {suite} | {method} | {count} | {transport or 'missing'} | {scope} | {quality} |")
    lines.extend(
        [
            "",
            "A row marked `inferred_legacy` derives transport from its runner command. Do not combine it with an "
            "explicitly recorded transport result without labeling the distinction.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--report_dir", type=Path, default=None)
    args = parser.parse_args()
    report_dir = args.report_dir or args.output_root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = collect(args.output_root)
    write_csv(report_dir / "experiment_registry.csv", rows)
    write_markdown(report_dir / "provenance_audit.md", rows)
    print(json.dumps({"runs": len(rows), "report_dir": str(report_dir)}, indent=2))


if __name__ == "__main__":
    main()
