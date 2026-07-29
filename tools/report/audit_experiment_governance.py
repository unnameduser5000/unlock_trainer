#!/usr/bin/env python3
"""Audit experiment registry and E2/E4 measurement hygiene.

This script is intentionally conservative. It does not decide scientific truth;
it prevents contaminated or under-specified runs from being promoted as final
throughput evidence.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


VALID_CLAIM_STATUS = {
    "final_evidence",
    "scheduler_runtime_evidence",
    "memory_evidence",
    "diagnostic",
    "historical_contaminated",
}

E2_REQUIRED_FIELDS = {
    "measurement_contract",
    "claim_status",
}

E4_REQUIRED_FIELDS = {
    "measurement_contract",
    "confounders",
    "pipeline_phase_policy",
    "claim_status",
}

CLEAN_THROUGHPUT_CONTRACT = {
    "gc_collect_in_measured_loop": False,
    "empty_cache_in_measured_loop": False,
    "activation_tracker": False,
    "torch_profiler": False,
    "autograd_profiler": False,
    "eval_during_train": False,
    "checkpointing": False,
    "retry_replay_cache": False,
    "failure_injection": False,
    "progress_print": "disabled_or_fixed",
    "timer_policy": "synchronized_max_wall_time",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def audit_registry(root: Path, errors: list[str], warnings: list[str]) -> None:
    path = root / "experiments" / "registry.json"
    if not path.is_file():
        fail(errors, "missing experiments/registry.json")
        return

    registry = load_json(path)
    if int(registry.get("schema_version", 0)) < 2:
        fail(errors, "registry schema_version must be >= 2")

    policy = registry.get("policy", {})
    for key in ["human_status_entry", "long_running_report", "measurement_rules"]:
        if not policy.get(key):
            fail(errors, f"registry policy missing {key}")
    if not policy.get("raw_output_roots") and not policy.get("raw_output_root"):
        fail(errors, "registry policy missing raw output roots")

    experiments = registry.get("experiments", [])
    seen: set[str] = set()
    for item in experiments:
        exp_id = item.get("id", "")
        if not exp_id:
            fail(errors, "experiment without id")
            continue
        if exp_id in seen:
            fail(errors, f"duplicate experiment id {exp_id}")
        seen.add(exp_id)

        for key in ["question", "output_root", "aggregation", "supported_claims", "unsupported_claims", "claim_status"]:
            if key not in item:
                fail(errors, f"{exp_id}: missing {key}")

        claim_status = item.get("claim_status")
        if claim_status not in VALID_CLAIM_STATUS:
            fail(errors, f"{exp_id}: invalid claim_status {claim_status!r}")

        if not item.get("config_authority") and not item.get("config_gap"):
            fail(errors, f"{exp_id}: missing config_authority or config_gap")

        if exp_id.startswith("E2"):
            for key in E2_REQUIRED_FIELDS:
                if key not in item:
                    fail(errors, f"{exp_id}: E2 experiment missing {key}")
            contract = item.get("measurement_contract", {})
            expected_memory = {
                "contract_type": "gpu_memory_exchange",
                "effective_batch": 32,
                "activation_tracker": True,
                "cuda_allocator_peak": True,
                "terminal_output_log_probabilities": False,
                "component_peak_semantics": "independent_non_additive_peaks",
            }
            for key, expected in expected_memory.items():
                if contract.get(key) != expected:
                    fail(
                        errors,
                        f"{exp_id}: E2 memory contract has {key}="
                        f"{contract.get(key)!r}, expected {expected!r}",
                    )

        if exp_id.startswith("E4"):
            for key in E4_REQUIRED_FIELDS:
                if key not in item:
                    fail(errors, f"{exp_id}: E4 experiment missing {key}")
            contract = item.get("measurement_contract", {})
            phase_policy = item.get("pipeline_phase_policy", {})
            if claim_status == "final_evidence":
                for key, expected in CLEAN_THROUGHPUT_CONTRACT.items():
                    if contract.get(key) != expected:
                        fail(
                            errors,
                            f"{exp_id}: final E4 evidence has {key}="
                            f"{contract.get(key)!r}, expected {expected!r}",
                        )
                if not phase_policy.get("steady_state_primary"):
                    fail(
                        errors,
                        f"{exp_id}: final E4 evidence must make "
                        "steady-state throughput primary",
                    )
            elif (
                phase_policy.get("phase_split_required_for_final_throughput")
                and not phase_policy.get("current_phase_split")
            ):
                warnings.append(
                    f"{exp_id}: phase split required but current_phase_split "
                    "is not described"
                )


def extract_function_block(text: str, function_name: str) -> str:
    pattern = re.compile(rf"^def {re.escape(function_name)}\b", re.MULTILINE)
    match = pattern.search(text)
    if not match:
        return ""
    start = match.start()
    next_match = re.search(r"^def \w+\b", text[match.end():], re.MULTILINE)
    end = match.end() + next_match.start() if next_match else len(text)
    return text[start:end]


def audit_runner_static(root: Path, errors: list[str], warnings: list[str]) -> None:
    one_f1b = root / "src" / "sg_exe_trainer" / "runtime" / "exactbp" / "distributed_runtime.py"
    bpfree_runner = root / "src" / "sg_exe_trainer" / "runtime" / "bpfree" / "gpu_runner.py"
    bpfree_phase = root / "src" / "sg_exe_trainer" / "runtime" / "bpfree" / "gpu_phase.py"

    if one_f1b.is_file():
        text = one_f1b.read_text(encoding="utf-8")
        run_phase = extract_function_block(text, "run_phase")
        if "gc.collect()" in run_phase and "gc_interval_batches" not in run_phase:
            fail(errors, "1F1B run_phase has unconditional measured-loop gc.collect()")
        if "--gc_interval_batches" not in text or "default=0" not in text:
            fail(errors, "1F1B runner must expose --gc_interval_batches with default 0")
        if "track_activation_memory" not in text:
            fail(errors, "1F1B runner must expose activation tracking control")
        if "timeline_events.csv" not in text:
            fail(errors, "1F1B runner must emit timeline_events.csv for phase analysis")
    else:
        fail(errors, "missing src/sg_exe_trainer/runtime/exactbp/distributed_runtime.py")

    if bpfree_runner.is_file() and bpfree_phase.is_file():
        runner_text = bpfree_runner.read_text(encoding="utf-8")
        phase_text = bpfree_phase.read_text(encoding="utf-8")
        run_phase = extract_function_block(phase_text, "run_phase_schedule_v3")
        if "gc.collect()" in run_phase:
            fail(errors, "BPFree GPU phase has measured-loop gc.collect()")
        for needle in ["track_activation_memory", "physical_batch_size", "gradient_accumulation_steps", "skip_eval_before", "skip_eval_after"]:
            if needle not in runner_text:
                fail(errors, f"BPFree GPU runner missing {needle}")
        if "throughput_per_s" in phase_text and "steady_state_throughput_per_s" not in phase_text:
            warnings.append("BPFree GPU phase records full-run throughput but no steady-state throughput field yet")
    else:
        fail(errors, "missing BPFree gpu_runner.py or gpu_phase.py")


def audit_e2_runtime_static(root: Path, errors: list[str]) -> None:
    files = {
        "BP-free": root / "src/sg_exe_trainer/runtime/bpfree/cpu_runner.py",
        "Exact-BP": root / "src/sg_exe_trainer/runtime/exactbp/cpu_runner.py",
        "PipeDream": root / "experiments/shared/baselines/pipedream_cpu.py",
    }
    required = {
        "BP-free": [
            "track_activation_memory",
        ],
        "Exact-BP": [
            "track_activation_memory",
            "memory_profile",
            "peak_activation_cache_bytes",
        ],
        "PipeDream": [
            "track_activation_memory",
            "SavedTensorTracker",
            "peak_runtime_delta_bytes",
        ],
    }
    for name, path in files.items():
        if not path.is_file():
            fail(errors, f"missing E2 {name} runner: {path}")
            continue
        text = path.read_text(encoding="utf-8")
        for needle in required[name]:
            if needle not in text:
                fail(errors, f"E2 {name} runner missing {needle}")

    bpfree_phase = root / "src/sg_exe_trainer/runtime/bpfree/cpu_phase.py"
    if (
        not bpfree_phase.is_file()
        or "cuda_peak_memory_allocated"
        not in bpfree_phase.read_text(encoding="utf-8")
    ):
        fail(errors, "E2 BP-free CPU phase missing CUDA peak instrumentation")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit experiment governance and E2/E4 measurement rules."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--strict", action="store_true", help="Treat warnings as failures.")
    args = parser.parse_args()

    root = args.root.resolve()
    errors: list[str] = []
    warnings: list[str] = []

    audit_registry(root, errors, warnings)
    audit_runner_static(root, errors, warnings)
    audit_e2_runtime_static(root, errors)

    for warning in warnings:
        print(f"WARN: {warning}")
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)

    if errors or (args.strict and warnings):
        return 1
    print(json.dumps({"status": "ok", "warnings": len(warnings)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
