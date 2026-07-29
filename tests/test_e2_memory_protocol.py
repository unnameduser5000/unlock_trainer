from __future__ import annotations

import json
from pathlib import Path

from experiments.e2_memory import run_gpu_memory as e2


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "experiments/e2_memory/configs/gpu_memory_v1.json"


def _arg(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def _jobs() -> list[e2.Job]:
    protocol = e2.load_protocol(CONFIG_PATH)
    return e2.build_jobs(
        protocol,
        {
            "train": Path("/artifact/train.jsonl"),
            "eval": Path("/artifact/eval.jsonl"),
        },
        Path("/results/e2"),
        list(e2.METHOD_IDS),
        [item[0] for item in e2.GEOMETRIES],
        "cuda:0,cuda:1,cuda:2",
        smoke=False,
    )


def test_e2_protocol_is_the_fixed_paper_memory_matrix() -> None:
    protocol = e2.load_protocol(CONFIG_PATH)

    assert [item["id"] for item in protocol["methods"]] == list(e2.METHOD_IDS)
    assert [
        (
            item["id"],
            item["physical_batch_size"],
            item["microbatches"],
        )
        for item in protocol["geometries"]
    ] == list(e2.GEOMETRIES)
    assert protocol["optimization"]["effective_batch"] == 32
    assert protocol["optimization"]["train_windows"] == 8
    assert protocol["repetitions"] == {
        "default": 1,
        "primary": 3,
        "primary_geometry": "b8_m4",
    }
    assert protocol["measurement"]["activation_tracking"] is True
    assert (
        protocol["measurement"]["terminal_output_log_probabilities"]
        is False
    )
    assert all(
        item["physical_batch_size"] * item["microbatches"] == 32
        for item in protocol["geometries"]
    )


def test_e2_formal_matrix_has_24_fresh_process_jobs() -> None:
    jobs = _jobs()

    assert len(jobs) == 24
    counts: dict[tuple[str, str], int] = {}
    for job in jobs:
        key = (job.geometry, job.method)
        counts[key] = counts.get(key, 0) + 1
        assert job.train_windows == 8
        assert job.physical_batch_size * job.microbatches == 32
    for geometry, _, _ in e2.GEOMETRIES:
        expected = 3 if geometry == "b8_m4" else 1
        for method in e2.METHOD_IDS:
            assert counts[(geometry, method)] == expected


def test_e2_commands_use_shared_runtimes_and_fixed_memory_flags() -> None:
    jobs = {
        job.method: job
        for job in _jobs()
        if job.geometry == "b1_m32" and job.repetition == 0
    }

    bpfree = jobs["bpfree"].command
    assert bpfree[1:3] == ["-m", e2.RUNNER_MODULES["bpfree"]]
    assert _arg(bpfree, "--belief_transport_mode") == "none"
    assert "--track_activation_memory" in bpfree

    for method in ("exactbp_1f1b", "exactbp_gpipe"):
        command = jobs[method].command
        assert "--module" in command
        assert _arg(command, "--module") == e2.RUNNER_MODULES["exactbp"]
        assert "--track_activation_memory" in command
        assert _arg(command, "--memory_warmup_windows") == "0"
    assert _arg(jobs["exactbp_1f1b"].command, "--pipeline_schedule") == "1f1b"
    assert _arg(jobs["exactbp_gpipe"].command, "--pipeline_schedule") == "gpipe"

    pipedream = jobs["pipedream"].command
    assert _arg(pipedream, "--module") == e2.RUNNER_MODULES["pipedream"]
    assert "--track_activation_memory" in pipedream

    for job in jobs.values():
        command_text = " ".join(job.command)
        assert "vendor_snapshot" not in command_text
        assert "experiments/e4_throughput" not in command_text
        assert _arg(job.command, "--train_limit") == "256"


def test_e2_smoke_changes_only_run_count_and_window_count() -> None:
    protocol = e2.load_protocol(CONFIG_PATH)
    jobs = e2.build_jobs(
        protocol,
        {
            "train": Path("/artifact/train.jsonl"),
            "eval": Path("/artifact/eval.jsonl"),
        },
        Path("/results/e2-smoke"),
        list(e2.METHOD_IDS),
        ["b8_m4"],
        "cuda:0,cuda:1,cuda:2",
        smoke=True,
    )

    assert len(jobs) == 4
    assert {job.train_windows for job in jobs} == {1}
    assert {_arg(job.command, "--train_limit") for job in jobs} == {"32"}


def test_e2_has_one_registry_entry_and_no_legacy_runtime_copy() -> None:
    registry = json.loads(
        (REPO_ROOT / "experiments/registry.json").read_text(encoding="utf-8")
    )
    e2_entries = [
        item for item in registry["experiments"]
        if str(item["id"]).startswith("E2")
    ]

    assert [item["id"] for item in e2_entries] == ["E2"]
    assert e2_entries[0]["launcher"] == (
        "experiments/e2_memory/run_gpu_memory.py"
    )
    assert not (REPO_ROOT / "experiments/e2_readout").exists()
    assert not (
        REPO_ROOT / "experiments/protocols/run_agnews_formal_protocol.py"
    ).exists()


def test_e2_runtime_instrumentation_lives_outside_experiment_driver() -> None:
    exact_source = e2.RUNNER_FILES["exactbp"].read_text(encoding="utf-8")
    pipedream_source = e2.RUNNER_FILES["pipedream"].read_text(encoding="utf-8")
    driver_source = (
        REPO_ROOT / "experiments/e2_memory/run_gpu_memory.py"
    ).read_text(encoding="utf-8")

    assert "memory_profile" in exact_source
    assert "SavedTensorTracker" in exact_source
    assert "peak_activation_cache_bytes" in exact_source
    assert "SavedTensorTracker" in pipedream_source
    assert "peak_runtime_delta_bytes" in pipedream_source
    assert "class ExactBPCpuCommFair" not in driver_source
    assert "class PipeDreamCpuComm" not in driver_source
