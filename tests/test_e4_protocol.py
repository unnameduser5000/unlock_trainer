from __future__ import annotations

import importlib
import json
from pathlib import Path

from experiments.e4_throughput import provenance


REPO_ROOT = Path(__file__).resolve().parents[1]
MAINTAINED_E4_MODULES = (
    "experiments.e4_throughput.run_e4_1_cpu_transport_scaling",
    "experiments.e4_throughput.run_e4_1_gpu_scaling",
    "experiments.e4_throughput.run_e4_2_cpu_transport",
    "experiments.e4_throughput.run_e4_3_mobile_network_sensitivity",
    "experiments.e4_throughput.run_e4_4_overhead_decomposition",
    "experiments.e4_throughput.run_e4_4_steady_trace",
    "experiments.e4_throughput.gpu_timeline.run_nsys_timeline",
    "experiments.e4_throughput.gpu_timeline.export_nsys_intervals",
    "experiments.e4_throughput.pipedream_async.run_formal_comparison",
    "experiments.e4_throughput.pipedream_async.run_formal_quality",
    "experiments.e4_throughput.pipedream_async.evaluate_pipeline_lora_state",
)
FORBIDDEN_E4_PATHS = (
    "experiments/e4_throughput/cpu_transport",
    "experiments/e4_throughput/run_1f1b_benchmark.py",
    "experiments/e4_throughput/run_e4_2a_batch_geometry.py",
    "experiments/e4_throughput/run_e4_2b_low_batch.py",
    "experiments/e4_throughput/simulate_forward_pipeline.py",
    "experiments/e4_throughput/profiles",
)

CANONICAL_CONFIGS = {
    "e4_1_gpu_scaling.json",
    "e4_1_scaling.json",
    "e4_2a_batch_geometry.json",
    "e4_2b_low_batch.json",
    "e4_3_network_sensitivity.json",
    "e4_4_overhead_decomposition.json",
    "e4_4_steady_trace.json",
}
PIPEDREAM_CONFIGS = {
    "comparison.json",
    "quality.json",
    "steady_state.json",
}


def test_maintained_e4_launchers_are_importable() -> None:
    for module_name in MAINTAINED_E4_MODULES:
        importlib.import_module(module_name)


def test_e4_provenance_sources_exist() -> None:
    missing = [
        relative
        for relative in provenance.E4_RUNTIME_SOURCES
        if not (REPO_ROOT / relative).is_file()
    ]
    assert not missing


def test_supplementary_e4_uses_shared_commands_and_provenance() -> None:
    gpu_scaling = importlib.import_module(
        "experiments.e4_throughput.run_e4_1_gpu_scaling"
    )
    nsys = importlib.import_module(
        "experiments.e4_throughput.gpu_timeline.run_nsys_timeline"
    )
    e4_2 = importlib.import_module(
        "experiments.e4_throughput.run_e4_2_cpu_transport"
    )

    assert nsys.build_command is e4_2.build_command
    missing = [
        relative
        for source_paths in (
            gpu_scaling.GPU_SCALING_SOURCE_PATHS,
            nsys.TIMELINE_SOURCE_PATHS,
        )
        for relative in source_paths
        if not (REPO_ROOT / relative).is_file()
    ]
    assert not missing


def test_e4_has_no_experiment_local_runtime_or_legacy_import() -> None:
    present = [
        relative
        for relative in FORBIDDEN_E4_PATHS
        if (REPO_ROOT / relative).exists()
    ]
    assert not present

    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(
            (REPO_ROOT / "experiments/e4_throughput").rglob("*.py")
        )
    )
    assert "sys.path.insert" not in source
    assert "--import-legacy-auto" not in source
    assert "transport_primitives.py" not in source


def test_e4_has_one_canonical_config_per_protocol() -> None:
    config_dir = REPO_ROOT / "experiments/e4_throughput/configs"
    assert {path.name for path in config_dir.glob("*.json")} == CANONICAL_CONFIGS

    baseline_dir = (
        REPO_ROOT
        / "experiments/e4_throughput/pipedream_async/configs"
    )
    assert {path.name for path in baseline_dir.glob("*.json")} == (
        PIPEDREAM_CONFIGS
    )

    experiment_ids: list[str] = []
    for name in sorted(CANONICAL_CONFIGS):
        config = json.loads((config_dir / name).read_text(encoding="utf-8"))
        experiment_ids.append(str(config["experiment_id"]))
        assert (
            int(
                config.get(
                    "omp_num_threads",
                    provenance.DEFAULT_OMP_NUM_THREADS,
                )
            )
            == 4
        )
    assert len(experiment_ids) == len(set(experiment_ids))


def test_e4_registry_has_one_entry_per_maintained_protocol() -> None:
    registry = json.loads(
        (REPO_ROOT / "experiments/registry.json").read_text(
            encoding="utf-8"
        )
    )
    entries = {
        item["id"]: item
        for item in registry["experiments"]
        if str(item["id"]).startswith("E4")
    }
    assert set(entries) == {
        "E4.1",
        "E4.1-G",
        "E4.2a",
        "E4.2b",
        "E4.3",
        "E4.4",
        "E4.4-N",
        "E4.4-S",
    }
    for entry in entries.values():
        assert (REPO_ROOT / entry["config_authority"]).is_file()
        assert (REPO_ROOT / entry["launcher"]).is_file()
        assert {
            "measurement_contract",
            "confounders",
            "pipeline_phase_policy",
        } <= entry.keys()
        assert entry["confounders"]
        assert entry["pipeline_phase_policy"]["current_phase_split"]

    for experiment_id in ("E4.1", "E4.2a", "E4.2b", "E4.3"):
        entry = entries[experiment_id]
        assert entry["measurement_contract"]["reported_metric"] == (
            "full_run_throughput_per_s"
        )
        assert entry["pipeline_phase_policy"]["steady_state_primary"] is False

    gpu_scaling = entries["E4.1-G"]
    assert gpu_scaling["claim_status"] == "diagnostic"
    assert gpu_scaling["status"] == (
        "smoke_validated_formal_matrix_pending"
    )

    for experiment_id in ("E4.4", "E4.4-S"):
        assert entries[experiment_id]["measurement_contract"]["action_trace"]

    nsys = entries["E4.4-N"]["measurement_contract"]
    assert nsys["nsys_profile"] is True
    assert nsys["action_trace"] is False


def test_e3_recovery_baseline_is_not_owned_by_e4() -> None:
    module = importlib.import_module(
        "experiments.e5_recovery.baselines.run_exactbp_batch_recovery"
    )
    assert module.RUNNER_SCRIPT == (
        REPO_ROOT
        / "src/sg_exe_trainer/runtime/exactbp/distributed_runtime.py"
    )
