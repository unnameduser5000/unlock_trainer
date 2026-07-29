from __future__ import annotations

from pathlib import Path

from experiments.e1_quality import run_agnews_quality as e1


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "experiments/e1_quality/configs/agnews_quality_v1.json"
CANONICAL_METHODS = [
    "full_bp_1gpu",
    "1f1b_3gpu",
    "bpfree_ce_3gpu",
    "bpfree_belief_3gpu",
]
REMOVED_SYNTHETIC_GRADIENT_LAUNCHERS = [
    "run_bpfree_lora_label_sg_pilot.py",
    "run_bpfree_lora_label_sweep.sh",
    "run_label_control_suite.sh",
    "run_label_optimizer_compare.sh",
    "run_label_sg_pilot.sh",
    "run_label_sgd_grid.sh",
    "run_label_window_compare.sh",
]


def _arg(command: list[str], name: str) -> str:
    index = command.index(name)
    return command[index + 1]


def _jobs() -> dict[str, e1.Job]:
    protocol = e1.load_protocol(CONFIG_PATH)
    manifests = {
        "train": Path("/artifact/train.jsonl"),
        "validation": Path("/artifact/validation.jsonl"),
        "test": Path("/artifact/test.jsonl"),
    }
    jobs = e1.build_jobs(
        protocol,
        manifests,
        Path("/results/e1"),
        [20260531],
        CANONICAL_METHODS,
        "cuda:0,cuda:1,cuda:2",
    )
    return {job.method: job for job in jobs}


def test_e1_protocol_restores_the_paper_training_contract() -> None:
    protocol = e1.load_protocol(CONFIG_PATH)

    assert [method["id"] for method in protocol["methods"]] == CANONICAL_METHODS
    assert protocol["seeds"] == [20260531, 20260532, 20260533]
    assert {name: split["records"] for name, split in protocol["manifests"].items()} == {
        "train": 10_000,
        "validation": 1_000,
        "test": 7_600,
    }
    assert protocol["optimization"]["effective_batch"] == 8
    assert protocol["optimization"]["optimizer_steps"] == 1_250
    assert protocol["optimization"]["validation_interval_steps"] == 125
    assert protocol["model"]["trainable_mode"] == "lora"
    assert protocol["model"]["lora_targets"] == "q_proj,v_proj"


def test_e1_commands_have_one_runtime_definition_per_paper_arm() -> None:
    jobs = _jobs()
    assert list(jobs) == CANONICAL_METHODS

    full_bp = jobs["full_bp_1gpu"].command
    assert _arg(full_bp, "--num_chunks") == "1"
    assert _arg(full_bp, "--batch_size") == "8"
    assert _arg(full_bp, "--microbatches") == "1"
    assert _arg(full_bp, "--pipeline_schedule") == "1f1b"

    one_f1b = jobs["1f1b_3gpu"].command
    assert _arg(one_f1b, "--num_chunks") == "3"
    assert _arg(one_f1b, "--microbatches") == "8"
    assert _arg(one_f1b, "--pipeline_schedule") == "1f1b"

    ce = jobs["bpfree_ce_3gpu"].command
    belief = jobs["bpfree_belief_3gpu"].command
    for command in (ce, belief):
        assert command[1:3] == ["-m", e1.BPFREE_MODULE]
        assert _arg(command, "--gradient_accumulation_steps") == "8"
        assert _arg(command, "--failure_mode") == "none"
        assert _arg(command, "--limit") == "10000"
        assert not any("experiments/e5_recovery" in token for token in command)
    assert _arg(ce, "--belief_transport_mode") == "none"
    assert _arg(ce, "--alpha") == "1.0"
    assert _arg(belief, "--belief_transport_mode") == "full"
    assert _arg(belief, "--alpha") == "0.5"

    for job in jobs.values():
        assert _arg(job.command, "--validation_interval_steps") == "125"
        assert _arg(job.command, "--validation_limit") == "1000"


def test_e1_audit_does_not_double_count_segment_and_aggregate_phases() -> None:
    segmented = {
        "phase_summaries": [
            {"phase": "train_to_001000", "mode": "train", "records": 1_000},
            {"phase": "train_to_010000", "mode": "train", "records": 9_000},
            {"phase": "train", "mode": "train", "records": 10_000},
        ]
    }
    selected = e1.authoritative_train_phases(segmented)
    assert [phase["phase"] for phase in selected] == ["train"]
    assert sum(phase["records"] for phase in selected) == 10_000


def test_e1_directory_has_no_early_synthetic_gradient_entry_points() -> None:
    root = REPO_ROOT / "experiments/e1_quality"
    remaining = [name for name in REMOVED_SYNTHETIC_GRADIENT_LAUNCHERS if (root / name).exists()]
    assert not remaining


def test_unregistered_local_objective_factorial_is_archival_only() -> None:
    root = REPO_ROOT / "experiments/e1_local_objective_factorial"
    maintained_sources = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix in {".py", ".sh"}
    ]
    assert not maintained_sources


def test_e1_root_exposes_only_the_canonical_driver() -> None:
    root = REPO_ROOT / "experiments/e1_quality"
    assert sorted(path.name for path in root.glob("*.py")) == [
        "__init__.py",
        "run_agnews_quality.py",
    ]
