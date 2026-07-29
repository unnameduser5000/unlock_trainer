from __future__ import annotations

import ast
import importlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = REPO_ROOT / "src" / "sg_exe_trainer" / "runtime"
TOOLS_ROOT = REPO_ROOT / "tools"
FORBIDDEN_DUPLICATE_RUNTIME_PATHS = (
    "experiments/e4_throughput/run_bpfree_schedule.py",
    "experiments/e4_throughput/cpu_transport/run_bpfree_cpu_comm.py",
    "experiments/e4_throughput/cpu_transport/run_exactbp_cpu_comm_fair.py",
    "experiments/e4_throughput/pipedream_async/run_pipedream_cpu_comm.py",
    "experiments/e4_throughput/cpu_transport/bpfree_cpu_stage.py",
    "experiments/e4_throughput/cpu_transport/cpu_transport.py",
    "experiments/e4_throughput/cpu_transport/phase_cpu.py",
    "experiments/e4_throughput/cpu_transport/transport_primitives.py",
    "experiments/e5_recovery/formal_v2/catchup_stream.py",
    "experiments/e5_recovery/formal_v2/checkpoint_store.py",
    "experiments/e5_recovery/formal_v2/durable_io.py",
    "experiments/e5_recovery/formal_v2/event_log.py",
    "experiments/e5_recovery/formal_v2/runtime_adapter.py",
    "experiments/e5_recovery/formal_v2/state_contract.py",
    "experiments/e5_recovery/formal_v2/volatile_backlog.py",
    "experiments/e5_recovery/formal_v2/window_journal.py",
    "experiments/e5_recovery/window_recovery",
    "src/sg_exe_trainer/runtime/bpfree/distributed_runtime.py",
    "src/sg_exe_trainer/runtime/bpfree/multigpu_runtime.py",
    "src/sg_exe_trainer/runtime/bpfree/phase_v0.py",
    "src/sg_exe_trainer/runtime/bpfree/phase_v1.py",
    "src/sg_exe_trainer/runtime/bpfree/phase_v2.py",
    "src/sg_exe_trainer/runtime/bpfree/phase_v3.py",
)

MAINTAINED_E5_LAUNCHER_MODULES = (
    "experiments.e5_recovery.formal_v2.run_bpfree_outage",
    "experiments.e5_recovery.formal_v2.run_bpfree_streamed_outage",
    "experiments.e5_recovery.formal_v2.run_bpfree_volatile_outage",
    "experiments.e5_recovery.formal_v2.run_bpfree_cpu_volatile_outage",
    "experiments.e5_recovery.formal_v2.run_exactbp_outage",
    "experiments.e5_recovery.formal_v2.run_exactbp_volatile_outage",
    "experiments.e5_recovery.formal_v2.run_exactbp_cpu_volatile_outage",
)


def _forbidden_imports(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""] if node.level == 0 else []
        else:
            continue
        for name in names:
            if name == "experiments" or name.startswith("experiments."):
                found.append((node.lineno, name))
    return found


def test_runtime_does_not_import_experiments() -> None:
    violations: list[str] = []
    for path in sorted(RUNTIME_ROOT.rglob("*.py")):
        for line, imported in _forbidden_imports(path):
            rel = path.relative_to(RUNTIME_ROOT)
            violations.append(f"{rel}:{line} imports {imported}")

    assert not violations, "runtime/experiment dependency inversion:\n" + "\n".join(
        violations
    )


def test_runtime_has_no_duplicate_or_alias_modules() -> None:
    duplicates = [
        relative
        for relative in FORBIDDEN_DUPLICATE_RUNTIME_PATHS
        if (REPO_ROOT / relative).exists()
    ]
    assert not duplicates, "duplicate or alias runtime paths:\n" + "\n".join(
        duplicates
    )


def test_maintained_tools_do_not_mutate_python_import_path() -> None:
    violations = [
        str(path.relative_to(REPO_ROOT))
        for path in sorted(TOOLS_ROOT.rglob("*.py"))
        if "sys.path.insert" in path.read_text(encoding="utf-8")
    ]
    assert not violations, "tools mutate sys.path:\n" + "\n".join(violations)


def test_maintained_e5_launchers_are_importable() -> None:
    for module_name in MAINTAINED_E5_LAUNCHER_MODULES:
        importlib.import_module(module_name)


def test_runtime_packages_are_importable() -> None:
    from sg_exe_trainer.runtime.bpfree import (
        cpu_phase,
        cpu_stage,
        gpu_phase,
        gpu_runner,
        gpu_stage,
        gpu_transport,
        model_runtime,
    )
    from sg_exe_trainer.runtime.recovery import state_contract, window_journal
    from sg_exe_trainer.runtime.transport import cpu

    assert cpu_phase.run_phase_schedule_cpu
    assert gpu_phase.run_phase_schedule_v3
    assert gpu_runner.distributed_worker
    assert gpu_stage.BPFreePipelineStageV0
    assert gpu_transport.post_batch_p2p
    assert model_runtime.build_stage_chunk
    assert cpu_stage.BPFreePipelineStageV0
    assert state_contract.StageCommitLedger
    assert window_journal.BPFreeWindowJournal
    assert cpu.CpuTransportBudget
