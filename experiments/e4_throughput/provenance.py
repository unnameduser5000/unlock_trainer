from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


E4_RUNTIME_SOURCES = (
    "experiments/e4_throughput/provenance.py",
    "experiments/e4_throughput/run_e4_1_cpu_transport_scaling.py",
    "experiments/e4_throughput/run_e4_2_cpu_transport.py",
    "experiments/e4_throughput/run_e4_3_mobile_network_sensitivity.py",
    "experiments/e4_throughput/run_e4_4_overhead_decomposition.py",
    "experiments/e4_throughput/run_e4_4_steady_trace.py",
    "src/sg_exe_trainer/runtime/bpfree/cpu_runner.py",
    "src/sg_exe_trainer/runtime/bpfree/cpu_phase.py",
    "src/sg_exe_trainer/runtime/bpfree/cpu_stage.py",
    "src/sg_exe_trainer/runtime/bpfree/chunk_split.py",
    "src/sg_exe_trainer/runtime/bpfree/model_runtime.py",
    "src/sg_exe_trainer/runtime/bpfree/schedule_runtime.py",
    "src/sg_exe_trainer/runtime/exactbp/cpu_runner.py",
    "src/sg_exe_trainer/runtime/exactbp/distributed_runtime.py",
    "src/sg_exe_trainer/runtime/transport/cpu.py",
    "experiments/shared/baselines/pipedream_cpu.py",
)

DEFAULT_OMP_NUM_THREADS = 4


def build_execution_environment(
    cfg: dict[str, Any],
) -> tuple[dict[str, str], dict[str, str]]:
    """Build the controlled subprocess environment and its audit record."""
    omp_num_threads = int(
        cfg.get("omp_num_threads", DEFAULT_OMP_NUM_THREADS)
    )
    if omp_num_threads <= 0:
        raise ValueError("omp_num_threads must be positive")

    controlled = {"OMP_NUM_THREADS": str(omp_num_threads)}
    environment = os.environ.copy()
    environment.update(controlled)
    return environment, controlled


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _gpu_inventory() -> list[str]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _relative(repo_root: Path, path: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def capture_provenance(
    *,
    repo_root: Path,
    config_path: Path,
    cfg: dict[str, Any],
    command: list[str],
    execution_environment: dict[str, str] | None = None,
    source_paths: Iterable[str] = E4_RUNTIME_SOURCES,
) -> dict[str, Any]:
    artifact_paths = [
        config_path.resolve(),
        (repo_root / str(cfg["train_manifest"])).resolve(),
        (repo_root / str(cfg["eval_manifest"])).resolve(),
    ]
    artifact_paths.extend((repo_root / item).resolve() for item in source_paths)

    files = []
    for path in artifact_paths:
        if not path.is_file():
            raise FileNotFoundError(f"provenance input is missing: {path}")
        files.append(
            {
                "path": _relative(repo_root, path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    files.sort(key=lambda item: item["path"])

    snapshot_payload = json.dumps(files, sort_keys=True, separators=(",", ":"))
    snapshot_sha256 = hashlib.sha256(snapshot_payload.encode("utf-8")).hexdigest()
    scoped_paths = [item["path"] for item in files]
    scoped_status = _git(
        repo_root,
        "status",
        "--short",
        "--untracked-files=all",
        "--",
        *scoped_paths,
    )

    return {
        "schema_version": 1,
        "captured_before_run": True,
        "git_head": _git(repo_root, "rev-parse", "HEAD"),
        "git_branch": _git(repo_root, "branch", "--show-current"),
        "scoped_git_dirty": bool(scoped_status),
        "scoped_git_status": scoped_status.splitlines(),
        "source_snapshot_sha256": snapshot_sha256,
        "files": files,
        "command": command,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": {
            name: _package_version(name)
            for name in ("torch", "transformers", "numpy")
        },
        "gpu_inventory": _gpu_inventory(),
        "hostname": socket.gethostname(),
        "execution_environment": dict(
            sorted((execution_environment or {}).items())
        ),
        "bpfree_environment": {
            key: value
            for key, value in sorted(os.environ.items())
            if key.startswith("BPFREE_")
        },
    }


def artifact_hashes(paths: Iterable[Path], *, repo_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(paths):
        if path.is_file():
            rows.append(
                {
                    "path": _relative(repo_root, path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return rows
