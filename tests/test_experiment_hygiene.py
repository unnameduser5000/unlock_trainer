from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_PATH_MARKERS = (
    "/home/user/",
    "/mnt/data/home/user/",
)


def test_experiment_launchers_do_not_pin_server_checkout() -> None:
    violations: list[str] = []
    experiment_root = REPO_ROOT / "experiments"
    for path in sorted(experiment_root.rglob("*")):
        if not path.is_file() or path.suffix not in {".py", ".sh"}:
            continue
        source = path.read_text(encoding="utf-8")
        for marker in SERVER_PATH_MARKERS:
            if marker in source:
                violations.append(
                    f"{path.relative_to(REPO_ROOT)} contains {marker}"
                )
    assert not violations, "\n".join(violations)
