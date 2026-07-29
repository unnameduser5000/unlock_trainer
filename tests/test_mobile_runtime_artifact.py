from __future__ import annotations

import ast
from pathlib import Path

from tools.android.verify_bpfree_aar import verify_artifact


REPO_ROOT = Path(__file__).resolve().parents[1]
AAR = REPO_ROOT / "app" / "libs" / "executorch-android-bpfree-1.2.0.aar"
METADATA = AAR.with_suffix(".json")
PATCH = (
    REPO_ROOT
    / "third_party"
    / "executorch_patches"
    / "executorch-v1.2.0-bpfree-boundary.patch"
)


def test_checked_in_bpfree_aar_matches_manifest() -> None:
    report = verify_artifact(AAR, METADATA)
    assert report["passed"] is True


def test_android_app_uses_only_the_patched_executorch_runtime() -> None:
    gradle = (REPO_ROOT / "app" / "build.gradle.kts").read_text(encoding="utf-8")
    assert 'implementation(files("libs/executorch-android-bpfree-1.2.0.aar"))' in gradle
    assert "org.pytorch:executorch-android" not in gradle


def test_source_patch_enables_the_required_runtime_features() -> None:
    source = PATCH.read_text(encoding="utf-8")
    assert "sid_bpfree_pipeline::pipeline_boundary.out" in source
    assert "BPFreeTrainingModule" in source
    assert "EXECUTORCH_BUILD_KERNELS_OPTIMIZED" in source


def test_mobile_export_defaults_to_terminal_belief_transport() -> None:
    exporter = REPO_ROOT / "tools" / "export" / "sid_export_mobile.py"
    tree = ast.parse(exporter.read_text(encoding="utf-8"), filename=str(exporter))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "normalize_belief_transport_mode"
    )
    namespace: dict[str, object] = {}
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(module, str(exporter), "exec"), namespace)
    normalize = namespace["normalize_belief_transport_mode"]
    assert callable(normalize)
    assert normalize("") == "terminal"
    assert normalize("terminal_only") == "terminal"
    assert normalize("full") == "full"
    assert normalize("none") == "none"

    argument = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and any(
            isinstance(value, ast.Constant) and value.value == "--belief_transport_mode"
            for value in node.args
        )
    )
    default = next(
        keyword.value for keyword in argument.keywords if keyword.arg == "default"
    )
    assert ast.literal_eval(default) == "terminal"
