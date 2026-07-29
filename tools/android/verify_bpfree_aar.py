#!/usr/bin/env python3
"""Verify the checked-in BP-free ExecuTorch Android runtime artifact."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AAR = REPO_ROOT / "app" / "libs" / "executorch-android-bpfree-1.2.0.aar"
DEFAULT_METADATA = DEFAULT_AAR.with_suffix(".json")


class VerificationError(RuntimeError):
    pass


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def verify_artifact(
    aar_path: Path = DEFAULT_AAR,
    metadata_path: Path = DEFAULT_METADATA,
    *,
    exact_hashes: bool = True,
) -> dict[str, object]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    _require(aar_path.is_file(), f"AAR not found: {aar_path}")
    _require(metadata.get("artifact") == aar_path.name, "artifact name does not match manifest")

    artifact_hash = _sha256_file(aar_path)
    artifact_size = aar_path.stat().st_size
    if exact_hashes:
        _require(artifact_hash == metadata.get("artifact_sha256"), "AAR SHA-256 mismatch")
        _require(artifact_size == metadata.get("artifact_size"), "AAR size mismatch")

    contents = metadata["contents"]
    class_archive = contents["class_archive"]
    native_library = contents["native_library"]
    with zipfile.ZipFile(aar_path) as aar:
        names = set(aar.namelist())
        _require(class_archive in names, f"missing AAR entry: {class_archive}")
        _require(native_library in names, f"missing AAR entry: {native_library}")
        class_bytes = aar.read(class_archive)
        native_bytes = aar.read(native_library)

    with zipfile.ZipFile(io.BytesIO(class_bytes)) as classes:
        class_names = set(classes.namelist())
    for required_class in contents["required_java_classes"]:
        _require(required_class in class_names, f"missing Java class: {required_class}")

    for marker in contents["required_native_markers"]:
        _require(marker.encode("utf-8") in native_bytes, f"missing native marker: {marker}")

    native_hash = _sha256_bytes(native_bytes)
    native_size = len(native_bytes)
    if exact_hashes:
        _require(native_hash == contents.get("native_library_sha256"), "native library SHA-256 mismatch")
        _require(native_size == contents.get("native_library_size"), "native library size mismatch")

    return {
        "artifact": str(aar_path),
        "artifact_sha256": artifact_hash,
        "artifact_size": artifact_size,
        "native_library_sha256": native_hash,
        "native_library_size": native_size,
        "exact_hashes": exact_hashes,
        "passed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aar", type=Path, default=DEFAULT_AAR)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument(
        "--structural-only",
        action="store_true",
        help="check required classes/operators without enforcing canonical byte hashes",
    )
    args = parser.parse_args()
    try:
        report = verify_artifact(
            args.aar.resolve(),
            args.metadata.resolve(),
            exact_hashes=not args.structural_only,
        )
    except (OSError, KeyError, json.JSONDecodeError, zipfile.BadZipFile, VerificationError) as error:
        print(f"BP-free AAR verification failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
