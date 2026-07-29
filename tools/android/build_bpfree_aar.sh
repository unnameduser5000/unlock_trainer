#!/usr/bin/env bash
set -euo pipefail

readonly UPSTREAM_COMMIT="0b0e2c5cdd67c8b4396a46ea1d1aa72ffb0128d7"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
readonly PATCH_FILE="${REPO_ROOT}/third_party/executorch_patches/executorch-v1.2.0-bpfree-boundary.patch"
readonly METADATA_FILE="${REPO_ROOT}/app/libs/executorch-android-bpfree-1.2.0.json"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 EXECUTORCH_CHECKOUT [OUTPUT_AAR]" >&2
  exit 2
fi

readonly SOURCE_CHECKOUT="$(cd -- "$1" && pwd)"
readonly OUTPUT_AAR="${2:-${REPO_ROOT}/app/libs/executorch-android-bpfree-1.2.0.aar}"

: "${ANDROID_NDK:?set ANDROID_NDK to an Android NDK directory}"
: "${ANDROID_SDK:?set ANDROID_SDK to an Android SDK directory}"
command -v git >/dev/null
command -v cmake >/dev/null
command -v java >/dev/null
command -v python3 >/dev/null
git -C "${SOURCE_CHECKOUT}" cat-file -e "${UPSTREAM_COMMIT}^{commit}"

readonly BUILD_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sid-bpfree-aar.XXXXXX")"
readonly BUILD_CHECKOUT="${BUILD_ROOT}/executorch"

cleanup() {
  git -C "${SOURCE_CHECKOUT}" worktree remove --force "${BUILD_CHECKOUT}" >/dev/null 2>&1 || true
  case "${BUILD_ROOT}" in
    "${TMPDIR:-/tmp}"/sid-bpfree-aar.*) rm -rf -- "${BUILD_ROOT}" ;;
  esac
}
trap cleanup EXIT

git -C "${SOURCE_CHECKOUT}" worktree add --detach "${BUILD_CHECKOUT}" "${UPSTREAM_COMMIT}"
git -C "${BUILD_CHECKOUT}" submodule update --init --recursive
git -C "${BUILD_CHECKOUT}" apply --check "${PATCH_FILE}"
git -C "${BUILD_CHECKOUT}" apply "${PATCH_FILE}"

(
  cd "${BUILD_CHECKOUT}"
  export ANDROID_ABIS="arm64-v8a"
  export EXECUTORCH_CMAKE_BUILD_TYPE="Release"
  export EXECUTORCH_BUILD_EXTENSION_LLM="OFF"
  export EXECUTORCH_BUILD_KERNELS_OPTIMIZED="ON"
  export CFLAGS="${CFLAGS:-} -ffile-prefix-map=${BUILD_CHECKOUT}=/usr/src/executorch"
  export CXXFLAGS="${CXXFLAGS:-} -ffile-prefix-map=${BUILD_CHECKOUT}=/usr/src/executorch"
  ./scripts/build_android_library.sh
)

readonly BUILT_AAR="${BUILD_CHECKOUT}/extension/android/executorch_android/build/outputs/aar/executorch_android-release.aar"
test -f "${BUILT_AAR}"
mkdir -p -- "$(dirname -- "${OUTPUT_AAR}")"
install -m 0644 "${BUILT_AAR}" "${OUTPUT_AAR}"
python3 "${SCRIPT_DIR}/verify_bpfree_aar.py" \
  --aar "${OUTPUT_AAR}" \
  --metadata "${METADATA_FILE}" \
  --structural-only
sha256sum "${OUTPUT_AAR}"
