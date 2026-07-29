# ExecuTorch source provenance

The Android worker uses a patched ExecuTorch runtime because upstream 1.2.0
does not expose a pause-at-boundary/resume-backward training API.

Upstream source:

- repository: `https://github.com/pytorch/executorch.git`
- tag: `v1.2.0`
- commit: `0b0e2c5cdd67c8b4396a46ea1d1aa72ffb0128d7`
- license: BSD, reproduced in `EXECUTORCH_LICENSE`

`executorch-v1.2.0-bpfree-boundary.patch` adds the boundary operator, the C++
training module, JNI bindings, Java API, native registration, and an explicit
optimized-kernel build switch. Apply it only to the exact commit above. The
upstream source tree is not vendored into this repository.

The canonical Android binary and its machine-readable build manifest live in
`app/libs/`. Use `tools/android/build_bpfree_aar.sh` for an isolated rebuild;
the script creates a temporary Git worktree and does not patch the supplied
ExecuTorch checkout in place.
