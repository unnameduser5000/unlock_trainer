# Patched ExecuTorch Android runtime

`executorch-android-bpfree-1.2.0.aar` is the runtime deployed by the Android
worker. The stock `org.pytorch:executorch-android:1.2.0` artifact does not
contain the split forward/backward JNI API or the BP-free boundary operator,
so it is not a compatible replacement.

The adjacent JSON file records the exact upstream revision, build switches,
archive hashes, and required contents. Verify the checked-in binary with:

```bash
python tools/android/verify_bpfree_aar.py
```

Compiler-generated absolute source paths in the native library use the
equal-length placeholder `anonymous`. The adjacent manifest records both the
checked-in artifact hash and the validated build hash; executable code and
archive structure are otherwise unchanged.

To rebuild from an existing ExecuTorch checkout, set `ANDROID_NDK` and
`ANDROID_SDK`, then run:

```bash
tools/android/build_bpfree_aar.sh /path/to/executorch
```

The build uses the source patch and provenance files under
`third_party/executorch_patches/`. A newly built archive is checked for the
required API and native operators. Its full byte hash can differ when the
Android NDK, SDK, CMake, Gradle, or host toolchain differs from the canonical
artifact recorded in the JSON manifest.
