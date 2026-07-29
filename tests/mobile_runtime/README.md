# Mobile split-runtime host check

`bpfree_module_harness` is the production semantic check for the patched
ExecuTorch module. Given one boundary-enabled training PTE, it verifies that:

1. forward execution pauses at the BP-free boundary;
2. a second forward is rejected while the method is paused;
3. backward resumes and produces named gradients;
4. split and atomic execution produce matching gradients; and
5. the split module can execute a second complete cycle.

Build against an already patched and installed host ExecuTorch tree:

```bash
cmake -S tests/mobile_runtime -B build/mobile_runtime \
  -DEXECUTORCH_ROOT=/path/to/executorch \
  -DCMAKE_PREFIX_PATH=/path/to/executorch/cmake-out
cmake --build build/mobile_runtime --target bpfree_module_harness
build/mobile_runtime/bpfree_module_harness /path/to/chunk_0_joint_bpfree.pte 4
```

The harness is intentionally separate from experiment launchers. It tests the
runtime contract and does not encode an E1-E5 workload or server topology.
