# BP-Free ExecuTorch Operator Notes

Last updated: 2026-05-29 Asia/Shanghai

Read this after `docs/CURRENT_DEBUG_STATE.md` when discussing operator-level work.

## Current Conclusion

Do not describe BP-free as "no backward". The current intended algorithm is:

- each phone runs local forward, local backward, and local LoRA optimizer;
- phones do not send backward gradients to previous stages;
- cross-stage traffic is forward-only hidden state plus optional belief/log-prob signal.

The exported graph probably already avoids a cross-stage hidden-gradient output because `tools/export/sid_export_mobile.py` builds `dummy_hidden` without `requires_grad=True`, and LoRA export only leaves LoRA A/B tensors trainable. Therefore "remove the redundant hidden-gradient op" is not yet a proven fix. In ExecuTorch AOT training there is no single `aten::backward` op to delete; backward is lowered into ordinary ATen ops such as `mm`, `mul`, `sum`, `slice`, and the graph signature marks which outputs are parameter gradients.

The real operator/runtime mismatch points are:

1. dense CE/KL/log-prob branch in every chunk;
2. duplicated full `final_norm + lm_head` in every chunk;
3. generic ExecuTorch training output layout: user outputs, gradient outputs, then parameter outputs;
4. Android `TrainingModule` loading via `FileDataLoader`, which allocates PTE segments into private memory;
5. Java/native `namedGradients()` to `SGD.step(map)` path, which is generic rather than LoRA-specific.

## Current Export Facts

Project exporter: `tools/export/sid_export_mobile.py`.

- LoRA path freezes all parameters, then re-enables only `LoRALinear.lora_a` and `LoRALinear.lora_b`.
- Example `hidden_states` input is a normal float32 tensor, not `requires_grad=True`.
- Nonzero chunks take `prev_log_probs`.
- Every chunk computes `logits = lm_head(final_norm(curr_hidden))`.
- Every chunk computes local CE.
- Nonzero chunks also compute KL against previous belief.
- Every chunk returns dense `next_hidden` and dense `next_log_probs`.

Current local three-stage LoRA PTE marker check:

```text
model/tinyllama_lora_chunk_0.pte: size=1496232320 __et_training=3 aten::_softmax=1 aten::_log_softmax=1
model/tinyllama_lora_chunk_1.pte: size=1496236800 __et_training=3 aten::_softmax=1 aten::_log_softmax=1
model/tinyllama_lora_chunk_2.pte: size=1672526848 __et_training=3 aten::_softmax=1 aten::_log_softmax=1
```

This proves the PTEs are training artifacts and include softmax/log-softmax symbols. It does not by itself prove which graph node produced each symbol.

## ExecuTorch Locations

Training graph model:

- `executorch/extension/training/README.md`: training captures the backward graph ahead of time; gradients become explicit graph outputs; weights are mutable and memory planned.
- `executorch/exir/emit/_emit_program.py`: emits hidden metadata functions `__et_training_gradients_index_forward`, `__et_training_parameters_index_forward`, and `__et_training_fqn_forward`.
- `executorch/exir/passes/weights_to_outputs_pass.py`: appends trainable weights to graph outputs so TrainingModule can expose/update them.
- `executorch/extension/training/module/training_module.cpp`: `execute_forward_backward()` executes the joint graph, returns only user outputs, and stores gradient tensor aliases for `named_gradients()`.

Android training bridge:

- `executorch/extension/android/jni/jni_layer_training.cpp`: `executeForwardBackwardNative`, `namedParametersNative`, `namedGradientsNative`, and SGD JNI bridge.
- Important memory finding: the training JNI constructor uses `FileDataLoader::from(modelPath)`.
- `executorch/extension/data_loader/file_data_loader.h/.cpp`: `FileDataLoader` loads segments from file and allocates memory for them.
- In contrast, regular `Module` has mmap load modes in `executorch/extension/module/module.cpp` and Java `Module.load(path, LOAD_MODE_MMAP)`, but Maven `TrainingModule` 1.2.0 exposes no load-mode parameter.

Kernel/operator targets:

- `executorch/kernels/optimized/cpu/op_log_softmax.cpp`
- `executorch/kernels/portable/cpu/op_log_softmax.cpp`
- `executorch/kernels/optimized/cpu/op_linear.cpp`
- `executorch/kernels/optimized/cpu/op_mm.cpp`
- custom op path: `executorch/examples/portable/custom_ops/README.md` plus `custom_ops.yaml`.

Memory planning:

- `executorch/exir/memory_planning.py`
- `executorch/exir/passes/memory_planning_pass.py`
- `executorch/runtime/executor/memory_manager.h`
- `executorch/runtime/core/hierarchical_allocator.h`

Do not start by changing the allocator. First reduce exported graph/PTE pressure and prove the memory change.

## What To Prove On Server

The exporter now has a diagnostic switch:

```bash
DUMP_JOINT_GRAPH=1 NUM_CHUNKS=3 CHUNK_IDX=-1 OUTPUT_DIR=model \
  bash tools/export/export_lora_tinyllama.sh
```

This writes `model/tinyllama_lora_chunk_<idx>.joint_graph.txt`.

Check:

```bash
grep -n "non_parameter_gradient_count\\|parameter_grad=\\|log_softmax\\|kl_div\\|cross_entropy" \
  model/tinyllama_lora_chunk_*.joint_graph.txt
```

Expected BP-free boundary result:

- `non_parameter_gradient_count=0`;
- gradient targets are only LoRA parameter names;
- no output spec corresponds to `hidden_states` or `prev_log_probs`.

If this is true, the paper claim should be: "the boundary gradient is elided by construction in the exported joint graph signature." It should not claim that we found and removed an existing hidden-gradient operator.

## Concrete Optimization Order

1. Export-level proof and cleanup:
   - run `DUMP_JOINT_GRAPH=1`;
   - add a CE-only export mode if belief remains ineffective;
   - remove `prev_log_probs`, KL, and dense `next_log_probs` from CE-only artifacts.

2. Android training loader:
   - build a custom ExecuTorch Android AAR or JNI bridge where `TrainingModule` can use `MmapDataLoader`;
   - expose `TrainingModule.load(modelPath, loadMode)` or default training load to mmap;
   - measure app PSS after model load before/after. This directly targets the 3.6GB PSS plateau.

3. Custom CE/stat operator:
   - do not globally patch `aten::_log_softmax`;
   - add a project op such as `sid::masked_lm_ce_stats.out`;
   - compute loss and token stats without returning dense `[B,S,V]` log-probs;
   - register Python-side op for export and C++ out-variant kernel for ExecuTorch.

4. LoRA-specific local update path:
   - current path materializes named gradients and crosses Java/native maps;
   - a custom native fused path can execute forward/backward and update LoRA tensors without `namedGradients()` and `SGD.step(map)`;
   - this is useful after loader/export memory pressure is under control.

5. Bigger algorithm choice:
   - if every chunk keeps a full `lm_head`, more phones reduce transformer-layer memory but do not remove duplicated vocabulary-head memory;
   - a true system contribution can be an auxiliary/local-head design, terminal-only head with a different local objective, or a fused/quantized head op, but this changes the algorithm tradeoff and must be evaluated against quality.

