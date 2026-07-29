#!/usr/bin/env python3
"""Continuous PipeDream baseline with LoRA weight stashing over CPU/Gloo."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.distributed as dist
from torch.func import functional_call
from transformers import AutoModelForCausalLM

from sg_exe_trainer.metrics.activation_memory import SavedTensorTracker
from sg_exe_trainer.runtime.exactbp.cpu_runner import (
    CpuActionTrace,
    ExactBPCpuCommFair,
    _drain_sends,
    _post_send_tensor,
)
from sg_exe_trainer.runtime.transport.cpu import (
    backward_tag,
    configure_link_emulation,
    forward_tag,
    link_emulation_summary,
    sync_max_wall_ms,
    tensor_nbytes,
)
from sg_exe_trainer.common.trainable_modes import (
    configure_model_trainable,
    gradient_storage_nbytes,
    module_param_stats,
    optimizer_state_nbytes,
)
from sg_exe_trainer.runtime.exactbp.distributed_runtime import (
    RankConfig,
    StageEventRecorder,
    build_optimizer,
    build_stage_module,
    causal_lm_loss,
    load_batch_tensors,
    parse_devices,
    set_optimizer_lr,
    stage_memory_ledger,
)
from sg_exe_trainer.tasks.label_experiment import (
    lora_parameter_fingerprint,
    read_manifest,
    resolve_dtype,
    resolve_model_name,
)


def _batched(records: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    usable = (len(records) // batch_size) * batch_size
    return [records[i : i + batch_size] for i in range(0, usable, batch_size)]


class PipeDreamCpuComm(ExactBPCpuCommFair):
    """Continuous 1F1B with LoRA weight stashing and local gradient coalescing.

    This is intentionally not optimizer-equivalent to synchronous Exact-BP. Each
    stage updates independently after ``gradient_accumulation_steps`` local
    backward completions. A minibatch backward uses the trainable parameter
    version captured by its forward, while frozen trunk parameters are shared.
    """

    def __init__(
        self,
        *,
        optimizer: torch.optim.Optimizer,
        gradient_accumulation_steps: int,
        grad_clip: float,
        manifest_dir: Path,
        stage0_input_embedding: Optional[torch.nn.Module],
        perf_minimal_metrics: bool,
        **kwargs: Any,
    ) -> None:
        super().__init__(pipeline_schedule="1f1b", **kwargs)
        self.optimizer = optimizer
        self.accumulation_steps = int(gradient_accumulation_steps)
        self.grad_clip = float(grad_clip)
        self.manifest_dir = manifest_dir
        self.stage0_input_embedding = stage0_input_embedding
        self.perf_minimal_metrics = bool(perf_minimal_metrics)

        self.master_params = {
            name: param
            for name, param in self.module.named_parameters()
            if param.requires_grad
        }
        if not self.master_params:
            raise ValueError("PipeDream runner requires local trainable parameters")
        self.initial_master_params = {
            name: param.detach().float().cpu().clone()
            for name, param in self.master_params.items()
        }

        self.master_version = 0
        self.weight_versions: dict[int, dict[str, Any]] = {}
        self.local_backward_count = 0
        self.local_optimizer_steps = 0
        self.loss_sum = 0.0
        self.loss_count = 0
        self.peak_activation_stash = 0
        self.peak_live_weight_versions = 0
        self.peak_weight_stash_bytes = 0
        self.version_forward_counts: dict[int, int] = {}
        self.version_backward_counts: dict[int, int] = {}
        self.backward_version_lag_sum = 0
        self.backward_version_lag_count = 0
        self.max_backward_version_lag = 0
        self.missing_snapshot_gradients = 0

    def _recv_spec(
        self,
        *,
        kind: str,
        payload: dict[str, Any],
    ) -> tuple[tuple[int, ...], int, str, int]:
        records = payload["batch_records"]
        if not records:
            raise RuntimeError("empty physical minibatch")
        seq_len = int(records[0].get("seq_len", 0))
        if seq_len <= 0:
            raise ValueError("manifest records must contain positive seq_len")
        shape = (len(records), seq_len, self.hidden_size)
        global_mb_id = int(payload["global_mb_id"])
        if kind == "fwd":
            return shape, self.rank - 1, "FWD_HIDDEN", forward_tag(self.rank - 1, global_mb_id)
        if kind == "bwd":
            return shape, self.rank + 1, "BWD_GRAD", backward_tag(self.rank, global_mb_id)
        raise ValueError(f"unknown action kind: {kind}")

    def _prepost_recv_action(
        self,
        *,
        kind: str,
        payload: dict[str, Any],
        phase: str,
        batch_seq: int,
    ) -> bool:
        # Preposted receives must retain the logical window of their own payload.
        return super()._prepost_recv_action(
            kind=kind,
            payload=payload,
            phase=phase,
            batch_seq=int(payload["update_seq"]),
        )

    def _snapshot_bytes(self) -> int:
        return sum(
            tensor_nbytes(param)
            for entry in self.weight_versions.values()
            for param in entry["params"].values()
        )

    def reset_memory_peaks(self) -> None:
        super().reset_memory_peaks()
        self.peak_activation_stash = 0
        self.peak_live_weight_versions = 0
        self.peak_weight_stash_bytes = 0

    def _update_stash_peaks(self) -> None:
        self._update_activation_cache_peak()
        self.peak_activation_stash = max(self.peak_activation_stash, len(self.cache))
        self.peak_live_weight_versions = max(
            self.peak_live_weight_versions,
            len(self.weight_versions),
        )
        self.peak_weight_stash_bytes = max(
            self.peak_weight_stash_bytes,
            self._snapshot_bytes(),
        )

    def _acquire_current_version(self, payload: dict[str, Any]) -> tuple[int, dict[str, torch.Tensor]]:
        version_id = self.master_version
        entry = self.weight_versions.get(version_id)
        if entry is None:
            batch_seq = int(payload["update_seq"])
            with self.trace.span(
                phase="train",
                batch_seq=batch_seq,
                mb_id=int(payload["mb_id"]),
                seq_start=int(payload["seq_start"]),
                records=len(payload["batch_records"]),
                action="WEIGHT_SNAPSHOT_CLONE",
            ):
                params = {
                    name: param.detach().clone().requires_grad_(True)
                    for name, param in self.master_params.items()
                }
                self._sync_cuda_for_trace(batch_seq)
            entry = {"params": params, "refcount": 0}
            self.weight_versions[version_id] = entry
        entry["refcount"] += 1
        self.version_forward_counts[version_id] = self.version_forward_counts.get(version_id, 0) + 1
        self._update_stash_peaks()
        return version_id, entry["params"]

    def _release_version(self, version_id: int) -> None:
        entry = self.weight_versions[version_id]
        entry["refcount"] -= 1
        if entry["refcount"] < 0:
            raise RuntimeError(f"negative weight-version refcount: {version_id}")
        if entry["refcount"] == 0 and version_id != self.master_version:
            del self.weight_versions[version_id]

    def _load_payload(self, payload: dict[str, Any]) -> dict[str, Optional[torch.Tensor]]:
        batch_seq = int(payload["update_seq"])
        with self.trace.span(
            phase="train",
            batch_seq=batch_seq,
            mb_id=int(payload["mb_id"]),
            seq_start=int(payload["seq_start"]),
            records=len(payload["batch_records"]),
            action="INPUT_LOAD_H2D",
        ):
            loaded = load_batch_tensors(
                records=payload["batch_records"],
                manifest_dir=self.manifest_dir,
                device=self.device,
                dtype=self.dtype,
                load_hidden=self.is_first,
                load_labels=self.is_last,
                input_embedding=self.stage0_input_embedding if self.is_first else None,
            )
            self._sync_cuda_for_trace(batch_seq)
        return loaded

    def forward_stream(self, payload: dict[str, Any]) -> None:
        batch_seq = int(payload["update_seq"])
        loaded = self._load_payload(payload)
        if self.is_first:
            hidden = loaded["hidden"]
            if hidden is None:
                raise RuntimeError("rank0 missing loaded hidden")
        else:
            hidden = self._take_recv(
                kind="fwd",
                payload=payload,
                phase="train",
                batch_seq=batch_seq,
            )
            hidden.requires_grad_(True)

        attention_mask = loaded["attention_mask"]
        position_ids = loaded["position_ids"]
        if attention_mask is None or position_ids is None:
            raise RuntimeError("missing attention_mask/position_ids")

        version_id, version_params = self._acquire_current_version(payload)
        with self.trace.span(
            phase="train",
            batch_seq=batch_seq,
            mb_id=int(payload["mb_id"]),
            seq_start=int(payload["seq_start"]),
            records=len(payload["batch_records"]),
            action="LOCAL_FORWARD_PIPEDREAM",
        ):
            output = functional_call(
                self.module,
                version_params,
                (hidden,),
                {
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                },
                strict=False,
            )
            self._sync_cuda_for_trace(batch_seq)

        if not self.is_last:
            _post_send_tensor(
                tensor_gpu=output,
                dst=self.rank + 1,
                pending=self.pending_sends,
                trace=self.trace,
                budget=self.transport_budget,
                phase="train",
                batch_seq=batch_seq,
                mb_id=int(payload["mb_id"]),
                seq_start=int(payload["seq_start"]),
                records=len(payload["batch_records"]),
                prefix="FWD_HIDDEN",
                tag=forward_tag(self.rank, int(payload["global_mb_id"])),
            )

        self.cache[int(payload["global_mb_id"])] = {
            "hidden": hidden,
            "output": output,
            "labels": loaded["labels"],
            "payload": payload,
            "version_id": version_id,
        }
        self._update_stash_peaks()

    def _accumulate_stashed_gradients(self, version_id: int, payload: dict[str, Any]) -> None:
        version_params = self.weight_versions[version_id]["params"]
        batch_seq = int(payload["update_seq"])
        with self.trace.span(
            phase="train",
            batch_seq=batch_seq,
            mb_id=int(payload["mb_id"]),
            seq_start=int(payload["seq_start"]),
            records=len(payload["batch_records"]),
            action="WEIGHT_GRAD_ACCUM",
        ):
            for name, snapshot_param in version_params.items():
                grad = snapshot_param.grad
                if grad is None:
                    self.missing_snapshot_gradients += 1
                    continue
                master = self.master_params[name]
                if master.grad is None:
                    master.grad = grad.detach().clone()
                else:
                    master.grad.add_(grad.detach())
                snapshot_param.grad = None
            self._sync_cuda_for_trace(batch_seq)

    def _maybe_optimizer_step(self, payload: dict[str, Any]) -> None:
        if self.local_backward_count % self.accumulation_steps != 0:
            return
        batch_seq = int(payload["update_seq"])
        if self.grad_clip > 0:
            with self.trace.span(
                phase="train",
                batch_seq=batch_seq,
                mb_id=int(payload["mb_id"]),
                seq_start=int(payload["seq_start"]),
                records=len(payload["batch_records"]),
                action="GRAD_CLIP",
            ):
                torch.nn.utils.clip_grad_norm_(self.master_params.values(), self.grad_clip)
                self._sync_cuda_for_trace(batch_seq)

        old_version = self.master_version
        with self.trace.span(
            phase="train",
            batch_seq=batch_seq,
            mb_id=int(payload["mb_id"]),
            seq_start=int(payload["seq_start"]),
            records=len(payload["batch_records"]),
            action="OPTIMIZER_STEP_ASYNC",
        ):
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self._sync_cuda_for_trace(batch_seq)
        self.local_optimizer_steps += 1
        self.master_version += 1
        old_entry = self.weight_versions.get(old_version)
        if old_entry is not None and old_entry["refcount"] == 0:
            del self.weight_versions[old_version]

    def backward_stream(self, payload: dict[str, Any]) -> None:
        global_mb_id = int(payload["global_mb_id"])
        entry = self.cache.pop(global_mb_id)
        hidden = entry["hidden"]
        output = entry["output"]
        labels = entry["labels"]
        version_id = int(entry["version_id"])
        batch_seq = int(payload["update_seq"])
        version_lag = self.master_version - version_id
        if version_lag < 0:
            raise RuntimeError(
                f"backward saw future weight version: master={self.master_version} forward={version_id}"
            )
        self.backward_version_lag_sum += version_lag
        self.backward_version_lag_count += 1
        self.max_backward_version_lag = max(self.max_backward_version_lag, version_lag)

        if self.is_last:
            if labels is None:
                raise RuntimeError("last stage missing labels")
            with self.trace.span(
                phase="train",
                batch_seq=batch_seq,
                mb_id=int(payload["mb_id"]),
                seq_start=int(payload["seq_start"]),
                records=len(payload["batch_records"]),
                action="LOCAL_BACKWARD_PIPEDREAM_LOSS",
            ):
                loss = causal_lm_loss(
                    output,
                    labels,
                    label_smoothing=self.label_smoothing,
                )
                if not self.perf_minimal_metrics:
                    self.loss_sum += float(loss.detach().cpu().item())
                    self.loss_count += 1
                (loss / self.accumulation_steps).backward()
                self._sync_cuda_for_trace(batch_seq)
        else:
            grad_out = self._take_recv(
                kind="bwd",
                payload=payload,
                phase="train",
                batch_seq=batch_seq,
            )
            with self.trace.span(
                phase="train",
                batch_seq=batch_seq,
                mb_id=int(payload["mb_id"]),
                seq_start=int(payload["seq_start"]),
                records=len(payload["batch_records"]),
                action="LOCAL_BACKWARD_PIPEDREAM_GRAD",
            ):
                output.backward(grad_out)
                self._sync_cuda_for_trace(batch_seq)

        if not self.is_first:
            grad_in = hidden.grad
            if grad_in is None:
                raise RuntimeError(f"missing hidden.grad stage={self.rank} mb={global_mb_id}")
            _post_send_tensor(
                tensor_gpu=grad_in,
                dst=self.rank - 1,
                pending=self.pending_sends,
                trace=self.trace,
                budget=self.transport_budget,
                phase="train",
                batch_seq=batch_seq,
                mb_id=int(payload["mb_id"]),
                seq_start=int(payload["seq_start"]),
                records=len(payload["batch_records"]),
                prefix="BWD_GRAD",
                tag=backward_tag(self.rank - 1, global_mb_id),
            )

        self._accumulate_stashed_gradients(version_id, payload)
        self.version_backward_counts[version_id] = self.version_backward_counts.get(version_id, 0) + 1
        self._release_version(version_id)
        self.local_backward_count += 1
        self._maybe_optimizer_step(payload)

    def run_stream(self, payloads: list[dict[str, Any]]) -> dict[str, Any]:
        plan = self._build_action_plan(payloads)
        self.optimizer.zero_grad(set_to_none=True)
        for index, (kind, payload) in enumerate(plan):
            self._maintain_recv_preposts(
                plan=plan,
                start_index=index,
                phase="train",
                batch_seq=int(payload["update_seq"]),
            )
            if kind == "fwd":
                self.forward_stream(payload)
            else:
                self.backward_stream(payload)

        if self.recv_entries:
            raise RuntimeError(f"stream ended with unconsumed receives: {sorted(self.recv_entries)}")
        _drain_sends(
            pending=self.pending_sends,
            trace=self.trace,
            budget=self.transport_budget,
        )
        if self.cache:
            raise RuntimeError(f"stream ended with activation cache entries: {sorted(self.cache)}")
        if self.local_backward_count % self.accumulation_steps:
            raise RuntimeError("stream ended with a partial local gradient window")
        for version_id, entry in self.weight_versions.items():
            if entry["refcount"] != 0:
                raise RuntimeError(f"live weight version at stream end: {version_id}")

        return {
            "local_backward_count": self.local_backward_count,
            "local_optimizer_steps": self.local_optimizer_steps,
            "final_master_version": self.master_version,
            "peak_activation_stash": self.peak_activation_stash,
            "peak_live_weight_versions": self.peak_live_weight_versions,
            "peak_weight_stash_bytes": self.peak_weight_stash_bytes,
            "version_forward_counts": self.version_forward_counts,
            "version_backward_counts": self.version_backward_counts,
            "mean_backward_version_lag": (
                self.backward_version_lag_sum / self.backward_version_lag_count
                if self.backward_version_lag_count
                else 0.0
            ),
            "max_backward_version_lag": self.max_backward_version_lag,
            "missing_snapshot_gradients": self.missing_snapshot_gradients,
            "avg_loss": self.loss_sum / self.loss_count if self.loss_count else "",
            "loss_count": self.loss_count,
        }

    def parameter_change_stats(self) -> dict[str, float]:
        delta_sq = 0.0
        initial_sq = 0.0
        for name, param in self.master_params.items():
            initial = self.initial_master_params[name]
            current = param.detach().float().cpu()
            delta_sq += float(torch.sum((current - initial) ** 2).item())
            initial_sq += float(torch.sum(initial**2).item())
        return {
            "trainable_delta_l2": delta_sq**0.5,
            "initial_trainable_l2": initial_sq**0.5,
        }


def _evaluator_compatible_trainable_state(
    module: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    layer_start = int(getattr(module, "layer_start", 0))
    state: dict[str, torch.Tensor] = {}
    for local_name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        parts = local_name.split(".")
        if len(parts) >= 3 and parts[0] == "layers" and parts[1].isdigit():
            global_layer = layer_start + int(parts[1])
            full_name = ".".join(["model", "layers", str(global_layer), *parts[2:]])
        else:
            full_name = local_name
        if full_name in state:
            raise RuntimeError(f"duplicate evaluator checkpoint key: {full_name}")
        state[full_name] = param.detach().cpu()
    return state


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exploratory continuous PipeDream 1F1B with LoRA weight stashing."
    )
    parser.add_argument("--model_name", default="tinyllama")
    parser.add_argument("--train_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_chunks", type=int, default=3)
    parser.add_argument("--stage_devices", default="cuda:0,cuda:1,cuda:2")
    parser.add_argument("--train_limit", type=int, default=96)
    parser.add_argument("--train_epochs", type=int, default=1)
    parser.add_argument("--physical_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--recv_prepost_depth", type=int, default=0)
    parser.add_argument("--max_pending_send_bytes", type=int, default=67_108_864)
    parser.add_argument("--max_posted_recv_bytes", type=int, default=67_108_864)
    parser.add_argument("--link_latency_ms", type=float, default=0.0)
    parser.add_argument("--link_bandwidth_mbps", type=float, default=0.0)
    parser.add_argument("--link_jitter_ms", type=float, default=0.0)
    parser.add_argument("--link_emulation_seed", type=int, default=20260531)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--sgd_momentum", type=float, default=0.0)
    parser.add_argument("--sgd_dampening", type=float, default=0.0)
    parser.add_argument("--sgd_weight_decay", type=float, default=0.0)
    parser.add_argument("--sgd_nesterov", action="store_true")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--trainable_mode", default="lora")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_targets", default="q_proj,v_proj")
    parser.add_argument("--lora_init_std", type=float, default=0.01)
    parser.add_argument("--lora_init_seed", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--enable_action_trace", action="store_true")
    parser.add_argument("--sync_action_trace", action="store_true")
    parser.add_argument("--action_trace_start_window", type=int, default=0)
    parser.add_argument("--action_trace_end_window", type=int, default=None)
    parser.add_argument("--perf_minimal_metrics", action="store_true")
    memory_group = parser.add_mutually_exclusive_group()
    memory_group.add_argument(
        "--track_activation_memory",
        dest="track_activation_memory",
        action="store_true",
    )
    memory_group.add_argument(
        "--no-track_activation_memory",
        dest="track_activation_memory",
        action="store_false",
    )
    parser.set_defaults(track_activation_memory=False)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if not dist.is_initialized():
        dist.init_process_group("gloo")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != args.num_chunks:
        raise ValueError(f"WORLD_SIZE={world_size} must equal num_chunks={args.num_chunks}")
    if args.gradient_accumulation_steps < args.num_chunks:
        raise ValueError("continuous 1F1B pilot requires accumulation steps >= num_chunks")
    if args.physical_batch_size <= 0 or args.gradient_accumulation_steps <= 0:
        raise ValueError("batch and accumulation values must be positive")

    torch.manual_seed(int(args.seed) + rank)
    devices = parse_devices(args.stage_devices, args.num_chunks)
    device = torch.device(devices[rank])
    if device.type == "cuda":
        torch.cuda.set_device(device)
    configure_link_emulation(
        one_way_latency_ms=args.link_latency_ms,
        bandwidth_mbps=args.link_bandwidth_mbps,
        jitter_ms=args.link_jitter_ms,
        seed=args.link_emulation_seed + rank,
    )

    dtype = resolve_dtype(args.dtype)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_model = resolve_model_name(args.model_name)
    if rank == 0:
        print(
            "Starting continuous PipeDream pilot "
            f"model={resolved_model} devices={devices} physical={args.physical_batch_size} "
            f"local_accum={args.gradient_accumulation_steps}",
            flush=True,
        )

    recorder = StageEventRecorder(
        stage_id=rank,
        rank=rank,
        device_name=devices[rank],
        enabled=False,
    )
    model = AutoModelForCausalLM.from_pretrained(resolved_model, torch_dtype=dtype)
    hidden_size = int(getattr(model.config, "hidden_size", 0) or getattr(model.config, "n_embd", 0) or 0)
    vocab_size = int(getattr(model.config, "vocab_size", 0) or 0)
    trainable_setup = configure_model_trainable(
        module=model,
        mode=args.trainable_mode,
        lora_targets=args.lora_targets,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_init_std=args.lora_init_std,
        lora_init_seed=args.lora_init_seed,
    )
    lora_init_fingerprint = lora_parameter_fingerprint(model)
    stage0_input_embedding = model.get_input_embeddings() if rank == 0 else None
    module = build_stage_module(
        model=model,
        stage_id=rank,
        num_chunks=args.num_chunks,
        recorder=recorder,
    )
    module.to(device)
    if stage0_input_embedding is not None:
        stage0_input_embedding.to(device)
        stage0_input_embedding.eval()
        for param in stage0_input_embedding.parameters():
            param.requires_grad = False

    local_param_stats = module_param_stats(module)
    memory_ledger = stage_memory_ledger(module=module, input_embedding=stage0_input_embedding)
    optimizer = build_optimizer(
        params=[param for param in module.parameters() if param.requires_grad],
        optimizer_name=args.optimizer,
        learning_rate=args.learning_rate,
        sgd_momentum=args.sgd_momentum,
        sgd_dampening=args.sgd_dampening,
        sgd_weight_decay=args.sgd_weight_decay,
        sgd_nesterov=args.sgd_nesterov,
    )
    set_optimizer_lr(optimizer, args.learning_rate)

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    cfg = RankConfig(
        stage_id=rank,
        rank=rank,
        world_size=world_size,
        device=device,
        device_name=devices[rank],
        dtype=dtype,
        label_smoothing=args.label_smoothing,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        trainable_mode=args.trainable_mode,
        local_params=local_param_stats.params,
        local_trainable_params=local_param_stats.trainable_params,
        local_param_bytes=local_param_stats.bytes,
        local_trainable_param_bytes=local_param_stats.trainable_bytes,
        **memory_ledger,
    )
    trace = CpuActionTrace(
        rank=rank,
        stage_id=rank,
        output_dir=output_dir,
        enabled=args.enable_action_trace,
        min_batch_seq=args.action_trace_start_window,
        max_batch_seq=args.action_trace_end_window,
    )
    schedule = PipeDreamCpuComm(
        module=module,
        rank=rank,
        world_size=world_size,
        device=device,
        dtype=dtype,
        hidden_size=hidden_size,
        label_smoothing=args.label_smoothing,
        trace=trace,
        recv_prepost_depth=args.recv_prepost_depth,
        max_pending_send_bytes=args.max_pending_send_bytes,
        max_posted_recv_bytes=args.max_posted_recv_bytes,
        sync_action_trace=args.sync_action_trace,
        optimizer=optimizer,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        grad_clip=args.grad_clip,
        manifest_dir=Path(args.train_manifest).resolve().parent,
        stage0_input_embedding=stage0_input_embedding,
        perf_minimal_metrics=args.perf_minimal_metrics,
    )

    base_records = read_manifest(Path(args.train_manifest), args.train_limit)
    records = base_records * int(args.train_epochs)
    physical_batches = _batched(records, args.physical_batch_size)
    if len(physical_batches) % args.gradient_accumulation_steps:
        raise ValueError(
            "number of physical minibatches must be divisible by gradient_accumulation_steps"
        )
    payloads = [
        {
            "mb_id": global_mb_id % args.gradient_accumulation_steps,
            "global_mb_id": global_mb_id,
            "update_seq": global_mb_id // args.gradient_accumulation_steps,
            "seq_start": global_mb_id * args.physical_batch_size,
            "batch_records": batch_records,
        }
        for global_mb_id, batch_records in enumerate(physical_batches)
    ]

    module.train(True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    dist.barrier()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        baseline_cuda_allocated = int(torch.cuda.memory_allocated(device))
        baseline_cuda_reserved = int(torch.cuda.memory_reserved(device))
    else:
        baseline_cuda_allocated = 0
        baseline_cuda_reserved = 0
    activation_tracker = SavedTensorTracker(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
    )
    activation_tracker.reset()
    schedule.reset_memory_peaks()
    hook_context = (
        torch.autograd.graph.saved_tensors_hooks(
            activation_tracker.pack,
            activation_tracker.unpack,
        )
        if args.track_activation_memory
        else nullcontext()
    )
    started = time.perf_counter()
    with torch.enable_grad(), hook_context:
        stream_stats = schedule.run_stream(payloads)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    local_wall_ms = (time.perf_counter() - started) * 1000.0
    wall_ms = sync_max_wall_ms(local_wall_ms)
    completed_records = len(physical_batches) * args.physical_batch_size
    peak_cuda_allocated = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0
    )
    peak_cuda_reserved = (
        int(torch.cuda.max_memory_reserved(device))
        if device.type == "cuda"
        else 0
    )
    activation_stats = (
        activation_tracker.snapshot()
        if args.track_activation_memory
        else {}
    )
    actions_csv = trace.flush("train")

    parameter_change = schedule.parameter_change_stats()
    trainable_state = _evaluator_compatible_trainable_state(module)
    torch.save(trainable_state, output_dir / f"stage{rank}.trainable.pt")

    local_summary = {
        "rank": rank,
        "stage_id": rank,
        "device": devices[rank],
        "local_wall_ms": local_wall_ms,
        "completed_records": completed_records,
        "throughput_per_s_using_global_wall": completed_records / (wall_ms / 1000.0),
        "baseline_cuda_allocated_bytes": baseline_cuda_allocated,
        "baseline_cuda_reserved_bytes": baseline_cuda_reserved,
        "peak_cuda_allocated_bytes": peak_cuda_allocated,
        "peak_cuda_reserved_bytes": peak_cuda_reserved,
        "peak_runtime_delta_bytes": max(
            0, peak_cuda_allocated - baseline_cuda_allocated
        ),
        "gradient_storage_bytes": gradient_storage_nbytes(module),
        "optimizer_state_bytes": optimizer_state_nbytes(optimizer),
        "activation_tracking_enabled": bool(args.track_activation_memory),
        "activation_memory": activation_stats,
        **schedule.activation_cache_summary(),
        "memory_ledger": memory_ledger,
        "transport_budget": schedule.transport_summary(),
        "actions_csv": actions_csv,
        **parameter_change,
        **stream_stats,
    }
    (output_dir / f"rank{rank}.summary.json").write_text(
        json.dumps(local_summary, indent=2),
        encoding="utf-8",
    )
    by_rank: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(by_rank, local_summary)
    dist.barrier()

    if rank == world_size - 1:
        summary = {
            "runner": "pipedream-continuous-1f1b-weight-stash-cpu-comm",
            "status": "exploratory_baseline",
            "optimization_semantics": (
                "continuous cross-window 1F1B; stage-local asynchronous optimizer steps; "
                "LoRA weight stashing binds each backward to its forward version; local "
                "gradient coalescing preserves the requested effective batch size"
            ),
            "not_equivalent_to": "synchronous Exact-BP / PipeDream-Flush",
            "transport": "gloo-cpu-hidden-and-grad-pinned",
            "model_name": args.model_name,
            "resolved_model": resolved_model,
            "num_chunks": args.num_chunks,
            "stage_devices": devices,
            "physical_request_batch": args.physical_batch_size,
            "local_gradient_coalescing": args.gradient_accumulation_steps,
            "effective_optimizer_batch": args.physical_batch_size * args.gradient_accumulation_steps,
            "completed_records": completed_records,
            "optimizer_steps_per_stage": len(physical_batches) // args.gradient_accumulation_steps,
            "wall_ms": wall_ms,
            "throughput_per_s": completed_records / (wall_ms / 1000.0),
            "dtype": str(dtype).replace("torch.", ""),
            "seed": args.seed,
            "lora": {
                "rank": args.lora_rank,
                "alpha": args.lora_alpha,
                "targets": args.lora_targets,
                "init_std": args.lora_init_std,
                "init_seed": args.lora_init_seed,
                "initialization_fingerprint": lora_init_fingerprint,
                "modules": trainable_setup.lora_modules,
            },
            "recv_prepost_depth": args.recv_prepost_depth,
            "link_emulation": link_emulation_summary(),
            "by_rank": by_rank,
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
        print(f"Wrote {output_dir / 'summary.json'}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
