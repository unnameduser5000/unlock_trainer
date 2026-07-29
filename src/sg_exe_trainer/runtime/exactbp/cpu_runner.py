#!/usr/bin/env python3
"""Run exact-BP GPipe/1F1B over pinned-CPU/Gloo stage transport."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM

from sg_exe_trainer.runtime.transport.cpu import (
    CpuTransportBudget,
    configure_link_emulation,
    backward_tag,
    cpu_irecv,
    cpu_isend,
    cpu_to_gpu,
    forward_tag,
    gpu_to_cpu,
    link_emulation_summary,
    shape_nbytes,
    sync_max_wall_ms,
    tensor_nbytes,
    wait_work,
)

from sg_exe_trainer.runtime.exactbp.distributed_runtime import (
    RankConfig,
    StageEventRecorder,
    SavedTensorTracker,
    build_optimizer,
    build_stage_module,
    causal_lm_loss,
    load_batch_tensors,
    parse_devices,
    set_optimizer_lr,
    stage_memory_ledger,
)
from sg_exe_trainer.tasks.label_experiment import (
    read_manifest,
    resolve_dtype,
    resolve_model_name,
    lora_parameter_fingerprint,
)
from sg_exe_trainer.common.trainable_modes import (
    configure_model_trainable,
    gradient_storage_nbytes,
    module_param_stats,
    optimizer_state_nbytes,
)


class CpuActionTrace:
    def __init__(
        self,
        *,
        rank: int,
        stage_id: int,
        output_dir: Path,
        enabled: bool,
        min_batch_seq: int = 0,
        max_batch_seq: int | None = None,
    ) -> None:
        self.rank = int(rank)
        self.stage_id = int(stage_id)
        self.output_dir = output_dir
        self.enabled = bool(enabled)
        self.min_batch_seq = max(0, int(min_batch_seq))
        self.max_batch_seq = (
            None if max_batch_seq is None else int(max_batch_seq)
        )
        if (
            self.max_batch_seq is not None
            and self.max_batch_seq <= self.min_batch_seq
        ):
            raise ValueError(
                "max_batch_seq must be greater than min_batch_seq"
            )
        self.rows: list[dict[str, Any]] = []

    def enabled_for_batch(self, batch_seq: int) -> bool:
        if not self.enabled:
            return False
        batch = int(batch_seq)
        if batch < self.min_batch_seq:
            return False
        if self.max_batch_seq is not None and batch >= self.max_batch_seq:
            return False
        return True

    @contextmanager
    def span(
        self,
        *,
        phase: str,
        batch_seq: int,
        mb_id: int,
        seq_start: int,
        records: int,
        action: str,
    ):
        if not self.enabled_for_batch(batch_seq):
            yield
            return
        start_perf = time.perf_counter()
        start_epoch_ms = time.time() * 1000.0
        try:
            yield
        finally:
            end_perf = time.perf_counter()
            end_epoch_ms = time.time() * 1000.0
            self.rows.append(
                {
                    "phase": phase,
                    "stage_id": self.stage_id,
                    "rank": self.rank,
                    "batch_seq": batch_seq,
                    "mb_id": mb_id,
                    "seq_start": seq_start,
                    "records": records,
                    "action": action,
                    "start_epoch_ms": start_epoch_ms,
                    "end_epoch_ms": end_epoch_ms,
                    "duration_ms": (end_perf - start_perf) * 1000.0,
                }
            )

    def flush(self, phase: str) -> str:
        if not self.enabled:
            return ""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"{phase}.stage{self.stage_id}.actions.csv"
        fieldnames = [
            "phase",
            "stage_id",
            "rank",
            "batch_seq",
            "mb_id",
            "seq_start",
            "records",
            "action",
            "start_epoch_ms",
            "end_epoch_ms",
            "duration_ms",
        ]
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in self.rows:
                w.writerow({k: row.get(k, "") for k in fieldnames})
        return str(path)


def _hidden_shape_from_loaded(loaded: dict[str, Optional[torch.Tensor]], *, batch_records: list[dict[str, Any]], hidden_size: int) -> tuple[int, int, int]:
    pos = loaded["position_ids"]
    if pos is None:
        raise RuntimeError("position_ids missing")
    seq_len = int(pos.shape[-1])
    return (len(batch_records), seq_len, int(hidden_size))


def _drain_one_send(
    *,
    pending: list[dict[str, Any]],
    trace: CpuActionTrace,
    budget: CpuTransportBudget,
) -> None:
    entry = pending.pop(0)
    with trace.span(
        phase=entry["phase"],
        batch_seq=entry["batch_seq"],
        mb_id=entry["mb_id"],
        seq_start=entry["seq_start"],
        records=entry["records"],
        action=f"{entry['prefix']}_SEND_WAIT_CPU",
    ):
        wait_work(entry["work"])
    budget.release_send(int(entry["nbytes"]))
    entry.clear()


def _drain_sends(
    *,
    pending: list[dict[str, Any]],
    trace: CpuActionTrace,
    budget: CpuTransportBudget,
) -> None:
    while pending:
        _drain_one_send(pending=pending, trace=trace, budget=budget)


def _post_send_tensor(
    *,
    tensor_gpu: torch.Tensor,
    dst: int,
    pending: list[dict[str, Any]],
    trace: CpuActionTrace,
    budget: CpuTransportBudget,
    phase: str,
    batch_seq: int,
    mb_id: int,
    seq_start: int,
    records: int,
    prefix: str,
    tag: int,
) -> None:
    nbytes = tensor_nbytes(tensor_gpu)
    while not budget.can_reserve_send(nbytes):
        if not pending:
            raise RuntimeError(
                "send budget cannot fit one message despite passing message-size validation"
            )
        budget.send_budget_waits += 1
        _drain_one_send(pending=pending, trace=trace, budget=budget)

    with trace.span(
        phase=phase,
        batch_seq=batch_seq,
        mb_id=mb_id,
        seq_start=seq_start,
        records=records,
        action=f"{prefix}_D2H",
    ):
        tensor_cpu, _ = gpu_to_cpu(tensor_gpu.detach(), pin_memory=True, sync=True)

    with trace.span(
        phase=phase,
        batch_seq=batch_seq,
        mb_id=mb_id,
        seq_start=seq_start,
        records=records,
        action=f"{prefix}_SEND_POST_CPU",
    ):
        work, _ = cpu_isend(tensor_cpu, dst=dst, tag=tag)

    budget.reserve_send(nbytes)
    pending.append(
        {
            "work": work,
            "keepalive_cpu": tensor_cpu,
            "nbytes": nbytes,
            "phase": phase,
            "batch_seq": batch_seq,
            "mb_id": mb_id,
            "seq_start": seq_start,
            "records": records,
            "prefix": prefix,
        }
    )


def _post_recv_cpu_entry(
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    src: int,
    trace: CpuActionTrace,
    budget: CpuTransportBudget,
    phase: str,
    batch_seq: int,
    mb_id: int,
    seq_start: int,
    records: int,
    prefix: str,
    tag: int,
) -> dict[str, Any] | None:
    nbytes = shape_nbytes(shape, dtype)
    if not budget.can_reserve_recv(nbytes):
        budget.recv_budget_stalls += 1
        return None

    budget.reserve_recv(nbytes)
    try:
        with trace.span(
            phase=phase,
            batch_seq=batch_seq,
            mb_id=mb_id,
            seq_start=seq_start,
            records=records,
            action=f"{prefix}_RECV_POST_CPU",
        ):
            tensor_cpu, work, _ = cpu_irecv(
                shape=shape,
                dtype=dtype,
                src=src,
                pin_memory=True,
                tag=tag,
            )
    except Exception:
        budget.release_recv(nbytes)
        raise

    return {
        "tensor_cpu": tensor_cpu,
        "work": work,
        "nbytes": nbytes,
        "phase": phase,
        "batch_seq": batch_seq,
        "mb_id": mb_id,
        "seq_start": seq_start,
        "records": records,
        "prefix": prefix,
    }


def _consume_recv_cpu_entry(
    *,
    entry: dict[str, Any],
    device: torch.device,
    trace: CpuActionTrace,
    budget: CpuTransportBudget,
) -> torch.Tensor:
    try:
        with trace.span(
            phase=entry["phase"],
            batch_seq=entry["batch_seq"],
            mb_id=entry["mb_id"],
            seq_start=entry["seq_start"],
            records=entry["records"],
            action=f"{entry['prefix']}_RECV_WAIT_CPU",
        ):
            wait_work(entry["work"])

        with trace.span(
            phase=entry["phase"],
            batch_seq=entry["batch_seq"],
            mb_id=entry["mb_id"],
            seq_start=entry["seq_start"],
            records=entry["records"],
            action=f"{entry['prefix']}_RECV_H2D",
        ):
            tensor_gpu, _ = cpu_to_gpu(
                entry["tensor_cpu"],
                device=device,
                non_blocking=True,
                sync=True,
            )
        return tensor_gpu
    finally:
        budget.release_recv(int(entry["nbytes"]))
        entry.clear()


def _recv_tensor_blocking(
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    src: int,
    device: torch.device,
    trace: CpuActionTrace,
    budget: CpuTransportBudget,
    phase: str,
    batch_seq: int,
    mb_id: int,
    seq_start: int,
    records: int,
    prefix: str,
    tag: int,
) -> torch.Tensor:
    nbytes = shape_nbytes(shape, dtype)
    if not budget.can_reserve_recv(nbytes):
        raise RuntimeError(
            "blocking receive cannot fit because future preposts still consume the "
            "shared receive budget"
        )
    budget.reserve_recv(nbytes)
    try:
        with trace.span(
            phase=phase,
            batch_seq=batch_seq,
            mb_id=mb_id,
            seq_start=seq_start,
            records=records,
            action=f"{prefix}_RECV_POST_CPU",
        ):
            tensor_cpu, work, _ = cpu_irecv(
                shape=shape,
                dtype=dtype,
                src=src,
                pin_memory=True,
                tag=tag,
            )

        with trace.span(
            phase=phase,
            batch_seq=batch_seq,
            mb_id=mb_id,
            seq_start=seq_start,
            records=records,
            action=f"{prefix}_RECV_WAIT_CPU",
        ):
            wait_work(work)

        with trace.span(
            phase=phase,
            batch_seq=batch_seq,
            mb_id=mb_id,
            seq_start=seq_start,
            records=records,
            action=f"{prefix}_RECV_H2D",
        ):
            tensor_gpu, _ = cpu_to_gpu(
                tensor_cpu,
                device=device,
                non_blocking=True,
                sync=True,
            )
        return tensor_gpu
    finally:
        budget.release_recv(nbytes)


class ExactBPCpuCommFair:
    def __init__(
        self,
        *,
        module: torch.nn.Module,
        rank: int,
        world_size: int,
        device: torch.device,
        dtype: torch.dtype,
        hidden_size: int,
        label_smoothing: float,
        trace: CpuActionTrace,
        pipeline_schedule: str,
        recv_prepost_depth: int,
        max_pending_send_bytes: int,
        max_posted_recv_bytes: int,
        sync_action_trace: bool = False,
    ) -> None:
        self.module = module
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.device = device
        self.dtype = dtype
        self.hidden_size = int(hidden_size)
        self.label_smoothing = float(label_smoothing)
        self.trace = trace
        self.pipeline_schedule = str(pipeline_schedule)
        self.recv_prepost_depth = max(0, int(recv_prepost_depth))
        self.sync_action_trace = bool(sync_action_trace)
        self.transport_budget = CpuTransportBudget(
            max_pending_send_bytes=max_pending_send_bytes,
            max_posted_recv_bytes=max_posted_recv_bytes,
        )
        self.perf_minimal_metrics = False
        # One queue and one byte budget across forward-hidden and backward-grad
        # channels.  Exact-BP does not receive two independent allowances.
        self.pending_sends: list[dict[str, Any]] = []
        self.recv_entries: dict[tuple[str, int], dict[str, Any]] = {}
        self.cache: dict[int, dict[str, Any]] = {}
        self.peak_activation_cache_entries = 0
        self.peak_activation_cache_bytes = 0

    @property
    def is_first(self) -> bool:
        return self.rank == 0

    @property
    def is_last(self) -> bool:
        return self.rank == self.world_size - 1

    def _sync_cuda_for_trace(self, batch_seq: int) -> None:
        if (
            self.sync_action_trace
            and self.device.type == "cuda"
            and self.trace.enabled_for_batch(batch_seq)
        ):
            torch.cuda.synchronize(self.device)

    def transport_summary(self) -> dict[str, Any]:
        return {
            "recv_prepost_depth": self.recv_prepost_depth,
            **self.transport_budget.summary(),
            "link_emulation": link_emulation_summary(),
        }

    @staticmethod
    def _unique_tensor_storage_bytes(values: Any) -> int:
        storages: dict[tuple[str, int], int] = {}

        def visit(value: Any) -> None:
            if isinstance(value, torch.Tensor):
                try:
                    storage = value.untyped_storage()
                    key = (str(value.device), int(storage.data_ptr()))
                    storages[key] = max(
                        storages.get(key, 0),
                        int(storage.nbytes()),
                    )
                except Exception:
                    key = (str(value.device), int(value.data_ptr()))
                    storages[key] = max(
                        storages.get(key, 0),
                        int(value.numel() * value.element_size()),
                    )
                return
            if isinstance(value, dict):
                for item in value.values():
                    visit(item)
                return
            if isinstance(value, (list, tuple)):
                for item in value:
                    visit(item)

        visit(values)
        return sum(storages.values())

    def reset_memory_peaks(self) -> None:
        if self.cache:
            raise RuntimeError(
                "cannot reset activation-cache peaks with live activations"
            )
        self.peak_activation_cache_entries = 0
        self.peak_activation_cache_bytes = 0

    def _update_activation_cache_peak(self) -> None:
        self.peak_activation_cache_entries = max(
            self.peak_activation_cache_entries,
            len(self.cache),
        )
        self.peak_activation_cache_bytes = max(
            self.peak_activation_cache_bytes,
            self._unique_tensor_storage_bytes(self.cache),
        )

    def activation_cache_summary(self) -> dict[str, int]:
        return {
            "peak_activation_cache_entries": self.peak_activation_cache_entries,
            "peak_activation_cache_bytes": self.peak_activation_cache_bytes,
        }

    def _needs_recv(self, kind: str) -> bool:
        if kind == "fwd":
            return not self.is_first
        if kind == "bwd":
            return not self.is_last
        raise ValueError(f"unknown action kind: {kind}")

    def _recv_key(self, kind: str, payload: dict[str, Any]) -> tuple[str, int]:
        return kind, int(payload["global_mb_id"])

    def _recv_spec(
        self,
        *,
        kind: str,
        payload: dict[str, Any],
    ) -> tuple[tuple[int, ...], int, str, int]:
        shape = _hidden_shape_from_loaded(
            payload["loaded"],
            batch_records=payload["batch_records"],
            hidden_size=self.hidden_size,
        )
        global_mb_id = int(payload["global_mb_id"])
        if kind == "fwd":
            return (
                shape,
                self.rank - 1,
                "FWD_HIDDEN",
                forward_tag(self.rank - 1, global_mb_id),
            )
        if kind == "bwd":
            return (
                shape,
                self.rank + 1,
                "BWD_GRAD",
                backward_tag(self.rank, global_mb_id),
            )
        raise ValueError(f"unknown action kind: {kind}")

    def _prepost_recv_action(
        self,
        *,
        kind: str,
        payload: dict[str, Any],
        phase: str,
        batch_seq: int,
    ) -> bool:
        if not self._needs_recv(kind):
            return False
        key = self._recv_key(kind, payload)
        if key in self.recv_entries:
            return True
        shape, src, prefix, tag = self._recv_spec(kind=kind, payload=payload)
        entry = _post_recv_cpu_entry(
            shape=shape,
            dtype=self.dtype,
            src=src,
            trace=self.trace,
            budget=self.transport_budget,
            phase=phase,
            batch_seq=batch_seq,
            mb_id=int(payload["mb_id"]),
            seq_start=int(payload["seq_start"]),
            records=len(payload["batch_records"]),
            prefix=prefix,
            tag=tag,
        )
        if entry is None:
            return False
        self.recv_entries[key] = entry
        return True

    def _maintain_recv_preposts(
        self,
        *,
        plan: list[tuple[str, dict[str, Any]]],
        start_index: int,
        phase: str,
        batch_seq: int,
    ) -> None:
        if self.recv_prepost_depth <= 0:
            return
        outstanding = 0
        for kind, payload in plan[start_index:]:
            if not self._needs_recv(kind):
                continue
            key = self._recv_key(kind, payload)
            if key in self.recv_entries:
                outstanding += 1
            else:
                if outstanding >= self.recv_prepost_depth:
                    break
                if not self._prepost_recv_action(
                    kind=kind,
                    payload=payload,
                    phase=phase,
                    batch_seq=batch_seq,
                ):
                    break
                outstanding += 1
            if outstanding >= self.recv_prepost_depth:
                break

    def _take_recv(
        self,
        *,
        kind: str,
        payload: dict[str, Any],
        phase: str,
        batch_seq: int,
    ) -> torch.Tensor:
        shape, src, prefix, tag = self._recv_spec(kind=kind, payload=payload)
        key = self._recv_key(kind, payload)
        entry = self.recv_entries.pop(key, None)
        if entry is not None:
            return _consume_recv_cpu_entry(
                entry=entry,
                device=self.device,
                trace=self.trace,
                budget=self.transport_budget,
            )
        return _recv_tensor_blocking(
            shape=shape,
            dtype=self.dtype,
            src=src,
            device=self.device,
            trace=self.trace,
            budget=self.transport_budget,
            phase=phase,
            batch_seq=batch_seq,
            mb_id=int(payload["mb_id"]),
            seq_start=int(payload["seq_start"]),
            records=len(payload["batch_records"]),
            prefix=prefix,
            tag=tag,
        )

    def forward_one(
        self,
        *,
        phase: str,
        batch_seq: int,
        mb_id: int,
        global_mb_id: int,
        seq_start: int,
        batch_records: list[dict[str, Any]],
        loaded: dict[str, Optional[torch.Tensor]],
    ) -> None:
        payload = {
            "mb_id": mb_id,
            "global_mb_id": global_mb_id,
            "seq_start": seq_start,
            "batch_records": batch_records,
            "loaded": loaded,
        }
        if self.is_first:
            hidden = loaded["hidden"]
            if hidden is None:
                raise RuntimeError("rank0 missing loaded hidden")
        else:
            hidden = self._take_recv(
                kind="fwd",
                payload=payload,
                phase=phase,
                batch_seq=batch_seq,
            )
            hidden.requires_grad_(True)

        attention_mask = loaded["attention_mask"]
        position_ids = loaded["position_ids"]
        if attention_mask is None or position_ids is None:
            raise RuntimeError("missing attention_mask/position_ids")

        with self.trace.span(
            phase=phase,
            batch_seq=batch_seq,
            mb_id=mb_id,
            seq_start=seq_start,
            records=len(batch_records),
            action="LOCAL_FORWARD_EXACT",
        ):
            output = self.module(
                hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            self._sync_cuda_for_trace(batch_seq)

        if not self.is_last:
            _post_send_tensor(
                tensor_gpu=output,
                dst=self.rank + 1,
                pending=self.pending_sends,
                trace=self.trace,
                budget=self.transport_budget,
                phase=phase,
                batch_seq=batch_seq,
                mb_id=mb_id,
                seq_start=seq_start,
                records=len(batch_records),
                prefix="FWD_HIDDEN",
                tag=forward_tag(self.rank, global_mb_id),
            )

        self.cache[global_mb_id] = {
            "hidden": hidden,
            "output": output,
            "labels": loaded["labels"],
            "batch_records": batch_records,
            "batch_seq": batch_seq,
            "mb_id": mb_id,
            "seq_start": seq_start,
            "payload": payload,
        }
        self._update_activation_cache_peak()

    def backward_one(
        self,
        *,
        phase: str,
        global_mb_id: int,
        microbatches: int,
    ) -> Optional[float]:
        entry = self.cache.pop(global_mb_id)
        hidden = entry["hidden"]
        output = entry["output"]
        batch_records = entry["batch_records"]
        batch_seq = entry["batch_seq"]
        mb_id = entry["mb_id"]
        seq_start = entry["seq_start"]
        payload = entry["payload"]

        loss_value: Optional[float] = None

        if self.is_last:
            labels = entry["labels"]
            if labels is None:
                raise RuntimeError("last stage missing labels")
            with self.trace.span(
                phase=phase,
                batch_seq=batch_seq,
                mb_id=mb_id,
                seq_start=seq_start,
                records=len(batch_records),
                action="LOCAL_BACKWARD_EXACT_LOSS",
            ):
                loss = causal_lm_loss(
                    output,
                    labels,
                    label_smoothing=self.label_smoothing,
                )
                if not self.perf_minimal_metrics:
                    loss_value = float(loss.detach().cpu().item())
                (loss / max(1, int(microbatches))).backward()
                self._sync_cuda_for_trace(batch_seq)

            if not self.is_first:
                grad_in = hidden.grad
                if grad_in is None:
                    raise RuntimeError(
                        f"missing hidden.grad on last stage mb={global_mb_id}"
                    )
                _post_send_tensor(
                    tensor_gpu=grad_in,
                    dst=self.rank - 1,
                    pending=self.pending_sends,
                    trace=self.trace,
                    budget=self.transport_budget,
                    phase=phase,
                    batch_seq=batch_seq,
                    mb_id=mb_id,
                    seq_start=seq_start,
                    records=len(batch_records),
                    prefix="BWD_GRAD",
                    tag=backward_tag(self.rank - 1, global_mb_id),
                )
        else:
            grad_out = self._take_recv(
                kind="bwd",
                payload=payload,
                phase=phase,
                batch_seq=batch_seq,
            )

            with self.trace.span(
                phase=phase,
                batch_seq=batch_seq,
                mb_id=mb_id,
                seq_start=seq_start,
                records=len(batch_records),
                action="LOCAL_BACKWARD_EXACT_GRAD",
            ):
                output.backward(grad_out)
                self._sync_cuda_for_trace(batch_seq)

            if not self.is_first:
                grad_in = hidden.grad
                if grad_in is None:
                    raise RuntimeError(
                        f"missing hidden.grad on stage={self.rank} mb={global_mb_id}"
                    )
                _post_send_tensor(
                    tensor_gpu=grad_in,
                    dst=self.rank - 1,
                    pending=self.pending_sends,
                    trace=self.trace,
                    budget=self.transport_budget,
                    phase=phase,
                    batch_seq=batch_seq,
                    mb_id=mb_id,
                    seq_start=seq_start,
                    records=len(batch_records),
                    prefix="BWD_GRAD",
                    tag=backward_tag(self.rank - 1, global_mb_id),
                )

        return loss_value

    def _build_action_plan(
        self,
        microbatch_payloads: list[dict[str, Any]],
    ) -> list[tuple[str, dict[str, Any]]]:
        m = len(microbatch_payloads)
        if self.pipeline_schedule == "gpipe":
            return [
                *(('fwd', payload) for payload in microbatch_payloads),
                *(('bwd', payload) for payload in reversed(microbatch_payloads)),
            ]
        if self.pipeline_schedule == "1f1b":
            warmup = min(self.world_size - self.rank - 1, m)
            steady = m - warmup
            plan: list[tuple[str, dict[str, Any]]] = []
            plan.extend(("fwd", payload) for payload in microbatch_payloads[:warmup])
            for i in range(steady):
                plan.append(("fwd", microbatch_payloads[warmup + i]))
                plan.append(("bwd", microbatch_payloads[i]))
            plan.extend(("bwd", payload) for payload in microbatch_payloads[steady:])
            return plan
        raise ValueError(f"Unsupported pipeline schedule: {self.pipeline_schedule}")

    def step_logical_batch(
        self,
        *,
        phase: str,
        batch_seq: int,
        microbatch_payloads: list[dict[str, Any]],
    ) -> list[float]:
        m = len(microbatch_payloads)
        losses: list[float] = []
        plan = self._build_action_plan(microbatch_payloads)

        for index, (kind, payload) in enumerate(plan):
            self._maintain_recv_preposts(
                plan=plan,
                start_index=index,
                phase=phase,
                batch_seq=batch_seq,
            )
            if kind == "fwd":
                self.forward_one(phase=phase, batch_seq=batch_seq, **payload)
            else:
                loss_value = self.backward_one(
                    phase=phase,
                    global_mb_id=payload["global_mb_id"],
                    microbatches=m,
                )
                if loss_value is not None:
                    losses.append(loss_value)

        if self.recv_entries:
            raise RuntimeError(
                f"logical batch ended with unconsumed recv entries: {sorted(self.recv_entries)}"
            )
        _drain_sends(
            pending=self.pending_sends,
            trace=self.trace,
            budget=self.transport_budget,
        )
        return losses

def _split_logical_batch(records: list[dict[str, Any]], microbatches: int) -> list[list[dict[str, Any]]]:
    if len(records) % microbatches != 0:
        raise ValueError(f"logical batch size {len(records)} not divisible by microbatches {microbatches}")
    physical = len(records) // microbatches
    return [records[i * physical : (i + 1) * physical] for i in range(microbatches)]


def _batched(records: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    usable = (len(records) // batch_size) * batch_size
    return [records[i : i + batch_size] for i in range(0, usable, batch_size)]


def _evaluator_compatible_trainable_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
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


def run_train_phase(
    *,
    records: list[dict[str, Any]],
    manifest_dir: Path,
    cfg: RankConfig,
    schedule: ExactBPCpuCommFair,
    module: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    stage0_input_embedding: Optional[torch.nn.Module],
    trace: CpuActionTrace,
) -> dict[str, Any]:
    logical_batch_size = int(args.physical_batch_size) * int(args.gradient_accumulation_steps)
    microbatches = int(args.gradient_accumulation_steps)
    batches = _batched(records, logical_batch_size)

    module.train(True)
    if cfg.device.type == "cuda":
        torch.cuda.synchronize(cfg.device)
    dist.barrier()
    if cfg.device.type == "cuda":
        torch.cuda.synchronize(cfg.device)
    started = time.perf_counter()

    completed_records = 0
    optimizer_steps = 0
    loss_sum = 0.0
    loss_count = 0
    activation_tracker = SavedTensorTracker(
        hidden_size=cfg.hidden_size,
        vocab_size=cfg.vocab_size,
    )
    memory_windows: list[dict[str, Any]] = []

    for batch_seq, batch_records in enumerate(batches):
        if cfg.device.type == "cuda":
            torch.cuda.synchronize(cfg.device)
            torch.cuda.reset_peak_memory_stats(cfg.device)
        activation_tracker.reset()
        schedule.reset_memory_peaks()

        learning_rate = args.learning_rate
        if learning_rate is None:
            learning_rate = batch_records[0].get("learning_rate")
        if learning_rate is not None:
            set_optimizer_lr(optimizer, float(learning_rate))

        with trace.span(
            phase="train",
            batch_seq=batch_seq,
            mb_id=-1,
            seq_start=batch_seq * logical_batch_size,
            records=len(batch_records),
            action="ZERO_GRAD",
        ):
            optimizer.zero_grad(set_to_none=True)

        if cfg.device.type == "cuda":
            torch.cuda.synchronize(cfg.device)
            baseline_allocated = int(torch.cuda.memory_allocated(cfg.device))
            baseline_reserved = int(torch.cuda.memory_reserved(cfg.device))
        else:
            baseline_allocated = 0
            baseline_reserved = 0

        micro_records = _split_logical_batch(batch_records, microbatches)
        payloads: list[dict[str, Any]] = []

        for mb_id, mb_records in enumerate(micro_records):
            global_mb_id = batch_seq * microbatches + mb_id
            seq_start = batch_seq * logical_batch_size + mb_id * len(mb_records)
            with trace.span(
                phase="train",
                batch_seq=batch_seq,
                mb_id=mb_id,
                seq_start=seq_start,
                records=len(mb_records),
                action="INPUT_LOAD_H2D",
            ):
                loaded = load_batch_tensors(
                    records=mb_records,
                    manifest_dir=manifest_dir,
                    device=cfg.device,
                    dtype=cfg.dtype,
                    load_hidden=(cfg.rank == 0),
                    load_labels=(cfg.rank == cfg.world_size - 1),
                    input_embedding=stage0_input_embedding if cfg.rank == 0 else None,
                )
                if (
                    args.sync_action_trace
                    and cfg.device.type == "cuda"
                    and trace.enabled_for_batch(batch_seq)
                ):
                    torch.cuda.synchronize(cfg.device)
            payloads.append(
                {
                    "mb_id": mb_id,
                    "global_mb_id": global_mb_id,
                    "seq_start": seq_start,
                    "batch_records": mb_records,
                    "loaded": loaded,
                }
            )

        if cfg.device.type == "cuda":
            torch.cuda.synchronize(cfg.device)
            pre_schedule_allocated = int(torch.cuda.memory_allocated(cfg.device))
        else:
            pre_schedule_allocated = 0

        hook_context = (
            torch.autograd.graph.saved_tensors_hooks(
                activation_tracker.pack,
                activation_tracker.unpack,
            )
            if args.track_activation_memory
            else nullcontext()
        )
        with torch.enable_grad(), hook_context:
            losses = schedule.step_logical_batch(
                phase="train",
                batch_seq=batch_seq,
                microbatch_payloads=payloads,
            )

            if args.grad_clip > 0:
                with trace.span(
                    phase="train",
                    batch_seq=batch_seq,
                    mb_id=-1,
                    seq_start=batch_seq * logical_batch_size,
                    records=len(batch_records),
                    action="GRAD_CLIP",
                ):
                    torch.nn.utils.clip_grad_norm_(module.parameters(), args.grad_clip)
                    if (
                        args.sync_action_trace
                        and cfg.device.type == "cuda"
                        and trace.enabled_for_batch(batch_seq)
                    ):
                        torch.cuda.synchronize(cfg.device)

            with trace.span(
                phase="train",
                batch_seq=batch_seq,
                mb_id=-1,
                seq_start=batch_seq * logical_batch_size,
                records=len(batch_records),
                action="OPTIMIZER_STEP",
            ):
                optimizer.step()
                if (
                    args.sync_action_trace
                    and cfg.device.type == "cuda"
                    and trace.enabled_for_batch(batch_seq)
                ):
                    torch.cuda.synchronize(cfg.device)

        if cfg.device.type == "cuda":
            torch.cuda.synchronize(cfg.device)
            peak_allocated = int(torch.cuda.max_memory_allocated(cfg.device))
            peak_reserved = int(torch.cuda.max_memory_reserved(cfg.device))
        else:
            peak_allocated = 0
            peak_reserved = 0

        activation_stats = (
            activation_tracker.snapshot()
            if args.track_activation_memory
            else {}
        )
        memory_windows.append(
            {
                "batch_seq": batch_seq,
                "measured": batch_seq >= int(args.memory_warmup_windows),
                "baseline_cuda_allocated_bytes": baseline_allocated,
                "baseline_cuda_reserved_bytes": baseline_reserved,
                "pre_schedule_cuda_allocated_bytes": pre_schedule_allocated,
                "peak_cuda_allocated_bytes": peak_allocated,
                "peak_cuda_reserved_bytes": peak_reserved,
                "peak_runtime_delta_bytes": max(
                    0, peak_allocated - baseline_allocated
                ),
                "gradient_storage_bytes": gradient_storage_nbytes(module),
                "optimizer_state_bytes": optimizer_state_nbytes(optimizer),
                **schedule.activation_cache_summary(),
                **activation_stats,
            }
        )

        if losses:
            loss_sum += sum(losses)
            loss_count += len(losses)

        optimizer_steps += 1
        completed_records += len(batch_records)

        checkpoint_interval = int(args.trainable_checkpoint_interval)
        if checkpoint_interval > 0 and optimizer_steps % checkpoint_interval == 0:
            checkpoint_dir = (
                Path(args.output_dir)
                / "trainable_checkpoints"
                / f"step_{optimizer_steps:04d}"
            )
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                _evaluator_compatible_trainable_state(module),
                checkpoint_dir / f"stage{cfg.rank}.trainable.pt",
            )

        if cfg.rank == cfg.world_size - 1 and args.progress_interval > 0:
            if completed_records % int(args.progress_interval) == 0 or completed_records == len(batches) * logical_batch_size:
                print(f"[rank {cfg.rank}] train: {completed_records}/{len(batches) * logical_batch_size}", flush=True)

    if cfg.device.type == "cuda":
        torch.cuda.synchronize(cfg.device)
    local_wall_ms = (time.perf_counter() - started) * 1000.0
    wall_ms = sync_max_wall_ms(local_wall_ms)
    actions_csv = trace.flush("train")

    measured_windows = [row for row in memory_windows if row["measured"]]
    if not measured_windows:
        measured_windows = memory_windows
    aggregate_keys = sorted(
        {
            key
            for row in measured_windows
            for key, value in row.items()
            if key not in {"batch_seq", "measured"}
            and isinstance(value, (int, float))
        }
    )
    memory_aggregate = {
        key: max(int(row.get(key, 0) or 0) for row in measured_windows)
        for key in aggregate_keys
    }

    return {
        "phase": "train",
        "mode": "train",
        "completed_records": completed_records,
        "optimizer_steps": optimizer_steps,
        "batches": len(batches),
        "wall_ms": wall_ms,
        "throughput_per_s": completed_records / (wall_ms / 1000.0) if wall_ms > 0 else 0.0,
        "avg_loss": loss_sum / loss_count if loss_count else "",
        "loss_count": loss_count,
        "actions_csv": actions_csv,
        "memory_profile": {
            "tracking_enabled": bool(args.track_activation_memory),
            "warmup_windows": int(args.memory_warmup_windows),
            "aggregate": memory_aggregate,
            "windows": memory_windows,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fair Exact-BP GPipe/1F1B with GPU compute and CPU/Gloo transport.")
    parser.add_argument("--model_name", default="tinyllama")
    parser.add_argument("--train_manifest", required=True)
    parser.add_argument("--eval_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_chunks", type=int, default=3)
    parser.add_argument("--stage_devices", default="cuda:0,cuda:1,cuda:2")
    parser.add_argument("--train_limit", type=int, default=72)
    parser.add_argument("--eval_limit", type=int, default=8)
    parser.add_argument("--train_epochs", type=int, default=1)
    parser.add_argument("--physical_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=3)
    parser.add_argument("--pipeline_schedule", choices=["auto", "gpipe", "1f1b"], default="auto")
    parser.add_argument(
        "--recv_prepost_depth",
        type=int,
        default=0,
        help="Maximum number of receive actions preposted on this rank; 0 is blocking.",
    )
    parser.add_argument("--max_pending_send_bytes", type=int, default=67_108_864)
    parser.add_argument("--max_posted_recv_bytes", type=int, default=67_108_864)
    parser.add_argument(
        "--link_latency_ms",
        type=float,
        default=0.0,
        help="Injected one-way sender-side latency per data message.",
    )
    parser.add_argument(
        "--link_bandwidth_mbps",
        type=float,
        default=0.0,
        help="Injected sender-side bandwidth cap; 0 means unlimited.",
    )
    parser.add_argument(
        "--link_jitter_ms",
        type=float,
        default=0.0,
        help="Deterministic per-message jitter amplitude.",
    )
    parser.add_argument("--link_emulation_seed", type=int, default=20260531)
    parser.add_argument("--learning_rate", type=float, default=None)
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
    parser.add_argument("--progress_interval", type=int, default=1000)
    parser.add_argument("--skip_eval_before", action="store_true")
    parser.add_argument("--skip_eval_after", action="store_true")
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
    parser.add_argument("--memory_warmup_windows", type=int, default=0)
    parser.add_argument("--enable_action_trace", action="store_true")
    parser.add_argument(
        "--sync_action_trace",
        action="store_true",
        help="Synchronize CUDA at Exact-BP compute trace boundaries for diagnostic timing.",
    )
    parser.add_argument(
        "--action_trace_start_window",
        type=int,
        default=0,
        help="First logical optimizer window recorded and synchronized.",
    )
    parser.add_argument(
        "--action_trace_end_window",
        type=int,
        default=None,
        help="Exclusive logical optimizer-window limit for action tracing.",
    )
    parser.add_argument("--perf_minimal_metrics", action="store_true")
    parser.add_argument(
        "--save_trainable_state",
        action="store_true",
        help="Save evaluator-compatible per-stage trainable tensors after timed training.",
    )
    parser.add_argument(
        "--trainable_checkpoint_interval",
        type=int,
        default=0,
        help=(
            "Save evaluator-compatible per-stage trainable tensors every N optimizer "
            "steps; zero disables periodic checkpoints."
        ),
    )
    args = parser.parse_args()

    if not dist.is_initialized():
        dist.init_process_group("gloo")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != args.num_chunks:
        raise ValueError(f"WORLD_SIZE={world_size} must equal --num_chunks={args.num_chunks}")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient_accumulation_steps must be positive")
    if args.physical_batch_size <= 0:
        raise ValueError("--physical_batch_size must be positive")
    if args.trainable_checkpoint_interval < 0:
        raise ValueError("--trainable_checkpoint_interval must be non-negative")
    if args.recv_prepost_depth < 0:
        raise ValueError("--recv_prepost_depth must be non-negative")
    if args.action_trace_start_window < 0:
        raise ValueError("--action_trace_start_window must be non-negative")
    if args.memory_warmup_windows < 0:
        raise ValueError("--memory_warmup_windows must be non-negative")
    if (
        args.action_trace_end_window is not None
        and args.action_trace_end_window <= args.action_trace_start_window
    ):
        raise ValueError(
            "--action_trace_end_window must be greater than the start window"
        )
    if args.max_pending_send_bytes <= 0:
        raise ValueError("--max_pending_send_bytes must be positive")
    if args.max_posted_recv_bytes <= 0:
        raise ValueError("--max_posted_recv_bytes must be positive")
    if args.link_latency_ms < 0:
        raise ValueError("--link_latency_ms must be non-negative")
    if args.link_bandwidth_mbps < 0:
        raise ValueError("--link_bandwidth_mbps must be non-negative")
    if args.link_jitter_ms < 0:
        raise ValueError("--link_jitter_ms must be non-negative")
    resolved_pipeline_schedule = (
        "1f1b"
        if args.pipeline_schedule == "auto" and args.gradient_accumulation_steps >= args.num_chunks
        else "gpipe"
        if args.pipeline_schedule == "auto"
        else args.pipeline_schedule
    )
    if resolved_pipeline_schedule == "1f1b" and args.gradient_accumulation_steps < args.num_chunks:
        raise ValueError("1f1b requires microbatches >= num_chunks")

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
            f"Starting 1F1B CPU-comm exact-BP model={resolved_model} chunks={args.num_chunks} "
            f"devices={devices} physical={args.physical_batch_size} micro={args.gradient_accumulation_steps} schedule={resolved_pipeline_schedule}",
            flush=True,
        )

    recorder = StageEventRecorder(
        stage_id=rank,
        rank=rank,
        device_name=devices[rank],
        enabled=False,
    )

    print(f"[rank {rank}] loading model={resolved_model} dtype={dtype} device={device}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(resolved_model, torch_dtype=dtype)

    hidden_size = int(getattr(model.config, "hidden_size", 0) or getattr(model.config, "n_embd", 0) or 0)
    vocab_size = int(getattr(model.config, "vocab_size", 0) or 0)
    if hidden_size <= 0:
        raise ValueError(f"Could not resolve hidden_size for model={resolved_model}")

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
    module = build_stage_module(model=model, stage_id=rank, num_chunks=args.num_chunks, recorder=recorder)
    module.to(device)

    if stage0_input_embedding is not None:
        stage0_input_embedding.to(device)
        stage0_input_embedding.eval()
        for param in stage0_input_embedding.parameters():
            param.requires_grad = False

    local_param_stats = module_param_stats(module)
    memory_ledger = stage_memory_ledger(module=module, input_embedding=stage0_input_embedding)

    params = [param for param in module.parameters() if param.requires_grad]
    optimizer = build_optimizer(
        params=params,
        optimizer_name=args.optimizer,
        learning_rate=args.learning_rate or 3e-4,
        sgd_momentum=args.sgd_momentum,
        sgd_dampening=args.sgd_dampening,
        sgd_weight_decay=args.sgd_weight_decay,
        sgd_nesterov=args.sgd_nesterov,
    )

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    print(
        f"[rank {rank}] ready stage={rank} trainable_mode={trainable_setup.mode} "
        f"lora_modules={trainable_setup.lora_modules} "
        f"all_trainable={trainable_setup.trainable_params} "
        f"frozen={trainable_setup.frozen_params} "
        f"local_trainable={local_param_stats.trainable_params} "
        f"lora_init={lora_init_fingerprint[:12]}",
        flush=True,
    )

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

    schedule = ExactBPCpuCommFair(
        module=module,
        rank=rank,
        world_size=world_size,
        device=device,
        dtype=dtype,
        hidden_size=hidden_size,
        label_smoothing=args.label_smoothing,
        trace=trace,
        pipeline_schedule=resolved_pipeline_schedule,
        recv_prepost_depth=args.recv_prepost_depth,
        max_pending_send_bytes=args.max_pending_send_bytes,
        max_posted_recv_bytes=args.max_posted_recv_bytes,
        sync_action_trace=args.sync_action_trace,
    )
    schedule.perf_minimal_metrics = bool(args.perf_minimal_metrics)

    base_train_records = read_manifest(Path(args.train_manifest), args.train_limit)
    train_records = base_train_records * int(args.train_epochs)

    train_summary = run_train_phase(
        records=train_records,
        manifest_dir=Path(args.train_manifest).resolve().parent,
        cfg=cfg,
        schedule=schedule,
        module=module,
        optimizer=optimizer,
        args=args,
        stage0_input_embedding=stage0_input_embedding,
        trace=trace,
    )

    if args.save_trainable_state:
        torch.save(
            _evaluator_compatible_trainable_state(module),
            output_dir / f"stage{rank}.trainable.pt",
        )

    local_transport_budget = schedule.transport_summary()
    transport_budget_by_rank: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(transport_budget_by_rank, local_transport_budget)

    local_summary = {
        "rank": rank,
        "stage_id": rank,
        "train": train_summary,
        "gradient_storage_bytes": gradient_storage_nbytes(module),
        "optimizer_state_bytes": optimizer_state_nbytes(optimizer),
        "memory_ledger": memory_ledger,
        "transport_budget": local_transport_budget,
    }
    (output_dir / f"rank{rank}.summary.json").write_text(json.dumps(local_summary, indent=2), encoding="utf-8")

    dist.barrier()

    if rank == world_size - 1:
        summary = {
            "runner": f"exactbp-cpu-comm-fair-{resolved_pipeline_schedule}",
            "transport": "gloo-cpu-hidden-and-grad-pinned",
            "transport_details": (
                "manual exact-BP schedule; GPU compute + pinned CPU/Gloo transport; "
                "forward hidden and backward hidden_grad share one per-rank send-byte "
                "budget and one per-rank preposted-receive-byte budget"
            ),
            "pipeline_schedule": resolved_pipeline_schedule,
            "recv_prepost_depth": args.recv_prepost_depth,
            "max_pending_send_bytes": args.max_pending_send_bytes,
            "max_posted_recv_bytes": args.max_posted_recv_bytes,
            "link_emulation": {
                "mode": "sender_side_pacing",
                "one_way_latency_ms": args.link_latency_ms,
                "bandwidth_mbps": args.link_bandwidth_mbps,
                "jitter_ms": args.link_jitter_ms,
                "seed": args.link_emulation_seed,
            },
            "transport_budget_by_rank": transport_budget_by_rank,
            "model_name": args.model_name,
            "resolved_model": resolved_model,
            "num_chunks": args.num_chunks,
            "stage_devices": devices,
            "physical_request_batch": args.physical_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "microbatches": args.gradient_accumulation_steps,
            "effective_optimizer_batch": args.physical_batch_size * args.gradient_accumulation_steps,
            "dtype": str(dtype).replace("torch.", ""),
            "seed": args.seed,
            "trainable_mode": args.trainable_mode,
            "lora": {
                "rank": args.lora_rank,
                "alpha": args.lora_alpha,
                "targets": args.lora_targets,
                "init_std": args.lora_init_std,
                "init_seed": args.lora_init_seed,
                "initialization_fingerprint": lora_init_fingerprint,
                "modules": trainable_setup.lora_modules,
                "trainable_params": trainable_setup.trainable_params,
            },
            "completed_records": train_summary["completed_records"],
            "optimizer_steps": train_summary["optimizer_steps"],
            "perf_minimal_metrics": bool(args.perf_minimal_metrics),
            "trainable_state_saved": bool(args.save_trainable_state),
            "action_trace_enabled": bool(args.enable_action_trace),
            "sync_action_trace": bool(args.sync_action_trace),
            "action_trace_start_window": int(args.action_trace_start_window),
            "action_trace_end_window": args.action_trace_end_window,
            "phases": [train_summary],
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
        print(f"Wrote {output_dir / 'summary.json'}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
