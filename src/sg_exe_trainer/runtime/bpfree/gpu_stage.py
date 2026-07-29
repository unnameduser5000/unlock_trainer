from __future__ import annotations

import os
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.distributed as dist

from sg_exe_trainer.common.trainable_modes import gradient_storage_nbytes
from sg_exe_trainer.runtime.bpfree import gpu_transport, pipeline_support
from sg_exe_trainer.runtime.bpfree.schedule import BPFreeMicrobatch, BPFreeUpdateWindow
from sg_exe_trainer.runtime.bpfree.trace import ActionTracer
from sg_exe_trainer.runtime.bpfree.chunk_split import (
    body_forward,
    local_head_loss_from_hidden,
)
from sg_exe_trainer.tasks.label_experiment import one_token_choice_ids


@dataclass
class ForwardInput:
    hidden: torch.Tensor
    prev_log_probs: Optional[torch.Tensor]
    recv_hidden_ms: float = 0.0
    recv_log_probs_ms: float = 0.0
    load_hidden_ms: float = 0.0


@dataclass
class CommonInputs:
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    labels: torch.Tensor
    load_input_ms: float


@dataclass
class WindowCommonInputCache:
    window_id: int
    first_global_batch_seq: int
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    labels: torch.Tensor
    slices_by_global_batch_seq: dict[int, tuple[int, int]]
    load_input_ms: float


@dataclass
class ForwardOutput:
    loss: torch.Tensor
    next_hidden: Optional[torch.Tensor]
    next_log_probs: Optional[torch.Tensor]
    forward_ms: float


@dataclass
class BodyForwardOutput:
    next_hidden: torch.Tensor
    body_forward_ms: float


@dataclass
class LocalHeadLossOutput:
    loss: torch.Tensor
    next_log_probs: Optional[torch.Tensor]
    local_head_loss_ms: float


@dataclass
class BackwardOutput:
    backward_ms: float
    gradient_storage_bytes: int


@dataclass
class OptimizerStepOutput:
    optimizer_ms: float
    applied: bool


@dataclass
class SendOutput:
    send_hidden_ms: float
    send_log_probs_ms: float


class BPFreePipelineStageV0:
    """
    BP-free stage runtime v0.

    This stage intentionally keeps the old semantic order:
      FWD_COMPUTE_INCLUDES_LOCAL_HEAD
      FWD_SEND_POST
      LOCAL_BACKWARD

    v1 later will split:
      BODY_FORWARD
      FWD_SEND_POST
      LOCAL_HEAD_LOSS
      LOCAL_BACKWARD
    """

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        phase: str,
        mode: str,
        request_prefix: str,
        chunk: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        train_this_rank: bool,
        dtype: torch.dtype,
        device: torch.device,
        belief_transport_mode: str,
        grad_clip: float,
        hidden_size: int,
        vocab_size: int,
        manifest_dir: Path,
        tracer: ActionTracer,
        max_pending_sends: int = 2,
        recv_inflight_depth: int = 1,
        sync_action_trace: bool = False,
        perf_minimal_metrics: bool = False,
    ) -> None:
        self.rank = rank
        self.world_size = world_size
        self.phase = phase
        self.mode = mode
        self.request_prefix = request_prefix
        self.chunk = chunk
        self.optimizer = optimizer
        self.train_this_rank = train_this_rank
        self.dtype = dtype
        self.device = device
        self.belief_transport_mode = belief_transport_mode
        self.grad_clip = grad_clip
        self.hidden_size = int(hidden_size)
        self.vocab_size = vocab_size
        self.manifest_dir = manifest_dir
        self.tracer = tracer
        self.max_pending_sends = max_pending_sends
        self.recv_inflight_depth = int(recv_inflight_depth)
        self.forward_recv_plan: list[BPFreeMicrobatch] = []
        self.forward_recv_plan_index_by_seq: dict[int, int] = {}
        self.pending_forward_recv_entries: dict[int, dict[str, Any]] = {}
        self.consumed_forward_recv_seqs: set[int] = set()
        self.sync_action_trace = sync_action_trace
        self.perf_minimal_metrics = perf_minimal_metrics
        self.prof_record_actions = os.environ.get("BPFREE_PROF_RECORD_ACTIONS", "0") == "1"
        self._window_common_input_cache: Optional[WindowCommonInputCache] = None

        self.compute_stream = torch.cuda.current_stream(device) if device.type == "cuda" else None
        self.comm_stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
        self.pending_send_entries: list[dict[str, Any]] = []

    @property
    def is_first(self) -> bool:
        return self.rank == 0

    @property
    def is_last(self) -> bool:
        return self.rank == self.world_size - 1

    def _sync_cuda(self, _mb: BPFreeMicrobatch | None = None) -> None:
        if self.sync_action_trace and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @contextmanager
    def _span(self, mb: BPFreeMicrobatch, action: str):
        label = (
            f"S{self.rank}/{action}"
            f"/w{getattr(mb, 'window_id', -1)}"
            f"/mb{getattr(mb, 'mb_id', -1)}"
            f"/seq{getattr(mb, 'seq_start', -1)}"
        )
        record_cm = (
            torch.profiler.record_function(label)
            if self.prof_record_actions
            else nullcontext()
        )

        with self.tracer.span(
            window_id=mb.window_id,
            mb_id=mb.mb_id,
            global_batch_seq=mb.global_batch_seq,
            seq_start=mb.seq_start,
            records=mb.num_records,
            action=action,
        ):
            with record_cm:
                yield

    def begin_window(
        self,
        *,
        window: BPFreeUpdateWindow,
        learning_rate_override: Optional[float],
    ) -> None:
        if not self.train_this_rank:
            return
        if self.optimizer is None:
            raise RuntimeError("train_this_rank=True but optimizer is None.")

        first_record = window.microbatches[0].records[0]
        lr = learning_rate_override
        if lr is None:
            lr = first_record.get("learning_rate")

        if lr is not None:
            for group in self.optimizer.param_groups:
                group["lr"] = float(lr)

        first_mb = window.microbatches[0]
        with self._span(first_mb, "BEGIN_WINDOW_ZERO_GRAD"):
            self.optimizer.zero_grad(set_to_none=True)
            self._sync_cuda()


    def set_forward_recv_plan(self, microbatches: list[BPFreeMicrobatch]) -> None:
        """Install flattened physical-microbatch plan for rolling recv prepost.

        This is receiver-side communication buffering only. It stores detached
        receive tensors/work handles, not autograd activation graphs.
        """
        self.forward_recv_plan = list(microbatches)
        self.forward_recv_plan_index_by_seq = {
            mb.global_batch_seq: idx for idx, mb in enumerate(self.forward_recv_plan)
        }
        self.pending_forward_recv_entries.clear()
        self.consumed_forward_recv_seqs.clear()


    def _make_forward_recv_entry(
        self,
        mb: BPFreeMicrobatch,
    ) -> tuple[list[dist.P2POp], dict[str, Any]]:
        """Create recv buffers and P2P ops for one future microbatch.

        Important: this method must NOT consume pending_forward_recv_entries.
        Consumption happens only in load_or_recv_forward_input().
        """
        batch_records = mb.records

        hidden_shape = pipeline_support.hidden_shape(
            batch_records[0],
            len(batch_records),
            self.hidden_size,
        )
        hidden = torch.empty(tuple(hidden_shape), dtype=self.dtype, device=self.device)

        recv_ops: list[dist.P2POp] = [
            dist.P2POp(dist.irecv, hidden, self.rank - 1)
        ]

        if self.belief_transport_mode == "full":
            lp_shape = pipeline_support.log_probs_shape(
                batch_records[0],
                len(batch_records),
                self.vocab_size,
            )
            prev_log_probs: Optional[torch.Tensor] = torch.empty(
                tuple(lp_shape),
                dtype=torch.float32,
                device=self.device,
            )
            recv_ops.append(dist.P2POp(dist.irecv, prev_log_probs, self.rank - 1))
        else:
            prev_log_probs = None

        entry: dict[str, Any] = {
            "hidden": hidden,
            "prev_log_probs": prev_log_probs,
            "works": [],
            "post_ms": 0.0,
            "window_id": mb.window_id,
            "mb_id": mb.mb_id,
            "batch_seq": mb.global_batch_seq,
            "seq_start": mb.seq_start,
            "records": mb.num_records,
        }
        return recv_ops, entry


    def prepost_forward_recv(self, mb: BPFreeMicrobatch) -> None:
        """Post recv for one future microbatch if not already posted."""
        if self.is_first:
            return
        if self.recv_inflight_depth <= 1:
            return
        if mb.global_batch_seq in self.consumed_forward_recv_seqs:
            return
        if mb.global_batch_seq in self.pending_forward_recv_entries:
            return

        recv_ops, entry = self._make_forward_recv_entry(mb)

        entry = self._post_forward_recv_entry(mb, recv_ops, entry)
        self.pending_forward_recv_entries[mb.global_batch_seq] = entry

    def _post_forward_recv_entry(
        self,
        mb: BPFreeMicrobatch,
        recv_ops: list[dist.P2POp],
        entry: dict[str, Any],
    ) -> dict[str, Any]:
        with self._span(mb, "FWD_RECV_POST"):
            works, post_ms = gpu_transport.post_batch_p2p(
                ops=recv_ops,
                device=self.device,
                comm_stream=self.comm_stream,
            )

        entry["works"] = works
        entry["post_ms"] = post_ms

        if self.device.type == "cuda" and self.comm_stream is not None:
            ready_event = torch.cuda.Event()
            ready_event.record(self.comm_stream)
            entry["ready_event"] = ready_event
        else:
            entry["ready_event"] = None

        return entry


    def maintain_forward_recv_inflight(self, mb: BPFreeMicrobatch) -> None:
        """Keep recv ops posted for [current, current + depth).

        Called before load_or_recv_forward_input(current_mb).
        """
        if self.is_first:
            return

        idx = self.forward_recv_plan_index_by_seq.get(mb.global_batch_seq)
        end = (
            min(len(self.forward_recv_plan), idx + self.recv_inflight_depth)
            if idx is not None
            else -1
        )

        with self._span(
            mb,
            (
                f"DEBUG_RECV_INFLIGHT_"
                f"D{self.recv_inflight_depth}_"
                f"PLAN{len(self.forward_recv_plan)}_"
                f"IDX{idx}_"
                f"END{end}_"
                f"PENDING{len(self.pending_forward_recv_entries)}"
            ),
        ):
            pass

        if self.recv_inflight_depth <= 1:
            return
        if not self.forward_recv_plan_index_by_seq:
            return
        if idx is None:
            return

        for j in range(idx, end):
            candidate = self.forward_recv_plan[j]
            if candidate.window_id != mb.window_id:
                break
            self.prepost_forward_recv(candidate)

    def _prepare_send_buffer(self, tensor: torch.Tensor) -> torch.Tensor:
        send_buffer = tensor.detach()
        if not send_buffer.is_contiguous():
            send_buffer = send_buffer.contiguous()
        return send_buffer


    def load_or_recv_forward_input(self, mb: BPFreeMicrobatch) -> ForwardInput:
        recv_entry = self.start_forward_input(mb)
        return self.finish_forward_input(mb, recv_entry)

    def start_forward_input(
        self,
        mb: BPFreeMicrobatch,
    ) -> Optional[dict[str, Any]]:
        if self.is_first:
            return None

        preposted_entry = self.pending_forward_recv_entries.pop(mb.global_batch_seq, None)
        if preposted_entry is not None:
            return preposted_entry

        batch_records = mb.records

        # Fallback path: depth=1 or missed prepost. This preserves original clean-v5
        # behavior and prevents silent failure.
        hidden_shape = pipeline_support.hidden_shape(
            batch_records[0],
            len(batch_records),
            self.hidden_size,
        )
        hidden = torch.empty(tuple(hidden_shape), dtype=self.dtype, device=self.device)

        recv_ops = [dist.P2POp(dist.irecv, hidden, self.rank - 1)]
        prev_log_probs: Optional[torch.Tensor]

        if self.belief_transport_mode == "full":
            lp_shape = pipeline_support.log_probs_shape(
                batch_records[0],
                len(batch_records),
                self.vocab_size,
            )
            prev_log_probs = torch.empty(tuple(lp_shape), dtype=torch.float32, device=self.device)
            recv_ops.append(dist.P2POp(dist.irecv, prev_log_probs, self.rank - 1))
        else:
            prev_log_probs = None

        return self._post_forward_recv_entry(
            mb,
            recv_ops,
            {
                "hidden": hidden,
                "prev_log_probs": prev_log_probs,
                "works": [],
                "post_ms": 0.0,
                "window_id": mb.window_id,
                "mb_id": mb.mb_id,
                "batch_seq": mb.global_batch_seq,
                "seq_start": mb.seq_start,
                "records": mb.num_records,
            },
        )

    def finish_forward_input(
        self,
        mb: BPFreeMicrobatch,
        recv_entry: Optional[dict[str, Any]],
    ) -> ForwardInput:
        batch_records = mb.records

        if self.is_first:
            started = time.perf_counter()
            with self._span(mb, "LOAD_STAGE0_HIDDEN"):
                hidden = pipeline_support.load_stage0_hidden(
                    records=batch_records,
                    manifest_dir=self.manifest_dir,
                    device=self.device,
                    dtype=self.dtype,
                )
                self._sync_cuda()

            return ForwardInput(
                hidden=hidden,
                prev_log_probs=None,
                load_hidden_ms=(time.perf_counter() - started) * 1000.0,
            )

        if recv_entry is None:
            raise RuntimeError(f"missing forward recv entry for seq={mb.global_batch_seq}")

        started = time.perf_counter()
        with self._span(mb, "FWD_RECV_WAIT"):
            wait_ms = gpu_transport.wait_batch_p2p(works=recv_entry["works"])
            ready_event = recv_entry.get("ready_event")
            if self.device.type == "cuda" and ready_event is not None:
                torch.cuda.current_stream(self.device).wait_event(ready_event)
            self._sync_cuda()

        self.consumed_forward_recv_seqs.add(mb.global_batch_seq)

        return ForwardInput(
            hidden=recv_entry["hidden"],
            prev_log_probs=recv_entry["prev_log_probs"],
            recv_hidden_ms=float(recv_entry.get("post_ms", 0.0)) + wait_ms,
            recv_log_probs_ms=0.0,
            load_hidden_ms=(time.perf_counter() - started) * 1000.0,
        )

    def load_common_inputs(self, mb: BPFreeMicrobatch) -> CommonInputs:
        started = time.perf_counter()
        with self._span(mb, "LOAD_COMMON_INPUTS"):
            attention_mask, position_ids, labels = pipeline_support.load_common_tensors(
                records=mb.records,
                manifest_dir=self.manifest_dir,
                device=self.device,
            )
            self._sync_cuda()

        return CommonInputs(
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            load_input_ms=(time.perf_counter() - started) * 1000.0,
        )

    def prepare_window_common_inputs(self, window: BPFreeUpdateWindow) -> None:
        """Load one update window's common tensors once, then keep GPU views.

        This method does not alter the current microbatch path by itself. The
        scheduler must explicitly call it and then use
        window_common_inputs_for_microbatch().
        """
        if self._window_common_input_cache is not None:
            cached_window_id = self._window_common_input_cache.window_id
            raise RuntimeError(
                "window common-input cache is already populated: "
                f"cached_window_id={cached_window_id}, requested_window_id={window.window_id}"
            )
        if not window.microbatches:
            raise ValueError(f"window {window.window_id} has no microbatches")

        records: list[dict[str, Any]] = []
        slices_by_global_batch_seq: dict[int, tuple[int, int]] = {}
        offset = 0
        for mb in window.microbatches:
            start = offset
            records.extend(mb.records)
            offset += mb.num_records
            slices_by_global_batch_seq[mb.global_batch_seq] = (start, offset)

        if offset != window.num_records:
            raise RuntimeError(
                f"window {window.window_id} record accounting mismatch: "
                f"built={offset}, expected={window.num_records}"
            )

        first_mb = window.microbatches[0]
        started = time.perf_counter()
        with self._span(first_mb, "LOAD_WINDOW_COMMON_INPUTS"):
            attention_mask, position_ids, labels = pipeline_support.load_common_tensors(
                records=records,
                manifest_dir=self.manifest_dir,
                device=self.device,
            )
            self._sync_cuda()
        load_input_ms = (time.perf_counter() - started) * 1000.0

        expected_batch = window.num_records
        for name, tensor in (
            ("attention_mask", attention_mask),
            ("position_ids", position_ids),
            ("labels", labels),
        ):
            if tensor.ndim == 0 or tensor.shape[0] != expected_batch:
                raise RuntimeError(
                    f"window {window.window_id} {name} has batch shape "
                    f"{tuple(tensor.shape)}, expected leading dimension {expected_batch}"
                )

        self._window_common_input_cache = WindowCommonInputCache(
            window_id=window.window_id,
            first_global_batch_seq=first_mb.global_batch_seq,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            slices_by_global_batch_seq=slices_by_global_batch_seq,
            load_input_ms=load_input_ms,
        )

    def window_common_inputs_for_microbatch(
        self,
        mb: BPFreeMicrobatch,
    ) -> CommonInputs:
        cache = self._window_common_input_cache
        if cache is None:
            raise RuntimeError(
                "window common-input cache is empty; "
                "prepare_window_common_inputs() must run first"
            )
        if cache.window_id != mb.window_id:
            raise RuntimeError(
                f"window common-input cache mismatch: cached_window_id={cache.window_id}, "
                f"microbatch_window_id={mb.window_id}"
            )

        bounds = cache.slices_by_global_batch_seq.get(mb.global_batch_seq)
        if bounds is None:
            raise KeyError(
                f"microbatch seq={mb.global_batch_seq} is absent from "
                f"window {cache.window_id} common-input cache"
            )
        start, end = bounds
        if end - start != mb.num_records:
            raise RuntimeError(
                f"microbatch seq={mb.global_batch_seq} slice size mismatch: "
                f"slice={end - start}, records={mb.num_records}"
            )

        return CommonInputs(
            attention_mask=cache.attention_mask[start:end],
            position_ids=cache.position_ids[start:end],
            labels=cache.labels[start:end],
            load_input_ms=(
                cache.load_input_ms
                if mb.global_batch_seq == cache.first_global_batch_seq
                else 0.0
            ),
        )

    def clear_window_common_inputs(self, window: BPFreeUpdateWindow) -> None:
        cache = self._window_common_input_cache
        if cache is None:
            return
        if cache.window_id != window.window_id:
            raise RuntimeError(
                f"refusing to clear window {window.window_id}; "
                f"cache belongs to window {cache.window_id}"
            )
        self._window_common_input_cache = None

    def forward_compute_includes_local_head(
        self,
        *,
        mb: BPFreeMicrobatch,
        fwd_input: ForwardInput,
        common: CommonInputs,
        hook_context=nullcontext(),
    ) -> ForwardOutput:
        self.chunk.train(self.train_this_rank)

        choice_ids = (
            one_token_choice_ids(mb.records[0])
            if self.is_last
            and len(mb.records) == 1
            and mb.records[0].get("label_choices")
            else None
        )

        started = time.perf_counter()

        with torch.set_grad_enabled(self.train_this_rank), hook_context:
            with self._span(mb, "FWD_COMPUTE_INCLUDES_LOCAL_HEAD"):
                loss, next_hidden, next_log_probs = self.chunk(
                    hidden_states=fwd_input.hidden,
                    attention_mask=common.attention_mask,
                    position_ids=common.position_ids,
                    labels=common.labels,
                    prev_log_probs=fwd_input.prev_log_probs,
                    choice_ids=choice_ids,
                    record_loss_components=not self.perf_minimal_metrics,
                )
                self._sync_cuda()

        return ForwardOutput(
            loss=loss,
            next_hidden=next_hidden,
            next_log_probs=next_log_probs,
            forward_ms=(time.perf_counter() - started) * 1000.0,
        )

    def post_forward_send(
        self,
        *,
        mb: BPFreeMicrobatch,
        fwd_output: ForwardOutput,
    ) -> SendOutput:
        if self.is_last:
            return SendOutput(send_hidden_ms=0.0, send_log_probs_ms=0.0)

        if fwd_output.next_hidden is None:
            raise RuntimeError("non-last BP-free stage must produce next_hidden.")

        send_hidden_buffer = self._prepare_send_buffer(fwd_output.next_hidden)
        send_ops = [dist.P2POp(dist.isend, send_hidden_buffer, self.rank + 1)]

        send_entry: dict[str, Any] = {
            "buffers": [send_hidden_buffer],
            "works": [],
            "window_id": mb.window_id,
            "mb_id": mb.mb_id,
            "batch_seq": mb.global_batch_seq,
            "seq_start": mb.seq_start,
            "records": mb.num_records,
        }

        if self.belief_transport_mode == "full":
            if fwd_output.next_log_probs is None:
                raise RuntimeError("full belief mode requires next_log_probs from every non-last stage.")
            send_log_probs_buffer = self._prepare_send_buffer(
                fwd_output.next_log_probs.float()
            )
            send_ops.append(dist.P2POp(dist.isend, send_log_probs_buffer, self.rank + 1))
            send_entry["buffers"].append(send_log_probs_buffer)

        if self.device.type == "cuda" and self.comm_stream is not None and self.compute_stream is not None:
            self.comm_stream.wait_stream(self.compute_stream)

        with self._span(mb, "FWD_SEND_POST"):
            works, post_ms = gpu_transport.post_batch_p2p(
                ops=send_ops,
                device=self.device,
                comm_stream=self.comm_stream,
            )

        send_entry["works"] = works
        self.pending_send_entries.append(send_entry)

        total_send_hidden_ms = post_ms
        total_send_log_probs_ms = 0.0

        while len(self.pending_send_entries) > self.max_pending_sends:
            drained = self.pending_send_entries.pop(0)
            total_send_hidden_ms += self._wait_send_entry(drained, action="FWD_SEND_WAIT")

        return SendOutput(
            send_hidden_ms=total_send_hidden_ms,
            send_log_probs_ms=total_send_log_probs_ms,
        )

    def _wait_send_entry(self, entry: dict[str, Any], *, action: str) -> float:
        fake_mb = BPFreeMicrobatch(
            window_id=int(entry["window_id"]),
            mb_id=int(entry["mb_id"]),
            global_batch_seq=int(entry["batch_seq"]),
            seq_start=int(entry["seq_start"]),
            records=[{} for _ in range(int(entry["records"]))],
        )
        with self._span(fake_mb, action):
            return gpu_transport.wait_batch_p2p(works=entry["works"])

    def body_forward_one_chunk(
        self,
        *,
        mb: BPFreeMicrobatch,
        fwd_input: ForwardInput,
        common: CommonInputs,
        hook_context=nullcontext(),
    ) -> BodyForwardOutput:
        started = time.perf_counter()

        with torch.set_grad_enabled(self.train_this_rank), hook_context:
            with self._span(mb, "BODY_FORWARD"):
                result = body_forward(
                    chunk=self.chunk,
                    hidden_states=fwd_input.hidden,
                    attention_mask=common.attention_mask,
                    position_ids=common.position_ids,
                )
                self._sync_cuda()

        return BodyForwardOutput(
            next_hidden=result.curr_hidden,
            body_forward_ms=(time.perf_counter() - started) * 1000.0,
        )

    def post_body_forward_send(
        self,
        *,
        mb: BPFreeMicrobatch,
        body_output: BodyForwardOutput,
    ) -> SendOutput:
        if self.is_last:
            return SendOutput(send_hidden_ms=0.0, send_log_probs_ms=0.0)

        send_hidden_buffer = self._prepare_send_buffer(body_output.next_hidden)
        send_ops = [dist.P2POp(dist.isend, send_hidden_buffer, self.rank + 1)]

        send_entry: dict[str, Any] = {
            "buffers": [send_hidden_buffer],
            "works": [],
            "window_id": mb.window_id,
            "mb_id": mb.mb_id,
            "batch_seq": mb.global_batch_seq,
            "seq_start": mb.seq_start,
            "records": mb.num_records,
        }

        # In full belief mode, output_log_probs is produced by LOCAL_HEAD_LOSS.
        # Therefore the first body-send-head version supports terminal mode first.
        if self.belief_transport_mode == "full":
            raise NotImplementedError(
                "BODY_FORWARD -> SEND -> LOCAL_HEAD_LOSS currently supports "
                "terminal belief transport only. Full belief transport requires "
                "sending log_probs after LOCAL_HEAD_LOSS or splitting hidden/logprob sends."
            )

        if self.device.type == "cuda" and self.comm_stream is not None and self.compute_stream is not None:
            self.comm_stream.wait_stream(self.compute_stream)

        with self._span(mb, "FWD_SEND_POST"):
            works, post_ms = gpu_transport.post_batch_p2p(
                ops=send_ops,
                device=self.device,
                comm_stream=self.comm_stream,
            )

        send_entry["works"] = works
        self.pending_send_entries.append(send_entry)

        total_send_hidden_ms = post_ms

        while len(self.pending_send_entries) > self.max_pending_sends:
            drained = self.pending_send_entries.pop(0)
            total_send_hidden_ms += self._wait_send_entry(drained, action="FWD_SEND_WAIT")

        return SendOutput(
            send_hidden_ms=total_send_hidden_ms,
            send_log_probs_ms=0.0,
        )

    def local_head_loss_one_chunk(
        self,
        *,
        mb: BPFreeMicrobatch,
        body_output: BodyForwardOutput,
        fwd_input: ForwardInput,
        common: CommonInputs,
        hook_context=nullcontext(),
    ) -> LocalHeadLossOutput:
        choice_ids = (
            one_token_choice_ids(mb.records[0])
            if self.is_last
            and len(mb.records) == 1
            and mb.records[0].get("label_choices")
            else None
        )

        started = time.perf_counter()

        with torch.set_grad_enabled(self.train_this_rank), hook_context:
            with self._span(mb, "LOCAL_HEAD_LOSS"):
                result = local_head_loss_from_hidden(
                    chunk=self.chunk,
                    curr_hidden=body_output.next_hidden,
                    labels=common.labels,
                    prev_log_probs=fwd_input.prev_log_probs,
                    choice_ids=choice_ids,
                    record_loss_components=not self.perf_minimal_metrics,
                )
                self._sync_cuda()

        return LocalHeadLossOutput(
            loss=result.loss,
            next_log_probs=result.output_log_probs,
            local_head_loss_ms=(time.perf_counter() - started) * 1000.0,
        )

    def local_backward(
        self,
        *,
        mb: BPFreeMicrobatch,
        window: BPFreeUpdateWindow,
        loss: torch.Tensor,
    ) -> BackwardOutput:
        if not self.train_this_rank:
            return BackwardOutput(backward_ms=0.0, gradient_storage_bytes=0)

        started = time.perf_counter()
        with self._span(mb, "LOCAL_BACKWARD"):
            (loss / window.num_microbatches).backward()
            self._sync_cuda()

        return BackwardOutput(
            backward_ms=(time.perf_counter() - started) * 1000.0,
            gradient_storage_bytes=gradient_storage_nbytes(self.chunk),
        )

    def maybe_optimizer_step(
        self,
        *,
        mb: BPFreeMicrobatch,
        window: BPFreeUpdateWindow,
    ) -> OptimizerStepOutput:
        if not self.train_this_rank:
            return OptimizerStepOutput(optimizer_ms=0.0, applied=False)

        if mb.mb_id != window.num_microbatches - 1:
            return OptimizerStepOutput(optimizer_ms=0.0, applied=False)

        if self.optimizer is None:
            raise RuntimeError("train_this_rank=True but optimizer is None.")

        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.chunk.parameters(), self.grad_clip)

        started = time.perf_counter()
        with self._span(mb, "LOCAL_OPTIMIZER_STEP"):
            self.optimizer.step()
            self._sync_cuda()

        with self._span(mb, "LOCAL_ZERO_GRAD_AFTER_STEP"):
            self.optimizer.zero_grad(set_to_none=True)
            self._sync_cuda()

        return OptimizerStepOutput(
            optimizer_ms=(time.perf_counter() - started) * 1000.0,
            applied=True,
        )

    def drain_pending_sends(self) -> float:
        total_ms = 0.0
        for entry in self.pending_send_entries:
            total_ms += self._wait_send_entry(entry, action="FWD_SEND_WAIT_FINAL_DRAIN")
        self.pending_send_entries.clear()
        return total_ms
