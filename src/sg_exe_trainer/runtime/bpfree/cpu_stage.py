from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch

from sg_exe_trainer.common.trainable_modes import gradient_storage_nbytes
from sg_exe_trainer.tasks.label_experiment import one_token_choice_ids
from sg_exe_trainer.runtime.transport.cpu import (
    CpuTransportBudget,
    gpu_to_cpu,
    cpu_to_gpu,
    cpu_isend,
    cpu_irecv,
    wait_work,
    forward_tag,
    link_emulation_summary,
    shape_nbytes,
    tensor_nbytes,
)
import torch.distributed as dist

from sg_exe_trainer.runtime.bpfree import gpu_transport, pipeline_support
from sg_exe_trainer.runtime.bpfree.schedule import BPFreeMicrobatch, BPFreeUpdateWindow
from sg_exe_trainer.runtime.bpfree.trace import ActionTracer
from sg_exe_trainer.runtime.bpfree.chunk_split import (
    body_forward,
    local_head_loss_from_hidden,
)


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
        vocab_size: int,
        hidden_size: int,
        input_embedding: Optional[torch.nn.Module] = None,
        manifest_dir: Path,
        tracer: ActionTracer,
        recv_prepost_depth: int = 0,
        max_pending_send_bytes: int = 67_108_864,
        max_posted_recv_bytes: int = 67_108_864,
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
        self.vocab_size = vocab_size
        self.hidden_size = int(hidden_size)
        self.input_embedding = input_embedding
        self.manifest_dir = manifest_dir
        self.tracer = tracer
        # recv_prepost_depth=0 is the strict blocking control.  Positive values
        # cap the total number of posted forward receives, while byte budgets
        # remain the authoritative resource limit.
        self.recv_prepost_depth = max(0, int(recv_prepost_depth))
        self.transport_budget = CpuTransportBudget(
            max_pending_send_bytes=max_pending_send_bytes,
            max_posted_recv_bytes=max_posted_recv_bytes,
        )
        # Retained only for the unused NCCL compatibility path below.
        self.max_pending_sends = 2
        self.forward_recv_plan: list[BPFreeMicrobatch] = []
        self.forward_recv_plan_index_by_seq: dict[int, int] = {}
        self.pending_forward_recv_entries: dict[int, dict[str, Any]] = {}
        self.consumed_forward_recv_seqs: set[int] = set()
        self.sync_action_trace = sync_action_trace
        self.perf_minimal_metrics = perf_minimal_metrics

        self.compute_stream = torch.cuda.current_stream(device) if device.type == "cuda" else None
        self.comm_stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
        self.pending_send_entries: list[dict[str, Any]] = []

    @property
    def is_first(self) -> bool:
        return self.rank == 0

    @property
    def is_last(self) -> bool:
        return self.rank == self.world_size - 1

    def transport_summary(self) -> dict[str, Any]:
        return {
            "recv_prepost_depth": self.recv_prepost_depth,
            **self.transport_budget.summary(),
            "link_emulation": link_emulation_summary(),
        }

    def _sync_cuda(self, mb: BPFreeMicrobatch) -> None:
        if (
            self.sync_action_trace
            and self.device.type == "cuda"
            and self.tracer.is_enabled_for_window(mb.window_id)
        ):
            torch.cuda.synchronize(self.device)

    def _span(self, mb: BPFreeMicrobatch, action: str):
        return self.tracer.span(
            window_id=mb.window_id,
            mb_id=mb.mb_id,
            global_batch_seq=mb.global_batch_seq,
            seq_start=mb.seq_start,
            records=mb.num_records,
            action=action,
        )

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
            self._sync_cuda(first_mb)


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
    ) -> dict[str, Any] | None:
        """Post one tagged CPU/Gloo hidden receive within the shared budget."""
        if self.is_first:
            raise RuntimeError("_make_forward_recv_entry called on first stage")
        if self.belief_transport_mode == "full":
            raise NotImplementedError(
                "CPU recv prepost currently supports terminal belief mode only."
            )

        hidden_shape = tuple(
            pipeline_support.hidden_shape(
                mb.records[0],
                len(mb.records),
                self.hidden_size,
            )
        )
        nbytes = shape_nbytes(hidden_shape, self.dtype)
        if not self.transport_budget.can_reserve_recv(nbytes):
            self.transport_budget.recv_budget_stalls += 1
            return None

        self.transport_budget.reserve_recv(nbytes)
        try:
            with self._span(mb, "FWD_RECV_POST_CPU"):
                hidden_cpu, work, post_ms = cpu_irecv(
                    shape=hidden_shape,
                    dtype=self.dtype,
                    src=self.rank - 1,
                    pin_memory=True,
                    tag=forward_tag(self.rank - 1, mb.global_batch_seq),
                )
        except Exception:
            self.transport_budget.release_recv(nbytes)
            raise

        return {
            "hidden_cpu": hidden_cpu,
            "work": work,
            "post_ms": float(post_ms),
            "nbytes": nbytes,
            "window_id": mb.window_id,
            "mb_id": mb.mb_id,
            "batch_seq": mb.global_batch_seq,
            "seq_start": mb.seq_start,
            "records": mb.num_records,
        }

    def prepost_forward_recv(self, mb: BPFreeMicrobatch) -> bool:
        """Post one future receive; return False when the byte budget is full."""
        if self.is_first or self.recv_prepost_depth <= 0:
            return False
        if mb.global_batch_seq in self.consumed_forward_recv_seqs:
            return True
        if mb.global_batch_seq in self.pending_forward_recv_entries:
            return True

        entry = self._make_forward_recv_entry(mb)
        if entry is None:
            return False
        self.pending_forward_recv_entries[mb.global_batch_seq] = entry
        return True

    def maintain_forward_recv_inflight(self, mb: BPFreeMicrobatch) -> None:
        """Keep at most recv_prepost_depth receive actions posted.

        Depth zero is the strict blocking control.  Positive depth counts the
        current receive action as one outstanding action.  The byte budget may
        reduce the realized depth for larger physical batches.
        """
        if self.is_first or self.recv_prepost_depth <= 0:
            return
        idx = self.forward_recv_plan_index_by_seq.get(mb.global_batch_seq)
        if idx is None:
            return

        outstanding = 0
        for planned in self.forward_recv_plan[idx:]:
            if planned.global_batch_seq in self.consumed_forward_recv_seqs:
                continue
            if planned.global_batch_seq in self.pending_forward_recv_entries:
                outstanding += 1
            else:
                if outstanding >= self.recv_prepost_depth:
                    break
                if not self.prepost_forward_recv(planned):
                    break
                outstanding += 1
            if outstanding >= self.recv_prepost_depth:
                break

    def _consume_forward_recv_entry(
        self,
        mb: BPFreeMicrobatch,
        entry: dict[str, Any],
    ) -> ForwardInput:
        with self._span(mb, "FWD_RECV_WAIT_CPU"):
            wait_ms = wait_work(entry["work"])

        with self._span(mb, "FWD_RECV_H2D"):
            hidden, h2d_ms = cpu_to_gpu(
                entry["hidden_cpu"],
                device=self.device,
                non_blocking=True,
                sync=True,
            )

        post_ms = float(entry.get("post_ms", 0.0))
        self.transport_budget.release_recv(int(entry["nbytes"]))
        entry.clear()
        self.consumed_forward_recv_seqs.add(mb.global_batch_seq)

        return ForwardInput(
            hidden=hidden,
            prev_log_probs=None,
            recv_hidden_ms=post_ms + wait_ms + h2d_ms,
            recv_log_probs_ms=0.0,
            load_hidden_ms=0.0,
        )


    def load_or_recv_forward_input(self, mb: BPFreeMicrobatch) -> ForwardInput:
        batch_records = mb.records

        if self.is_first:
            started = time.perf_counter()
            with self._span(mb, "LOAD_STAGE0_HIDDEN"):
                hidden = pipeline_support.load_stage0_hidden(
                    records=batch_records,
                    manifest_dir=self.manifest_dir,
                    device=self.device,
                    dtype=self.dtype,
                    input_embedding=self.input_embedding,
                )
                self._sync_cuda(mb)

            return ForwardInput(
                hidden=hidden,
                prev_log_probs=None,
                load_hidden_ms=(time.perf_counter() - started) * 1000.0,
            )

        if self.belief_transport_mode == "full":
            raise NotImplementedError(
                "CP-CPU CPU hidden transport supports terminal belief mode only. "
                "Full belief mode needs a separate CPU log_probs transport path."
            )

        # CPU-comm v1 path: consume a preposted CPU/Gloo recv if available.
        entry = self.pending_forward_recv_entries.pop(mb.global_batch_seq, None)
        if entry is not None:
            return self._consume_forward_recv_entry(mb, entry)

        # Blocking fallback, also charged to the shared receive-byte budget.
        hidden_shape = tuple(
            pipeline_support.hidden_shape(
                batch_records[0],
                len(batch_records),
                self.hidden_size,
            )
        )
        nbytes = shape_nbytes(hidden_shape, self.dtype)
        if not self.transport_budget.can_reserve_recv(nbytes):
            raise RuntimeError(
                "blocking BP-free receive cannot fit because future preposts still "
                "consume the shared receive budget"
            )
        self.transport_budget.reserve_recv(nbytes)
        try:
            with self._span(mb, "FWD_RECV_POST_CPU"):
                hidden_cpu, work, post_ms = cpu_irecv(
                    shape=hidden_shape,
                    dtype=self.dtype,
                    src=self.rank - 1,
                    pin_memory=True,
                    tag=forward_tag(self.rank - 1, mb.global_batch_seq),
                )

            with self._span(mb, "FWD_RECV_WAIT_CPU"):
                wait_ms = wait_work(work)

            with self._span(mb, "FWD_RECV_H2D"):
                hidden, h2d_ms = cpu_to_gpu(
                    hidden_cpu,
                    device=self.device,
                    non_blocking=True,
                    sync=True,
                )
        finally:
            self.transport_budget.release_recv(nbytes)

        self.consumed_forward_recv_seqs.add(mb.global_batch_seq)

        return ForwardInput(
            hidden=hidden,
            prev_log_probs=None,
            recv_hidden_ms=post_ms + wait_ms + h2d_ms,
            recv_log_probs_ms=0.0,
            load_hidden_ms=0.0,
        )


    def load_common_inputs(self, mb: BPFreeMicrobatch) -> CommonInputs:
        started = time.perf_counter()
        with self._span(mb, "LOAD_COMMON_INPUTS"):
            attention_mask, position_ids, labels = pipeline_support.load_common_tensors(
                records=mb.records,
                manifest_dir=self.manifest_dir,
                device=self.device,
            )
            self._sync_cuda(mb)

        return CommonInputs(
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            load_input_ms=(time.perf_counter() - started) * 1000.0,
        )

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
                self._sync_cuda(mb)

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

        send_hidden_buffer = fwd_output.next_hidden.detach().contiguous()
        send_ops = [dist.P2POp(dist.isend, send_hidden_buffer, self.rank + 1)]

        send_entry: dict[str, Any] = {
            "buffers": [send_hidden_buffer],
            "works": [],
            "mb": mb,
            "window_id": mb.window_id,
            "mb_id": mb.mb_id,
            "batch_seq": mb.global_batch_seq,
            "seq_start": mb.seq_start,
            "records": mb.num_records,
        }

        if self.belief_transport_mode == "full":
            if fwd_output.next_log_probs is None:
                raise RuntimeError("full belief mode requires next_log_probs from every non-last stage.")
            send_log_probs_buffer = fwd_output.next_log_probs.detach().float().contiguous()
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
                self._sync_cuda(mb)

        return BodyForwardOutput(
            next_hidden=result.curr_hidden,
            body_forward_ms=(time.perf_counter() - started) * 1000.0,
        )

    def _drain_oldest_cpu_send(self, *, action: str) -> float:
        entry = self.pending_send_entries.pop(0)
        with self._span(entry["mb"], action):
            wait_ms = sum(wait_work(work) for work in entry["works"])
        self.transport_budget.release_send(int(entry["nbytes"]))
        entry.clear()
        return wait_ms

    def post_body_forward_send(
        self,
        *,
        mb: BPFreeMicrobatch,
        body_output: BodyForwardOutput,
    ) -> SendOutput:
        if self.is_last:
            return SendOutput(send_hidden_ms=0.0, send_log_probs_ms=0.0)

        if self.belief_transport_mode == "full":
            raise NotImplementedError(
                "BODY_FORWARD -> CPU SEND -> LOCAL_HEAD_LOSS currently supports "
                "terminal belief transport only. Full belief transport requires "
                "sending log_probs after LOCAL_HEAD_LOSS or splitting hidden/logprob sends."
            )

        nbytes = tensor_nbytes(body_output.next_hidden)
        total_send_hidden_ms = 0.0
        while not self.transport_budget.can_reserve_send(nbytes):
            if not self.pending_send_entries:
                raise RuntimeError(
                    "send budget cannot fit one BP-free hidden message despite "
                    "passing message-size validation"
                )
            self.transport_budget.send_budget_waits += 1
            total_send_hidden_ms += self._drain_oldest_cpu_send(
                action="FWD_SEND_WAIT_BUDGET_CPU"
            )

        with self._span(mb, "FWD_SEND_D2H"):
            send_hidden_cpu, d2h_ms = gpu_to_cpu(
                body_output.next_hidden,
                pin_memory=True,
                sync=True,
            )

        with self._span(mb, "FWD_SEND_POST_CPU"):
            work, post_ms = cpu_isend(
                send_hidden_cpu,
                dst=self.rank + 1,
                tag=forward_tag(self.rank, mb.global_batch_seq),
            )

        self.transport_budget.reserve_send(nbytes)
        self.pending_send_entries.append(
            {
                "works": [work],
                "hidden_cpu": send_hidden_cpu,
                "nbytes": nbytes,
                "d2h_ms": d2h_ms,
                "post_ms": post_ms,
                "mb": mb,
                "window_id": mb.window_id,
                "mb_id": mb.mb_id,
                "batch_seq": mb.global_batch_seq,
                "seq_start": mb.seq_start,
                "records": mb.num_records,
            }
        )

        total_send_hidden_ms += d2h_ms + post_ms
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
                    emit_output_log_probs=not (
                        self.perf_minimal_metrics and self.is_last
                    ),
                )
                self._sync_cuda(mb)

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
            self._sync_cuda(mb)

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
            self._sync_cuda(mb)

        with self._span(mb, "LOCAL_ZERO_GRAD_AFTER_STEP"):
            self.optimizer.zero_grad(set_to_none=True)
            self._sync_cuda(mb)

        return OptimizerStepOutput(
            optimizer_ms=(time.perf_counter() - started) * 1000.0,
            applied=True,
        )

    def drain_pending_sends(self) -> float:
        total_wait_ms = 0.0
        while self.pending_send_entries:
            total_wait_ms += self._drain_oldest_cpu_send(
                action="FWD_SEND_WAIT_FINAL_CPU"
            )
        if self.transport_budget.pending_send_bytes != 0:
            raise RuntimeError("BP-free send budget did not return to zero")
        if self.transport_budget.posted_recv_bytes != 0:
            raise RuntimeError("BP-free receive budget did not return to zero")
        return total_wait_ms
