from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch.distributed as dist

from sg_exe_trainer.runtime.bpfree.schedule import BPFreeMicrobatch, BPFreeUpdateWindow
from sg_exe_trainer.runtime.bpfree.cpu_stage import (
    BPFreePipelineStageV0 as CpuBPFreePipelineStageV0,
    BodyForwardOutput,
    SendOutput,
)
from sg_exe_trainer.runtime.recovery.volatile_backlog import (
    SkipP2PPolicy,
    VolatileBoundaryBuffer,
)
from sg_exe_trainer.runtime.transport.cpu import (
    cpu_isend,
    forward_tag,
    gpu_to_cpu,
)


class CpuVolatileCaptureBPFreePipelineStage(CpuBPFreePipelineStageV0):
    """E5 transient-outage adapter for the E4 CPU transport stage.

    During the outage the normal GPU-to-pinned-CPU boundary copy still occurs,
    but the resulting CPU tensor is retained instead of posted to Gloo. At
    rejoin, replay posts that exact CPU tensor directly to Gloo.
    """

    def __init__(
        self,
        *,
        volatile_buffer: VolatileBoundaryBuffer,
        skip_p2p_policy: SkipP2PPolicy,
        **kwargs: Any,
    ) -> None:
        # E5's older GPU-P2P runner used these names. Keep the compatibility at
        # this adapter boundary rather than teaching the E4 stage about E5.
        kwargs.pop("max_pending_sends", None)
        recv_depth = int(kwargs.pop("recv_inflight_depth", 0))
        kwargs.setdefault("recv_prepost_depth", recv_depth)
        kwargs.setdefault("max_pending_send_bytes", 67_108_864)
        kwargs.setdefault("max_posted_recv_bytes", 67_108_864)
        super().__init__(**kwargs)
        self.volatile_buffer = volatile_buffer
        self.skip_p2p_policy = skip_p2p_policy

        if dist.is_initialized() and dist.get_backend() != "gloo":
            raise RuntimeError(
                "CpuVolatileCaptureBPFreePipelineStage requires a Gloo process group"
            )

    def transport_summary(self) -> dict[str, Any]:
        return {
            "transport": "gloo-cpu-hidden-pinned-budgeted",
            "outage_boundary_state": "retained-pinned-cpu-hidden",
            **super().transport_summary(),
        }

    def post_body_forward_send(
        self,
        *,
        mb: BPFreeMicrobatch,
        body_output: BodyForwardOutput,
    ) -> SendOutput:
        if self.is_last or not self.skip_p2p_policy(mb):
            return super().post_body_forward_send(mb=mb, body_output=body_output)

        with self._span(mb, "OUTAGE_HIDDEN_D2H"):
            hidden_cpu, d2h_ms = gpu_to_cpu(
                body_output.next_hidden,
                pin_memory=True,
                sync=True,
            )
        self.volatile_buffer.capture_prepared_cpu(
            mb,
            hidden_cpu,
            capture_ms=d2h_ms,
        )
        return SendOutput(send_hidden_ms=d2h_ms, send_log_probs_ms=0.0)

    def prepare_buffered_replay(
        self,
        windows: list[BPFreeUpdateWindow],
    ) -> dict[str, float | int]:
        started = time.perf_counter()
        microbatches = 0
        tensor_bytes = 0
        for window in windows:
            self.volatile_buffer.validate_window(window)
            for mb in window.microbatches:
                entry = self.volatile_buffer.get(mb)
                microbatches += 1
                tensor_bytes += entry.tensor_bytes
        return {
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
            "microbatches": microbatches,
            "tensor_bytes": tensor_bytes,
        }

    def replay_buffered_hidden(self, mb: BPFreeMicrobatch) -> dict[str, float | int]:
        if self.is_last:
            raise RuntimeError("terminal stage cannot replay a downstream boundary")

        entry = self.volatile_buffer.get(mb)
        nbytes = entry.tensor_bytes
        wait_ms = 0.0
        while not self.transport_budget.can_reserve_send(nbytes):
            if not self.pending_send_entries:
                raise RuntimeError(
                    "recovery hidden does not fit the configured pending-send budget"
                )
            self.transport_budget.send_budget_waits += 1
            wait_ms += self._drain_oldest_cpu_send(
                action="RECOVERY_SEND_WAIT_BUDGET_CPU"
            )

        with self._span(mb, "RECOVERY_SEND_POST_CPU"):
            work, post_ms = cpu_isend(
                entry.hidden,
                dst=self.rank + 1,
                tag=forward_tag(self.rank, mb.global_batch_seq),
            )

        self.transport_budget.reserve_send(nbytes)
        self.pending_send_entries.append(
            {
                "works": [work],
                "hidden_cpu": entry.hidden,
                "nbytes": nbytes,
                "d2h_ms": 0.0,
                "post_ms": post_ms,
                "mb": mb,
                "window_id": mb.window_id,
                "mb_id": mb.mb_id,
                "batch_seq": mb.global_batch_seq,
                "seq_start": mb.seq_start,
                "records": mb.num_records,
            }
        )
        return {
            "window_id": int(mb.window_id),
            "microbatch_id": int(mb.mb_id),
            "h2d_ms": 0.0,
            "send_hidden_ms": wait_ms + post_ms,
            "tensor_bytes": nbytes,
        }

    def clear_replay_cache(self) -> None:
        # CPU replay owns no second staging cache; the volatile buffer is the
        # transport-ready state and is cleared by the runner after catch-up.
        return None
