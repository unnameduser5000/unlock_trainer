from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

import torch

from sg_exe_trainer.runtime.bpfree.schedule import BPFreeMicrobatch, BPFreeUpdateWindow
from sg_exe_trainer.runtime.bpfree.gpu_stage import (
    BPFreePipelineStageV0,
    BodyForwardOutput,
    SendOutput,
)


SkipP2PPolicy = Callable[[BPFreeMicrobatch], bool]


@dataclass(frozen=True)
class VolatileBoundary:
    window_id: int
    microbatch_id: int
    hidden: torch.Tensor
    tensor_bytes: int
    capture_ms: float


class VolatileBoundaryBuffer:
    """Process-local CPU hidden backlog for a transient service outage."""

    def __init__(self, *, max_pending_windows: int) -> None:
        if max_pending_windows <= 0:
            raise ValueError("max_pending_windows must be positive")
        self.max_pending_windows = int(max_pending_windows)
        self._entries: dict[tuple[int, int], VolatileBoundary] = {}

    def _validate_capture(self, mb: BPFreeMicrobatch) -> tuple[int, int]:
        key = (int(mb.window_id), int(mb.mb_id))
        if key in self._entries:
            raise RuntimeError(f"volatile boundary already captured: {key}")
        pending_windows = set(self.window_ids())
        if mb.window_id not in pending_windows and len(pending_windows) >= self.max_pending_windows:
            raise RuntimeError(
                "volatile boundary buffer capacity reached before window "
                f"{mb.window_id}: capacity={self.max_pending_windows}"
            )
        return key

    def capture_prepared_cpu(
        self,
        mb: BPFreeMicrobatch,
        hidden_cpu: torch.Tensor,
        *,
        capture_ms: float,
    ) -> VolatileBoundary:
        """Take ownership of an already staged CPU boundary tensor."""
        key = self._validate_capture(mb)
        if hidden_cpu.device.type != "cpu":
            raise ValueError(
                f"prepared volatile boundary must be on CPU, got {hidden_cpu.device}"
            )
        if not hidden_cpu.is_contiguous():
            raise ValueError("prepared volatile boundary must be contiguous")

        entry = VolatileBoundary(
            window_id=int(mb.window_id),
            microbatch_id=int(mb.mb_id),
            hidden=hidden_cpu,
            tensor_bytes=int(hidden_cpu.numel() * hidden_cpu.element_size()),
            capture_ms=float(capture_ms),
        )
        self._entries[key] = entry
        return entry

    def capture(self, mb: BPFreeMicrobatch, hidden: torch.Tensor) -> VolatileBoundary:
        self._validate_capture(mb)

        started = time.perf_counter()
        cpu_hidden = hidden.detach().to(device="cpu").contiguous().clone()
        capture_ms = (time.perf_counter() - started) * 1000.0
        # Validation happened above so this call cannot race with a duplicate in
        # the process-local runner.
        key = (int(mb.window_id), int(mb.mb_id))
        entry = VolatileBoundary(
            window_id=key[0],
            microbatch_id=key[1],
            hidden=cpu_hidden,
            tensor_bytes=int(cpu_hidden.numel() * cpu_hidden.element_size()),
            capture_ms=float(capture_ms),
        )
        self._entries[key] = entry
        return entry

    def get(self, mb: BPFreeMicrobatch) -> VolatileBoundary:
        key = (int(mb.window_id), int(mb.mb_id))
        try:
            return self._entries[key]
        except KeyError as exc:
            raise KeyError(f"volatile boundary is missing: {key}") from exc

    def validate_window(self, window: BPFreeUpdateWindow) -> None:
        missing = [
            mb.mb_id
            for mb in window.microbatches
            if (int(window.window_id), int(mb.mb_id)) not in self._entries
        ]
        if missing:
            raise RuntimeError(
                f"window {window.window_id} has missing volatile microbatches: {missing}"
            )

    def window_ids(self) -> list[int]:
        return sorted({window_id for window_id, _ in self._entries})

    @property
    def tensor_bytes(self) -> int:
        return sum(entry.tensor_bytes for entry in self._entries.values())

    @property
    def capture_ms(self) -> float:
        return sum(entry.capture_ms for entry in self._entries.values())

    @property
    def microbatch_count(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()


class VolatileCaptureBPFreePipelineStage(BPFreePipelineStageV0):
    """Captures skipped forward sends in RAM and replays them without recompute."""

    def __init__(
        self,
        *,
        volatile_buffer: VolatileBoundaryBuffer,
        skip_p2p_policy: SkipP2PPolicy,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.volatile_buffer = volatile_buffer
        self.skip_p2p_policy = skip_p2p_policy
        self._replay_cache: dict[tuple[int, int], torch.Tensor] = {}

    def post_body_forward_send(
        self,
        *,
        mb: BPFreeMicrobatch,
        body_output: BodyForwardOutput,
    ) -> SendOutput:
        if self.is_last or not self.skip_p2p_policy(mb):
            return super().post_body_forward_send(mb=mb, body_output=body_output)
        self.volatile_buffer.capture(mb, body_output.next_hidden)
        return SendOutput(send_hidden_ms=0.0, send_log_probs_ms=0.0)

    def prepare_buffered_replay(
        self,
        windows: list[BPFreeUpdateWindow],
    ) -> dict[str, float | int]:
        if self._replay_cache:
            raise RuntimeError("volatile replay cache is already populated")
        started = time.perf_counter()
        tensor_bytes = 0
        for window in windows:
            self.volatile_buffer.validate_window(window)
            for mb in window.microbatches:
                entry = self.volatile_buffer.get(mb)
                self._replay_cache[(mb.window_id, mb.mb_id)] = entry.hidden.to(
                    device=self.device,
                    dtype=self.dtype,
                )
                tensor_bytes += entry.tensor_bytes
        self._sync_cuda()
        return {
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
            "microbatches": len(self._replay_cache),
            "tensor_bytes": tensor_bytes,
        }

    def replay_buffered_hidden(self, mb: BPFreeMicrobatch) -> dict[str, float | int]:
        if self.is_last:
            raise RuntimeError("terminal stage cannot replay a downstream boundary")
        entry = self.volatile_buffer.get(mb)
        key = (int(mb.window_id), int(mb.mb_id))
        try:
            hidden = self._replay_cache[key]
        except KeyError as exc:
            raise RuntimeError(f"volatile boundary was not staged for replay: {key}") from exc
        send = super().post_body_forward_send(
            mb=mb,
            body_output=BodyForwardOutput(next_hidden=hidden, body_forward_ms=0.0),
        )
        return {
            "window_id": int(mb.window_id),
            "microbatch_id": int(mb.mb_id),
            "h2d_ms": 0.0,
            "send_hidden_ms": float(send.send_hidden_ms),
            "tensor_bytes": entry.tensor_bytes,
        }

    def clear_replay_cache(self) -> None:
        self._replay_cache.clear()
