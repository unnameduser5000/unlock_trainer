from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class CpuLinkEmulator:
    """Deterministic sender-side pacing for mobile-link sensitivity tests.

    This is deliberately a userspace pacing model rather than a kernel-level
    network emulator.  Each data ``isend`` is delayed by one-way latency plus
    message serialization time.  The same implementation is shared by
    BP-free and Exact-BP, and defaults to disabled.
    """

    one_way_latency_ms: float = 0.0
    bandwidth_mbps: float = 0.0
    jitter_ms: float = 0.0
    seed: int = 0
    send_calls: int = 0
    paced_bytes: int = 0
    injected_delay_ms: float = 0.0
    injected_latency_ms: float = 0.0
    injected_serialization_ms: float = 0.0
    injected_jitter_ms: float = 0.0

    def __post_init__(self) -> None:
        self.one_way_latency_ms = float(self.one_way_latency_ms)
        self.bandwidth_mbps = float(self.bandwidth_mbps)
        self.jitter_ms = float(self.jitter_ms)
        self.seed = int(self.seed)
        if self.one_way_latency_ms < 0:
            raise ValueError("one_way_latency_ms must be non-negative")
        if self.bandwidth_mbps < 0:
            raise ValueError("bandwidth_mbps must be non-negative; 0 means unlimited")
        if self.jitter_ms < 0:
            raise ValueError("jitter_ms must be non-negative")

    @property
    def enabled(self) -> bool:
        return (
            self.one_way_latency_ms > 0
            or self.bandwidth_mbps > 0
            or self.jitter_ms > 0
        )

    def _deterministic_jitter_ms(self, *, dst: int, tag: int) -> float:
        if self.jitter_ms <= 0:
            return 0.0
        # Stable integer mixing: independent of Python's randomized hash seed.
        value = (
            (self.seed & 0xFFFFFFFFFFFFFFFF)
            ^ ((int(dst) + 1) * 0x9E3779B185EBCA87)
            ^ ((int(tag) + 1) * 0xC2B2AE3D27D4EB4F)
            ^ ((self.send_calls + 1) * 0x165667B19E3779F9)
        ) & 0xFFFFFFFFFFFFFFFF
        value ^= value >> 30
        value = (value * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        value ^= value >> 27
        value = (value * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        value ^= value >> 31
        unit = value / float(0xFFFFFFFFFFFFFFFF)
        return (2.0 * unit - 1.0) * self.jitter_ms

    def pace_send(self, *, nbytes: int, dst: int, tag: int) -> float:
        size = int(nbytes)
        if size <= 0:
            raise ValueError(f"paced message size must be positive, got {size}")

        latency_ms = self.one_way_latency_ms
        serialization_ms = (
            size * 8.0 / (self.bandwidth_mbps * 1000.0)
            if self.bandwidth_mbps > 0
            else 0.0
        )
        jitter_ms = self._deterministic_jitter_ms(dst=dst, tag=tag)
        delay_ms = max(0.0, latency_ms + serialization_ms + jitter_ms)

        self.send_calls += 1
        self.paced_bytes += size
        self.injected_delay_ms += delay_ms
        self.injected_latency_ms += latency_ms
        self.injected_serialization_ms += serialization_ms
        self.injected_jitter_ms += (
            delay_ms - latency_ms - serialization_ms
            if self.jitter_ms > 0
            else 0.0
        )

        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)
        return delay_ms

    def summary(self) -> dict[str, int | float | bool | str]:
        return {
            "mode": "sender_side_pacing",
            "enabled": self.enabled,
            "one_way_latency_ms": self.one_way_latency_ms,
            "bandwidth_mbps": self.bandwidth_mbps,
            "jitter_ms": self.jitter_ms,
            "seed": self.seed,
            "send_calls": self.send_calls,
            "paced_bytes": self.paced_bytes,
            "injected_delay_ms": self.injected_delay_ms,
            "injected_latency_ms": self.injected_latency_ms,
            "injected_serialization_ms": self.injected_serialization_ms,
            "injected_jitter_ms": self.injected_jitter_ms,
        }


_LINK_EMULATOR = CpuLinkEmulator()


def configure_link_emulation(
    *,
    one_way_latency_ms: float = 0.0,
    bandwidth_mbps: float = 0.0,
    jitter_ms: float = 0.0,
    seed: int = 0,
) -> None:
    """Reset and configure the process-local shared link pacing model."""
    global _LINK_EMULATOR
    _LINK_EMULATOR = CpuLinkEmulator(
        one_way_latency_ms=one_way_latency_ms,
        bandwidth_mbps=bandwidth_mbps,
        jitter_ms=jitter_ms,
        seed=seed,
    )


def link_emulation_summary() -> dict[str, int | float | bool | str]:
    return _LINK_EMULATOR.summary()


@dataclass
class CpuTransportTiming:
    d2h_ms: float = 0.0
    post_ms: float = 0.0
    wait_ms: float = 0.0
    h2d_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.d2h_ms + self.post_ms + self.wait_ms + self.h2d_ms


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _elapsed_ms(t0: float, t1: float) -> float:
    return (t1 - t0) * 1000.0


def gpu_to_cpu(
    x_gpu: torch.Tensor,
    *,
    pin_memory: bool = False,
    sync: bool = True,
) -> tuple[torch.Tensor, float]:
    device = x_gpu.device
    t0 = time.perf_counter()

    if pin_memory:
        x_cpu = torch.empty(
            tuple(x_gpu.shape),
            dtype=x_gpu.dtype,
            device="cpu",
            pin_memory=True,
        )
        x_cpu.copy_(x_gpu.detach(), non_blocking=True)
        if sync:
            _sync_device(device)
    else:
        x_cpu = x_gpu.detach().cpu().contiguous()

    t1 = time.perf_counter()
    return x_cpu, _elapsed_ms(t0, t1)


def cpu_to_gpu(
    x_cpu: torch.Tensor,
    *,
    device: torch.device,
    non_blocking: bool = False,
    sync: bool = True,
) -> tuple[torch.Tensor, float]:
    t0 = time.perf_counter()
    x_gpu = x_cpu.to(device=device, non_blocking=non_blocking)
    if sync:
        _sync_device(device)
    t1 = time.perf_counter()
    return x_gpu, _elapsed_ms(t0, t1)


def cpu_isend(
    x_cpu: torch.Tensor,
    *,
    dst: int,
    tag: int = 0,
) -> tuple[dist.Work, float]:
    assert x_cpu.device.type == "cpu", x_cpu.device
    contiguous = x_cpu.contiguous()
    _LINK_EMULATOR.pace_send(
        nbytes=tensor_nbytes(contiguous),
        dst=dst,
        tag=int(tag),
    )
    t0 = time.perf_counter()
    work = dist.isend(contiguous, dst=dst, tag=int(tag))
    t1 = time.perf_counter()
    return work, _elapsed_ms(t0, t1)


def cpu_irecv(
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    src: int,
    pin_memory: bool = False,
    tag: int = 0,
) -> tuple[torch.Tensor, dist.Work, float]:
    x_cpu = torch.empty(
        shape,
        dtype=dtype,
        device="cpu",
        pin_memory=pin_memory,
    )

    t0 = time.perf_counter()
    work = dist.irecv(x_cpu, src=src, tag=int(tag))
    t1 = time.perf_counter()
    return x_cpu, work, _elapsed_ms(t0, t1)


def wait_work(work: dist.Work) -> float:
    t0 = time.perf_counter()
    work.wait()
    t1 = time.perf_counter()
    return _elapsed_ms(t0, t1)


def send_gpu_tensor_cpu_transport(
    x_gpu: torch.Tensor,
    *,
    dst: int,
    pin_memory: bool = False,
) -> tuple[dist.Work, torch.Tensor, CpuTransportTiming]:
    x_cpu, d2h_ms = gpu_to_cpu(x_gpu, pin_memory=pin_memory, sync=True)
    work, post_ms = cpu_isend(x_cpu, dst=dst)

    return work, x_cpu, CpuTransportTiming(
        d2h_ms=d2h_ms,
        post_ms=post_ms,
    )


def recv_gpu_tensor_cpu_transport(
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    src: int,
    device: torch.device,
    pin_memory: bool = False,
) -> tuple[torch.Tensor, CpuTransportTiming]:
    x_cpu, work, post_ms = cpu_irecv(
        shape=shape,
        dtype=dtype,
        src=src,
        pin_memory=pin_memory,
    )
    wait_ms = wait_work(work)
    x_gpu, h2d_ms = cpu_to_gpu(
        x_cpu,
        device=device,
        non_blocking=pin_memory,
        sync=True,
    )

    return x_gpu, CpuTransportTiming(
        post_ms=post_ms,
        wait_ms=wait_ms,
        h2d_ms=h2d_ms,
    )



def tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def shape_nbytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    elements = 1
    for dim in shape:
        elements *= int(dim)
    return int(elements * torch.empty((), dtype=dtype).element_size())


@dataclass
class CpuTransportBudget:
    """Shared per-rank byte budget for pinned CPU transport state.

    The send budget is shared by all logical channels on the rank.  In
    particular, Exact-BP forward-hidden and backward-gradient sends do not get
    separate allowances.  The receive budget is likewise shared by every
    preposted receive action.
    """

    max_pending_send_bytes: int
    max_posted_recv_bytes: int
    pending_send_bytes: int = 0
    posted_recv_bytes: int = 0
    peak_pending_send_bytes: int = 0
    peak_posted_recv_bytes: int = 0
    total_send_bytes: int = 0
    total_recv_bytes: int = 0
    send_budget_waits: int = 0
    recv_budget_stalls: int = 0

    def __post_init__(self) -> None:
        self.max_pending_send_bytes = int(self.max_pending_send_bytes)
        self.max_posted_recv_bytes = int(self.max_posted_recv_bytes)
        if self.max_pending_send_bytes <= 0:
            raise ValueError("max_pending_send_bytes must be positive")
        if self.max_posted_recv_bytes <= 0:
            raise ValueError("max_posted_recv_bytes must be positive")

    def _validate_message(self, nbytes: int, limit: int, label: str) -> int:
        value = int(nbytes)
        if value <= 0:
            raise ValueError(f"{label} message size must be positive, got {value}")
        if value > limit:
            raise RuntimeError(
                f"{label} message needs {value} bytes but the configured budget is "
                f"only {limit} bytes; raise the shared transport budget"
            )
        return value

    def can_reserve_send(self, nbytes: int) -> bool:
        value = self._validate_message(
            nbytes, self.max_pending_send_bytes, "pending send"
        )
        return self.pending_send_bytes + value <= self.max_pending_send_bytes

    def reserve_send(self, nbytes: int) -> None:
        value = self._validate_message(
            nbytes, self.max_pending_send_bytes, "pending send"
        )
        if self.pending_send_bytes + value > self.max_pending_send_bytes:
            raise RuntimeError("pending-send budget exceeded without draining")
        self.pending_send_bytes += value
        self.total_send_bytes += value
        self.peak_pending_send_bytes = max(
            self.peak_pending_send_bytes, self.pending_send_bytes
        )

    def release_send(self, nbytes: int) -> None:
        self.pending_send_bytes -= int(nbytes)
        if self.pending_send_bytes < 0:
            raise RuntimeError("pending-send byte accounting underflow")

    def can_reserve_recv(self, nbytes: int) -> bool:
        value = self._validate_message(
            nbytes, self.max_posted_recv_bytes, "posted receive"
        )
        return self.posted_recv_bytes + value <= self.max_posted_recv_bytes

    def reserve_recv(self, nbytes: int) -> None:
        value = self._validate_message(
            nbytes, self.max_posted_recv_bytes, "posted receive"
        )
        if self.posted_recv_bytes + value > self.max_posted_recv_bytes:
            raise RuntimeError("posted-receive budget exceeded")
        self.posted_recv_bytes += value
        self.total_recv_bytes += value
        self.peak_posted_recv_bytes = max(
            self.peak_posted_recv_bytes, self.posted_recv_bytes
        )

    def release_recv(self, nbytes: int) -> None:
        self.posted_recv_bytes -= int(nbytes)
        if self.posted_recv_bytes < 0:
            raise RuntimeError("posted-receive byte accounting underflow")

    def summary(self) -> dict[str, int]:
        return {
            "max_pending_send_bytes": self.max_pending_send_bytes,
            "max_posted_recv_bytes": self.max_posted_recv_bytes,
            "peak_pending_send_bytes": self.peak_pending_send_bytes,
            "peak_posted_recv_bytes": self.peak_posted_recv_bytes,
            "total_send_bytes": self.total_send_bytes,
            "total_recv_bytes": self.total_recv_bytes,
            "send_budget_waits": self.send_budget_waits,
            "recv_budget_stalls": self.recv_budget_stalls,
            "pending_send_bytes_at_end": self.pending_send_bytes,
            "posted_recv_bytes_at_end": self.posted_recv_bytes,
        }

def forward_tag(edge: int, global_microbatch: int) -> int:
    """Stable Gloo tag for one forward hidden transfer."""
    return 100_000_000 + int(edge) * 1_000_000 + int(global_microbatch)


def backward_tag(edge: int, global_microbatch: int) -> int:
    """Stable Gloo tag for one backward hidden-gradient transfer."""
    return 200_000_000 + int(edge) * 1_000_000 + int(global_microbatch)


def sync_max_wall_ms(local_ms: float) -> float:
    """Return the slowest rank wall time using a CPU collective.

    The CPU transport runners initialize a Gloo process group, so the timing
    reduction must use a CPU tensor even when stage computation runs on CUDA.
    """
    value = torch.tensor([float(local_ms)], dtype=torch.float64, device="cpu")
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return float(value.item())
