"""GPU/NCCL point-to-point transport primitives for BPFree stages."""

from __future__ import annotations

import time
from typing import Any, Optional

import torch
import torch.distributed as dist


def recv_tensor(shape: list[int], dtype: torch.dtype, device: torch.device, src: int) -> tuple[torch.Tensor, float]:
    tensor = torch.empty(tuple(shape), dtype=dtype, device=device)
    started = time.perf_counter()
    dist.recv(tensor=tensor, src=src)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return tensor, (time.perf_counter() - started) * 1000.0


def send_tensor(tensor: torch.Tensor, dst: int, device: torch.device) -> float:
    started = time.perf_counter()
    dist.send(tensor=tensor.contiguous(), dst=dst)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (time.perf_counter() - started) * 1000.0


def post_recv_tensor(
    *,
    tensor: torch.Tensor,
    src: int,
    device: torch.device,
    comm_stream: Optional[torch.cuda.Stream],
) -> tuple[Any, float]:
    started = time.perf_counter()
    if device.type == "cuda" and comm_stream is not None:
        with torch.cuda.stream(comm_stream):
            work = dist.irecv(tensor=tensor, src=src)
    else:
        work = dist.irecv(tensor=tensor, src=src)
    return work, (time.perf_counter() - started) * 1000.0


def wait_recv_tensor(
    *,
    work: Any,
    device: torch.device,
    comm_stream: Optional[torch.cuda.Stream],
) -> float:
    started = time.perf_counter()
    work.wait()
    return (time.perf_counter() - started) * 1000.0


def post_send_tensor(
    *,
    tensor: torch.Tensor,
    dst: int,
    device: torch.device,
    comm_stream: Optional[torch.cuda.Stream],
) -> tuple[torch.Tensor, Any, float]:
    send_buffer = tensor.detach().contiguous()
    started = time.perf_counter()
    if device.type == "cuda" and comm_stream is not None:
        compute_stream = torch.cuda.current_stream(device)
        with torch.cuda.stream(comm_stream):
            comm_stream.wait_stream(compute_stream)
            work = dist.isend(tensor=send_buffer, dst=dst)
    else:
        work = dist.isend(tensor=send_buffer, dst=dst)
    return send_buffer, work, (time.perf_counter() - started) * 1000.0


def wait_send_tensor(
    *,
    work: Any,
    device: torch.device,
    comm_stream: Optional[torch.cuda.Stream],
) -> float:
    started = time.perf_counter()
    work.wait()
    return (time.perf_counter() - started) * 1000.0


def post_batch_p2p(
    *,
    ops: list[Any],
    device: torch.device,
    comm_stream: Optional[torch.cuda.Stream],
) -> tuple[list[Any], float]:
    if not ops:
        return [], 0.0
    started = time.perf_counter()
    if device.type == "cuda" and comm_stream is not None:
        with torch.cuda.stream(comm_stream):
            works = dist.batch_isend_irecv(ops)
    else:
        works = dist.batch_isend_irecv(ops)
    return works, (time.perf_counter() - started) * 1000.0


def wait_batch_p2p(
    *,
    works: list[Any],
) -> float:
    started = time.perf_counter()
    for work in works:
        work.wait()
    return (time.perf_counter() - started) * 1000.0


def start_region_timer(
    *,
    device: torch.device,
    stream: Optional[torch.cuda.Stream],
) -> tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event], float]:
    if device.type == "cuda" and stream is not None:
        start_event = torch.cuda.Event(enable_timing=True)
        start_event.record(stream)
        return start_event, None, 0.0
    return None, None, time.perf_counter()


def stop_region_timer(
    *,
    timer: tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event], float],
    device: torch.device,
    stream: Optional[torch.cuda.Stream],
) -> tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event], float]:
    start_event, _, cpu_value = timer
    if start_event is not None and device.type == "cuda" and stream is not None:
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record(stream)
        return start_event, end_event, 0.0
    return None, None, (time.perf_counter() - cpu_value) * 1000.0


def resolve_region_timer_ms(
    timer: tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event], float],
) -> float:
    start_event, end_event, cpu_value = timer
    if start_event is not None and end_event is not None:
        return float(start_event.elapsed_time(end_event))
    return float(cpu_value)


def sync_max_ms(local_ms: float, device: torch.device) -> float:
    tensor = torch.tensor([local_ms], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())
