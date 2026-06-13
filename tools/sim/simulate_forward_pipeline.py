#!/usr/bin/env python3
"""Discrete-event simulator for the SID forward-only stage pipeline.

This models the current coordinator-managed stage runner:

    stage0(request i) -> stage1(request i) -> ... -> terminal

Each stage task is atomic from the scheduler's point of view. A task may run
local CE, local backward, and a local optimizer step inside the worker, but no
cross-stage backward task is scheduled.

The runtime semantics intentionally match RunPreparedStagePipelineExperimentMain:

* one bounded input queue per stage;
* one or more workers per stage;
* FIFO within each stage queue;
* sending to a full downstream queue blocks the upstream worker;
* admission into Q0 blocks when Q0 is full.

The simulator is stdlib-only so it can run on a server before the model/runtime
stack is installed.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import random
import statistics
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Iterable, Optional


@dataclass
class StageProfile:
    stage_id: int
    duration_ms: float
    duration_samples_ms: list[float] = field(default_factory=list)
    device_id: str = ""
    output_hidden_bytes: int = 0
    output_shift_log_p_bytes: int = 0
    pss_peak_kb: int = 0
    java_heap_peak_kb: int = 0

    @property
    def output_bytes(self) -> int:
        return self.output_hidden_bytes + self.output_shift_log_p_bytes


@dataclass
class WorkItem:
    seq: int
    request_id: str
    admitted_ms: float
    queue_enter_ms: list[float]
    stage_wait_ms: list[float]
    stage_start_ms: list[float]
    stage_finish_ms: list[float]
    stage_duration_ms: list[float]
    stage_output_block_ms: list[float]
    completed_ms: Optional[float] = None


@dataclass(order=True)
class CompletionEvent:
    finish_ms: float
    stage_sort: int
    serial: int
    stage_id: int = field(compare=False)
    worker_id: int = field(compare=False)
    work: WorkItem = field(compare=False)
    duration_ms: float = field(compare=False)


@dataclass
class BlockedOutput:
    work: WorkItem
    worker_id: int
    blocked_since_ms: float


@dataclass
class SimulationResult:
    policy: str
    requests: int
    stage_count: int
    buffer_capacity: int
    replicas: list[int]
    makespan_ms: float
    throughput_per_s: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    max_latency_ms: float
    stage_exec_busy_ms: list[float]
    stage_output_blocked_ms: list[float]
    stage_exec_utilization: list[float]
    stage_occupied_utilization: list[float]
    max_queue_depth: list[int]
    peak_buffer_bytes: int
    completed: list[WorkItem]


class DurationSampler:
    def __init__(self, profiles: list[StageProfile], mode: str, seed: int) -> None:
        self.profiles = profiles
        self.mode = mode
        self.random = random.Random(seed)

    def duration_for(self, stage_id: int, seq: int) -> float:
        profile = self.profiles[stage_id]
        samples = profile.duration_samples_ms
        if self.mode == "mean" or not samples:
            return profile.duration_ms
        if self.mode == "cycle":
            return samples[seq % len(samples)]
        if self.mode == "sample":
            return self.random.choice(samples)
        raise ValueError(f"Unsupported duration mode: {self.mode}")


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    frac = rank - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def parse_int_list(raw: str, expected_len: int, name: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if len(values) == 1 and expected_len > 1:
        values = values * expected_len
    if len(values) != expected_len:
        raise ValueError(f"{name} must contain 1 or {expected_len} integers, got {raw!r}.")
    if any(value <= 0 for value in values):
        raise ValueError(f"{name} values must be positive.")
    return values


def parse_capacity_list(raw: str, expected_len: int) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if len(values) == 1 and expected_len > 1:
        values = values * expected_len
    if len(values) != expected_len:
        raise ValueError(
            f"buffer capacity must contain 1 or {expected_len} integers, got {raw!r}."
        )
    if any(value < 0 for value in values):
        raise ValueError("buffer capacity values must be non-negative.")
    return values


def load_profile_json(path: Path) -> list[StageProfile]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_stages = payload.get("stages")
    if not isinstance(raw_stages, list) or not raw_stages:
        raise ValueError(f"Profile JSON {path} must contain a non-empty 'stages' list.")

    profiles: list[StageProfile] = []
    for raw_stage in raw_stages:
        samples = [
            float(value)
            for value in raw_stage.get("duration_samples_ms", [])
            if value is not None
        ]
        duration = float(raw_stage.get("duration_ms") or (statistics.mean(samples) if samples else 0.0))
        if duration <= 0:
            raise ValueError(f"Stage profile has non-positive duration: {raw_stage}")
        profiles.append(
            StageProfile(
                stage_id=int(raw_stage.get("stage_id", len(profiles))),
                duration_ms=duration,
                duration_samples_ms=samples,
                device_id=str(raw_stage.get("device_id", "")),
                output_hidden_bytes=int(raw_stage.get("output_hidden_bytes", 0)),
                output_shift_log_p_bytes=int(raw_stage.get("output_shift_log_p_bytes", 0)),
                pss_peak_kb=int(raw_stage.get("pss_peak_kb", 0)),
                java_heap_peak_kb=int(raw_stage.get("java_heap_peak_kb", 0)),
            )
        )

    return sorted(profiles, key=lambda item: item.stage_id)


def as_float(raw: Optional[str], default: float = 0.0) -> float:
    if raw is None or raw == "":
        return default
    return float(raw)


def as_int(raw: Optional[str], default: int = 0) -> int:
    if raw is None or raw == "":
        return default
    return int(float(raw))


def mean_int(values: Iterable[int]) -> int:
    concrete = list(values)
    if not concrete:
        return 0
    return int(round(statistics.mean(concrete)))


def load_profile_from_stage_memory_csv(paths: list[Path], duration_column: str) -> list[StageProfile]:
    durations: dict[int, list[float]] = defaultdict(list)
    hidden_bytes: dict[int, list[int]] = defaultdict(list)
    belief_bytes: dict[int, list[int]] = defaultdict(list)
    pss_kb: dict[int, list[int]] = defaultdict(list)
    java_kb: dict[int, list[int]] = defaultdict(list)
    devices: dict[int, Counter[str]] = defaultdict(Counter)

    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if duration_column not in (reader.fieldnames or []):
                raise ValueError(
                    f"{path} does not contain duration column {duration_column!r}."
                )
            for row in reader:
                stage_id = int(row["stage_id"])
                duration = as_float(row.get(duration_column))
                if duration <= 0:
                    continue
                durations[stage_id].append(duration)
                hidden_bytes[stage_id].append(as_int(row.get("output_hidden_bytes")))
                belief_bytes[stage_id].append(as_int(row.get("output_shift_log_p_bytes")))
                pss_kb[stage_id].append(as_int(row.get("pss_peak_kb")))
                java_kb[stage_id].append(as_int(row.get("java_heap_peak_kb")))
                device_id = row.get("device_id", "")
                if device_id:
                    devices[stage_id][device_id] += 1

    if not durations:
        raise ValueError("No positive stage durations were found in the CSV input.")

    profiles: list[StageProfile] = []
    for stage_id in sorted(durations):
        stage_durations = durations[stage_id]
        device_id = devices[stage_id].most_common(1)[0][0] if devices[stage_id] else ""
        profiles.append(
            StageProfile(
                stage_id=stage_id,
                duration_ms=statistics.mean(stage_durations),
                duration_samples_ms=stage_durations,
                device_id=device_id,
                output_hidden_bytes=mean_int(hidden_bytes[stage_id]),
                output_shift_log_p_bytes=mean_int(belief_bytes[stage_id]),
                pss_peak_kb=max(pss_kb[stage_id]) if pss_kb[stage_id] else 0,
                java_heap_peak_kb=max(java_kb[stage_id]) if java_kb[stage_id] else 0,
            )
        )
    return profiles


def validate_profiles(profiles: list[StageProfile]) -> list[StageProfile]:
    if not profiles:
        raise ValueError("At least one stage profile is required.")
    expected = list(range(len(profiles)))
    actual = [profile.stage_id for profile in profiles]
    if actual != expected:
        raise ValueError(f"Stage IDs must be contiguous from 0. Got {actual}.")
    return profiles


def queue_item_bytes_for_stage(
    stage_id: int,
    profiles: list[StageProfile],
    input_bytes: int,
) -> int:
    if stage_id == 0:
        return input_bytes
    return profiles[stage_id - 1].output_bytes


def new_work_item(seq: int, stage_count: int, admitted_ms: float) -> WorkItem:
    return WorkItem(
        seq=seq,
        request_id=f"sim-{seq:06d}",
        admitted_ms=admitted_ms,
        queue_enter_ms=[math.nan] * stage_count,
        stage_wait_ms=[0.0] * stage_count,
        stage_start_ms=[math.nan] * stage_count,
        stage_finish_ms=[math.nan] * stage_count,
        stage_duration_ms=[0.0] * stage_count,
        stage_output_block_ms=[0.0] * stage_count,
    )


def simulate_serial(
    profiles: list[StageProfile],
    requests: int,
    duration_mode: str,
    seed: int,
) -> SimulationResult:
    stage_count = len(profiles)
    sampler = DurationSampler(profiles, duration_mode, seed)
    completed: list[WorkItem] = []
    stage_busy = [0.0] * stage_count
    now = 0.0
    for seq in range(requests):
        work = new_work_item(seq, stage_count, now)
        for stage_id in range(stage_count):
            duration = sampler.duration_for(stage_id, seq)
            work.queue_enter_ms[stage_id] = now
            work.stage_start_ms[stage_id] = now
            work.stage_duration_ms[stage_id] = duration
            now += duration
            work.stage_finish_ms[stage_id] = now
            stage_busy[stage_id] += duration
        work.completed_ms = now
        completed.append(work)
    return build_result(
        policy="serial",
        requests=requests,
        stage_count=stage_count,
        buffer_capacity=0,
        replicas=[1] * stage_count,
        makespan_ms=now,
        stage_exec_busy_ms=stage_busy,
        stage_output_blocked_ms=[0.0] * stage_count,
        max_queue_depth=[0] * stage_count,
        peak_buffer_bytes=0,
        completed=completed,
    )


def simulate_bounded_fifo(
    profiles: list[StageProfile],
    requests: int,
    buffer_capacities: list[int],
    replicas: list[int],
    duration_mode: str,
    seed: int,
    input_bytes: int,
    admit_delay_ms: float,
) -> SimulationResult:
    stage_count = len(profiles)
    sampler = DurationSampler(profiles, duration_mode, seed)
    queues: list[Deque[WorkItem]] = [deque() for _ in range(stage_count)]
    active_counts = [0] * stage_count
    blocked_outputs: list[Deque[BlockedOutput]] = [deque() for _ in range(stage_count)]
    active_events: list[CompletionEvent] = []
    completed: list[WorkItem] = []
    stage_exec_busy = [0.0] * stage_count
    stage_output_blocked = [0.0] * stage_count
    max_queue_depth = [0] * stage_count
    peak_buffer_bytes = 0
    serial = 0
    next_seq = 0
    next_admit_ms = 0.0
    now = 0.0

    queue_item_bytes = [
        queue_item_bytes_for_stage(stage_id, profiles, input_bytes)
        for stage_id in range(stage_count)
    ]

    def current_buffer_bytes() -> int:
        queued = sum(len(queues[idx]) * queue_item_bytes[idx] for idx in range(stage_count))
        blocked = 0
        for stage_id in range(stage_count - 1):
            blocked += len(blocked_outputs[stage_id]) * queue_item_bytes[stage_id + 1]
        return queued + blocked

    def note_buffers() -> None:
        nonlocal peak_buffer_bytes
        for idx, queue in enumerate(queues):
            max_queue_depth[idx] = max(max_queue_depth[idx], len(queue))
        peak_buffer_bytes = max(peak_buffer_bytes, current_buffer_bytes())

    def has_queue_credit(stage_id: int) -> bool:
        return len(queues[stage_id]) < buffer_capacities[stage_id]

    def enqueue(stage_id: int, work: WorkItem, enqueue_ms: float) -> None:
        work.queue_enter_ms[stage_id] = enqueue_ms
        queues[stage_id].append(work)
        note_buffers()

    def free_worker_count(stage_id: int) -> int:
        return replicas[stage_id] - active_counts[stage_id] - len(blocked_outputs[stage_id])

    def unblock_outputs() -> bool:
        progressed = False
        for stage_id in range(stage_count - 2, -1, -1):
            next_stage = stage_id + 1
            while blocked_outputs[stage_id] and has_queue_credit(next_stage):
                blocked = blocked_outputs[stage_id].popleft()
                blocked_ms = now - blocked.blocked_since_ms
                blocked.work.stage_output_block_ms[stage_id] += blocked_ms
                stage_output_blocked[stage_id] += blocked_ms
                enqueue(next_stage, blocked.work, now)
                progressed = True
        return progressed

    def admit_ready() -> bool:
        nonlocal next_seq, next_admit_ms
        progressed = False
        while (
            next_seq < requests
            and now + 1e-9 >= next_admit_ms
            and has_queue_credit(0)
        ):
            work = new_work_item(next_seq, stage_count, now)
            enqueue(0, work, now)
            next_seq += 1
            next_admit_ms = now + admit_delay_ms
            progressed = True
            if admit_delay_ms > 0:
                break
        return progressed

    def dispatch_ready() -> bool:
        nonlocal serial
        progressed = False
        for stage_id in range(stage_count - 1, -1, -1):
            while free_worker_count(stage_id) > 0 and queues[stage_id]:
                worker_id = replicas[stage_id] - free_worker_count(stage_id)
                work = queues[stage_id].popleft()
                note_buffers()
                wait_ms = now - work.queue_enter_ms[stage_id]
                duration = sampler.duration_for(stage_id, work.seq)
                finish_ms = now + duration
                work.stage_wait_ms[stage_id] += wait_ms
                work.stage_start_ms[stage_id] = now
                work.stage_duration_ms[stage_id] = duration
                active_counts[stage_id] += 1
                stage_exec_busy[stage_id] += duration
                heapq.heappush(
                    active_events,
                    CompletionEvent(
                        finish_ms=finish_ms,
                        stage_sort=-stage_id,
                        serial=serial,
                        stage_id=stage_id,
                        worker_id=worker_id,
                        work=work,
                        duration_ms=duration,
                    ),
                )
                serial += 1
                progressed = True
        return progressed

    def process_completions() -> None:
        nonlocal now
        if not active_events:
            return
        now = active_events[0].finish_ms
        ready: list[CompletionEvent] = []
        while active_events and abs(active_events[0].finish_ms - now) < 1e-9:
            ready.append(heapq.heappop(active_events))
        ready.sort(key=lambda event: (-event.stage_id, event.serial))
        for event in ready:
            stage_id = event.stage_id
            active_counts[stage_id] -= 1
            event.work.stage_finish_ms[stage_id] = now
            if stage_id == stage_count - 1:
                event.work.completed_ms = now
                completed.append(event.work)
                continue
            next_stage = stage_id + 1
            if has_queue_credit(next_stage):
                enqueue(next_stage, event.work, now)
            else:
                blocked_outputs[stage_id].append(
                    BlockedOutput(
                        work=event.work,
                        worker_id=event.worker_id,
                        blocked_since_ms=now,
                    )
                )
                note_buffers()

    note_buffers()
    while len(completed) < requests:
        progressed = True
        while progressed:
            progressed = False
            if unblock_outputs():
                progressed = True
            if admit_ready():
                progressed = True
            if dispatch_ready():
                progressed = True

        if len(completed) >= requests:
            break

        if active_events:
            process_completions()
            continue

        if next_seq < requests and has_queue_credit(0) and next_admit_ms > now:
            now = next_admit_ms
            continue

        raise RuntimeError(
            "Simulation deadlocked. Check buffer capacities and replica settings."
        )

    makespan = max((work.completed_ms or 0.0) for work in completed) if completed else 0.0
    return build_result(
        policy="bounded_fifo",
        requests=requests,
        stage_count=stage_count,
        buffer_capacity=max(buffer_capacities) if buffer_capacities else 0,
        replicas=replicas,
        makespan_ms=makespan,
        stage_exec_busy_ms=stage_exec_busy,
        stage_output_blocked_ms=stage_output_blocked,
        max_queue_depth=max_queue_depth,
        peak_buffer_bytes=peak_buffer_bytes,
        completed=completed,
    )


def build_result(
    policy: str,
    requests: int,
    stage_count: int,
    buffer_capacity: int,
    replicas: list[int],
    makespan_ms: float,
    stage_exec_busy_ms: list[float],
    stage_output_blocked_ms: list[float],
    max_queue_depth: list[int],
    peak_buffer_bytes: int,
    completed: list[WorkItem],
) -> SimulationResult:
    latencies = [
        (work.completed_ms or work.admitted_ms) - work.admitted_ms
        for work in completed
    ]
    throughput = 0.0 if makespan_ms <= 0 else requests / (makespan_ms / 1000.0)
    stage_exec_util = []
    stage_occupied_util = []
    for stage_id in range(stage_count):
        denom = makespan_ms * replicas[stage_id] if makespan_ms > 0 else 0.0
        exec_busy = stage_exec_busy_ms[stage_id]
        blocked = stage_output_blocked_ms[stage_id]
        stage_exec_util.append(0.0 if denom <= 0 else exec_busy / denom)
        stage_occupied_util.append(0.0 if denom <= 0 else (exec_busy + blocked) / denom)

    return SimulationResult(
        policy=policy,
        requests=requests,
        stage_count=stage_count,
        buffer_capacity=buffer_capacity,
        replicas=replicas,
        makespan_ms=makespan_ms,
        throughput_per_s=throughput,
        avg_latency_ms=statistics.mean(latencies) if latencies else 0.0,
        p50_latency_ms=percentile(latencies, 0.50),
        p95_latency_ms=percentile(latencies, 0.95),
        max_latency_ms=max(latencies) if latencies else 0.0,
        stage_exec_busy_ms=stage_exec_busy_ms,
        stage_output_blocked_ms=stage_output_blocked_ms,
        stage_exec_utilization=stage_exec_util,
        stage_occupied_utilization=stage_occupied_util,
        max_queue_depth=max_queue_depth,
        peak_buffer_bytes=peak_buffer_bytes,
        completed=completed,
    )


def write_request_csv(path: Path, result: SimulationResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stage_count = result.stage_count
    headers = [
        "seq",
        "request_id",
        "admitted_ms",
        "completed_ms",
        "latency_ms",
    ]
    for stage_id in range(stage_count):
        headers.extend(
            [
                f"stage{stage_id}_wait_ms",
                f"stage{stage_id}_start_ms",
                f"stage{stage_id}_finish_ms",
                f"stage{stage_id}_duration_ms",
                f"stage{stage_id}_output_block_ms",
            ]
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for work in sorted(result.completed, key=lambda item: item.seq):
            completed_ms = work.completed_ms or work.admitted_ms
            row: list[Any] = [
                work.seq,
                work.request_id,
                round(work.admitted_ms, 6),
                round(completed_ms, 6),
                round(completed_ms - work.admitted_ms, 6),
            ]
            for stage_id in range(stage_count):
                row.extend(
                    [
                        round(work.stage_wait_ms[stage_id], 6),
                        round(work.stage_start_ms[stage_id], 6),
                        round(work.stage_finish_ms[stage_id], 6),
                        round(work.stage_duration_ms[stage_id], 6),
                        round(work.stage_output_block_ms[stage_id], 6),
                    ]
                )
            writer.writerow(row)


def result_to_summary_dict(result: SimulationResult, profiles: list[StageProfile]) -> dict[str, Any]:
    return {
        "policy": result.policy,
        "requests": result.requests,
        "stage_count": result.stage_count,
        "buffer_capacity": result.buffer_capacity,
        "replicas": result.replicas,
        "makespan_ms": result.makespan_ms,
        "throughput_per_s": result.throughput_per_s,
        "avg_latency_ms": result.avg_latency_ms,
        "p50_latency_ms": result.p50_latency_ms,
        "p95_latency_ms": result.p95_latency_ms,
        "max_latency_ms": result.max_latency_ms,
        "peak_buffer_bytes": result.peak_buffer_bytes,
        "stages": [
            {
                "stage_id": profile.stage_id,
                "device_id": profile.device_id,
                "duration_ms": profile.duration_ms,
                "replicas": result.replicas[profile.stage_id],
                "exec_busy_ms": result.stage_exec_busy_ms[profile.stage_id],
                "output_blocked_ms": result.stage_output_blocked_ms[profile.stage_id],
                "exec_utilization": result.stage_exec_utilization[profile.stage_id],
                "occupied_utilization": result.stage_occupied_utilization[profile.stage_id],
                "max_queue_depth": result.max_queue_depth[profile.stage_id],
                "output_hidden_bytes": profile.output_hidden_bytes,
                "output_shift_log_p_bytes": profile.output_shift_log_p_bytes,
                "pss_peak_kb": profile.pss_peak_kb,
                "java_heap_peak_kb": profile.java_heap_peak_kb,
            }
            for profile in profiles
        ],
    }


def write_summary_json(path: Path, result: SimulationResult, profiles: list[StageProfile]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = result_to_summary_dict(result, profiles)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def print_profile(profiles: list[StageProfile]) -> None:
    print("profile:")
    for profile in profiles:
        sample_note = ""
        if profile.duration_samples_ms:
            sample_note = (
                f" samples={len(profile.duration_samples_ms)}"
                f" p50={percentile(profile.duration_samples_ms, 0.50):.1f}"
                f" p95={percentile(profile.duration_samples_ms, 0.95):.1f}"
            )
        print(
            "  "
            f"stage={profile.stage_id} device={profile.device_id or '-'} "
            f"duration_ms={profile.duration_ms:.1f}{sample_note} "
            f"output_bytes={profile.output_bytes} "
            f"pss_peak_kb={profile.pss_peak_kb}"
        )


def print_result(result: SimulationResult) -> None:
    print("simulation:")
    print(f"  policy={result.policy}")
    print(f"  requests={result.requests}")
    print(f"  stage_count={result.stage_count}")
    print(f"  replicas={','.join(str(value) for value in result.replicas)}")
    print(f"  makespan_ms={result.makespan_ms:.3f}")
    print(f"  throughput_per_s={result.throughput_per_s:.6f}")
    print(f"  avg_latency_ms={result.avg_latency_ms:.3f}")
    print(f"  p50_latency_ms={result.p50_latency_ms:.3f}")
    print(f"  p95_latency_ms={result.p95_latency_ms:.3f}")
    print(f"  max_latency_ms={result.max_latency_ms:.3f}")
    print(f"  peak_buffer_bytes={result.peak_buffer_bytes}")
    print("  stages:")
    for stage_id in range(result.stage_count):
        print(
            "    "
            f"stage={stage_id} "
            f"exec_util={result.stage_exec_utilization[stage_id]:.4f} "
            f"occupied_util={result.stage_occupied_utilization[stage_id]:.4f} "
            f"exec_busy_ms={result.stage_exec_busy_ms[stage_id]:.1f} "
            f"output_blocked_ms={result.stage_output_blocked_ms[stage_id]:.1f} "
            f"max_queue_depth={result.max_queue_depth[stage_id]}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate the SID forward-only bounded FIFO stage pipeline."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--profile-json",
        type=Path,
        help="Stage profile JSON with a 'stages' list.",
    )
    source.add_argument(
        "--stage-memory-csv",
        type=Path,
        action="append",
        help="results.stage_memory.csv from RunPreparedStagePipelineExperimentMain. Can be repeated.",
    )
    parser.add_argument("--requests", type=int, default=512)
    parser.add_argument(
        "--policy",
        choices=["bounded_fifo", "serial"],
        default="bounded_fifo",
    )
    parser.add_argument(
        "--buffer",
        default="3",
        help="Per-stage queue capacity. Use one value or comma-separated values.",
    )
    parser.add_argument(
        "--replicas",
        default="1",
        help="Per-stage worker replicas. Use one value or comma-separated values.",
    )
    parser.add_argument(
        "--duration-column",
        default="stage_total_ms",
        help="CSV duration column to use when --stage-memory-csv is provided.",
    )
    parser.add_argument(
        "--duration-mode",
        choices=["mean", "cycle", "sample"],
        default="mean",
        help="Use mean duration, cycle through empirical rows, or sample rows.",
    )
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument(
        "--input-bytes",
        type=int,
        default=0,
        help="Bytes held by each Q0 input item. Defaults to stage0 output bytes.",
    )
    parser.add_argument(
        "--admit-delay-ms",
        type=float,
        default=0.0,
        help="Optional delay after each Q0 admission, matching the runner delay argument.",
    )
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.requests <= 0:
        raise ValueError("--requests must be positive.")
    if args.profile_json:
        profiles = validate_profiles(load_profile_json(args.profile_json))
    else:
        profiles = validate_profiles(
            load_profile_from_stage_memory_csv(args.stage_memory_csv, args.duration_column)
        )

    stage_count = len(profiles)
    replicas = parse_int_list(args.replicas, stage_count, "replicas")
    buffer_capacities = parse_capacity_list(args.buffer, stage_count)
    input_bytes = args.input_bytes if args.input_bytes > 0 else max(profiles[0].output_bytes, 1)

    if args.policy == "serial":
        result = simulate_serial(
            profiles=profiles,
            requests=args.requests,
            duration_mode=args.duration_mode,
            seed=args.seed,
        )
    else:
        result = simulate_bounded_fifo(
            profiles=profiles,
            requests=args.requests,
            buffer_capacities=buffer_capacities,
            replicas=replicas,
            duration_mode=args.duration_mode,
            seed=args.seed,
            input_bytes=input_bytes,
            admit_delay_ms=args.admit_delay_ms,
        )

    print_profile(profiles)
    print_result(result)

    if args.output_csv:
        write_request_csv(args.output_csv, result)
        print(f"wrote_request_csv={args.output_csv}")
    if args.output_json:
        write_summary_json(args.output_json, result, profiles)
        print(f"wrote_summary_json={args.output_json}")


if __name__ == "__main__":
    main()
