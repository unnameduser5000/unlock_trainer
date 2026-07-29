from __future__ import annotations

import json
import os
import socket
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from .durable_io import atomic_write_json


class RecoveryEventName(str, Enum):
    OUTAGE_INJECTED = "outage_injected"
    OUTAGE_DETECTED = "outage_detected"
    PREFIX_WINDOW_COMMIT = "prefix_window_commit"
    OUTBOX_BACKPRESSURE = "outbox_backpressure"
    STAGE_REJOINED = "stage_rejoined"
    CATCHUP_STAGE1_START = "catchup_stage1_start"
    CATCHUP_STAGE1_DONE = "catchup_stage1_done"
    CATCHUP_STAGE2_START = "catchup_stage2_start"
    CATCHUP_STAGE2_DONE = "catchup_stage2_done"
    TERMINAL_TARGET_REACHED = "terminal_target_reached"
    LIVE_P2P_RESUMED = "live_p2p_resumed"


@dataclass(frozen=True)
class RecoveryEvent:
    schema_version: int
    run_id: str
    rank: int
    stage_id: int
    sequence: int
    name: str
    window_id: int
    microbatch_id: int
    monotonic_ns: int
    epoch_ns: int
    hostname: str
    process_id: int
    details: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RecoveryEvent":
        return cls(**{key: payload[key] for key in cls.__dataclass_fields__})


class RecoveryEventRecorder:
    SCHEMA_VERSION = 1

    def __init__(self, root: Path, *, run_id: str, rank: int, stage_id: int) -> None:
        if rank < 0 or stage_id < 0:
            raise ValueError("rank and stage_id must be non-negative")
        self.run_id = str(run_id)
        self.rank = rank
        self.stage_id = stage_id
        self.directory = Path(root) / self.run_id / "events" / f"rank-{rank}"
        self.directory.mkdir(parents=True, exist_ok=True)
        existing = [int(path.stem.split("-")[-1]) for path in self.directory.glob("event-*.json")]
        self.next_sequence = max(existing, default=-1) + 1

    def record(
        self,
        name: RecoveryEventName,
        *,
        window_id: int = -1,
        microbatch_id: int = -1,
        details: Optional[dict[str, Any]] = None,
    ) -> RecoveryEvent:
        sequence = self.next_sequence
        event = RecoveryEvent(
            schema_version=self.SCHEMA_VERSION,
            run_id=self.run_id,
            rank=self.rank,
            stage_id=self.stage_id,
            sequence=sequence,
            name=name.value,
            window_id=int(window_id),
            microbatch_id=int(microbatch_id),
            monotonic_ns=time.monotonic_ns(),
            epoch_ns=time.time_ns(),
            hostname=socket.gethostname(),
            process_id=os.getpid(),
            details=dict(details or {}),
        )
        path = self.directory / f"event-{sequence:08d}.json"
        atomic_write_json(path, asdict(event))
        self.next_sequence += 1
        return event


class RecoveryTimeline:
    def __init__(self, events: list[RecoveryEvent]) -> None:
        self.events = sorted(events, key=lambda event: (event.monotonic_ns, event.rank, event.sequence))
        hostnames = {event.hostname for event in self.events}
        if len(hostnames) > 1:
            raise ValueError("cross-rank monotonic timing is only valid on one host")

    @classmethod
    def load(cls, root: Path, run_id: str) -> "RecoveryTimeline":
        directory = Path(root) / run_id / "events"
        events = [
            RecoveryEvent.from_dict(json.loads(path.read_text(encoding="utf-8")))
            for path in directory.glob("rank-*/event-*.json")
        ]
        if not events:
            raise FileNotFoundError(f"no recovery events under {directory}")
        return cls(events)

    def select(self, name: RecoveryEventName, *, rank: Optional[int] = None) -> list[RecoveryEvent]:
        return [
            event
            for event in self.events
            if event.name == name.value and (rank is None or event.rank == rank)
        ]

    def unique(self, name: RecoveryEventName, *, rank: Optional[int] = None) -> RecoveryEvent:
        matches = self.select(name, rank=rank)
        if len(matches) != 1:
            raise ValueError(f"expected one {name.value} event, found {len(matches)}")
        return matches[0]

    def duration_ms(
        self,
        start: RecoveryEventName,
        end: RecoveryEventName,
        *,
        start_rank: Optional[int] = None,
        end_rank: Optional[int] = None,
    ) -> float:
        start_event = self.unique(start, rank=start_rank)
        end_event = self.unique(end, rank=end_rank)
        if end_event.monotonic_ns < start_event.monotonic_ns:
            raise ValueError(f"event {end.value} precedes {start.value}")
        return (end_event.monotonic_ns - start_event.monotonic_ns) / 1_000_000.0

    def recovery_summary(self) -> dict[str, float]:
        return {
            "outage_duration_ms": self.duration_ms(
                RecoveryEventName.OUTAGE_INJECTED,
                RecoveryEventName.STAGE_REJOINED,
            ),
            "stage1_catchup_ms": self.duration_ms(
                RecoveryEventName.STAGE_REJOINED,
                RecoveryEventName.CATCHUP_STAGE1_DONE,
            ),
            "all_stage_catchup_ms": self.duration_ms(
                RecoveryEventName.STAGE_REJOINED,
                RecoveryEventName.CATCHUP_STAGE2_DONE,
            ),
            "live_resume_ms": self.duration_ms(
                RecoveryEventName.STAGE_REJOINED,
                RecoveryEventName.LIVE_P2P_RESUMED,
            ),
        }

    def common_recovery_summary(self) -> dict[str, float]:
        """Method-neutral intervals used by the BP-free/Exact-BP comparison."""
        return {
            "rejoin_to_terminal_target_ms": self.duration_ms(
                RecoveryEventName.STAGE_REJOINED,
                RecoveryEventName.TERMINAL_TARGET_REACHED,
            ),
            "rejoin_to_live_resume_ms": self.duration_ms(
                RecoveryEventName.STAGE_REJOINED,
                RecoveryEventName.LIVE_P2P_RESUMED,
            ),
        }
