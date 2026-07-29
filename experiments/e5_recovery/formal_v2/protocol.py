from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

from sg_exe_trainer.runtime.recovery.durable_io import atomic_write_json


class StageAction(str, Enum):
    LIVE_P2P = "live_p2p"
    PREFIX_LOCAL_TO_OUTBOX = "prefix_local_to_outbox"
    FAILED_STAGE_OFFLINE = "failed_stage_offline"
    SUFFIX_IDLE = "suffix_idle"
    PRESERVED_PREFIX_IDLE = "preserved_prefix_idle"
    CATCHUP_FROM_OUTBOX = "catchup_from_outbox"
    CATCHUP_STREAM_PRODUCER = "catchup_stream_producer"
    CATCHUP_STREAM_CONSUMER = "catchup_stream_consumer"


@dataclass(frozen=True)
class E5OutageProtocol:
    run_id: str
    num_stages: int = 3
    failure_stage: int = 1
    prelude_windows: int = 4
    outage_windows: int = 4
    resumed_windows: int = 2
    physical_batch_size: int = 1
    microbatches_per_window: int = 8
    max_pending_windows: int = 4
    belief_transport_mode: str = "terminal"
    catchup_policy: str = "drain_first"
    failure_model: str = "controlled_stage_service_outage"

    def __post_init__(self) -> None:
        if self.num_stages != 3:
            raise ValueError("formal v2 first implementation is fixed to three stages")
        if self.failure_stage != 1:
            raise ValueError("formal v2 first implementation injects the outage at stage 1")
        if min(self.prelude_windows, self.outage_windows, self.resumed_windows) <= 0:
            raise ValueError("all phase window counts must be positive")
        if min(self.physical_batch_size, self.microbatches_per_window) <= 0:
            raise ValueError("batch dimensions must be positive")
        if self.max_pending_windows < self.outage_windows:
            raise ValueError(
                "max_pending_windows must cover outage_windows for the no-backpressure main point"
            )
        if self.belief_transport_mode != "terminal":
            raise ValueError("formal v2 requires terminal belief mode to keep stages locally independent")
        if self.catchup_policy not in {"drain_first", "window_streamed"}:
            raise ValueError(f"unsupported catch-up policy: {self.catchup_policy}")
        if self.failure_model != "controlled_stage_service_outage":
            raise ValueError("unsupported failure model")

    @property
    def effective_batch_size(self) -> int:
        return self.physical_batch_size * self.microbatches_per_window

    @property
    def outage_start_window(self) -> int:
        return self.prelude_windows

    @property
    def outage_end_window_exclusive(self) -> int:
        return self.prelude_windows + self.outage_windows

    @property
    def total_logical_windows(self) -> int:
        return self.prelude_windows + self.outage_windows + self.resumed_windows

    def phase_actions(self) -> dict[str, list[str]]:
        common = {
            "prelude": [StageAction.LIVE_P2P.value] * self.num_stages,
            "outage": [
                StageAction.PREFIX_LOCAL_TO_OUTBOX.value,
                StageAction.FAILED_STAGE_OFFLINE.value,
                StageAction.SUFFIX_IDLE.value,
            ],
            "resumed": [StageAction.LIVE_P2P.value] * self.num_stages,
        }
        if self.catchup_policy == "window_streamed":
            common["catchup_streamed"] = [
                StageAction.PRESERVED_PREFIX_IDLE.value,
                StageAction.CATCHUP_STREAM_PRODUCER.value,
                StageAction.CATCHUP_STREAM_CONSUMER.value,
            ]
            return common
        common.update({
            "catchup_stage1": [
                StageAction.PRESERVED_PREFIX_IDLE.value,
                StageAction.CATCHUP_FROM_OUTBOX.value,
                StageAction.SUFFIX_IDLE.value,
            ],
            "catchup_stage2": [
                StageAction.PRESERVED_PREFIX_IDLE.value,
                StageAction.PRESERVED_PREFIX_IDLE.value,
                StageAction.CATCHUP_FROM_OUTBOX.value,
            ],
        })
        return common

    def expected_invariants(self) -> dict[str, int | bool]:
        return {
            "prefix_stage0_commits_at_rejoin": self.outage_windows,
            "stage1_commits_at_rejoin": 0,
            "terminal_stage2_commits_at_rejoin": 0,
            "outage_boundary_windows": self.outage_windows,
            "all_stages_caught_up_before_resume": True,
            "rollback_of_stage0_commits": 0,
        }

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload.update(
            {
                "effective_batch_size": self.effective_batch_size,
                "outage_start_window": self.outage_start_window,
                "outage_end_window_exclusive": self.outage_end_window_exclusive,
                "total_logical_windows": self.total_logical_windows,
                "phase_actions": self.phase_actions(),
                "expected_invariants": self.expected_invariants(),
            }
        )
        return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Emit the E5 formal-v2 outage protocol")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prelude-windows", type=int, default=4)
    parser.add_argument("--outage-windows", type=int, default=4)
    parser.add_argument("--resumed-windows", type=int, default=2)
    parser.add_argument("--physical-batch-size", type=int, default=1)
    parser.add_argument("--microbatches-per-window", type=int, default=8)
    parser.add_argument("--max-pending-windows", type=int, default=4)
    parser.add_argument(
        "--catchup-policy",
        choices=("drain_first", "window_streamed"),
        default="drain_first",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    protocol = E5OutageProtocol(
        run_id=args.run_id,
        prelude_windows=args.prelude_windows,
        outage_windows=args.outage_windows,
        resumed_windows=args.resumed_windows,
        physical_batch_size=args.physical_batch_size,
        microbatches_per_window=args.microbatches_per_window,
        max_pending_windows=args.max_pending_windows,
        catchup_policy=args.catchup_policy,
    )
    payload = protocol.to_dict()
    if args.output is not None:
        atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
