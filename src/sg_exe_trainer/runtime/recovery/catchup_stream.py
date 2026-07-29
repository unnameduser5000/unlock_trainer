from __future__ import annotations

import time
from dataclasses import dataclass

from .state_contract import StageCommit, StageCommitLedger


class CommitWaitTimeout(TimeoutError):
    """The streamed consumer did not observe its producer's durable commit."""


@dataclass(frozen=True)
class CommitWaitResult:
    commit: StageCommit
    wait_ms: float
    polls: int


def wait_for_stage_commit(
    *,
    ledger: StageCommitLedger,
    stage_id: int,
    window_id: int,
    timeout_s: float,
    poll_ms: float,
) -> CommitWaitResult:
    if min(stage_id, window_id) < 0:
        raise ValueError("stage_id and window_id must be non-negative")
    if timeout_s <= 0 or poll_ms <= 0:
        raise ValueError("timeout_s and poll_ms must be positive")

    started = time.monotonic_ns()
    deadline = started + int(timeout_s * 1_000_000_000)
    polls = 0
    while True:
        polls += 1
        try:
            commit = ledger.get(stage_id, window_id)
            return CommitWaitResult(
                commit=commit,
                wait_ms=(time.monotonic_ns() - started) / 1_000_000.0,
                polls=polls,
            )
        except FileNotFoundError:
            if time.monotonic_ns() >= deadline:
                raise CommitWaitTimeout(
                    f"timed out waiting for stage {stage_id} window {window_id} commit"
                )
            time.sleep(poll_ms / 1000.0)
