from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .protocol import E5OutageProtocol


@dataclass(frozen=True)
class ExactBPBacklogContract:
    """Exact-BP behavior under the shared stage-1 service outage."""

    protocol: E5OutageProtocol

    def __post_init__(self) -> None:
        if self.protocol.microbatches_per_window < self.protocol.num_stages:
            raise ValueError(
                "Schedule1F1B requires microbatches_per_window >= num_stages"
            )

    def execution_plan(self) -> dict[str, list[str]]:
        stages = self.protocol.num_stages
        return {
            "prelude": ["live_1f1b"] * stages,
            "outage": ["queue_raw_requests_no_commit"] * stages,
            "catchup": ["full_window_1f1b_global_commit"] * stages,
            "resumed": ["live_1f1b"] * stages,
        }

    def expected_invariants(self) -> dict[str, int | bool]:
        return {
            "stage_commits_at_rejoin": 0,
            "queued_windows_at_rejoin": self.protocol.outage_windows,
            "queued_records_at_rejoin": (
                self.protocol.outage_windows * self.protocol.effective_batch_size
            ),
            "full_windows_replayed": self.protocol.outage_windows,
            "all_stages_commit_every_replayed_window": True,
            "partial_stage_commit_allowed": False,
        }

    def validate(
        self,
        *,
        commits_at_rejoin: Mapping[int, int],
        final_commit_counts: Mapping[int, int],
    ) -> None:
        expected_stages = set(range(self.protocol.num_stages))
        if set(commits_at_rejoin) != expected_stages:
            raise RuntimeError("commits_at_rejoin does not cover every stage")
        if set(final_commit_counts) != expected_stages:
            raise RuntimeError("final_commit_counts does not cover every stage")
        if any(int(value) != 0 for value in commits_at_rejoin.values()):
            raise RuntimeError("Exact BP cannot commit an outage window before stage 1 rejoins")
        if any(
            int(value) != self.protocol.outage_windows
            for value in final_commit_counts.values()
        ):
            raise RuntimeError("every Exact-BP stage must commit every replayed global window")
