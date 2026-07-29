from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from sg_exe_trainer.runtime.recovery.state_contract import (
    BoundaryConflictError,
    BoundaryKey,
    CommitConflictError,
    DurableBoundaryOutbox,
    OutboxCapacityError,
    StageCommit,
    StageCommitLedger,
)


class DurableBoundaryOutboxTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.outbox = DurableBoundaryOutbox(self.root, "run-a", max_pending_windows=2)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    @staticmethod
    def key(window_id: int, microbatch_id: int = 0) -> BoundaryKey:
        return BoundaryKey("run-a", 0, 1, window_id, microbatch_id)

    def test_round_trip_and_ack_preserve_versioned_tensor(self) -> None:
        hidden = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
        metadata = self.outbox.put(
            self.key(7),
            hidden=hidden,
            request_ids=["r-7"],
            producer_version=11,
        )

        loaded = self.outbox.load(self.key(7))
        self.assertTrue(torch.equal(loaded.hidden, hidden))
        self.assertEqual(loaded.metadata, metadata)
        self.assertEqual(self.outbox.pending_window_ids(0, 1), [7])

        self.outbox.acknowledge(self.key(7), consumer_version=4)
        self.assertEqual(self.outbox.pending_window_ids(0, 1), [])

    def test_identical_put_is_idempotent_but_different_payload_conflicts(self) -> None:
        hidden = torch.ones((1, 2, 3))
        first = self.outbox.put(
            self.key(1),
            hidden=hidden,
            request_ids=["r-1"],
            producer_version=2,
        )
        second = self.outbox.put(
            self.key(1),
            hidden=hidden,
            request_ids=["r-1"],
            producer_version=2,
        )
        self.assertEqual(first, second)

        with self.assertRaises(BoundaryConflictError):
            self.outbox.put(
                self.key(1),
                hidden=torch.zeros_like(hidden),
                request_ids=["r-1"],
                producer_version=2,
            )

    def test_pending_window_capacity_applies_backpressure(self) -> None:
        for window_id in (1, 2):
            self.outbox.put(
                self.key(window_id),
                hidden=torch.tensor([float(window_id)]),
                request_ids=[f"r-{window_id}"],
                producer_version=window_id,
            )

        with self.assertRaises(OutboxCapacityError):
            self.outbox.put(
                self.key(3),
                hidden=torch.tensor([3.0]),
                request_ids=["r-3"],
                producer_version=3,
            )

        self.outbox.acknowledge(self.key(1), consumer_version=1)
        self.outbox.put(
            self.key(3),
            hidden=torch.tensor([3.0]),
            request_ids=["r-3"],
            producer_version=3,
        )
        self.assertEqual(self.outbox.pending_window_ids(0, 1), [2, 3])


class StageCommitLedgerTest(unittest.TestCase):
    def test_commit_is_idempotent_and_rejects_double_update(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            ledger = StageCommitLedger(Path(temp_dir), "run-a")
            commit = StageCommit(
                run_id="run-a",
                stage_id=0,
                window_id=5,
                optimizer_step=6,
                request_ids=("r-40", "r-41"),
                input_producer_versions=(),
                checkpoint_id="stage0-step6",
            )
            first = ledger.record(commit)
            second = ledger.record(commit)
            self.assertEqual(first, second)

            conflicting = StageCommit(
                run_id="run-a",
                stage_id=0,
                window_id=5,
                optimizer_step=7,
                request_ids=("r-40", "r-41"),
                input_producer_versions=(),
                checkpoint_id="stage0-step7",
            )
            with self.assertRaises(CommitConflictError):
                ledger.record(conflicting)


if __name__ == "__main__":
    unittest.main()
