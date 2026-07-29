from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from sg_exe_trainer.runtime.recovery.state_contract import (
    BoundaryKey,
    DurableBoundaryOutbox,
    OutboxCapacityError,
    StageCommitLedger,
)
from sg_exe_trainer.runtime.recovery.window_journal import (
    BPFreeWindowJournal,
    CommittedBoundaryReader,
    IncompleteWindowError,
)


class BPFreeWindowJournalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.outbox = DurableBoundaryOutbox(root, "run-a", max_pending_windows=2)
        self.ledger = StageCommitLedger(root, "run-a")
        self.journal = BPFreeWindowJournal(
            run_id="run-a",
            stage_id=0,
            world_size=3,
            outbox=self.outbox,
            ledger=self.ledger,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_commit_publishes_complete_window_and_reader_checks_ledger(self) -> None:
        self.journal.begin(window_id=3, microbatch_ids=[0, 1], producer_version=3)
        for mb_id in (0, 1):
            self.journal.capture_output(
                microbatch_id=mb_id,
                request_ids=[f"r-{mb_id}"],
                hidden=torch.full((1, 2), float(mb_id)),
            )

        result = self.journal.commit_after_optimizer(
            optimizer_step=4,
            request_ids=["r-0", "r-1"],
            checkpoint_id="stage0-step4",
        )
        self.assertEqual(len(result.boundaries), 2)
        self.assertEqual(result.commit.optimizer_step, 4)

        reader = CommittedBoundaryReader(outbox=self.outbox, ledger=self.ledger)
        loaded = reader.load(BoundaryKey("run-a", 0, 1, 3, 1))
        self.assertTrue(torch.equal(loaded.hidden, torch.full((1, 2), 1.0)))

    def test_incomplete_window_does_not_publish_commit_marker(self) -> None:
        self.journal.begin(window_id=3, microbatch_ids=[0, 1], producer_version=3)
        self.journal.capture_output(
            microbatch_id=0,
            request_ids=["r-0"],
            hidden=torch.zeros((1, 2)),
        )
        with self.assertRaises(IncompleteWindowError):
            self.journal.commit_after_optimizer(
                optimizer_step=4,
                request_ids=["r-0", "r-1"],
                checkpoint_id="stage0-step4",
            )
        with self.assertRaises(FileNotFoundError):
            self.ledger.get(0, 3)

    def test_capacity_is_checked_before_window_opens(self) -> None:
        for window_id in (0, 1):
            key = BoundaryKey("run-a", 0, 1, window_id, 0)
            self.outbox.put(
                key,
                hidden=torch.tensor([float(window_id)]),
                request_ids=[f"r-{window_id}"],
                producer_version=window_id,
            )

        with self.assertRaises(OutboxCapacityError):
            self.journal.begin(window_id=2, microbatch_ids=[0], producer_version=2)
        self.assertFalse(self.journal.has_open_window)

    def test_terminal_stage_commits_without_output_boundary(self) -> None:
        terminal = BPFreeWindowJournal(
            run_id="run-a",
            stage_id=2,
            world_size=3,
            outbox=self.outbox,
            ledger=self.ledger,
        )
        terminal.begin(
            window_id=0,
            microbatch_ids=[0],
            producer_version=0,
            input_producer_versions=[0],
        )
        result = terminal.commit_after_optimizer(
            optimizer_step=1,
            request_ids=["r-0"],
            checkpoint_id="stage2-step1",
        )
        self.assertEqual(result.boundaries, ())
        self.assertEqual(self.ledger.get(2, 0).input_producer_versions, (0,))


if __name__ == "__main__":
    unittest.main()
