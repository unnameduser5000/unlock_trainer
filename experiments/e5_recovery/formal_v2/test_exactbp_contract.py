from __future__ import annotations

import unittest

from experiments.e5_recovery.formal_v2.exactbp_contract import ExactBPBacklogContract
from experiments.e5_recovery.formal_v2.protocol import E5OutageProtocol


class ExactBPBacklogContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = ExactBPBacklogContract(
            E5OutageProtocol(
                run_id="exact-a",
                prelude_windows=2,
                outage_windows=4,
                resumed_windows=2,
                physical_batch_size=1,
                microbatches_per_window=8,
                max_pending_windows=4,
            )
        )

    def test_invariants_describe_global_backlog(self) -> None:
        expected = self.contract.expected_invariants()
        self.assertEqual(expected["queued_windows_at_rejoin"], 4)
        self.assertEqual(expected["queued_records_at_rejoin"], 32)
        self.assertFalse(expected["partial_stage_commit_allowed"])

    def test_validate_accepts_only_global_commits(self) -> None:
        self.contract.validate(
            commits_at_rejoin={0: 0, 1: 0, 2: 0},
            final_commit_counts={0: 4, 1: 4, 2: 4},
        )
        with self.assertRaises(RuntimeError):
            self.contract.validate(
                commits_at_rejoin={0: 1, 1: 0, 2: 0},
                final_commit_counts={0: 4, 1: 4, 2: 4},
            )

    def test_rejects_too_few_microbatches_for_1f1b(self) -> None:
        with self.assertRaises(ValueError):
            ExactBPBacklogContract(
                E5OutageProtocol(
                    run_id="exact-small-m",
                    prelude_windows=1,
                    outage_windows=1,
                    resumed_windows=1,
                    physical_batch_size=1,
                    microbatches_per_window=2,
                    max_pending_windows=1,
                )
            )


if __name__ == "__main__":
    unittest.main()
