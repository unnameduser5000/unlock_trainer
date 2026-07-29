from __future__ import annotations

import unittest

from experiments.e5_recovery.formal_v2.protocol import E5OutageProtocol, StageAction


class E5OutageProtocolTest(unittest.TestCase):
    def test_stage1_outage_has_explicit_prefix_and_suffix_roles(self) -> None:
        protocol = E5OutageProtocol(run_id="formal-v2", outage_windows=4, max_pending_windows=4)
        actions = protocol.phase_actions()
        self.assertEqual(
            actions["outage"],
            [
                StageAction.PREFIX_LOCAL_TO_OUTBOX.value,
                StageAction.FAILED_STAGE_OFFLINE.value,
                StageAction.SUFFIX_IDLE.value,
            ],
        )
        self.assertEqual(protocol.expected_invariants()["prefix_stage0_commits_at_rejoin"], 4)
        self.assertEqual(protocol.expected_invariants()["terminal_stage2_commits_at_rejoin"], 0)

    def test_main_point_rejects_outbox_smaller_than_outage(self) -> None:
        with self.assertRaises(ValueError):
            E5OutageProtocol(run_id="formal-v2", outage_windows=4, max_pending_windows=3)

    def test_stage0_and_stage2_faults_are_not_silently_relabelled(self) -> None:
        for failure_stage in (0, 2):
            with self.assertRaises(ValueError):
                E5OutageProtocol(run_id="formal-v2", failure_stage=failure_stage)

    def test_window_streamed_catchup_runs_stage1_and_stage2_together(self) -> None:
        protocol = E5OutageProtocol(
            run_id="formal-streamed",
            outage_windows=4,
            max_pending_windows=4,
            catchup_policy="window_streamed",
        )
        actions = protocol.phase_actions()
        self.assertNotIn("catchup_stage1", actions)
        self.assertEqual(
            actions["catchup_streamed"],
            [
                StageAction.PRESERVED_PREFIX_IDLE.value,
                StageAction.CATCHUP_STREAM_PRODUCER.value,
                StageAction.CATCHUP_STREAM_CONSUMER.value,
            ],
        )

    def test_unknown_catchup_policy_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            E5OutageProtocol(run_id="formal-v2", catchup_policy="not-a-policy")


if __name__ == "__main__":
    unittest.main()
