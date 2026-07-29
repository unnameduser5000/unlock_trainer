from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sg_exe_trainer.runtime.recovery.event_log import (
    RecoveryEventName,
    RecoveryEventRecorder,
    RecoveryTimeline,
)


class RecoveryEventLogTest(unittest.TestCase):
    def test_recovery_summary_uses_rejoin_and_catchup_events(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rank0 = RecoveryEventRecorder(root, run_id="run-a", rank=0, stage_id=0)
            rank1 = RecoveryEventRecorder(root, run_id="run-a", rank=1, stage_id=1)
            rank2 = RecoveryEventRecorder(root, run_id="run-a", rank=2, stage_id=2)
            rank1.record(RecoveryEventName.OUTAGE_INJECTED, window_id=4)
            rank1.record(RecoveryEventName.OUTAGE_DETECTED, window_id=4)
            rank0.record(RecoveryEventName.PREFIX_WINDOW_COMMIT, window_id=4)
            rank1.record(RecoveryEventName.STAGE_REJOINED, window_id=8)
            rank1.record(RecoveryEventName.CATCHUP_STAGE1_START, window_id=4)
            rank1.record(RecoveryEventName.CATCHUP_STAGE1_DONE, window_id=7)
            rank2.record(RecoveryEventName.CATCHUP_STAGE2_START, window_id=4)
            rank2.record(RecoveryEventName.CATCHUP_STAGE2_DONE, window_id=7)
            rank2.record(RecoveryEventName.TERMINAL_TARGET_REACHED, window_id=7)
            rank0.record(RecoveryEventName.LIVE_P2P_RESUMED, window_id=8)

            timeline = RecoveryTimeline.load(root, "run-a")
            summary = timeline.recovery_summary()
            self.assertGreaterEqual(summary["outage_duration_ms"], 0.0)
            self.assertGreaterEqual(summary["stage1_catchup_ms"], 0.0)
            self.assertGreaterEqual(summary["all_stage_catchup_ms"], summary["stage1_catchup_ms"])
            self.assertGreaterEqual(summary["live_resume_ms"], summary["all_stage_catchup_ms"])
            common = timeline.common_recovery_summary()
            self.assertGreaterEqual(common["rejoin_to_terminal_target_ms"], 0.0)
            self.assertGreaterEqual(
                common["rejoin_to_live_resume_ms"],
                common["rejoin_to_terminal_target_ms"],
            )

    def test_recorder_continues_sequence_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = RecoveryEventRecorder(root, run_id="run-a", rank=0, stage_id=0)
            self.assertEqual(first.record(RecoveryEventName.OUTAGE_INJECTED).sequence, 0)
            restarted = RecoveryEventRecorder(root, run_id="run-a", rank=0, stage_id=0)
            self.assertEqual(restarted.record(RecoveryEventName.STAGE_REJOINED).sequence, 1)


if __name__ == "__main__":
    unittest.main()
