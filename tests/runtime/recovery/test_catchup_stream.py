from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path

from sg_exe_trainer.runtime.recovery.catchup_stream import (
    CommitWaitTimeout,
    wait_for_stage_commit,
)
from sg_exe_trainer.runtime.recovery.state_contract import StageCommit, StageCommitLedger


class CatchupStreamTest(unittest.TestCase):
    def test_consumer_observes_only_the_durable_commit_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            ledger = StageCommitLedger(Path(temp_dir), "run-a")

            def publish() -> None:
                time.sleep(0.02)
                ledger.record(
                    StageCommit(
                        run_id="run-a",
                        stage_id=1,
                        window_id=4,
                        optimizer_step=5,
                        request_ids=("r4",),
                        input_producer_versions=(4,),
                        checkpoint_id="stage1-step5",
                    )
                )

            publisher = threading.Thread(target=publish)
            publisher.start()
            result = wait_for_stage_commit(
                ledger=ledger,
                stage_id=1,
                window_id=4,
                timeout_s=1.0,
                poll_ms=1.0,
            )
            publisher.join()
            self.assertEqual(result.commit.window_id, 4)
            self.assertGreater(result.wait_ms, 0.0)
            self.assertGreater(result.polls, 1)

    def test_missing_commit_times_out(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            ledger = StageCommitLedger(Path(temp_dir), "run-a")
            with self.assertRaises(CommitWaitTimeout):
                wait_for_stage_commit(
                    ledger=ledger,
                    stage_id=1,
                    window_id=7,
                    timeout_s=0.01,
                    poll_ms=1.0,
                )


if __name__ == "__main__":
    unittest.main()
