from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from sg_exe_trainer.runtime.recovery.checkpoint_store import StageCheckpointStore
from sg_exe_trainer.runtime.recovery.runtime_adapter import (
    BPFreeStageJournalObserver,
    JournaledBPFreePipelineStage,
    JournalWindowSelection,
    request_ids_for_microbatch,
)
from sg_exe_trainer.runtime.recovery.state_contract import (
    BoundaryKey,
    DurableBoundaryOutbox,
    StageCommitLedger,
)
from sg_exe_trainer.runtime.recovery.window_journal import (
    BPFreeWindowJournal,
    CommittedBoundaryReader,
)
from sg_exe_trainer.runtime.bpfree.schedule import BPFreeMicrobatch, BPFreeUpdateWindow
from sg_exe_trainer.runtime.bpfree.gpu_stage import BodyForwardOutput


def make_window(window_id: int) -> BPFreeUpdateWindow:
    return BPFreeUpdateWindow(
        window_id=window_id,
        microbatches=[
            BPFreeMicrobatch(
                window_id=window_id,
                mb_id=0,
                global_batch_seq=window_id * 2,
                seq_start=window_id * 2,
                records=[{"request_id": f"request-{window_id}-0"}],
            ),
            BPFreeMicrobatch(
                window_id=window_id,
                mb_id=1,
                global_batch_seq=window_id * 2 + 1,
                seq_start=window_id * 2 + 1,
                records=[{}],
            ),
        ],
    )


class RuntimeAdapterTest(unittest.TestCase):
    def test_terminal_stage_ignores_downstream_skip_policy(self) -> None:
        stage = JournaledBPFreePipelineStage.__new__(JournaledBPFreePipelineStage)
        stage.rank = 2
        stage.world_size = 3
        stage.skip_p2p_policy = lambda _mb: True
        mb = make_window(0).microbatches[0]
        output = stage.post_body_forward_send(
            mb=mb,
            body_output=BodyForwardOutput(
                next_hidden=torch.ones((1, 2)),
                body_forward_ms=0.0,
            ),
        )
        self.assertEqual(output.send_hidden_ms, 0.0)

    def test_selected_window_checkpoints_then_publishes_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            outbox = DurableBoundaryOutbox(root, "run-a", max_pending_windows=2)
            ledger = StageCommitLedger(root, "run-a")
            observer = BPFreeStageJournalObserver(
                stage_id=0,
                journal=BPFreeWindowJournal(
                    run_id="run-a",
                    stage_id=0,
                    world_size=3,
                    outbox=outbox,
                    ledger=ledger,
                ),
                checkpoint_store=StageCheckpointStore(root, "run-a"),
                selection=JournalWindowSelection(1, 2),
            )
            module = torch.nn.Linear(2, 2)
            optimizer = torch.optim.SGD(module.parameters(), lr=0.1)

            window0 = make_window(0)
            self.assertFalse(observer.before_window(window0))
            optimizer.step()
            observer.after_optimizer(
                window=window0,
                module=module,
                optimizer=optimizer,
                device=torch.device("cpu"),
            )

            window1 = make_window(1)
            self.assertTrue(observer.before_window(window1))
            for mb in window1.microbatches:
                observer.capture_output(
                    mb,
                    BodyForwardOutput(
                        next_hidden=torch.full((1, 2), float(mb.mb_id)),
                        body_forward_ms=0.0,
                    ),
                )
            optimizer.step()
            result = observer.after_optimizer(
                window=window1,
                module=module,
                optimizer=optimizer,
                device=torch.device("cpu"),
            )

            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(result.commit.optimizer_step, 2)
            self.assertEqual(result.commit.checkpoint_id, "stage0-step2")
            reader = CommittedBoundaryReader(outbox=outbox, ledger=ledger)
            loaded = reader.load(BoundaryKey("run-a", 0, 1, 1, 1))
            self.assertTrue(torch.equal(loaded.hidden, torch.ones((1, 2))))

    def test_request_id_fallback_uses_global_sequence_position(self) -> None:
        mb = BPFreeMicrobatch(
            window_id=0,
            mb_id=0,
            global_batch_seq=0,
            seq_start=12,
            records=[{"id": "provided"}, {}],
        )
        self.assertEqual(request_ids_for_microbatch(mb), ("provided", "seq-13"))


if __name__ == "__main__":
    unittest.main()
