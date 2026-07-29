from __future__ import annotations

import unittest

import torch

from sg_exe_trainer.runtime.bpfree.schedule import BPFreeMicrobatch, BPFreeUpdateWindow

from sg_exe_trainer.runtime.recovery.volatile_backlog import VolatileBoundaryBuffer


def _microbatch(window_id: int, mb_id: int) -> BPFreeMicrobatch:
    return BPFreeMicrobatch(
        window_id=window_id,
        mb_id=mb_id,
        global_batch_seq=window_id * 10 + mb_id,
        seq_start=window_id * 10 + mb_id,
        records=[{"request_id": f"r-{window_id}-{mb_id}"}],
    )


class VolatileBoundaryBufferTest(unittest.TestCase):
    def test_captures_exact_cpu_tensor_and_validates_complete_window(self) -> None:
        buffer = VolatileBoundaryBuffer(max_pending_windows=2)
        microbatches = [_microbatch(4, 0), _microbatch(4, 1)]
        source = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)

        first = buffer.capture(microbatches[0], source)
        source.add_(100)
        buffer.capture(microbatches[1], source)

        self.assertTrue(torch.equal(first.hidden, torch.arange(12).reshape(1, 3, 4)))
        self.assertEqual(first.hidden.device.type, "cpu")
        self.assertEqual(buffer.window_ids(), [4])
        self.assertEqual(buffer.microbatch_count, 2)
        self.assertEqual(buffer.tensor_bytes, 2 * 12 * 4)
        buffer.validate_window(BPFreeUpdateWindow(window_id=4, microbatches=microbatches))

    def test_accepts_transport_ready_cpu_tensor_without_copy(self) -> None:
        buffer = VolatileBoundaryBuffer(max_pending_windows=1)
        mb = _microbatch(2, 0)
        hidden = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)

        entry = buffer.capture_prepared_cpu(mb, hidden, capture_ms=1.25)

        self.assertIs(entry.hidden, hidden)
        self.assertEqual(entry.capture_ms, 1.25)
        self.assertEqual(entry.tensor_bytes, hidden.numel() * hidden.element_size())

    def test_capacity_is_counted_by_pending_window(self) -> None:
        buffer = VolatileBoundaryBuffer(max_pending_windows=1)
        buffer.capture(_microbatch(4, 0), torch.zeros(1, 2))
        buffer.capture(_microbatch(4, 1), torch.zeros(1, 2))

        with self.assertRaisesRegex(RuntimeError, "capacity reached"):
            buffer.capture(_microbatch(5, 0), torch.zeros(1, 2))

    def test_missing_or_duplicate_boundary_is_rejected(self) -> None:
        buffer = VolatileBoundaryBuffer(max_pending_windows=1)
        mb = _microbatch(4, 0)
        buffer.capture(mb, torch.zeros(1, 2))

        with self.assertRaisesRegex(RuntimeError, "already captured"):
            buffer.capture(mb, torch.zeros(1, 2))
        with self.assertRaisesRegex(RuntimeError, "missing volatile microbatches"):
            buffer.validate_window(
                BPFreeUpdateWindow(window_id=4, microbatches=[mb, _microbatch(4, 1)])
            )


if __name__ == "__main__":
    unittest.main()
