from __future__ import annotations

import unittest

from experiments.e5_recovery.formal_v2.run_bpfree_outage import build_parser


class BPFreeRunnerPolicyTest(unittest.TestCase):
    def test_baseline_entry_keeps_drain_first_default(self) -> None:
        parser = build_parser()
        self.assertEqual(parser.get_default("catchup_policy"), "drain_first")
        self.assertEqual(parser.get_default("recovery_state_mode"), "durable")

    def test_streamed_entry_can_override_only_the_default(self) -> None:
        parser = build_parser(default_catchup_policy="window_streamed")
        self.assertEqual(parser.get_default("catchup_policy"), "window_streamed")
        action = next(
            item for item in parser._actions if item.dest == "catchup_policy"
        )
        self.assertEqual(set(action.choices), {"drain_first", "window_streamed"})

    def test_volatile_entry_selects_streaming_and_ram_state(self) -> None:
        parser = build_parser(
            default_catchup_policy="window_streamed",
            default_recovery_state_mode="volatile",
        )
        self.assertEqual(parser.get_default("catchup_policy"), "window_streamed")
        self.assertEqual(parser.get_default("recovery_state_mode"), "volatile")


if __name__ == "__main__":
    unittest.main()
