from __future__ import annotations

import unittest

from experiments.e5_recovery.formal_v2.run_exactbp_outage import build_parser


class ExactBPRunnerModeTest(unittest.TestCase):
    def test_default_entry_remains_durable(self) -> None:
        self.assertEqual(build_parser().get_default("recovery_state_mode"), "durable")

    def test_volatile_entry_changes_only_the_default(self) -> None:
        parser = build_parser(default_recovery_state_mode="volatile")
        self.assertEqual(parser.get_default("recovery_state_mode"), "volatile")
        action = next(
            item for item in parser._actions if item.dest == "recovery_state_mode"
        )
        self.assertEqual(set(action.choices), {"durable", "volatile"})


if __name__ == "__main__":
    unittest.main()
