from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from experiments.e5_recovery.formal_v2.toy_outage_oracle import run_toy_scenario


class ToyOutageOracleTest(unittest.TestCase):
    def test_stage_major_catchup_matches_fault_free_logical_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reference = run_toy_scenario(
                root,
                run_id="reference",
                execution="window_major",
            )
            outage = run_toy_scenario(
                root,
                run_id="outage",
                execution="stage_major",
            )

            self.assertEqual(outage.prefix_commits_at_rejoin, 3)
            self.assertEqual(outage.terminal_commits_at_rejoin, 0)
            self.assertEqual(outage.terminal_commits_final, 3)
            for reference_stage, outage_stage in zip(reference.stage_states, outage.stage_states):
                self.assertEqual(reference_stage.keys(), outage_stage.keys())
                for name in reference_stage:
                    self.assertTrue(
                        torch.equal(reference_stage[name], outage_stage[name]),
                        msg=f"stage parameter mismatch: {name}",
                    )


if __name__ == "__main__":
    unittest.main()
