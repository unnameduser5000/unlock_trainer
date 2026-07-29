from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import torch

from sg_exe_trainer.runtime.recovery.checkpoint_store import (
    CheckpointConflictError,
    StageCheckpointStore,
)


def optimizer_tensor_state(optimizer: torch.optim.Optimizer) -> dict:
    return copy.deepcopy(optimizer.state_dict())


class StageCheckpointStoreTest(unittest.TestCase):
    def test_restores_trainable_parameters_optimizer_and_rng(self) -> None:
        torch.manual_seed(17)
        module = torch.nn.Linear(4, 2)
        optimizer = torch.optim.AdamW(module.parameters(), lr=0.01)
        loss = module(torch.ones((3, 4))).square().mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        expected_params = {name: value.detach().clone() for name, value in module.named_parameters()}
        expected_optimizer = optimizer_tensor_state(optimizer)
        expected_rng = torch.get_rng_state().clone()

        with tempfile.TemporaryDirectory() as temp_dir:
            store = StageCheckpointStore(Path(temp_dir), "run-a")
            metadata = store.save(
                module=module,
                optimizer=optimizer,
                stage_id=1,
                window_id=4,
                optimizer_step=5,
            )
            self.assertEqual(store.list_stage(1), [metadata])
            self.assertEqual(store.list_stage(0), [])

            with torch.no_grad():
                for parameter in module.parameters():
                    parameter.add_(100.0)
            torch.manual_seed(999)
            optimizer.param_groups[0]["lr"] = 0.5

            restored = store.restore(
                module=module,
                optimizer=optimizer,
                stage_id=1,
                optimizer_step=5,
            )
            self.assertEqual(metadata, restored)
            for name, parameter in module.named_parameters():
                self.assertTrue(torch.equal(parameter, expected_params[name]))
            self.assertEqual(optimizer.state_dict()["param_groups"], expected_optimizer["param_groups"])
            for state_id, state in optimizer.state_dict()["state"].items():
                for key, value in state.items():
                    expected = expected_optimizer["state"][state_id][key]
                    if torch.is_tensor(value):
                        self.assertTrue(torch.equal(value, expected))
                    else:
                        self.assertEqual(value, expected)
            self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))

    def test_same_version_rejects_different_state(self) -> None:
        module = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
        with tempfile.TemporaryDirectory() as temp_dir:
            store = StageCheckpointStore(Path(temp_dir), "run-a")
            store.save(
                module=module,
                optimizer=optimizer,
                stage_id=0,
                window_id=0,
                optimizer_step=0,
            )
            with torch.no_grad():
                module.weight.add_(1.0)
            with self.assertRaises(CheckpointConflictError):
                store.save(
                    module=module,
                    optimizer=optimizer,
                    stage_id=0,
                    window_id=0,
                    optimizer_step=0,
                )


if __name__ == "__main__":
    unittest.main()
