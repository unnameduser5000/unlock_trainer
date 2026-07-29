from __future__ import annotations

import copy
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from sg_exe_trainer.runtime.bpfree.schedule import BPFreeMicrobatch, BPFreeUpdateWindow
from sg_exe_trainer.runtime.bpfree.gpu_stage import BodyForwardOutput

from sg_exe_trainer.runtime.recovery.checkpoint_store import StageCheckpointStore
from sg_exe_trainer.runtime.recovery.runtime_adapter import (
    BPFreeStageJournalObserver,
    JournalWindowSelection,
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


@dataclass(frozen=True)
class ToyScenarioResult:
    stage_states: tuple[dict[str, torch.Tensor], ...]
    prefix_commits_at_rejoin: int
    terminal_commits_at_rejoin: int
    terminal_commits_final: int


def _make_windows(window_count: int, microbatches: int) -> list[BPFreeUpdateWindow]:
    windows: list[BPFreeUpdateWindow] = []
    for window_id in range(window_count):
        items: list[BPFreeMicrobatch] = []
        for mb_id in range(microbatches):
            seq = window_id * microbatches + mb_id
            items.append(
                BPFreeMicrobatch(
                    window_id=window_id,
                    mb_id=mb_id,
                    global_batch_seq=seq,
                    seq_start=seq,
                    records=[{"request_id": f"request-{seq}"}],
                )
            )
        windows.append(BPFreeUpdateWindow(window_id=window_id, microbatches=items))
    return windows


def _stage_input(seq: int) -> torch.Tensor:
    return torch.tensor(
        [[seq / 10.0, (seq + 1) / 10.0, (seq + 2) / 10.0, 1.0]],
        dtype=torch.float32,
    )


def _stage_target(seq: int, stage_id: int) -> torch.Tensor:
    base = _stage_input(seq)
    return torch.tanh(base + float(stage_id + 1) / 10.0)


def run_toy_scenario(
    root: Path,
    *,
    run_id: str,
    execution: Literal["window_major", "stage_major"],
    window_count: int = 3,
    microbatches: int = 2,
) -> ToyScenarioResult:
    torch.manual_seed(71)
    initial_modules = [torch.nn.Linear(4, 4, bias=False) for _ in range(3)]
    modules = [copy.deepcopy(module) for module in initial_modules]
    optimizers = [torch.optim.SGD(module.parameters(), lr=0.05) for module in modules]
    windows = _make_windows(window_count, microbatches)

    outbox = DurableBoundaryOutbox(root, run_id, max_pending_windows=window_count)
    ledger = StageCommitLedger(root, run_id)
    checkpoint_store = StageCheckpointStore(root, run_id)
    reader = CommittedBoundaryReader(outbox=outbox, ledger=ledger)

    observers: list[BPFreeStageJournalObserver] = []
    for stage_id in range(3):
        input_provider = None
        if stage_id > 0:
            def input_provider(
                window: BPFreeUpdateWindow,
                stage_id: int = stage_id,
            ) -> tuple[int, ...]:
                return tuple(
                    reader.load(
                        BoundaryKey(
                            run_id,
                            stage_id - 1,
                            stage_id,
                            window.window_id,
                            mb.mb_id,
                        )
                    ).metadata.producer_version
                    for mb in window.microbatches
                )

        observers.append(
            BPFreeStageJournalObserver(
                stage_id=stage_id,
                journal=BPFreeWindowJournal(
                    run_id=run_id,
                    stage_id=stage_id,
                    world_size=3,
                    outbox=outbox,
                    ledger=ledger,
                ),
                checkpoint_store=checkpoint_store,
                selection=JournalWindowSelection(0, window_count),
                input_version_provider=input_provider,
            )
        )

    def process(stage_id: int, window: BPFreeUpdateWindow) -> None:
        module = modules[stage_id]
        optimizer = optimizers[stage_id]
        observer = observers[stage_id]
        observer.before_window(window)
        optimizer.zero_grad(set_to_none=True)

        input_keys: list[BoundaryKey] = []
        for mb in window.microbatches:
            seq = mb.seq_start
            if stage_id == 0:
                stage_input = _stage_input(seq)
            else:
                key = BoundaryKey(
                    run_id,
                    stage_id - 1,
                    stage_id,
                    window.window_id,
                    mb.mb_id,
                )
                input_keys.append(key)
                stage_input = reader.load(key).hidden

            hidden = module(stage_input)
            if stage_id < 2:
                observer.capture_output(
                    mb,
                    BodyForwardOutput(next_hidden=hidden, body_forward_ms=0.0),
                )
            loss = (hidden - _stage_target(seq, stage_id)).square().mean()
            (loss / window.num_microbatches).backward()

        optimizer.step()
        observer.after_optimizer(
            window=window,
            module=module,
            optimizer=optimizer,
            device=torch.device("cpu"),
        )
        for key in input_keys:
            outbox.acknowledge(key, consumer_version=observer.optimizer_step)

    prefix_commits_at_rejoin = 0
    terminal_commits_at_rejoin = 0
    if execution == "window_major":
        for window in windows:
            for stage_id in range(3):
                process(stage_id, window)
    elif execution == "stage_major":
        for window in windows:
            process(0, window)
        prefix_commits_at_rejoin = len(ledger.list_stage(0))
        terminal_commits_at_rejoin = len(ledger.list_stage(2))
        for stage_id in (1, 2):
            for window in windows:
                process(stage_id, window)
    else:
        raise ValueError(f"unknown execution={execution!r}")

    return ToyScenarioResult(
        stage_states=tuple(
            {name: value.detach().clone() for name, value in module.state_dict().items()}
            for module in modules
        ),
        prefix_commits_at_rejoin=prefix_commits_at_rejoin,
        terminal_commits_at_rejoin=terminal_commits_at_rejoin,
        terminal_commits_final=len(ledger.list_stage(2)),
    )


def run_self_check() -> ToyScenarioResult:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        reference = run_toy_scenario(root, run_id="reference", execution="window_major")
        outage = run_toy_scenario(root, run_id="outage", execution="stage_major")
        for reference_stage, outage_stage in zip(reference.stage_states, outage.stage_states):
            for name in reference_stage:
                if not torch.equal(reference_stage[name], outage_stage[name]):
                    raise AssertionError(f"outage catch-up changed final parameter {name}")
        return outage


if __name__ == "__main__":
    result = run_self_check()
    print(
        {
            "prefix_commits_at_rejoin": result.prefix_commits_at_rejoin,
            "terminal_commits_at_rejoin": result.terminal_commits_at_rejoin,
            "terminal_commits_final": result.terminal_commits_final,
            "oracle_equal": True,
        }
    )
