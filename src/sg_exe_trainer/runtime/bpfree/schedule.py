from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BPFreeMicrobatch:
    window_id: int
    mb_id: int
    global_batch_seq: int
    seq_start: int
    records: list[dict[str, Any]]

    @property
    def num_records(self) -> int:
        return len(self.records)


@dataclass(frozen=True)
class BPFreeUpdateWindow:
    window_id: int
    microbatches: list[BPFreeMicrobatch]

    @property
    def num_microbatches(self) -> int:
        return len(self.microbatches)

    @property
    def num_records(self) -> int:
        return sum(mb.num_records for mb in self.microbatches)


def split_records_into_update_windows(
    *,
    records: list[dict[str, Any]],
    effective_batch_size: int,
    n_microbatches: int,
    drop_last: bool = True,
) -> list[BPFreeUpdateWindow]:
    """
    Canonical BP-free schedule split.

    1F1B:
      logical/global batch
        -> split into microbatches
        -> schedule

    BP-free:
      local update window / effective batch
        -> split into physical microbatches
        -> schedule

    Here:
      effective_batch_size = physical_batch_size * n_microbatches
      physical_batch_size = effective_batch_size // n_microbatches
    """
    if effective_batch_size <= 0:
        raise ValueError("effective_batch_size must be positive.")
    if n_microbatches <= 0:
        raise ValueError("n_microbatches must be positive.")
    if effective_batch_size % n_microbatches != 0:
        raise ValueError(
            f"effective_batch_size={effective_batch_size} must be divisible by "
            f"n_microbatches={n_microbatches}."
        )

    physical_batch_size = effective_batch_size // n_microbatches

    usable_records = len(records)
    if drop_last:
        usable_records = (usable_records // effective_batch_size) * effective_batch_size
    elif usable_records % effective_batch_size != 0:
        raise ValueError(
            f"records={len(records)} is not divisible by "
            f"effective_batch_size={effective_batch_size}; use drop_last=True."
        )

    windows: list[BPFreeUpdateWindow] = []

    for window_start in range(0, usable_records, effective_batch_size):
        window_id = window_start // effective_batch_size
        window_records = records[window_start : window_start + effective_batch_size]

        microbatches: list[BPFreeMicrobatch] = []
        for mb_id in range(n_microbatches):
            mb_start = mb_id * physical_batch_size
            mb_end = mb_start + physical_batch_size
            mb_records = window_records[mb_start:mb_end]

            global_batch_seq = window_id * n_microbatches + mb_id
            seq_start = window_start + mb_start

            microbatches.append(
                BPFreeMicrobatch(
                    window_id=window_id,
                    mb_id=mb_id,
                    global_batch_seq=global_batch_seq,
                    seq_start=seq_start,
                    records=mb_records,
                )
            )

        windows.append(
            BPFreeUpdateWindow(
                window_id=window_id,
                microbatches=microbatches,
            )
        )

    return windows


def flatten_update_windows(
    windows: list[BPFreeUpdateWindow],
) -> list[BPFreeMicrobatch]:
    return [mb for window in windows for mb in window.microbatches]


__all__ = [
    "BPFreeMicrobatch",
    "BPFreeUpdateWindow",
    "split_records_into_update_windows",
    "flatten_update_windows",
]
