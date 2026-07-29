from __future__ import annotations

import csv
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass
class ActionEvent:
    phase: str
    rank: int
    stage_id: int
    window_id: int
    mb_id: int
    global_batch_seq: int
    seq_start: int
    records: int
    action: str
    start_perf: float
    end_perf: float
    start_epoch_ms: float
    end_epoch_ms: float
    duration_ms: float
    extra: str = ""


class ActionTracer:
    def __init__(
        self,
        *,
        path: Path,
        phase: str,
        rank: int,
        stage_id: int,
        min_window_id: int = 0,
        max_window_id: int | None = None,
    ) -> None:
        self.path = path
        self.phase = phase
        self.rank = rank
        self.stage_id = stage_id
        self.min_window_id = max(0, int(min_window_id))
        self.max_window_id = (
            None if max_window_id is None else int(max_window_id)
        )
        if (
            self.max_window_id is not None
            and self.max_window_id <= self.min_window_id
        ):
            raise ValueError(
                "max_window_id must be greater than min_window_id"
            )
        self.rows: list[ActionEvent] = []

    def is_enabled_for_window(self, window_id: int) -> bool:
        window = int(window_id)
        if window < self.min_window_id:
            return False
        if self.max_window_id is not None and window >= self.max_window_id:
            return False
        return True

    def record(
        self,
        *,
        window_id: int,
        mb_id: int,
        global_batch_seq: int,
        seq_start: int,
        records: int,
        action: str,
        start_perf: float,
        end_perf: float,
        start_epoch_ms: float,
        end_epoch_ms: float,
        extra: str = "",
    ) -> None:
        if not self.is_enabled_for_window(window_id):
            return
        self.rows.append(
            ActionEvent(
                phase=self.phase,
                rank=self.rank,
                stage_id=self.stage_id,
                window_id=window_id,
                mb_id=mb_id,
                global_batch_seq=global_batch_seq,
                seq_start=seq_start,
                records=records,
                action=action,
                start_perf=start_perf,
                end_perf=end_perf,
                start_epoch_ms=start_epoch_ms,
                end_epoch_ms=end_epoch_ms,
                duration_ms=max(0.0, (end_perf - start_perf) * 1000.0),
                extra=extra,
            )
        )

    def span(
        self,
        *,
        window_id: int,
        mb_id: int,
        global_batch_seq: int,
        seq_start: int,
        records: int,
        action: str,
        extra: str = "",
    ):
        if not self.is_enabled_for_window(window_id):
            return NoOpActionTracer().span()

        tracer = self

        class _Span:
            def __enter__(self):
                self.start_perf = time.perf_counter()
                self.start_epoch_ms = time.time() * 1000.0
                return self

            def __exit__(self, exc_type, exc, tb):
                end_perf = time.perf_counter()
                end_epoch_ms = time.time() * 1000.0
                tracer.record(
                    window_id=window_id,
                    mb_id=mb_id,
                    global_batch_seq=global_batch_seq,
                    seq_start=seq_start,
                    records=records,
                    action=action,
                    start_perf=self.start_perf,
                    end_perf=end_perf,
                    start_epoch_ms=self.start_epoch_ms,
                    end_epoch_ms=end_epoch_ms,
                    extra=extra,
                )
                return False

        return _Span()

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(ActionEvent.__dataclass_fields__.keys())
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.rows:
                writer.writerow(asdict(row))



class NoOpActionTracer:
    def is_enabled_for_window(self, window_id: int) -> bool:
        return False

    def span(self, **kwargs):
        class _Span:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        return _Span()

    def flush(self) -> None:
        pass
