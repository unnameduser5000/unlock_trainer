#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


@dataclass
class CoarseSpan:
    stage_id: int
    update_id: int
    start_ms: float
    end_ms: float
    label: str


def read_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    return rows


def as_int(x) -> int:
    return int(float(str(x)))


def as_float(x) -> float:
    return float(str(x))


def event_end_ms(row: dict) -> float:
    start = as_float(row["start_epoch_ms"])
    if row.get("duration_ms") not in ("", None):
        return start + as_float(row["duration_ms"])
    return as_float(row["end_epoch_ms"])


def load_bpfree_action_csvs(paths: list[Path], update_ids: set[int]) -> list[CoarseSpan]:
    rows = read_rows(paths)

    grouped: dict[tuple[int, int], list[dict]] = {}

    for r in rows:
        if r.get("phase") != "train":
            continue

        window_id = as_int(r["window_id"])
        if window_id not in update_ids:
            continue

        stage_id = as_int(r["stage_id"])
        grouped.setdefault((stage_id, window_id), []).append(r)

    spans = []
    for (stage_id, window_id), group in sorted(grouped.items()):
        start = min(as_float(r["start_epoch_ms"]) for r in group)
        end = max(event_end_ms(r) for r in group)

        spans.append(
            CoarseSpan(
                stage_id=stage_id,
                update_id=window_id,
                start_ms=start,
                end_ms=end,
                label=f"u{window_id}",
            )
        )

    return spans


def load_1f1b_stage_metrics(paths: list[Path], batch_ids: set[int]) -> list[CoarseSpan]:
    rows = read_rows(paths)

    spans = []
    for r in rows:
        if r.get("phase") != "train":
            continue

        if "batch_seq" not in r:
            continue

        batch_seq = as_int(r["batch_seq"])
        if batch_seq not in batch_ids:
            continue

        stage_id = as_int(r.get("stage_id", r.get("rank")))

        start = as_float(r["start_epoch_ms"])
        if r.get("end_epoch_ms") not in ("", None):
            end = as_float(r["end_epoch_ms"])
        elif r.get("duration_ms") not in ("", None):
            end = start + as_float(r["duration_ms"])
        elif r.get("step_ms") not in ("", None):
            end = start + as_float(r["step_ms"])
        else:
            raise KeyError("stage_metrics row has no end_epoch_ms/duration_ms/step_ms")

        spans.append(
            CoarseSpan(
                stage_id=stage_id,
                update_id=batch_seq,
                start_ms=start,
                end_ms=end,
                label=f"b{batch_seq}",
            )
        )

    return spans


def plot(spans: list[CoarseSpan], title: str, output: Path) -> None:
    if not spans:
        raise SystemExit("No spans found for requested updates/batches.")

    t0 = min(s.start_ms for s in spans)
    stage_ids = sorted({s.stage_id for s in spans}, reverse=True)
    y_for_stage = {sid: i for i, sid in enumerate(stage_ids)}

    fig, ax = plt.subplots(figsize=(14, max(4, 1.4 * len(stage_ids))))

    for s in sorted(spans, key=lambda x: (x.start_ms, x.stage_id)):
        y = y_for_stage[s.stage_id]
        left = s.start_ms - t0
        width = max(0.1, s.end_ms - s.start_ms)
        ax.barh(y=y, width=width, left=left, height=0.58)
        ax.text(left + width / 2, y, s.label, ha="center", va="center", fontsize=10)

    ax.set_yticks(list(y_for_stage.values()))
    ax.set_yticklabels([f"stage {sid}" for sid in stage_ids])
    ax.set_xlabel("relative wall time (ms)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.35)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)

    print(f"Wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", choices=["bpfree", "1f1b"], required=True)
    parser.add_argument("--csv", type=Path, action="append", required=True)
    parser.add_argument("--updates", nargs=2, type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    update_ids = set(args.updates)

    if args.runner == "bpfree":
        spans = load_bpfree_action_csvs(args.csv, update_ids)
        title = args.title or f"BP-free coarse Gantt: update windows {args.updates[0]} and {args.updates[1]}"
    else:
        spans = load_1f1b_stage_metrics(args.csv, update_ids)
        title = args.title or f"1F1B coarse Gantt: logical batches {args.updates[0]} and {args.updates[1]}"

    plot(spans, title, args.output)


if __name__ == "__main__":
    main()
