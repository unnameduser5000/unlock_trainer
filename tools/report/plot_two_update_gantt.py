#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


@dataclass
class Span:
    stage_id: int
    update_id: int
    microbatch_id: int
    action: str
    start_ms: float
    end_ms: float
    label: str


def _read_csvs(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


def _first_present(row: dict, keys: Iterable[str], required: bool = True, default=None):
    for key in keys:
        if key in row and row[key] not in ("", None):
            return row[key]
    if required:
        raise KeyError(f"Missing any of columns: {list(keys)}")
    return default


def _to_int(x) -> int:
    return int(float(str(x).strip()))


def _to_float(x) -> float:
    return float(str(x).strip())


def load_bpfree(paths: list[Path], updates: set[int]) -> list[Span]:
    rows = _read_csvs(paths)
    spans: list[Span] = []
    for r in rows:
        window_id = _to_int(_first_present(r, ["window_id"]))
        if window_id not in updates:
            continue
        stage_id = _to_int(_first_present(r, ["stage_id"]))
        mb_id = _to_int(_first_present(r, ["mb_id"]))
        action = str(_first_present(r, ["action"]))

        keep_prefixes = (
            "LOAD_STAGE0_HIDDEN",
            "LOAD_COMMON_INPUTS",
            "FWD_RECV_POST",
            "FWD_RECV_WAIT",
            "FWD_COMPUTE",
            "FWD_SEND_POST",
            "FWD_SEND_WAIT",
            "FWD_SEND_WAIT_FINAL_DRAIN",
            "LOCAL_BACKWARD",
            "LOCAL_OPTIMIZER_STEP",
        )
        if not any(action.startswith(prefix) for prefix in keep_prefixes):
            continue

        start_ms = _to_float(_first_present(r, ["start_epoch_ms"]))
        duration = _to_float(_first_present(r, ["duration_ms"], required=False, default="")) if r.get("duration_ms") not in ("", None) else None
        end_ms = start_ms + duration if duration is not None else _to_float(_first_present(r, ["end_epoch_ms"]))
        short = action
        short = short.replace("FWD_COMPUTE_INCLUDES_LOCAL_HEAD", "FWD+HEAD")
        short = short.replace("LOCAL_BACKWARD", "BWD")
        short = short.replace("LOCAL_OPTIMIZER_STEP", "OPT")
        short = short.replace("LOAD_STAGE0_HIDDEN", "LOAD_H0")
        short = short.replace("LOAD_COMMON_INPUTS", "LOAD_IN")
        short = short.replace("FWD_RECV_POST", "RECV_POST")
        short = short.replace("FWD_RECV_WAIT", "RECV_WAIT")
        short = short.replace("FWD_SEND_POST", "SEND_POST")
        short = short.replace("FWD_SEND_WAIT_FINAL_DRAIN", "SEND_WAIT_DRAIN")
        short = short.replace("FWD_SEND_WAIT", "SEND_WAIT")

        spans.append(
            Span(
                stage_id=stage_id,
                update_id=window_id,
                microbatch_id=mb_id,
                action=action,
                start_ms=start_ms,
                end_ms=end_ms,
                label=f"u{window_id}.mb{mb_id} {short}",
            )
        )
    return spans


def load_1f1b(paths: list[Path], updates: set[int]) -> list[Span]:
    rows = _read_csvs(paths)
    spans: list[Span] = []
    for r in rows:
        try:
            batch_seq = _to_int(_first_present(r, ["batch_seq", "global_batch_seq"]))
        except KeyError:
            continue
        if batch_seq not in updates:
            continue

        try:
            stage_id = _to_int(_first_present(r, ["stage_id", "rank"]))
            mb_id = _to_int(_first_present(r, ["microbatch_id", "microbatch", "microbatch_index", "chunk_id", "mb_id"]))
            action = str(_first_present(r, ["action", "event", "kind", "name"]))
            start_ms = _to_float(_first_present(r, ["start_epoch_ms", "start_ms"]))
            duration = _to_float(_first_present(r, ["duration_ms"], required=False, default="")) if r.get("duration_ms") not in ("", None) else None
            end_ms = start_ms + duration if duration is not None else _to_float(_first_present(r, ["end_epoch_ms", "end_ms"]))
        except KeyError:
            continue

        lowered = action.lower()
        keep = any(
            key in lowered
            for key in [
                "forward",
                "backward",
                "fwd",
                "bwd",
                "recv",
                "send",
                "loss",
                "optimizer",
                "step",
            ]
        )
        if not keep:
            continue

        short = action
        short = short.replace("forward_one_chunk", "FWD")
        short = short.replace("backward_one_chunk", "BWD")
        short = short.replace("optimizer_step", "OPT")

        spans.append(
            Span(
                stage_id=stage_id,
                update_id=batch_seq,
                microbatch_id=mb_id,
                action=action,
                start_ms=start_ms,
                end_ms=end_ms,
                label=f"b{batch_seq}.mb{mb_id} {short}",
            )
        )
    return spans


def plot_spans(spans: list[Span], title: str, output: Path) -> None:
    if not spans:
        raise SystemExit("No spans matched the requested updates.")

    spans = sorted(spans, key=lambda s: (s.start_ms, s.stage_id, s.microbatch_id))
    t0 = min(s.start_ms for s in spans)
    stage_ids = sorted({s.stage_id for s in spans}, reverse=True)
    stage_y = {stage: i for i, stage in enumerate(stage_ids)}

    fig_w = 16
    fig_h = max(4.5, 1.6 * len(stage_ids))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    for s in spans:
        y = stage_y[s.stage_id]
        left = s.start_ms - t0
        width = max(0.1, s.end_ms - s.start_ms)
        ax.barh(y=y, width=width, left=left, height=0.65)
        if width >= 5:
            ax.text(left + width / 2, y, s.label, va="center", ha="center", fontsize=8)

    ax.set_yticks(list(stage_y.values()))
    ax.set_yticklabels([f"stage {sid}" for sid in stage_ids])
    ax.set_xlabel("relative time (ms)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.35)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", choices=["bpfree", "1f1b"], required=True)
    parser.add_argument("--csv", type=Path, action="append", required=True)
    parser.add_argument("--updates", nargs=2, type=int, required=True, metavar=("U0", "U1"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    update_set = set(args.updates)
    if args.runner == "bpfree":
        spans = load_bpfree(args.csv, update_set)
        title = args.title or f"BP-free Gantt: update windows {args.updates[0]} and {args.updates[1]}"
    else:
        spans = load_1f1b(args.csv, update_set)
        title = args.title or f"1F1B Gantt: batches {args.updates[0]} and {args.updates[1]}"

    plot_spans(spans, title, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
