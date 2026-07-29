#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


@dataclass
class Event:
    stage: int
    window: int
    mb: int
    seq: int
    records: int
    action: str
    start_ms: float
    end_ms: float


def f(x) -> float:
    return float(str(x))


def i(x) -> int:
    return int(float(str(x)))


def load_events(paths: list[Path]) -> list[Event]:
    events: list[Event] = []

    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for r in csv.DictReader(handle):
                if r.get("phase") != "train":
                    continue

                action = r["action"]

                # Keep real stage-local scheduling actions.
                keep = (
                    action.startswith("LOAD_STAGE0_HIDDEN")
                    or action.startswith("LOAD_COMMON_INPUTS")
                    or action.startswith("FWD_RECV_WAIT")
                    or action.startswith("FWD_RECV_POST")
                    or action.startswith("BODY_FORWARD")
                    or action.startswith("LOCAL_HEAD_LOSS")
                    or action.startswith("FWD_COMPUTE_INCLUDES_LOCAL_HEAD")
                    or action.startswith("FWD_SEND_POST")
                    or action.startswith("LOCAL_BACKWARD")
                    or action.startswith("LOCAL_OPTIMIZER_STEP")
                )

                if not keep:
                    continue

                start = f(r["start_epoch_ms"])

                if r.get("duration_ms") not in ("", None):
                    end = start + f(r["duration_ms"])
                else:
                    end = f(r["end_epoch_ms"])

                events.append(
                    Event(
                        stage=i(r["stage_id"]),
                        window=i(r["window_id"]),
                        mb=i(r["mb_id"]),
                        seq=i(r["seq_start"]),
                        records=i(r["records"]),
                        action=action,
                        start_ms=start,
                        end_ms=end,
                    )
                )

    return events


def short_action(action: str) -> str:
    return (
        action.replace("LOAD_STAGE0_HIDDEN", "LOAD_H0")
        .replace("LOAD_COMMON_INPUTS", "LOAD_IN")
        .replace("FWD_RECV_POST", "RECV_POST")
        .replace("FWD_RECV_WAIT", "RECV")
        .replace("BODY_FORWARD", "BODY")
        .replace("LOCAL_HEAD_LOSS", "HEAD")
        .replace("FWD_COMPUTE_INCLUDES_LOCAL_HEAD", "FWD+HEAD")
        .replace("FWD_SEND_POST", "SEND")
        .replace("LOCAL_BACKWARD", "BWD")
        .replace("LOCAL_OPTIMIZER_STEP", "OPT")
    )


def action_alpha(action: str) -> float:
    if action.startswith("BODY_FORWARD") or action.startswith("FWD_COMPUTE"):
        return 0.90
    if action.startswith("LOCAL_HEAD_LOSS"):
        return 0.55
    if action.startswith("LOCAL_BACKWARD"):
        return 0.35
    if action.startswith("FWD_SEND_POST"):
        return 0.95
    if action.startswith("FWD_RECV_WAIT"):
        return 0.75
    if action.startswith("LOCAL_OPTIMIZER_STEP"):
        return 0.95
    return 0.25


def plot(events: list[Event], windows: set[int], output: Path, title: str | None) -> None:
    events = [e for e in events if e.window in windows]
    if not events:
        raise SystemExit(f"No events found for windows={sorted(windows)}")

    # Use first real local execution as origin.
    t0 = min(e.start_ms for e in events)

    stages = sorted({e.stage for e in events}, reverse=True)
    y_for_stage = {stage: idx for idx, stage in enumerate(stages)}

    seqs = sorted({e.seq for e in events})
    cmap = plt.get_cmap("tab20")
    color_for_seq = {seq: cmap(idx % 20) for idx, seq in enumerate(seqs)}

    fig, ax = plt.subplots(figsize=(18, max(4, 1.4 * len(stages))))

    # Draw events.
    for e in sorted(events, key=lambda x: (x.start_ms, x.stage, x.seq, x.action)):
        y = y_for_stage[e.stage]
        left = (e.start_ms - t0) / 1000.0
        width = max(0.0005, (e.end_ms - e.start_ms) / 1000.0)

        ax.barh(
            y=y,
            width=width,
            left=left,
            height=0.58,
            color=color_for_seq[e.seq],
            alpha=action_alpha(e.action),
            edgecolor="black",
            linewidth=0.4,
        )

        label = short_action(e.action)
        if width >= 0.010:
            ax.text(
                left + width / 2,
                y,
                f"{e.seq} {label}",
                ha="center",
                va="center",
                fontsize=8,
            )

    # Terminal completion markers from last stage per seq.
    last_stage = max(stages)
    seq_terminal_end: dict[int, float] = {}

    for seq in seqs:
        seq_stage_events = [e for e in events if e.seq == seq and e.stage == last_stage]
        if not seq_stage_events:
            continue
        seq_terminal_end[seq] = max(e.end_ms for e in seq_stage_events)

    for seq, end_ms in sorted(seq_terminal_end.items()):
        x = (end_ms - t0) / 1000.0
        ax.axvline(x, color="purple", linestyle=":", linewidth=1.0, alpha=0.8)
        ax.text(
            x,
            len(stages) - 0.35,
            f"seq{seq} end",
            rotation=90,
            ha="center",
            va="bottom",
            fontsize=8,
            color="purple",
        )

    # Window terminal completion markers: last seq in each window on terminal stage.
    window_end: dict[int, float] = {}
    for window in windows:
        window_events = [e for e in events if e.window == window and e.stage == last_stage]
        if not window_events:
            continue
        window_end[window] = max(e.end_ms for e in window_events)

    for window, end_ms in sorted(window_end.items()):
        x = (end_ms - t0) / 1000.0
        ax.axvline(x, color="red", linestyle="--", linewidth=1.2, alpha=0.9)
        ax.text(
            x,
            -0.35,
            f"terminal completion window{window}",
            rotation=90,
            ha="center",
            va="top",
            fontsize=8,
            color="darkred",
        )

    # Arrow between two adjacent window completions.
    if len(window_end) >= 2:
        ordered = sorted(window_end.items())
        (w0, e0), (w1, e1) = ordered[0], ordered[1]
        x0 = (e0 - t0) / 1000.0
        x1 = (e1 - t0) / 1000.0
        y = -0.55
        ax.annotate(
            "",
            xy=(x1, y),
            xytext=(x0, y),
            arrowprops=dict(arrowstyle="<->", color="saddlebrown", linewidth=1.2),
        )
        ax.text(
            (x0 + x1) / 2,
            y - 0.08,
            f"terminal interval = {(e1 - e0):.1f} ms",
            ha="center",
            va="top",
            color="saddlebrown",
            fontsize=10,
        )

    # Print overlap audit numbers.
    print("\nStage productive intervals by window:")
    by_stage_window = defaultdict(list)
    for e in events:
        # Exclude POST-only bookkeeping from productive interval.
        if e.action.startswith("FWD_RECV_POST") or e.action.startswith("FWD_SEND_POST"):
            continue
        by_stage_window[(e.stage, e.window)].append(e)

    for window in sorted(windows):
        print(f"\nwindow {window}")
        intervals = {}
        for stage in sorted(stages):
            evs = by_stage_window.get((stage, window), [])
            if not evs:
                continue
            s = min(e.start_ms for e in evs)
            e = max(e.end_ms for e in evs)
            intervals[stage] = (s, e)
            print(f"  stage {stage}: start={(s-t0):8.2f} ms end={(e-t0):8.2f} ms dur={(e-s):8.2f} ms")

        for a, b in zip(sorted(intervals), sorted(intervals)[1:]):
            sa, ea = intervals[a]
            sb, eb = intervals[b]
            ov = max(0.0, min(ea, eb) - max(sa, sb))
            print(f"  overlap stage{a}-stage{b}: {ov:.2f} ms")

    ax.set_yticks([y_for_stage[s] for s in stages])
    ax.set_yticklabels([f"stage {s}" for s in stages])
    ax.set_xlabel("Seconds from first local execution in selected windows")
    ax.set_title(title or f"BP-free audit Gantt: windows {min(windows)}-{max(windows)}")
    ax.grid(axis="x", alpha=0.25)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)

    print(f"\nWrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, action="append", required=True)
    parser.add_argument("--windows", nargs=2, type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    events = load_events(args.csv)
    plot(events, set(args.windows), args.output, args.title)


if __name__ == "__main__":
    main()
