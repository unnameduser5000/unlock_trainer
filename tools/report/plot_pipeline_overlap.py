#!/usr/bin/env python3
"""Plot stage-level pipeline overlap from coordinator stage-timings.csv."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Interval:
    request_id: str
    request_suffix: str
    stage_id: int
    event_epoch_ms: int
    duration_ms: int
    local_ms: int
    execute_ms: int

    @property
    def start_ms(self) -> int:
        return self.event_epoch_ms - self.duration_ms

    @property
    def end_ms(self) -> int:
        return self.event_epoch_ms

    @property
    def queue_wait_ms(self) -> int:
        return max(0, self.local_ms - self.duration_ms)


def read_intervals(path: Path) -> list[Interval]:
    intervals: list[Interval] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("event_type") != "LOCAL_COMPLETED":
                continue
            request_id = row["request_id"]
            suffix = request_id.rsplit("-", 1)[-1]
            local_ms = parse_int(row.get("local_ms"))
            total_measured_ms = parse_int(row.get("total_measured_ms"))
            duration_ms = total_measured_ms or local_ms
            intervals.append(
                Interval(
                    request_id=request_id,
                    request_suffix=suffix,
                    stage_id=parse_int(row.get("stage_id")),
                    event_epoch_ms=parse_int(row.get("event_epoch_ms")),
                    duration_ms=duration_ms,
                    local_ms=local_ms,
                    execute_ms=parse_int(row.get("execute_ms")),
                )
            )
    return sorted(intervals, key=lambda item: (item.start_ms, item.stage_id, item.request_id))


def parse_int(value: str | None) -> int:
    if value is None or value == "":
        return 0
    return int(float(value))


def compute_overlaps(intervals: list[Interval]) -> list[tuple[Interval, Interval, int]]:
    overlaps: list[tuple[Interval, Interval, int]] = []
    for i, left in enumerate(intervals):
        for right in intervals[i + 1 :]:
            if left.stage_id == right.stage_id or left.request_id == right.request_id:
                continue
            overlap_ms = min(left.end_ms, right.end_ms) - max(left.start_ms, right.start_ms)
            if overlap_ms > 0:
                overlaps.append((left, right, overlap_ms))
    return sorted(overlaps, key=lambda item: item[2], reverse=True)


def write_intervals(path: Path, intervals: list[Interval]) -> None:
    base = min((item.start_ms for item in intervals), default=0)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "request_suffix",
                "request_id",
                "stage_id",
                "start_s",
                "end_s",
                "duration_ms",
                "execute_ms",
                "queue_wait_ms",
            ]
        )
        for item in intervals:
            writer.writerow(
                [
                    item.request_suffix,
                    item.request_id,
                    item.stage_id,
                    round((item.start_ms - base) / 1000.0, 3),
                    round((item.end_ms - base) / 1000.0, 3),
                    item.duration_ms,
                    item.execute_ms,
                    item.queue_wait_ms,
                ]
            )


def write_summary(path: Path, intervals: list[Interval], overlaps: list[tuple[Interval, Interval, int]]) -> None:
    lines = [
        f"local_completed_intervals={len(intervals)}",
        f"overlap_pair_count={len(overlaps)}",
        f"max_overlap_ms={overlaps[0][2] if overlaps else 0}",
        "",
        "Top overlaps:",
    ]
    for left, right, overlap_ms in overlaps[:12]:
        lines.append(
            f"{left.request_suffix}:stage{left.stage_id} overlaps "
            f"{right.request_suffix}:stage{right.stage_id} by {overlap_ms} ms"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_gantt(path: Path, intervals: list[Interval], title: str) -> None:
    if not intervals:
        return
    base = min(item.start_ms for item in intervals)
    request_suffixes = sorted({item.request_suffix for item in intervals})
    palette = plt.get_cmap("tab10")
    colors = {suffix: palette(i % 10) for i, suffix in enumerate(request_suffixes)}

    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    for item in intervals:
        start_s = (item.start_ms - base) / 1000.0
        duration_s = item.duration_ms / 1000.0
        y = item.stage_id * 10
        ax.broken_barh(
            [(start_s, duration_s)],
            (y, 6),
            facecolors=colors[item.request_suffix],
            edgecolors="#111827",
            linewidth=0.7,
            alpha=0.88,
        )
        if duration_s >= 2.0:
            ax.text(
                start_s + duration_s / 2,
                y + 3,
                item.request_suffix,
                ha="center",
                va="center",
                fontsize=8,
                color="#111827",
            )

    stages = sorted({item.stage_id for item in intervals})
    ax.set_yticks([stage * 10 + 3 for stage in stages])
    ax.set_yticklabels([f"stage {stage}" for stage in stages])
    ax.set_xlabel("Seconds from first local execution")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[suffix], label=suffix)
        for suffix in request_suffixes
    ]
    ax.legend(handles=handles, title="request", ncol=min(6, len(handles)), frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage_timings_csv", type=Path)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--title", default="Pipeline overlap from coordinator-observed stage timings")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.stage_timings_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    intervals = read_intervals(args.stage_timings_csv)
    overlaps = compute_overlaps(intervals)
    write_intervals(output_dir / "pipeline_overlap_intervals.csv", intervals)
    write_summary(output_dir / "pipeline_overlap_summary.txt", intervals, overlaps)
    plot_gantt(output_dir / "pipeline_overlap_gantt.png", intervals, args.title)
    print(f"Wrote {output_dir / 'pipeline_overlap_gantt.png'}")
    print(f"Wrote {output_dir / 'pipeline_overlap_summary.txt'}")


if __name__ == "__main__":
    main()
