#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def is_gpu_event(e: dict) -> bool:
    if e.get("ph") != "X":
        return False

    dur = float(e.get("dur", 0) or 0)
    if dur <= 0:
        return False

    cat = str(e.get("cat", "")).lower()
    name = str(e.get("name", "")).lower()

    if "kernel" in cat:
        return True
    if "gpu_memcpy" in cat or "gpu_memset" in cat:
        return True
    if "memcpy" in name or "memset" in name:
        return True
    if "nccl" in name:
        return True
    if "void" in name and ("kernel" in cat or "cuda" in cat):
        return True

    return False


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []

    intervals = sorted(intervals)
    merged = [intervals[0]]

    for s, e in intervals[1:]:
        ps, pe = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))

    return merged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    args = ap.parse_args()

    paths = sorted(args.root.rglob("*.trace.json")) + sorted(args.root.rglob("*.pt.trace.json"))

    if not paths:
        raise SystemExit(f"No trace json found under {args.root}")

    for path in paths:
        data = json.loads(path.read_text())
        events = data.get("traceEvents", data if isinstance(data, list) else [])

        intervals = []
        for e in events:
            if not isinstance(e, dict):
                continue
            if not is_gpu_event(e):
                continue

            ts = float(e.get("ts", 0) or 0)
            dur = float(e.get("dur", 0) or 0)
            intervals.append((ts, ts + dur))

        if not intervals:
            print(f"\n{path}")
            print("  no GPU events found")
            continue

        merged = merge_intervals(intervals)
        span_us = max(e for _, e in intervals) - min(s for s, _ in intervals)
        busy_us = sum(e - s for s, e in merged)
        idle_us = max(0.0, span_us - busy_us)

        print(f"\n{path}")
        print(f"  gpu_event_intervals: {len(intervals)}")
        print(f"  merged_busy_regions: {len(merged)}")
        print(f"  span_ms: {span_us / 1000.0:.2f}")
        print(f"  busy_ms: {busy_us / 1000.0:.2f}")
        print(f"  idle_ms: {idle_us / 1000.0:.2f}")
        print(f"  busy_ratio: {busy_us / span_us:.2%}")
        print(f"  idle_ratio: {idle_us / span_us:.2%}")

        print("  largest idle gaps:")
        gaps = []
        for (_, prev_e), (next_s, _) in zip(merged, merged[1:]):
            gap = next_s - prev_e
            if gap > 0:
                gaps.append((gap, prev_e, next_s))

        for gap, s, e in sorted(gaps, reverse=True)[:10]:
            print(f"    gap_ms={gap / 1000.0:8.2f}")


if __name__ == "__main__":
    main()
