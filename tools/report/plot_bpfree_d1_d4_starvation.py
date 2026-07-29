#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt


KEEP = {
    "LOAD_STAGE0_HIDDEN",
    "LOAD_COMMON_INPUTS",
    "FWD_RECV_POST",
    "FWD_RECV_WAIT",
    "BODY_FORWARD",
    "FWD_SEND_POST",
    "FWD_SEND_WAIT",
    "FWD_SEND_WAIT_FINAL_DRAIN",
    "LOCAL_HEAD_LOSS",
    "LOCAL_BACKWARD",
    "LOCAL_OPTIMIZER_STEP",
}

SHORT = {
    "LOAD_STAGE0_HIDDEN": "LOAD_H",
    "LOAD_COMMON_INPUTS": "LOAD_IN",
    "FWD_RECV_POST": "RPOST",
    "FWD_RECV_WAIT": "RWAIT",
    "BODY_FORWARD": "BODY",
    "FWD_SEND_POST": "SEND",
    "FWD_SEND_WAIT": "SWAIT",
    "FWD_SEND_WAIT_FINAL_DRAIN": "SFINAL",
    "LOCAL_HEAD_LOSS": "HEAD",
    "LOCAL_BACKWARD": "BWD",
    "LOCAL_OPTIMIZER_STEP": "OPT",
}


def load_events(root: Path, windows: set[int]):
    rows = []
    for p in sorted(root.glob("train.stage*.actions.csv")):
        with p.open() as f:
            for r in csv.DictReader(f):
                if r.get("phase") != "train":
                    continue
                action = r["action"]
                if action.startswith("DEBUG"):
                    continue
                if action not in KEEP:
                    continue
                win = int(r["window_id"])
                if win not in windows:
                    continue
                start = float(r["start_epoch_ms"])
                dur = float(r["duration_ms"])
                rows.append({
                    "stage": int(r["stage_id"]),
                    "win": win,
                    "mb": int(r["mb_id"]),
                    "seq": int(r["seq_start"]),
                    "action": action,
                    "start": start,
                    "end": start + dur,
                    "dur": dur,
                })

    if not rows:
        raise RuntimeError(f"No action rows loaded from {root}")

    t0 = min(r["start"] for r in rows)
    for r in rows:
        r["start"] -= t0
        r["end"] -= t0

    return rows


def plot_one(ax, rows, title):
    by_lane = defaultdict(list)
    for r in rows:
        # lane: stage plus physical microbatch seq, so staggered bars are readable
        lane = (r["stage"], r["win"], r["mb"], r["seq"])
        by_lane[lane].append(r)

    lanes = sorted(by_lane.keys())
    y_for = {lane: i for i, lane in enumerate(lanes)}

    for lane, rs in by_lane.items():
        y = y_for[lane]
        for r in sorted(rs, key=lambda x: x["start"]):
            ax.broken_barh(
                [(r["start"], max(0.05, r["end"] - r["start"]))],
                (y - 0.35, 0.7),
            )
            if r["dur"] > 1.0:
                ax.text(
                    r["start"],
                    y,
                    SHORT.get(r["action"], r["action"]),
                    va="center",
                    ha="left",
                    fontsize=6,
                )

    labels = [
        f"S{stage} W{win} mb{mb} seq{seq}"
        for stage, win, mb, seq in lanes
    ]

    ax.set_yticks(range(len(lanes)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("relative time ms")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d1", required=True)
    ap.add_argument("--d4", required=True)
    ap.add_argument("--windows", nargs="+", type=int, default=[2, 3, 4, 5])
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    windows = set(args.windows)
    d1 = load_events(Path(args.d1), windows)
    d4 = load_events(Path(args.d4), windows)

    fig, axes = plt.subplots(2, 1, figsize=(22, 16), sharex=False)
    plot_one(axes[0], d1, f"D1 recv_inflight_depth=1 windows={sorted(windows)}")
    plot_one(axes[1], d4, f"D4_event recv_inflight_depth=4 windows={sorted(windows)}")

    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    print(out)


if __name__ == "__main__":
    main()