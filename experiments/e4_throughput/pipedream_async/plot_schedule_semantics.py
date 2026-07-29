#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


METHODS = [
    ("exactbp_gpipe", "Synchronous GPipe"),
    ("exactbp_1f1b", "Synchronous 1F1B (flush per window)"),
    ("pipedream", "PipeDream (continuous across windows)"),
]
COLORS = {"F": "#3973C6", "B": "#E88632", "U": "#333333"}


def _kind(action: str) -> str | None:
    if "FORWARD" in action:
        return "F"
    if "BACKWARD" in action:
        return "B"
    if action in {"OPTIMIZER_STEP", "OPTIMIZER_STEP_ASYNC"}:
        return "U"
    return None


def _read_method(root: Path, method: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage in range(3):
        path = root / method / f"train.stage{stage}.actions.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            for raw in csv.DictReader(handle):
                kind = _kind(raw["action"])
                if kind is None:
                    continue
                rows.append(
                    {
                        "stage": stage,
                        "kind": kind,
                        "seq": int(raw["seq_start"]),
                        "start": float(raw["start_epoch_ms"]),
                        "end": float(raw["end_epoch_ms"]),
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    method_rows = {method: _read_method(args.input_root, method) for method, _ in METHODS}
    spans = {
        method: max(row["end"] for row in rows) - min(row["start"] for row in rows)
        for method, rows in method_rows.items()
    }
    max_span = max(spans.values())

    fig, axes = plt.subplots(3, 1, figsize=(13.2, 8.4), sharex=True)
    for ax, (method, title) in zip(axes, METHODS):
        rows = method_rows[method]
        origin = min(row["start"] for row in rows)
        for row in rows:
            start = row["start"] - origin
            width = max(3.0 if row["kind"] == "U" else 0.4, row["end"] - row["start"])
            height = 0.62 if row["kind"] != "U" else 0.22
            y = row["stage"] - height / 2
            ax.broken_barh(
                [(start, width)],
                (y, height),
                facecolors=COLORS[row["kind"]],
                edgecolors="white",
                linewidth=0.35,
            )
            if row["kind"] in {"F", "B"} and width >= 8:
                ax.text(
                    start + width / 2,
                    row["stage"],
                    f"{row['kind']}{row['seq']}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white",
                    clip_on=True,
                )
        ax.set_yticks([0, 1, 2], ["S0", "S1", "S2"])
        ax.invert_yaxis()
        ax.set_xlim(0, max_span * 1.01)
        ax.set_title(f"{title}  (trace span {spans[method]:.0f} ms)", fontsize=11, loc="left")
        ax.grid(axis="x", color="#D6D6D6", linewidth=0.6, alpha=0.65)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    axes[-1].set_xlabel("Time from first recorded compute action (ms)")
    fig.legend(
        handles=[
            Patch(facecolor=COLORS["F"], label="Forward"),
            Patch(facecolor=COLORS["B"], label="Backward"),
            Patch(facecolor=COLORS["U"], label="Optimizer update"),
        ],
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.suptitle(
        "Steady schedule semantics (windows 1-2, b=1, m=4, three colocated ranks)",
        fontsize=14,
        y=1.045,
    )
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(args.output_dir / f"schedule_semantics.{suffix}", dpi=220, bbox_inches="tight")
    print(args.output_dir / "schedule_semantics.png")


if __name__ == "__main__":
    main()
