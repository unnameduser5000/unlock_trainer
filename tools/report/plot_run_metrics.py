#!/usr/bin/env python3
"""Plot per-request metrics for a single coordinator run.

The input CSV can be either a terminal results CSV or a coordinator metrics CSV.
Successful rows are used for the main curves. If the input includes failed
attempts, they are counted in the summary but excluded from the success curves.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
        }
    )


def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in [
        "request_index",
        "record_index",
        "submission_seq",
        "attempt",
        "elapsed_ms",
        "local_loss",
        "token_correct",
        "token_count",
        "token_accuracy",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "success" in df.columns:
        df["success"] = df["success"].astype(str).str.lower().eq("true")
    if "row_idx" not in df.columns:
        df["row_idx"] = np.arange(len(df))
    return df


def rolling(series: pd.Series, window: int = 25) -> pd.Series:
    if len(series) < 3:
        return series
    effective = min(window, max(3, len(series) // 8))
    return series.rolling(effective, min_periods=1).mean()


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Wrote {path}")


def x_axis(df: pd.DataFrame) -> tuple[pd.Series, str]:
    for col in ["submission_seq", "request_index", "record_index"]:
        if col in df.columns and df[col].notna().any():
            return df[col], col
    return df["row_idx"], "row_idx"


def summarize(df: pd.DataFrame) -> str:
    success = df[df["success"]] if "success" in df.columns else df
    lines = [
        f"rows={len(df)}",
        f"success_rows={len(success)}",
    ]
    if "success" in df.columns:
        lines.append(f"failed_rows={int((~df['success']).sum())}")
    for col in ["elapsed_ms", "local_loss", "token_accuracy"]:
        if col in success.columns and len(success):
            val = float(success[col].mean())
            lines.append(f"avg_{col}={val}")
    if "token_correct" in success.columns and "token_count" in success.columns and len(success):
        denom = pd.to_numeric(success["token_count"], errors="coerce").fillna(0).sum()
        num = pd.to_numeric(success["token_correct"], errors="coerce").fillna(0).sum()
        if denom > 0:
            lines.append(f"weighted_token_accuracy={num / denom}")
    if "local_loss" in success.columns and len(success):
        lines.append(f"loss_min={float(success['local_loss'].min())}")
        lines.append(f"loss_max={float(success['local_loss'].max())}")
    if "elapsed_ms" in success.columns and len(success):
        lines.append(f"elapsed_p95_ms={float(success['elapsed_ms'].quantile(0.95))}")
    return "\n".join(lines) + "\n"


def plot_run_metrics(df: pd.DataFrame, out_dir: Path, title: str) -> None:
    if "success" in df.columns:
        ok = df[df["success"]].copy()
    else:
        ok = df.copy()
    if ok.empty:
        raise SystemExit("No successful rows found in input CSV.")

    x, x_label = x_axis(ok)
    x = pd.to_numeric(x, errors="coerce")
    x = x if x.notna().any() else pd.Series(np.arange(len(ok)), index=ok.index)

    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    ax.plot(x, ok["local_loss"], color="#9CA3AF", linewidth=0.9, alpha=0.55, label="per request")
    ax.plot(x, rolling(ok["local_loss"]), color="#2563EB", linewidth=2.0, label="rolling mean")
    ax.set_title(f"{title} terminal loss")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Terminal local loss")
    ax.legend(frameon=False)
    savefig(out_dir / "training_loss_curve.png")

    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    ax.plot(x, ok["elapsed_ms"] / 1000.0, color="#9CA3AF", linewidth=0.9, alpha=0.55, label="per request")
    ax.plot(x, rolling(ok["elapsed_ms"]) / 1000.0, color="#059669", linewidth=2.0, label="rolling mean")
    ax.set_title(f"{title} latency")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Elapsed time (s)")
    ax.legend(frameon=False)
    savefig(out_dir / "training_latency_curve.png")

    if "token_count" in ok.columns and "token_correct" in ok.columns:
        cumulative_correct = ok["token_correct"].fillna(0).cumsum()
        cumulative_tokens = ok["token_count"].fillna(0).cumsum().replace(0, np.nan)
        fig, ax = plt.subplots(figsize=(9.5, 4.4))
        ax.plot(x, cumulative_correct / cumulative_tokens, color="#7C3AED", linewidth=2.0)
        ax.set_title(f"{title} cumulative token accuracy")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Cumulative token accuracy")
        ax.set_ylim(bottom=0)
        savefig(out_dir / "training_token_accuracy_curve.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="Run metrics or results CSV.")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--title", type=str, default="Run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = read_csv(args.csv)
    (args.output_dir / "run_metrics_summary.txt").write_text(summarize(df), encoding="utf-8")
    plot_run_metrics(df, args.output_dir, args.title)


if __name__ == "__main__":
    main()
