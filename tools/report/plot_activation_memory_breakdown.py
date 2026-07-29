#!/usr/bin/env python3
"""Plot activation-memory breakdowns for BP-free and 1F1B smoke/formal runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MIB = 1024.0 * 1024.0


def numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def aggregate_bpfree(path: Path, label: str) -> pd.DataFrame:
    metrics = pd.read_csv(path / "scheduler_stage_metrics.csv")
    numeric(
        metrics,
        [
            "stage_id",
            "input_state_bytes",
            "output_state_bytes",
            "output_log_probs_bytes",
            "local_param_bytes",
            "local_trainable_param_bytes",
            "optimizer_state_bytes",
            "cuda_peak_memory_allocated",
            "autograd_saved_cuda_nonleaf_unique_bytes_peak",
            "autograd_saved_cuda_leaf_unique_bytes_peak",
            "autograd_saved_cuda_nonleaf_bytes_peak",
            "autograd_saved_cuda_leaf_bytes_peak",
            "autograd_saved_cuda_nonleaf_unique_hidden_bytes_peak",
            "autograd_saved_cuda_nonleaf_unique_vocab_bytes_peak",
            "autograd_saved_cuda_nonleaf_unique_attention_bytes_peak",
            "autograd_saved_cuda_nonleaf_unique_other_bytes_peak",
        ],
    )
    rows: list[dict[str, float | int | str]] = []
    for stage_id, group in metrics.groupby("stage_id"):
        rows.append(
            {
                "policy": label,
                "runner": "bpfree",
                "stage_id": int(stage_id),
                "rows": int(len(group)),
                "cuda_peak_mib": group["cuda_peak_memory_allocated"].max() / MIB,
                "local_param_mib": max_bytes(group, "local_param_bytes") / MIB,
                "local_trainable_param_mib": max_bytes(group, "local_trainable_param_bytes") / MIB,
                "optimizer_state_mib": max_bytes(group, "optimizer_state_bytes") / MIB,
                "saved_nonleaf_unique_mib": group["autograd_saved_cuda_nonleaf_unique_bytes_peak"].max() / MIB,
                "saved_leaf_unique_mib": group["autograd_saved_cuda_leaf_unique_bytes_peak"].max() / MIB,
                "saved_nonleaf_logical_mib": group["autograd_saved_cuda_nonleaf_bytes_peak"].max() / MIB,
                "saved_leaf_logical_mib": group["autograd_saved_cuda_leaf_bytes_peak"].max() / MIB,
                "saved_nonleaf_hidden_mib": max_bytes(group, "autograd_saved_cuda_nonleaf_unique_hidden_bytes_peak") / MIB,
                "saved_nonleaf_vocab_mib": max_bytes(group, "autograd_saved_cuda_nonleaf_unique_vocab_bytes_peak") / MIB,
                "saved_nonleaf_attention_mib": max_bytes(group, "autograd_saved_cuda_nonleaf_unique_attention_bytes_peak") / MIB,
                "saved_nonleaf_other_mib": max_bytes(group, "autograd_saved_cuda_nonleaf_unique_other_bytes_peak") / MIB,
                "input_payload_mib": group["input_state_bytes"].max() / MIB,
                "output_hidden_payload_mib": group["output_state_bytes"].max() / MIB,
                "output_log_probs_payload_mib": group["output_log_probs_bytes"].max() / MIB,
            }
        )
    return pd.DataFrame(rows)


def aggregate_1f1b(path: Path, label: str) -> pd.DataFrame:
    metrics = pd.read_csv(path / "stage_metrics.csv")
    if "phase" in metrics.columns:
        metrics = metrics[metrics["phase"] == "train"].copy()
    numeric(
        metrics,
        [
            "stage_id",
            "records",
            "microbatches",
            "local_param_bytes",
            "local_trainable_param_bytes",
            "optimizer_state_bytes",
            "cuda_peak_memory_allocated",
            "autograd_saved_cuda_nonleaf_unique_bytes_peak",
            "autograd_saved_cuda_leaf_unique_bytes_peak",
            "autograd_saved_cuda_nonleaf_bytes_peak",
            "autograd_saved_cuda_leaf_bytes_peak",
            "autograd_saved_cuda_nonleaf_unique_hidden_bytes_peak",
            "autograd_saved_cuda_nonleaf_unique_vocab_bytes_peak",
            "autograd_saved_cuda_nonleaf_unique_attention_bytes_peak",
            "autograd_saved_cuda_nonleaf_unique_other_bytes_peak",
        ],
    )
    rows: list[dict[str, float | int | str]] = []
    for stage_id, group in metrics.groupby("stage_id"):
        rows.append(
            {
                "policy": label,
                "runner": "1f1b",
                "stage_id": int(stage_id),
                "rows": int(len(group)),
                "microbatches": int(group["microbatches"].dropna().iloc[0]) if "microbatches" in group else np.nan,
                "records_per_step": int(group["records"].dropna().iloc[0]) if "records" in group else np.nan,
                "cuda_peak_mib": group["cuda_peak_memory_allocated"].max() / MIB,
                "local_param_mib": max_bytes(group, "local_param_bytes") / MIB,
                "local_trainable_param_mib": max_bytes(group, "local_trainable_param_bytes") / MIB,
                "optimizer_state_mib": max_bytes(group, "optimizer_state_bytes") / MIB,
                "saved_nonleaf_unique_mib": group["autograd_saved_cuda_nonleaf_unique_bytes_peak"].max() / MIB,
                "saved_leaf_unique_mib": group["autograd_saved_cuda_leaf_unique_bytes_peak"].max() / MIB,
                "saved_nonleaf_logical_mib": group["autograd_saved_cuda_nonleaf_bytes_peak"].max() / MIB,
                "saved_leaf_logical_mib": group["autograd_saved_cuda_leaf_bytes_peak"].max() / MIB,
                "saved_nonleaf_hidden_mib": max_bytes(group, "autograd_saved_cuda_nonleaf_unique_hidden_bytes_peak") / MIB,
                "saved_nonleaf_vocab_mib": max_bytes(group, "autograd_saved_cuda_nonleaf_unique_vocab_bytes_peak") / MIB,
                "saved_nonleaf_attention_mib": max_bytes(group, "autograd_saved_cuda_nonleaf_unique_attention_bytes_peak") / MIB,
                "saved_nonleaf_other_mib": max_bytes(group, "autograd_saved_cuda_nonleaf_unique_other_bytes_peak") / MIB,
                "input_payload_mib": np.nan,
                "output_hidden_payload_mib": np.nan,
                "output_log_probs_payload_mib": np.nan,
            }
        )
    return pd.DataFrame(rows)


def max_bytes(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns:
        return 0.0
    return float(df[column].max() or 0.0)


def add_residual(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cuda_other_residual_mib"] = (
        df["cuda_peak_mib"] - df["saved_nonleaf_unique_mib"] - df["saved_leaf_unique_mib"]
    ).clip(lower=0)
    return df


def write_csv(df: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "activation_memory_breakdown.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def plot_nonleaf_categories(df: pd.DataFrame, output_dir: Path) -> Path | None:
    category_columns = [
        "saved_nonleaf_hidden_mib",
        "saved_nonleaf_vocab_mib",
        "saved_nonleaf_attention_mib",
        "saved_nonleaf_other_mib",
    ]
    if not set(category_columns).issubset(df.columns):
        return None
    if float(df[category_columns].sum().sum()) <= 0.0:
        return None

    labels = [f"{row.policy}\nstage {int(row.stage_id)}" for row in df.itertuples()]
    x = np.arange(len(df))
    width = 0.72
    bottom = np.zeros(len(df))
    series = [
        ("hidden-state-like saved tensors", "saved_nonleaf_hidden_mib", "#2f80ed"),
        ("vocab logits/log-probs saved tensors", "saved_nonleaf_vocab_mib", "#bb6bd9"),
        ("attention-like saved tensors", "saved_nonleaf_attention_mib", "#27ae60"),
        ("other non-leaf saved tensors", "saved_nonleaf_other_mib", "#f2994a"),
    ]

    plt.figure(figsize=(max(12, len(df) * 0.9), 5.8))
    for label, column, color in series:
        values = df[column].fillna(0).to_numpy()
        plt.bar(x, values, width=width, bottom=bottom, color=color, label=label)
        bottom += values
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel("Unique CUDA storage peak (MiB)")
    plt.title("Non-leaf saved tensor peak by shape category")
    plt.legend(loc="upper left", frameon=False)
    path = output_dir / "activation_nonleaf_category_breakdown.png"
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def plot_saved_unique(df: pd.DataFrame, output_dir: Path) -> Path:
    labels = [f"{row.policy}\nstage {int(row.stage_id)}" for row in df.itertuples()]
    x = np.arange(len(df))
    width = 0.72
    nonleaf = df["saved_nonleaf_unique_mib"].to_numpy()
    leaf = df["saved_leaf_unique_mib"].to_numpy()

    plt.figure(figsize=(max(12, len(df) * 0.9), 5.8))
    plt.bar(x, nonleaf, width=width, color="#2f80ed", label="non-leaf saved tensors: activations/logits")
    plt.bar(x, leaf, width=width, bottom=nonleaf, color="#9aa0a6", label="leaf saved tensors: params/weights saved by backward")
    for index, value in enumerate(nonleaf):
        plt.text(index, value / 2, f"{value:.0f}", ha="center", va="center", color="white", fontsize=8)
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel("Unique CUDA storage peak (MiB)")
    plt.title("Autograd saved tensor peak, split by tensor type")
    plt.legend(loc="upper left", frameon=False)
    path = output_dir / "activation_saved_unique_breakdown.png"
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def plot_cuda_residual(df: pd.DataFrame, output_dir: Path) -> Path:
    labels = [f"{row.policy}\nstage {int(row.stage_id)}" for row in df.itertuples()]
    x = np.arange(len(df))
    width = 0.72
    nonleaf = df["saved_nonleaf_unique_mib"].to_numpy()
    leaf = df["saved_leaf_unique_mib"].to_numpy()
    other = df["cuda_other_residual_mib"].to_numpy()

    plt.figure(figsize=(max(12, len(df) * 0.9), 5.8))
    plt.bar(x, nonleaf, width=width, color="#2f80ed", label="saved non-leaf activation/logit storage")
    plt.bar(x, leaf, width=width, bottom=nonleaf, color="#9aa0a6", label="saved leaf param/weight storage")
    plt.bar(x, other, width=width, bottom=nonleaf + leaf, color="#f2994a", label="other CUDA allocation at peak")
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel("MiB")
    plt.title("CUDA peak allocated, with measured autograd-saved components")
    plt.legend(loc="upper left", frameon=False)
    path = output_dir / "activation_cuda_peak_breakdown.png"
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def plot_payloads(df: pd.DataFrame, output_dir: Path) -> Path:
    payload = df[df["runner"] == "bpfree"].copy()
    if payload.empty:
        return output_dir / "activation_bpfree_payloads.png"
    labels = [f"{row.policy}\nstage {int(row.stage_id)}" for row in payload.itertuples()]
    x = np.arange(len(payload))
    width = 0.72
    input_payload = payload["input_payload_mib"].fillna(0).to_numpy()
    output_hidden = payload["output_hidden_payload_mib"].fillna(0).to_numpy()
    output_log_probs = payload["output_log_probs_payload_mib"].fillna(0).to_numpy()

    plt.figure(figsize=(max(8, len(payload) * 1.1), 4.8))
    plt.bar(x, input_payload, width=width, color="#6fcf97", label="input hidden/mask/labels payload")
    plt.bar(x, output_hidden, width=width, bottom=input_payload, color="#56ccf2", label="output hidden boundary payload")
    plt.bar(
        x,
        output_log_probs,
        width=width,
        bottom=input_payload + output_hidden,
        color="#bb6bd9",
        label="output log-probs payload",
    )
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("MiB per request")
    plt.title("BP-free boundary/output payload size")
    plt.legend(loc="upper left", frameon=False)
    path = output_dir / "activation_bpfree_payloads.png"
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def plot_trainable_state(df: pd.DataFrame, output_dir: Path) -> Path | None:
    required = {"local_trainable_param_mib", "optimizer_state_mib", "saved_nonleaf_unique_mib"}
    if not required.issubset(df.columns):
        return None
    if float(df[["local_trainable_param_mib", "optimizer_state_mib"]].fillna(0).sum().sum()) <= 0.0:
        return None

    labels = [f"{row.policy}\nstage {int(row.stage_id)}" for row in df.itertuples()]
    x = np.arange(len(df))
    width = 0.72
    trainable = df["local_trainable_param_mib"].fillna(0).to_numpy()
    optimizer = df["optimizer_state_mib"].fillna(0).to_numpy()
    activation = df["saved_nonleaf_unique_mib"].fillna(0).to_numpy()

    plt.figure(figsize=(max(12, len(df) * 0.9), 5.8))
    plt.bar(x, trainable, width=width, color="#607d8b", label="trainable parameter storage")
    plt.bar(x, optimizer, width=width, bottom=trainable, color="#ffb74d", label="optimizer state storage")
    plt.bar(x, activation, width=width, bottom=trainable + optimizer, color="#2f80ed", label="saved non-leaf activation/logit storage")
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel("MiB")
    plt.title("Trainable state and activation stash by stage")
    plt.legend(loc="upper left", frameon=False)
    path = output_dir / "trainable_state_vs_activation.png"
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def parse_label_path(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        path = Path(raw)
        return path.name, path
    label, path = raw.split("=", 1)
    return label, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bpfree", action="append", default=[], help="label=run_dir containing scheduler_stage_metrics.csv")
    parser.add_argument("--onef1b", action="append", default=[], help="label=run_dir containing stage_metrics.csv")
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames: list[pd.DataFrame] = []
    for raw in args.bpfree:
        label, path = parse_label_path(raw)
        frames.append(aggregate_bpfree(path, label))
    for raw in args.onef1b:
        label, path = parse_label_path(raw)
        frames.append(aggregate_1f1b(path, label))
    if not frames:
        raise SystemExit("No inputs supplied.")
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["runner", "policy", "stage_id"]).reset_index(drop=True)
    df = add_residual(df)
    csv_path = write_csv(df, args.output_dir)
    category_path = plot_nonleaf_categories(df, args.output_dir)
    saved_path = plot_saved_unique(df, args.output_dir)
    cuda_path = plot_cuda_residual(df, args.output_dir)
    payload_path = plot_payloads(df, args.output_dir)
    trainable_path = plot_trainable_state(df, args.output_dir)
    print(f"Wrote {csv_path}")
    if category_path is not None:
        print(f"Wrote {category_path}")
    print(f"Wrote {saved_path}")
    print(f"Wrote {cuda_path}")
    print(f"Wrote {payload_path}")
    if trainable_path is not None:
        print(f"Wrote {trainable_path}")


if __name__ == "__main__":
    main()
