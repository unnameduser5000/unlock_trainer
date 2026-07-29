#!/usr/bin/env python3
"""Build the current E4 formal figures from the completed PipeDream matrix."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FORMAL_ROOTS = {
    "E4.1": "results/e4_throughput/raw/e4_1_pipedream_scaling_omp4_v2",
    "E4.2a": "results/e4_throughput/raw/e4_2a_pipedream_formal_omp4_v2",
    "E4.2b": "results/e4_throughput/raw/e4_2b_pipedream_formal_omp4_v2",
    "E4.3": "results/e4_throughput/raw/e4_3_pipedream_formal_omp4_v2",
}
CONFIGS = {
    "E4.1": "experiments/e4_throughput/configs/e4_1_scaling.json",
    "E4.2a": "experiments/e4_throughput/configs/e4_2a_batch_geometry.json",
    "E4.2b": "experiments/e4_throughput/configs/e4_2b_low_batch.json",
    "E4.3": "experiments/e4_throughput/configs/e4_3_network_sensitivity.json",
}
E44_BREAKDOWN = (
    "results/e4_throughput/analysis/e4_4_pipedream_trace_omp4_v2/"
    "case_breakdown.csv"
)

METHODS = ("bpfree", "exactbp_gpipe", "exactbp_1f1b", "pipedream")
METHOD_LABELS = {
    "bpfree": "BP-free",
    "exactbp_gpipe": "GPipe",
    "exactbp_1f1b": "1F1B",
    "pipedream": "PipeDream",
}
METHOD_COLORS = {
    "bpfree": "#2A9D5B",
    "exactbp_gpipe": "#4C78A8",
    "exactbp_1f1b": "#F28E2B",
    "pipedream": "#D95555",
}
METHOD_MARKERS = {
    "bpfree": "s",
    "exactbp_gpipe": "o",
    "exactbp_1f1b": "^",
    "pipedream": "D",
}
PROFILES = ("local", "wifi", "mobile", "constrained")
PROFILE_LABELS = {
    "local": "Local\n0 ms / uncapped",
    "wifi": "Wi-Fi\n2 ms / 1 Gb/s",
    "mobile": "Mobile\n10 ms / 200 Mb/s",
    "constrained": "Constrained\n30 ms / 50 Mb/s",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def formal_tables(repo_root: Path) -> tuple[dict[str, list[dict[str, str]]], dict[str, list[dict[str, str]]]]:
    summaries: dict[str, list[dict[str, str]]] = {}
    runs: dict[str, list[dict[str, str]]] = {}
    for experiment, relative in FORMAL_ROOTS.items():
        root = repo_root / relative
        summary_rows = read_csv(root / "summary.csv")
        summaries[experiment] = [
            row for row in summary_rows
            if not row["method"].startswith("ratio:")
        ]
        runs[experiment] = read_csv(root / "runs.csv")
    return summaries, runs


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def find_one(rows: Iterable[dict[str, str]], **filters: Any) -> dict[str, str]:
    found = [
        row for row in rows
        if all(str(row.get(key)) == str(value) for key, value in filters.items())
    ]
    if len(found) != 1:
        raise ValueError(f"expected one row for {filters}, found {len(found)}")
    return found[0]


def mean_key(row: dict[str, str]) -> str:
    if "mean_throughput_per_s" in row:
        return "mean_throughput_per_s"
    return "throughput_mean"


def std_key(row: dict[str, str]) -> str:
    if "std_throughput_per_s" in row:
        return "std_throughput_per_s"
    return "throughput_std"


def throughput(row: dict[str, str]) -> float:
    return as_float(row, mean_key(row))


def throughput_std(row: dict[str, str]) -> float:
    return as_float(row, std_key(row))


def paired_ratio(
    rows: list[dict[str, str]],
    *,
    numerator: str,
    denominator: str,
    filters: dict[str, Any],
) -> tuple[float, float, int]:
    selected = [
        row for row in rows
        if all(str(row.get(key)) == str(value) for key, value in filters.items())
    ]
    by_rep: dict[int, dict[str, float]] = defaultdict(dict)
    for row in selected:
        if row["method"] not in (numerator, denominator):
            continue
        by_rep[int(row["rep"])][row["method"]] = float(row["throughput_per_s"])
    values = [
        samples[numerator] / samples[denominator]
        for _, samples in sorted(by_rep.items())
        if numerator in samples and denominator in samples
    ]
    if not values:
        raise ValueError(
            f"no paired ratios for {numerator}/{denominator}, {filters}"
        )
    return (
        statistics.mean(values),
        statistics.stdev(values) if len(values) > 1 else 0.0,
        len(values),
    )


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.linewidth": 0.8,
            "lines.linewidth": 2.0,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def style_axis(ax: Any, *, grid_axis: str = "both") -> None:
    ax.grid(True, axis=grid_axis, color="#D9D9D9", linewidth=0.6, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_method(
    ax: Any,
    xs: list[float],
    ys: list[float],
    yerr: list[float] | None,
    method: str,
    *,
    label: str | None = None,
) -> None:
    ax.errorbar(
        xs,
        ys,
        yerr=yerr,
        label=label or METHOD_LABELS[method],
        color=METHOD_COLORS[method],
        marker=METHOD_MARKERS[method],
        markersize=6.5,
        markeredgecolor="white",
        markeredgewidth=0.7,
        capsize=3,
        linewidth=2.0,
        zorder=3,
    )


def save(fig: Any, output_dir: Path, stem: str, dpi: int) -> None:
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_e41(rows: list[dict[str, str]], out: Path, dpi: int) -> None:
    stages = (2, 3, 4)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))
    for method in METHODS:
        points = [
            find_one(rows, method=method, pipeline_stages=stage)
            for stage in stages
        ]
        means = [throughput(row) for row in points]
        stds = [throughput_std(row) for row in points]
        draw_method(axes[0], list(stages), means, stds, method)
        normalized = [value / means[0] for value in means]
        draw_method(axes[1], list(stages), normalized, None, method)

    axes[0].set_title("(a) Absolute throughput")
    axes[0].set_ylabel("Training throughput (requests/s)")
    axes[1].set_title("(b) Scaling relative to two stages")
    axes[1].set_ylabel("Throughput / method throughput at P=2")
    axes[1].axhline(1.0, color="#777777", linestyle="--", linewidth=1.0)
    for ax in axes:
        ax.set_xlabel("Pipeline stages / GPUs (P)")
        ax.set_xticks(stages)
        style_axis(ax)
    axes[0].legend(frameon=False, ncol=2, loc="upper left")
    fig.suptitle("E4.1 Stage-count scaling (b=8, m=4, B=32; mean +/- s.d., n=3)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, out, "e4_1_stage_scaling", dpi)


def plot_e42a(
    summary: list[dict[str, str]],
    runs: list[dict[str, str]],
    out: Path,
    dpi: int,
) -> None:
    cases = ((1, 32), (2, 16), (4, 8), (8, 4))
    xs = list(range(len(cases)))
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 6.8), sharex=True)
    for method in METHODS:
        points = [
            find_one(
                summary,
                method=method,
                physical_request_batch=b,
                microbatches_per_update=m,
            )
            for b, m in cases
        ]
        draw_method(
            axes[0], xs,
            [throughput(row) for row in points],
            [throughput_std(row) for row in points],
            method,
        )

    for method in METHODS[1:]:
        values = [
            paired_ratio(
                runs,
                numerator=method,
                denominator="bpfree",
                filters={"case": f"b{b}_m{m}"},
            )
            for b, m in cases
        ]
        draw_method(
            axes[1], xs,
            [item[0] for item in values],
            [item[1] for item in values],
            method,
            label=f"{METHOD_LABELS[method]} / BP-free",
        )

    axes[0].set_title("(a) Throughput at fixed effective batch")
    axes[0].set_ylabel("Training throughput (requests/s)")
    axes[0].legend(frameon=False, ncol=2)
    axes[1].set_title("(b) Paired ratio to BP-free at the same geometry")
    axes[1].set_ylabel("Baseline throughput / BP-free throughput")
    axes[1].axhline(1.0, color="#777777", linestyle="--", linewidth=1.0)
    axes[1].legend(frameon=False, ncol=3)
    axes[1].set_xticks(xs, [f"b={b}\nm={m}" for b, m in cases])
    axes[1].set_xlabel("Physical request batch b and microbatches m (B=b*m=32)")
    for ax in axes:
        style_axis(ax)
    fig.suptitle("E4.2a Batch geometry (mean +/- s.d., n=3)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, out, "e4_2a_batch_geometry", dpi)


def plot_e42b(
    summary: list[dict[str, str]],
    runs: list[dict[str, str]],
    out: Path,
    dpi: int,
) -> None:
    windows = (1, 2, 3, 4, 8, 16, 32)
    xs = list(range(len(windows)))
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 7.0), sharex=True)
    for method in METHODS:
        mapping = {
            int(row["microbatches_per_update"]): row
            for row in summary if row["method"] == method
        }
        available = [value for value in windows if value in mapping]
        points = [mapping[value] for value in available]
        method_xs = [windows.index(value) for value in available]
        draw_method(
            axes[0], method_xs,
            [throughput(row) for row in points],
            [throughput_std(row) for row in points],
            method,
        )

    for method in METHODS[1:]:
        available = windows if method == "exactbp_gpipe" else windows[2:]
        values = [
            paired_ratio(
                runs,
                numerator=method,
                denominator="bpfree",
                filters={"case": f"b1_m{window}"},
            )
            for window in available
        ]
        draw_method(
            axes[1], [windows.index(value) for value in available],
            [item[0] for item in values],
            [item[1] for item in values],
            method,
            label=f"{METHOD_LABELS[method]} / BP-free",
        )

    axes[0].set_title("(a) Absolute throughput")
    axes[0].set_ylabel("Training throughput (requests/s)")
    axes[0].legend(frameon=False, ncol=2)
    axes[1].set_title("(b) Paired ratio to BP-free")
    axes[1].set_ylabel("Baseline throughput / BP-free throughput")
    axes[1].axhline(1.0, color="#777777", linestyle="--", linewidth=1.0)
    axes[1].legend(frameon=False, ncol=3)
    axes[1].set_xticks(xs, [str(value) for value in windows])
    axes[1].set_xlabel("Microbatches per update m (physical batch b=1)")
    for ax in axes:
        style_axis(ax)
    fig.suptitle(
        "E4.2b Window sweep (mean +/- s.d., n=3; 1F1B/PipeDream require m>=P=3)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, out, "e4_2b_microbatch_sweep", dpi)


def plot_e43(
    summary: list[dict[str, str]],
    runs: list[dict[str, str]],
    out: Path,
    dpi: int,
) -> None:
    xs = list(range(len(PROFILES)))
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.5), sharex="col")

    for method in ("bpfree", "exactbp_gpipe"):
        points = [
            find_one(summary, profile=profile, regime="online_b1_m1", method=method)
            for profile in PROFILES
        ]
        draw_method(
            axes[0, 0], xs,
            [throughput(row) for row in points],
            [throughput_std(row) for row in points],
            method,
        )

    for method in METHODS:
        points = [
            find_one(
                summary,
                profile=profile,
                regime="throughput_b8_m4",
                method=method,
            )
            for profile in PROFILES
        ]
        means = [throughput(row) for row in points]
        draw_method(
            axes[0, 1], xs, means,
            [throughput_std(row) for row in points],
            method,
        )
        draw_method(
            axes[1, 1], xs,
            [100.0 * value / means[0] for value in means],
            None,
            method,
        )

    for method in METHODS[1:]:
        ratios = [
            paired_ratio(
                runs,
                numerator=method,
                denominator="bpfree",
                filters={"profile": profile, "regime": "throughput_b8_m4"},
            )
            for profile in PROFILES
        ]
        draw_method(
            axes[1, 0], xs,
            [item[0] for item in ratios],
            [item[1] for item in ratios],
            method,
            label=f"{METHOD_LABELS[method]} / BP-free",
        )

    axes[0, 0].set_title("(a) Online update: b=1, m=1")
    axes[0, 0].set_ylabel("Throughput (requests/s, log scale)")
    axes[0, 0].set_yscale("log")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].set_title("(b) Long stream: b=8, m=4, B=32")
    axes[0, 1].set_ylabel("Throughput (requests/s, log scale)")
    axes[0, 1].set_yscale("log")
    axes[0, 1].legend(frameon=False, ncol=2)
    axes[1, 0].set_title("(c) Paired ratio to BP-free for B=32")
    axes[1, 0].set_ylabel("Baseline throughput / BP-free throughput")
    axes[1, 0].axhline(1.0, color="#777777", linestyle="--", linewidth=1.0)
    axes[1, 0].legend(frameon=False, ncol=1)
    axes[1, 1].set_title("(d) Throughput retained from the local profile")
    axes[1, 1].set_ylabel("Local throughput retained (%)")
    axes[1, 1].legend(frameon=False, ncol=2)

    for ax in axes.flat:
        style_axis(ax)
    for ax in axes[1, :]:
        ax.set_xticks(xs, [PROFILE_LABELS[item] for item in PROFILES])
        ax.set_xlabel("Synthetic sender-side link profile")
    fig.suptitle("E4.3 Network sensitivity (mean +/- s.d., n=4)")
    fig.text(
        0.5,
        0.005,
        "Sender-side message pacing is a controlled sensitivity model, not a packet-level network emulator.",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    save(fig, out, "e4_3_network_sensitivity", dpi)


def plot_e44(rows: list[dict[str, str]], out: Path, dpi: int) -> None:
    profile_order = {"local": 0, "constrained": 1}
    method_order = {method: index for index, method in enumerate(METHODS)}
    rows = sorted(
        rows,
        key=lambda row: (profile_order[row["profile"]], method_order[row["method"]]),
    )
    categories = (
        ("Input / H2D", "#7AA6C2"),
        ("Forward + backward", "#2A9D5B"),
        ("Optimizer", "#F2C14E"),
        ("Receive wait", "#D95555"),
        ("Link pacing", "#8E6CBB"),
        ("Transfer / runtime", "#E78AC3"),
        ("Bookkeeping / other", "#A9A9A9"),
    )

    def components(row: dict[str, str]) -> list[float]:
        get = lambda name: float(row[f"critical_{name}_percent"])
        return [
            get("input_h2d"),
            get("forward_compute") + get("backward_compute"),
            get("optimizer"),
            get("transport_recv_wait"),
            get("link_pacing"),
            get("transport_d2h")
            + get("transport_recv_post")
            + get("transport_recv_h2d")
            + get("transport_send_post_runtime")
            + get("transport_send_wait"),
            get("weight_stash")
            + get("gradient_accumulation")
            + get("control")
            + get("untraced_idle"),
        ]

    xs = [0, 1, 2, 3, 5, 6, 7, 8]
    fig, ax = plt.subplots(figsize=(11.4, 5.2))
    bottoms = [0.0] * len(rows)
    for index, (label, color) in enumerate(categories):
        values = [components(row)[index] for row in rows]
        ax.bar(
            xs,
            values,
            bottom=bottoms,
            width=0.78,
            color=color,
            edgecolor="white",
            linewidth=0.4,
            label=label,
        )
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

    tick_labels = [
        f"{METHOD_LABELS[row['method']]}\n{float(row['throughput_per_s']):.1f} req/s"
        for row in rows
    ]
    ax.set_xticks(xs, tick_labels)
    ax.set_ylabel("Critical-stage trace span (%)")
    ax.set_ylim(0, 113)
    ax.axvline(4, color="#BBBBBB", linewidth=0.8)
    ax.text(1.5, 108, "Local", ha="center", fontweight="bold")
    ax.text(6.5, 108, "30 ms / 50 Mb/s", ha="center", fontweight="bold")
    for x, row in zip(xs, rows):
        ax.text(
            x,
            102.0,
            f"critical S{row['critical_stage_id']}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#555555",
        )
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    style_axis(ax, grid_axis="y")
    fig.suptitle("E4.4 Where critical-stage time is spent (synchronized diagnostic trace, n=1)")
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    save(fig, out, "e4_4_critical_stage_breakdown", dpi)


def consolidated_summary(
    summaries: dict[str, list[dict[str, str]]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for experiment, rows in summaries.items():
        for row in rows:
            output.append({"experiment": experiment, **row})
    return output


def key_result_rows(
    summaries: dict[str, list[dict[str, str]]]
) -> list[dict[str, Any]]:
    specs = [
        ("E4.1", "P=2, b=8, m=4", {"pipeline_stages": 2}),
        ("E4.1", "P=3, b=8, m=4", {"pipeline_stages": 3}),
        ("E4.1", "P=4, b=8, m=4", {"pipeline_stages": 4}),
        (
            "E4.2a",
            "b=1, m=32, B=32",
            {"physical_request_batch": 1, "microbatches_per_update": 32},
        ),
        (
            "E4.2a",
            "b=8, m=4, B=32",
            {"physical_request_batch": 8, "microbatches_per_update": 4},
        ),
        ("E4.2b", "b=1, m=1", {"microbatches_per_update": 1}),
        ("E4.2b", "b=1, m=8", {"microbatches_per_update": 8}),
        ("E4.2b", "b=1, m=32", {"microbatches_per_update": 32}),
        (
            "E4.3",
            "local, b=8, m=4",
            {"profile": "local", "regime": "throughput_b8_m4"},
        ),
        (
            "E4.3",
            "mobile, b=8, m=4",
            {"profile": "mobile", "regime": "throughput_b8_m4"},
        ),
        (
            "E4.3",
            "constrained, b=8, m=4",
            {"profile": "constrained", "regime": "throughput_b8_m4"},
        ),
    ]
    output: list[dict[str, Any]] = []
    for experiment, case, filters in specs:
        item: dict[str, Any] = {"experiment": experiment, "case": case}
        for method in METHODS:
            found = [
                row for row in summaries[experiment]
                if row["method"] == method
                and all(str(row.get(key)) == str(value) for key, value in filters.items())
            ]
            item[METHOD_LABELS[method]] = (
                f"{throughput(found[0]):.2f} +/- {throughput_std(found[0]):.2f}"
                if found else "--"
            )
        output.append(item)
    return output


def latex_key_results(rows: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Exp. & Case & BP-free & GPipe & 1F1B & PipeDream \\",
        r"\midrule",
    ]
    for row in rows:
        case = str(row["case"]).replace("=", r"{=}")
        values = [str(row[METHOD_LABELS[method]]).replace("+/-", r"$\pm$") for method in METHODS]
        lines.append(
            f"{row['experiment']} & {case} & " + " & ".join(values) + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_inputs(repo_root: Path, table_dir: Path) -> list[Path]:
    consumed: list[Path] = []
    for experiment, relative in FORMAL_ROOTS.items():
        root = repo_root / relative
        stem = experiment.lower().replace(".", "_")
        for filename in ("runs.csv", "summary.csv", "paired_summary.csv"):
            source = root / filename
            if source.is_file():
                shutil.copy2(source, table_dir / f"{stem}_{filename}")
                consumed.append(source)
    for relative in CONFIGS.values():
        consumed.append(repo_root / relative)
    consumed.append(repo_root / E44_BREAKDOWN)
    return consumed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="results/e4_throughput/paper_artifacts/e4_formal_v3_omp4",
        type=Path,
    )
    parser.add_argument("--dpi", default=300, type=int)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    figure_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    setup_style()
    summaries, runs = formal_tables(repo_root)
    e44_rows = read_csv(repo_root / E44_BREAKDOWN)

    plot_e41(summaries["E4.1"], figure_dir, args.dpi)
    plot_e42a(summaries["E4.2a"], runs["E4.2a"], figure_dir, args.dpi)
    plot_e42b(summaries["E4.2b"], runs["E4.2b"], figure_dir, args.dpi)
    plot_e43(summaries["E4.3"], runs["E4.3"], figure_dir, args.dpi)
    plot_e44(e44_rows, figure_dir, args.dpi)

    all_summary = consolidated_summary(summaries)
    write_csv(table_dir / "e4_formal_summary.csv", all_summary)
    key_rows = key_result_rows(summaries)
    write_csv(table_dir / "e4_key_results.csv", key_rows)
    (table_dir / "e4_key_results.tex").write_text(
        latex_key_results(key_rows),
        encoding="utf-8",
    )
    consumed = copy_inputs(repo_root, table_dir)
    builder = Path(__file__).resolve()
    consumed.append(builder)

    readme = """# E4 formal v3 OMP4 artifacts

This directory is generated only from the completed PipeDream-aware formal
matrices. Historical `budget_v2` inputs are deliberately excluded.

## Figures

- `e4_1_stage_scaling`: four-schedule stage-count scaling at b=8, m=4.
- `e4_2a_batch_geometry`: fixed-B=32 physical-batch/microbatch geometry.
- `e4_2b_microbatch_sweep`: b=1 window-length response and crossovers.
- `e4_3_network_sensitivity`: absolute throughput, paired ratios, and local
  throughput retention under sender-side link pacing.
- `e4_4_critical_stage_breakdown`: diagnostic synchronized-trace attribution
  for local and constrained profiles. It is n=1 and is not the formal
  throughput estimator.

PNG files are previews. PDF files are vector paper artifacts. Error bars in
E4.1-E4.2 are sample standard deviations over n=3; E4.3 uses n=4.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    manifest = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "builder": builder.relative_to(repo_root).as_posix(),
        "formal_run_counts": {name: len(rows) for name, rows in runs.items()},
        "inputs": [
            {
                "path": path.relative_to(repo_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(set(consumed))
        ],
    }
    (output_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "formal run rows:",
        ", ".join(f"{name}={len(rows)}" for name, rows in runs.items()),
    )
    print(output_dir)


if __name__ == "__main__":
    main()
