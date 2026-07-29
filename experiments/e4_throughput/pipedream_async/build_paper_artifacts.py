#!/usr/bin/env python3
"""Build paper-ready E4 PipeDream comparison figures and tables."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHODS = ("exactbp_gpipe", "exactbp_1f1b", "pipedream", "bpfree")
METHOD_LABELS = {
    "exactbp_gpipe": "GPipe",
    "exactbp_1f1b": "1F1B",
    "pipedream": "PipeDream",
    "bpfree": "BP-free",
}
METHOD_TABLE_LABELS = {
    "exactbp_gpipe": "Exact BP (GPipe)",
    "exactbp_1f1b": "Exact BP (1F1B)",
    "pipedream": "PipeDream",
    "bpfree": "BP-free",
}
METHOD_COLORS = {
    "exactbp_gpipe": "#4C78A8",
    "exactbp_1f1b": "#F58518",
    "pipedream": "#E45756",
    "bpfree": "#2CA02C",
}
METHOD_MARKERS = {
    "exactbp_gpipe": "o",
    "exactbp_1f1b": "^",
    "pipedream": "D",
    "bpfree": "s",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--steady-root",
        type=Path,
        default=Path("results/e4_throughput/raw/e4_pipedream_steady_formal_v1"),
    )
    parser.add_argument(
        "--quality-root",
        type=Path,
        default=Path("results/e4_throughput/raw/e4_pipedream_quality_formal_v1"),
    )
    parser.add_argument(
        "--short-root",
        type=Path,
        default=Path("results/e4_throughput/raw/e4_pipedream_formal_v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/e4_throughput/paper_artifacts/e4_pipedream_v1"),
    )
    parser.add_argument("--dpi", type=int, default=360)
    return parser.parse_args()


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("cannot aggregate an empty sample")
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def grouped_values(
    rows: list[dict[str, str]], value_key: str
) -> dict[str, list[float]]:
    output: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        output[row["method"]].append(float(row[value_key]))
    return dict(output)


def require_methods(groups: dict[str, Any], source: str) -> None:
    missing = set(METHODS) - set(groups)
    if missing:
        raise ValueError(f"{source} is missing methods: {sorted(missing)}")


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.7,
            "grid.linewidth": 0.45,
            "grid.alpha": 0.35,
            "lines.linewidth": 1.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig: Any, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def draw_measurement_panel(
    ax: Any,
    samples: dict[str, list[float]],
    *,
    title: str,
    xlabel: str,
    scale: float,
    value_format: str,
    xlim: tuple[float, float],
) -> None:
    for y, method in enumerate(METHODS):
        values = [value * scale for value in samples[method]]
        mean, stdev = mean_std(values)
        count = len(values)
        if count == 1:
            offsets = [0.0]
        else:
            step = 0.22 / (count - 1)
            offsets = [-0.11 + index * step for index in range(count)]
        ax.scatter(
            values,
            [y + offset for offset in offsets],
            s=19,
            marker=METHOD_MARKERS[method],
            facecolor=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.45,
            alpha=0.58,
            zorder=3,
        )
        ax.errorbar(
            mean,
            y,
            xerr=stdev,
            fmt=METHOD_MARKERS[method],
            markersize=5.6,
            color=METHOD_COLORS[method],
            markeredgecolor="#222222",
            markeredgewidth=0.55,
            capsize=2.2,
            elinewidth=1.0,
            zorder=4,
        )
        ax.text(
            mean + 0.012 * (xlim[1] - xlim[0]),
            y - 0.19,
            value_format.format(mean),
            color="#333333",
            fontsize=6.8,
            va="center",
        )
    ax.set_yticks(range(len(METHODS)), [METHOD_LABELS[method] for method in METHODS])
    ax.invert_yaxis()
    ax.set_xlim(*xlim)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def plot_quality_throughput(
    quality: dict[str, list[float]],
    throughput: dict[str, list[float]],
    output_dir: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.55), sharey=True)
    draw_measurement_panel(
        axes[0],
        quality,
        title="(a) Held-out quality",
        xlabel="AG News test accuracy (%)",
        scale=100.0,
        value_format="{:.2f}%",
        xlim=(88.75, 91.85),
    )
    draw_measurement_panel(
        axes[1],
        throughput,
        title="(b) Long-stream training throughput",
        xlabel="Throughput (records/s)",
        scale=1.0,
        value_format="{:.1f}",
        xlim=(125.0, 242.0),
    )
    axes[1].tick_params(axis="y", labelleft=False)
    fig.text(
        0.5,
        0.01,
        "Faint markers are individual runs; solid markers and bars show mean +/- sample SD.",
        ha="center",
        fontsize=6.8,
        color="#444444",
    )
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 1.0), w_pad=2.0)
    save_figure(fig, output_dir / "e4_pipedream_quality_throughput", dpi)


def plot_tradeoff(
    quality: dict[str, list[float]],
    throughput: dict[str, list[float]],
    output_dir: Path,
    dpi: int,
) -> None:
    stats = {}
    for method in METHODS:
        t_mean, t_std = mean_std(throughput[method])
        a_mean, a_std = mean_std([100.0 * value for value in quality[method]])
        stats[method] = (t_mean, t_std, a_mean, a_std)

    fig, ax = plt.subplots(figsize=(4.4, 3.05))
    label_offsets = {
        "exactbp_gpipe": (5, 7),
        "exactbp_1f1b": (5, -15),
        "pipedream": (-8, 9),
        "bpfree": (-7, 9),
    }
    alignments = {
        "exactbp_gpipe": "left",
        "exactbp_1f1b": "left",
        "pipedream": "right",
        "bpfree": "right",
    }
    for method in METHODS:
        t_mean, t_std, a_mean, a_std = stats[method]
        ax.errorbar(
            t_mean,
            a_mean,
            xerr=t_std,
            yerr=a_std,
            fmt=METHOD_MARKERS[method],
            markersize=6.5,
            color=METHOD_COLORS[method],
            markeredgecolor="#222222",
            markeredgewidth=0.55,
            capsize=2.3,
            elinewidth=1.0,
            zorder=3,
        )
        ax.annotate(
            METHOD_LABELS[method],
            (t_mean, a_mean),
            xytext=label_offsets[method],
            textcoords="offset points",
            ha=alignments[method],
            va="center",
            fontsize=7.2,
        )
    ax.text(
        0.98,
        0.96,
        "higher is better",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.8,
        color="#555555",
    )
    ax.set_xlim(128.0, 241.0)
    ax.set_ylim(89.0, 91.72)
    ax.set_xlabel("Long-stream throughput (records/s)")
    ax.set_ylabel("AG News test accuracy (%)")
    ax.set_title("Quality-throughput operating points", loc="left", fontweight="bold")
    ax.grid()
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, output_dir / "e4_pipedream_tradeoff", dpi)


def plot_system_cost(
    main_rows: list[dict[str, Any]],
    pipedream_state: dict[str, float],
    output_dir: Path,
    dpi: int,
) -> None:
    by_method = {row["method"]: row for row in main_rows}
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.15, 2.7),
        gridspec_kw={"width_ratios": (1.65, 1.0)},
    )

    ax = axes[0]
    x = list(range(len(METHODS)))
    communication = [by_method[method]["total_send_gib"] for method in METHODS]
    bars = ax.bar(
        x,
        communication,
        width=0.62,
        color=[METHOD_COLORS[method] for method in METHODS],
        edgecolor="white",
        linewidth=0.55,
    )
    base = by_method["bpfree"]["total_send_gib"]
    for bar, value in zip(bars, communication):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.55,
            f"{value:.2f} GiB\n({value / base:.1f}x)",
            ha="center",
            va="bottom",
            fontsize=6.8,
        )
    ax.set_xticks(x, [METHOD_LABELS[method] for method in METHODS])
    ax.set_ylim(0.0, 23.2)
    ax.set_ylabel("Total bytes sent per run (GiB)")
    ax.set_title("(a) End-to-end communication", loc="left", fontweight="bold")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    state_labels = (
        "Activation snapshots\n(max per stage)",
        "Live LoRA versions\n(max per stage)",
        "Backward version lag\n(max)",
    )
    state_values = (
        pipedream_state["peak_activation_stash"],
        pipedream_state["peak_live_weight_versions"],
        pipedream_state["max_backward_version_lag"],
    )
    y = list(range(len(state_labels)))
    bars = ax.barh(y, state_values, height=0.54, color=METHOD_COLORS["pipedream"])
    for bar, value in zip(bars, state_values):
        ax.text(
            value + 0.08,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.0f}",
            va="center",
            fontsize=7.2,
        )
    ax.set_yticks(y, state_labels)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 3.7)
    ax.set_xlabel("Count / version distance")
    ax.set_title("(b) PipeDream state and staleness", loc="left", fontweight="bold")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.02,
        -0.29,
        f"Peak per-stage LoRA weight stash: {pipedream_state['peak_weight_stash_mib']:.2f} MiB; "
        f"max stage-mean lag: {pipedream_state['max_stage_mean_version_lag']:.2f} versions.",
        transform=ax.transAxes,
        fontsize=6.6,
        color="#444444",
    )
    fig.tight_layout(w_pad=2.2)
    save_figure(fig, output_dir / "e4_pipedream_system_cost", dpi)


def plot_stream_length(
    short: dict[str, list[float]],
    steady: dict[str, list[float]],
    output_dir: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(4.7, 2.75))
    for y, method in enumerate(METHODS):
        short_mean, short_std = mean_std(short[method])
        long_mean, long_std = mean_std(steady[method])
        ax.plot(
            (short_mean, long_mean),
            (y, y),
            color=METHOD_COLORS[method],
            alpha=0.52,
            linewidth=1.5,
            zorder=1,
        )
        ax.errorbar(
            short_mean,
            y,
            xerr=short_std,
            fmt="o",
            markersize=5.2,
            markerfacecolor="white",
            markeredgecolor=METHOD_COLORS[method],
            color=METHOD_COLORS[method],
            capsize=2.0,
            elinewidth=0.9,
            zorder=3,
        )
        ax.errorbar(
            long_mean,
            y,
            xerr=long_std,
            fmt="s",
            markersize=5.2,
            color=METHOD_COLORS[method],
            markeredgecolor="#222222",
            markeredgewidth=0.45,
            capsize=2.0,
            elinewidth=0.9,
            zorder=3,
        )
        delta = 100.0 * (long_mean / short_mean - 1.0)
        ax.text(long_mean + 3.0, y, f"+{delta:.1f}%", va="center", fontsize=6.8)
    ax.scatter([], [], marker="o", facecolor="white", edgecolor="#555555", label="n=1,024")
    ax.scatter([], [], marker="s", facecolor="#555555", edgecolor="#222222", label="n=9,984")
    ax.set_yticks(range(len(METHODS)), [METHOD_LABELS[method] for method in METHODS])
    ax.invert_yaxis()
    ax.set_xlim(108.0, 250.0)
    ax.set_xlabel("Measured throughput (records/s)")
    ax.set_title("Stream-length sensitivity", loc="left", fontweight="bold")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right", ncol=2)
    fig.tight_layout()
    save_figure(fig, output_dir / "e4_pipedream_stream_length_sensitivity", dpi)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_latex(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & Accuracy (\%) & Choice NLL & Throughput & Sent (GiB) \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{METHOD_TABLE_LABELS[row['method']]} & "
            f"{100.0 * row['accuracy_mean']:.2f} $\\pm$ {100.0 * row['accuracy_stdev']:.2f} & "
            f"{row['nll_mean']:.3f} & "
            f"{row['throughput_mean']:.1f} $\\pm$ {row['throughput_stdev']:.1f} & "
            f"{row['total_send_gib']:.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    steady_root = resolve(repo_root, args.steady_root)
    quality_root = resolve(repo_root, args.quality_root)
    short_root = resolve(repo_root, args.short_root)
    output_dir = resolve(repo_root, args.output_dir)
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"

    source_paths = {
        "steady_runs": steady_root / "runs.csv",
        "steady_summary": steady_root / "summary.csv",
        "steady_audit": steady_root / "throughput_audit.json",
        "quality_runs": quality_root / "quality_runs.csv",
        "quality_summary": quality_root / "quality_summary.csv",
        "quality_audit": quality_root / "quality_audit.json",
        "short_runs": short_root / "runs.csv",
        "short_summary": short_root / "summary.csv",
    }
    for name, path in source_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"missing {name}: {path}")

    steady_rows = read_csv(source_paths["steady_runs"])
    quality_rows = read_csv(source_paths["quality_runs"])
    short_rows = read_csv(source_paths["short_runs"])
    steady_samples = grouped_values(steady_rows, "throughput_per_s")
    quality_samples = grouped_values(quality_rows, "test_accuracy")
    nll_samples = grouped_values(quality_rows, "test_choice_nll")
    short_samples = grouped_values(short_rows, "throughput_per_s")
    require_methods(steady_samples, "steady runs")
    require_methods(quality_samples, "quality runs")
    require_methods(short_samples, "short runs")

    sent_samples = grouped_values(steady_rows, "total_send_bytes")
    main_rows: list[dict[str, Any]] = []
    baseline_throughput = mean_std(steady_samples["exactbp_1f1b"])[0]
    for method in METHODS:
        accuracy_mean, accuracy_stdev = mean_std(quality_samples[method])
        nll_mean, nll_stdev = mean_std(nll_samples[method])
        throughput_mean, throughput_stdev = mean_std(steady_samples[method])
        short_mean, short_stdev = mean_std(short_samples[method])
        sent_mean, sent_stdev = mean_std(sent_samples[method])
        main_rows.append(
            {
                "method": method,
                "quality_seeds": len(quality_samples[method]),
                "accuracy_mean": accuracy_mean,
                "accuracy_stdev": accuracy_stdev,
                "nll_mean": nll_mean,
                "nll_stdev": nll_stdev,
                "throughput_repetitions": len(steady_samples[method]),
                "throughput_mean": throughput_mean,
                "throughput_stdev": throughput_stdev,
                "throughput_min": min(steady_samples[method]),
                "throughput_max": max(steady_samples[method]),
                "short_throughput_mean": short_mean,
                "short_throughput_stdev": short_stdev,
                "stream_length_delta_pct": 100.0 * (throughput_mean / short_mean - 1.0),
                "speedup_vs_1f1b": throughput_mean / baseline_throughput,
                "total_send_gib": sent_mean / 2**30,
                "total_send_gib_stdev": sent_stdev / 2**30,
            }
        )

    pipedream_rows = [row for row in steady_rows if row["method"] == "pipedream"]
    pipedream_state = {
        "peak_activation_stash": max(float(row["peak_activation_stash"]) for row in pipedream_rows),
        "peak_live_weight_versions": max(
            float(row["peak_live_weight_versions"]) for row in pipedream_rows
        ),
        "peak_weight_stash_mib": max(
            float(row["peak_weight_stash_bytes"]) for row in pipedream_rows
        )
        / 2**20,
        "max_stage_mean_version_lag": statistics.mean(
            float(row["max_stage_mean_version_lag"]) for row in pipedream_rows
        ),
        "max_backward_version_lag": max(
            float(row["max_backward_version_lag"]) for row in pipedream_rows
        ),
    }

    configure_matplotlib()
    plot_quality_throughput(quality_samples, steady_samples, figures_dir, args.dpi)
    plot_tradeoff(quality_samples, steady_samples, figures_dir, args.dpi)
    plot_system_cost(main_rows, pipedream_state, figures_dir, args.dpi)
    plot_stream_length(short_samples, steady_samples, figures_dir, args.dpi)

    write_csv(tables_dir / "e4_pipedream_main_results.csv", main_rows)
    write_csv(tables_dir / "e4_pipedream_state.csv", [pipedream_state])
    write_latex(tables_dir / "e4_pipedream_main_results.tex", main_rows)

    notes = """# E4 PipeDream comparison artifacts

## Setup

- Model: TinyLlama split over three L40 GPUs.
- Geometry: physical batch b=8, four microbatches per update, effective batch B=32.
- Long-stream throughput: 9,984 records, 312 optimizer steps, four order-balanced repetitions.
- Quality: three initialization/data seeds and the shared 7,600-record AG News test evaluator.
- Error bars: sample standard deviation.

## Figure roles

- `e4_pipedream_quality_throughput`: main measured comparison with every seed/rep visible.
- `e4_pipedream_tradeoff`: joint quality-throughput operating points.
- `e4_pipedream_system_cost`: communication volume and PipeDream-specific retained state.
- `e4_pipedream_stream_length_sensitivity`: diagnostic evidence that n=1,024 is startup-biased.

## Interpretation boundary

These are implementation-level measurements of the pinned-CPU/Gloo runners. Throughput combines
schedule, local objective, communication direction, retained activations, and implementation costs;
it is not a scheduler-only ablation.

## Suggested captions

**Quality and throughput.** AG News held-out quality and long-stream training throughput under the
shared b=8, m=4, B=32 geometry. Quality uses three seeds and the common 7,600-record test set;
throughput uses four order-balanced repetitions over 9,984 records. Faint markers show individual
runs, while solid markers and error bars show the mean and sample standard deviation.

**System cost.** Communication volume and PipeDream-specific retained state for the same 9,984-record
run. Sent bytes are summed over all three ranks. Exact BP and PipeDream return hidden gradients and
therefore send 2x the bytes of BP-free. PipeDream additionally retains activation snapshots and
versioned LoRA weights to bind each backward pass to its forward-pass weights.

**Stream-length diagnostic.** Throughput measured over 1,024 and 9,984 records. The larger increase
for PipeDream shows that the short run overweights startup and fill costs; the 9,984-record result is
used for the main comparison.
"""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "README.md").write_text(notes, encoding="utf-8")

    generated = sorted(
        path for path in output_dir.rglob("*") if path.is_file() and path.name != "artifact_manifest.json"
    )
    manifest = {
        "sources": {
            name: {"path": path.relative_to(repo_root).as_posix(), "sha256": sha256(path)}
            for name, path in source_paths.items()
        },
        "generated": [
            {"path": path.relative_to(repo_root).as_posix(), "sha256": sha256(path)}
            for path in generated
        ],
        "protocol": {
            "train_records": 9984,
            "optimizer_steps": 312,
            "physical_batch": 8,
            "microbatches_per_update": 4,
            "effective_batch": 32,
            "throughput_repetitions": 4,
            "quality_seeds": 3,
            "test_records": 7600,
        },
    }
    (output_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
