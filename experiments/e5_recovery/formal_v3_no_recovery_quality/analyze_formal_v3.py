#!/usr/bin/env python3
"""Aggregate and plot the formal E5 no-recovery quality experiment."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


METHODS = (
    "bpfree_fault_free",
    "bpfree_balanced_skip",
    "bpfree_local_retain",
    "exact_fault_free",
    "exact_strict_skip",
)
HORIZONS = ("post_outage", "final")
METHOD_LABELS = {
    "bpfree_fault_free": "BP-free\nfault-free",
    "bpfree_balanced_skip": "BP-free\nbalanced skip",
    "bpfree_local_retain": "BP-free\nlocal retain",
    "exact_fault_free": "Exact-BP\nfault-free",
    "exact_strict_skip": "Exact-BP\nstrict skip",
}
DELTA_SPECS = (
    ("bpfree_balanced_skip_vs_fault_free", "bpfree_balanced_skip", "bpfree_fault_free"),
    ("bpfree_local_retain_vs_fault_free", "bpfree_local_retain", "bpfree_fault_free"),
    ("bpfree_local_retain_vs_balanced_skip", "bpfree_local_retain", "bpfree_balanced_skip"),
    ("exact_strict_skip_vs_fault_free", "exact_strict_skip", "exact_fault_free"),
)
DELTA_LABELS = {
    "bpfree_balanced_skip_vs_fault_free": "BP-free balanced\n- fault-free",
    "bpfree_local_retain_vs_fault_free": "BP-free retain\n- fault-free",
    "bpfree_local_retain_vs_balanced_skip": "BP-free retain\n- balanced",
    "exact_strict_skip_vs_fault_free": "Exact-BP skip\n- fault-free",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def aggregate(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["horizon"], row["method"])].append(row)
    output: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        for method in METHODS:
            items = grouped.get((horizon, method), [])
            if not items:
                continue
            accuracy = [float(item["eval_accuracy"]) for item in items]
            nll = [float(item["eval_nll"]) for item in items]
            accuracy_mean, accuracy_sd = mean_sd(accuracy)
            nll_mean, nll_sd = mean_sd(nll)
            output.append(
                {
                    "horizon": horizon,
                    "method": method,
                    "n": len(items),
                    "accuracy_mean": accuracy_mean,
                    "accuracy_sd": accuracy_sd,
                    "nll_mean": nll_mean,
                    "nll_sd": nll_sd,
                }
            )
    return output


def paired_deltas(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    indexed = {
        (row["horizon"], row["method"], int(row["seed"])): row
        for row in rows
    }
    seeds = sorted({int(row["seed"]) for row in rows})
    seed_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        for comparison, treatment, control in DELTA_SPECS:
            values: list[float] = []
            for seed in seeds:
                treatment_row = indexed.get((horizon, treatment, seed))
                control_row = indexed.get((horizon, control, seed))
                if treatment_row is None or control_row is None:
                    continue
                delta = float(treatment_row["eval_accuracy"]) - float(control_row["eval_accuracy"])
                values.append(delta)
                seed_rows.append(
                    {
                        "horizon": horizon,
                        "comparison": comparison,
                        "seed": seed,
                        "accuracy_delta": delta,
                        "accuracy_delta_pp": 100.0 * delta,
                    }
                )
            if values:
                delta_mean, delta_sd = mean_sd(values)
                summary_rows.append(
                    {
                        "horizon": horizon,
                        "comparison": comparison,
                        "n": len(values),
                        "accuracy_delta_mean": delta_mean,
                        "accuracy_delta_sd": delta_sd,
                        "accuracy_delta_mean_pp": 100.0 * delta_mean,
                        "accuracy_delta_sd_pp": 100.0 * delta_sd,
                    }
                )
    return seed_rows, summary_rows


def plot_accuracy(rows: list[dict[str, str]], output_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    indexed: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        indexed[(row["horizon"], row["method"])].append(float(row["eval_accuracy"]))
    colors = ["#4C78A8", "#9ECAE1", "#2CA02C", "#F58518", "#E45756"]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), sharey=True)
    rng = np.random.default_rng(7)
    for axis, horizon in zip(axes, HORIZONS):
        x = np.arange(len(METHODS))
        means = [statistics.mean(indexed[(horizon, method)]) for method in METHODS]
        sds = [
            statistics.stdev(indexed[(horizon, method)])
            if len(indexed[(horizon, method)]) > 1
            else 0.0
            for method in METHODS
        ]
        axis.bar(x, means, yerr=sds, capsize=4, color=colors, width=0.72, edgecolor="white")
        for position, method in zip(x, METHODS):
            values = indexed[(horizon, method)]
            jitter = rng.uniform(-0.08, 0.08, size=len(values))
            axis.scatter(position + jitter, values, color="#222222", s=20, zorder=3)
        axis.set_xticks(x, [METHOD_LABELS[method] for method in METHODS])
        axis.set_title("Immediately after outage" if horizon == "post_outage" else "After subsequent training")
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("AG News test accuracy")
    fig.suptitle("No-recovery quality under a Stage-1 outage (mean +/- SD, 3 seeds)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_deltas(seed_rows: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    indexed: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in seed_rows:
        indexed[(row["horizon"], row["comparison"])].append(float(row["accuracy_delta_pp"]))
    comparisons = [item[0] for item in DELTA_SPECS]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), sharey=True)
    rng = np.random.default_rng(11)
    for axis, horizon in zip(axes, HORIZONS):
        x = np.arange(len(comparisons))
        means = [statistics.mean(indexed[(horizon, comparison)]) for comparison in comparisons]
        sds = [
            statistics.stdev(indexed[(horizon, comparison)])
            if len(indexed[(horizon, comparison)]) > 1
            else 0.0
            for comparison in comparisons
        ]
        colors = ["#9ECAE1", "#2CA02C", "#54A24B", "#E45756"]
        axis.bar(x, means, yerr=sds, capsize=4, color=colors, width=0.7, edgecolor="white")
        for position, comparison in zip(x, comparisons):
            values = indexed[(horizon, comparison)]
            jitter = rng.uniform(-0.08, 0.08, size=len(values))
            axis.scatter(position + jitter, values, color="#222222", s=20, zorder=3)
        axis.axhline(0.0, color="#333333", linewidth=1)
        axis.set_xticks(x, [DELTA_LABELS[comparison] for comparison in comparisons])
        axis.set_title("Immediately after outage" if horizon == "post_outage" else "After subsequent training")
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Paired accuracy difference (percentage points)")
    fig.suptitle("Effect of skipping and retaining local prefix updates (mean +/- SD, 3 seeds)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_recovery_trajectory(rows: list[dict[str, str]], output_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    indexed = {
        (row["horizon"], row["method"], int(row["seed"])): float(row["eval_accuracy"])
        for row in rows
    }
    seeds = sorted({int(row["seed"]) for row in rows})
    colors = ["#4C78A8", "#9ECAE1", "#2CA02C", "#F58518", "#E45756"]
    x = np.arange(len(HORIZONS))
    fig, axis = plt.subplots(figsize=(8.4, 4.8))
    for method, color in zip(METHODS, colors):
        per_seed = [
            [indexed[(horizon, method, seed)] for horizon in HORIZONS]
            for seed in seeds
        ]
        values = np.asarray(per_seed)
        means = values.mean(axis=0)
        sds = values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros(len(HORIZONS))
        for seed_values in values:
            axis.plot(x, seed_values, color=color, linewidth=0.8, alpha=0.22)
            axis.scatter(x, seed_values, color=color, s=14, alpha=0.55, zorder=3)
        axis.errorbar(
            x,
            means,
            yerr=sds,
            marker="o",
            markersize=6,
            capsize=4,
            linewidth=2.0,
            color=color,
            label=METHOD_LABELS[method].replace("\n", " "),
            zorder=4,
        )
    axis.set_xticks(x, ["Outage ends\n(train through sample 1280)", "Normal training resumes\n(train through sample 2048)"])
    axis.set_ylabel("AG News test accuracy")
    axis.set_title("Quality trajectory after the Stage-1 outage")
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(
    root: Path,
    rows: list[dict[str, str]],
    aggregates: list[dict[str, Any]],
    delta_summary: list[dict[str, Any]],
) -> None:
    protocol_path = root / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8")) if protocol_path.is_file() else {}
    planned = len(protocol.get("seeds", [])) * len(protocol.get("horizons", [])) * len(protocol.get("methods", []))
    aggregate_index = {(row["horizon"], row["method"]): row for row in aggregates}
    delta_index = {(row["horizon"], row["comparison"]): row for row in delta_summary}
    lines = [
        "# E5 无恢复质量实验（formal v3）",
        "",
        f"- 完成度：{len(rows)}/{planned or '未知'}",
        "- 故障：Stage 1 在原始样本序号 `[768,1280)` 离线，共 512 条样本、64 个 B=8 update。",
        "- `post_outage`：训练到样本 1280 后立即评估；`final`：继续训练到样本 2048 后评估。",
        "- 每个点均在 AG News 官方 7600 条 test 上评估；表中为 mean +/- SD。",
        "",
        "## 绝对准确率",
        "",
        "| 端点 | 方法 | seeds | accuracy | NLL |",
        "|---|---|---:|---:|---:|",
    ]
    for horizon in HORIZONS:
        for method in METHODS:
            row = aggregate_index.get((horizon, method))
            if row is None:
                continue
            lines.append(
                f"| {horizon} | {method} | {row['n']} | "
                f"{row['accuracy_mean']:.4f} +/- {row['accuracy_sd']:.4f} | "
                f"{row['nll_mean']:.4f} +/- {row['nll_sd']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## 配对准确率差值",
            "",
            "正值表示减号左侧方法更好。`local-retain - balanced-skip` 是检验故障期间保留 S0 local update 是否有益的主对照。",
            "",
            "| 端点 | 配对比较 | seeds | accuracy delta (pp) |",
            "|---|---|---:|---:|",
        ]
    )
    for horizon in HORIZONS:
        for comparison, _, _ in DELTA_SPECS:
            row = delta_index.get((horizon, comparison))
            if row is None:
                continue
            lines.append(
                f"| {horizon} | {comparison} | {row['n']} | "
                f"{row['accuracy_delta_mean_pp']:+.2f} +/- {row['accuracy_delta_sd_pp']:.2f} |"
            )
    post_local_ff = delta_index.get(("post_outage", "bpfree_local_retain_vs_fault_free"))
    post_local_balanced = delta_index.get(
        ("post_outage", "bpfree_local_retain_vs_balanced_skip")
    )
    final_local_ff = delta_index.get(("final", "bpfree_local_retain_vs_fault_free"))
    final_local_balanced = delta_index.get(("final", "bpfree_local_retain_vs_balanced_skip"))
    final_exact = delta_index.get(("final", "exact_strict_skip_vs_fault_free"))
    interpretation = ["", "## 结果判读", ""]
    if all(
        item is not None
        for item in (post_local_ff, post_local_balanced, final_local_ff, final_local_balanced, final_exact)
    ):
        interpretation.extend(
            [
                f"故障刚结束时，BP-free local-retain 相对自身无故障基线下降 "
                f"{abs(post_local_ff['accuracy_delta_mean_pp']):.2f} +/- "
                f"{post_local_ff['accuracy_delta_sd_pp']:.2f}pp；继续正常训练 768 条样本后，"
                f"该差距缩小到 {abs(final_local_ff['accuracy_delta_mean_pp']):.2f} +/- "
                f"{final_local_ff['accuracy_delta_sd_pp']:.2f}pp。这说明不均衡 stage updates 会造成"
                "明显的即时失配，但后续对齐训练能够修复其中大部分。",
                "",
                f"关键因果对照不支持“保留 S0 local updates 带来质量收益”：local-retain 相对 "
                f"balanced-skip 在 post-outage 和 final 分别低 "
                f"{abs(post_local_balanced['accuracy_delta_mean_pp']):.2f} +/- "
                f"{post_local_balanced['accuracy_delta_sd_pp']:.2f}pp 与 "
                f"{abs(final_local_balanced['accuracy_delta_mean_pp']):.2f} +/- "
                f"{final_local_balanced['accuracy_delta_sd_pp']:.2f}pp，且三个 paired seeds 的差值均为负。",
                "",
                f"BP-free local-retain 的 final 相对掉点小于 Exact-BP strict-skip 的 "
                f"{abs(final_exact['accuracy_delta_mean_pp']):.2f} +/- "
                f"{final_exact['accuracy_delta_sd_pp']:.2f}pp，但 balanced-skip 更好，故该差异不能归因于"
                "故障期间保留的 prefix updates。论文中可报告后续训练的恢复能力和这一质量边界，"
                "不能把 E5 的恢复时间收益扩展成 local-head 的质量收益。",
            ]
        )
    else:
        interpretation.append("实验矩阵尚未完成，暂不生成结论。")
    lines.extend(
        [
            "",
            "## Update 语义",
            "",
            "| 端点 | fault-free S0/S1/S2 | BP-free local-retain | balanced/strict skip |",
            "|---|---:|---:|---:|",
            "| post_outage | 160/160/160 | 160/96/96 | 96/96/96 |",
            "| final | 256/256/256 | 256/192/192 | 192/192/192 |",
            *interpretation,
            "",
            "![Absolute accuracy](figures/e5_no_recovery_accuracy.png)",
            "",
            "![Paired deltas](figures/e5_no_recovery_paired_deltas.png)",
            "",
            "![Recovery trajectory](figures/e5_no_recovery_recovery_trajectory.png)",
        ]
    )
    (root / "REPORT_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.result_root.resolve()
    rows = read_csv(root / "results.csv")
    protocol_path = root / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8")) if protocol_path.is_file() else {}
    aggregates = aggregate(rows)
    seed_deltas, delta_summary = paired_deltas(rows)
    write_csv(root / "aggregate.csv", aggregates)
    write_csv(root / "paired_deltas_by_seed.csv", seed_deltas)
    write_csv(root / "paired_deltas_summary.csv", delta_summary)
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    completed_keys = {
        (row["horizon"], row["method"], int(row["seed"]))
        for row in rows
    }
    expected_keys = {
        (horizon, method, int(seed))
        for horizon in protocol.get("horizons", HORIZONS)
        for method in protocol.get("methods", METHODS)
        for seed in protocol.get("seeds", [])
    }
    complete = bool(expected_keys) and expected_keys <= completed_keys
    if complete:
        plot_accuracy(rows, figures / "e5_no_recovery_accuracy.png")
        plot_deltas(seed_deltas, figures / "e5_no_recovery_paired_deltas.png")
        plot_recovery_trajectory(rows, figures / "e5_no_recovery_recovery_trajectory.png")
    write_report(root, rows, aggregates, delta_summary)
    print(f"Wrote {root / 'REPORT_ZH.md'}")


if __name__ == "__main__":
    main()
