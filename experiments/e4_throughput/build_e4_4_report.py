#!/usr/bin/env python3
"""Build the audited E4.4 report from the late-window v3 traces."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


METHODS = ("bpfree", "exactbp_1f1b")
METHOD_LABELS = {"bpfree": "BP-free", "exactbp_1f1b": "Exact BP (1F1B)"}
CONFIG = "experiments/e4_throughput/configs/e4_4_steady_trace.json"
RAW_ROOT = "results/e4_throughput/raw/e4_4_steady_trace_v3"
ANALYSIS_ROOT = "results/e4_throughput/analysis/e4_4_steady_trace_v3"
FIGURE_ROOT = "results/e4_throughput/figures/e4_4_steady_trace_v3"
CODE_INPUTS = (
    "experiments/e4_throughput/run_e4_4_steady_trace.py",
    "experiments/e4_throughput/run_e4_4_overhead_decomposition.py",
    "experiments/e4_throughput/analyze_e4_4_steady_trace.py",
    "experiments/e4_throughput/analyze_e4_4_traces.py",
    "experiments/e4_throughput/build_e4_4_paper_artifacts.py",
    "src/sg_exe_trainer/runtime/bpfree/cpu_runner.py",
    "src/sg_exe_trainer/runtime/bpfree/cpu_phase.py",
    "src/sg_exe_trainer/runtime/bpfree/cpu_stage.py",
    "src/sg_exe_trainer/runtime/bpfree/chunk_split.py",
    "src/sg_exe_trainer/runtime/bpfree/model_runtime.py",
    "src/sg_exe_trainer/runtime/bpfree/schedule_runtime.py",
    "src/sg_exe_trainer/runtime/exactbp/cpu_runner.py",
    "src/sg_exe_trainer/runtime/exactbp/distributed_runtime.py",
    "src/sg_exe_trainer/runtime/transport/cpu.py",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_phase(summary: dict[str, Any]) -> dict[str, Any]:
    return next(item for item in summary["phases"] if item["phase"] == "train")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(repo_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo_root, check=True, capture_output=True, text=True
    ).stdout.strip()


def stage_row(
    rows: list[dict[str, str]], method: str, stage_id: int
) -> dict[str, str]:
    found = [
        row
        for row in rows
        if row["method"] == method and int(row["stage_id"]) == stage_id
    ]
    if len(found) != 1:
        raise ValueError(f"expected one stage row for {method} S{stage_id}")
    return found[0]


def stability_rows(
    *, raw_root: Path, steady: dict[str, Any]
) -> list[dict[str, Any]]:
    reports = {
        (item["method"], int(item["rep"])): item for item in steady["runs"]
    }
    rows = []
    for method in METHODS:
        for run_dir in sorted((raw_root / "local" / "throughput_b8_m4" / method).glob("rep_*")):
            rep = int(run_dir.name.removeprefix("rep_"))
            summary = read_json(run_dir / "summary.json")
            phase = train_phase(summary)
            metadata = read_json(run_dir / "run_metadata.json")
            report = reports[(method, rep)]
            rows.append(
                {
                    "method": method,
                    "rep": rep,
                    "completed_records": int(phase["completed_records"]),
                    "optimizer_steps": int(phase["optimizer_steps"]),
                    "wall_ms": float(phase["wall_ms"]),
                    "throughput_records_per_s": float(phase["throughput_per_s"]),
                    "stable": bool(report["stable"]),
                    "analyzed_window_start": int(report["analyzed_windows"][0]),
                    "analyzed_window_end_exclusive": int(report["analyzed_windows"][1]),
                    "median_period_ms_per_window": float(report["median_period_ms_per_window"]),
                    "relative_stage_period_spread": float(report["relative_stage_period_spread"]),
                    "max_abs_lag_drift_ms_per_window": float(
                        report["max_abs_lag_drift_ms_per_window"]
                    ),
                    "relative_lag_drift": float(report["relative_lag_drift"]),
                    "returncode": int(metadata["returncode"]),
                    "source_snapshot_sha256": metadata["provenance"]["source_snapshot_sha256"],
                }
            )
    return rows


def representative_lags(
    lag_rows: list[dict[str, str]], *, rep: int
) -> dict[tuple[str, int], dict[str, str]]:
    return {
        (row["method"], int(row["stage_id"])): row
        for row in lag_rows
        if int(row["rep"]) == rep
    }


def method_stats(rows: list[dict[str, Any]], method: str) -> dict[str, float]:
    selected = [row for row in rows if row["method"] == method]
    throughputs = [float(row["throughput_records_per_s"]) for row in selected]
    return {
        "n": len(selected),
        "throughput_mean": statistics.mean(throughputs),
        "throughput_std": statistics.stdev(throughputs),
        "stable_count": sum(bool(row["stable"]) for row in selected),
        "period_median": statistics.median(
            float(row["median_period_ms_per_window"]) for row in selected
        ),
    }


def latex_stage_table(stage_rows: list[dict[str, str]]) -> str:
    categories = (
        ("input_percent", "Input"),
        ("forward_percent", "Fwd+loss"),
        ("recv_wait_percent", "Recv wait"),
        ("communication_percent", "Transfer"),
        ("backward_percent", "Backward"),
        ("optimizer_percent", "Opt."),
        ("idle_other_percent", "Idle/other"),
    )
    lines = [
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        "Method & Stage & " + " & ".join(label for _, label in categories) + r" \\",
        r"\midrule",
    ]
    for method in METHODS:
        for stage_id in range(3):
            row = stage_row(stage_rows, method, stage_id)
            values = " & ".join(f"{float(row[key]):.1f}" for key, _ in categories)
            lines.append(f"{METHOD_LABELS[method]} & S{stage_id} & {values}" + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def english_report(
    *,
    run_rows: list[dict[str, Any]],
    stage_rows: list[dict[str, str]],
    lag_rows: list[dict[str, str]],
    headline: dict[str, Any],
) -> str:
    bp = method_stats(run_rows, "bpfree")
    exact = method_stats(run_rows, "exactbp_1f1b")
    rep = int(headline["representative_rep"])
    lags = representative_lags(lag_rows, rep=rep)
    bp_period = float(lags[("bpfree", 0)]["period_ms_per_window"])
    s1_windows = float(lags[("bpfree", 1)]["mean_same_window_lag_ms"]) / bp_period
    s2_windows = float(lags[("bpfree", 2)]["mean_same_window_lag_ms"]) / bp_period
    bp_s0 = stage_row(stage_rows, "bpfree", 0)
    exact_recv = [float(stage_row(stage_rows, "exactbp_1f1b", stage)["recv_wait_percent"]) for stage in range(3)]
    return f"""# E4.4 Execution and Overhead Audit

## Question

Where does the current BP-free/1F1B execution time go, and how do the two methods advance across stages after buffers and backpressure have formed?

## Setup

TinyLlama is split over three L40 GPUs. Both methods use b=8, m=4, effective batch B=32, bfloat16, and q_proj/v_proj LoRA. Each repetition trains 4096 records (128 optimizer windows). Only late windows W64-W95 are synchronously traced; the stability fit uses W66-W93. Three repetitions use alternating method order. All six runs share the same source snapshot.

## Fixed wall-clock timeline

![E4.4 fixed wall-clock timeline](figures/e4_4_steady_timeline.png)

The figure compares independently rebased 1600 ms wall-clock slices. It does not join the start of BP-free Stage 0 window W with the completion of Stage 2 window W, because BP-free windows are stage-local rather than a global transaction. In representative stable rep {rep}, BP-free Stage 1 is about {s1_windows:.1f} local windows behind Stage 0 and Stage 2 is about {s2_windows:.1f} windows behind. Exact BP stages remain on the same global 1F1B windows, with only tens of milliseconds of optimizer-completion offset.

## Per-stage time decomposition

![E4.4 stage time breakdown](figures/e4_4_stage_time_breakdown.png)

The percentages are wall-time buckets within each stage trace, averaged over three repetitions. Exact BP spends {exact_recv[0]:.1f}%, {exact_recv[1]:.1f}%, and {exact_recv[2]:.1f}% of Stage 0/1/2 span in blocking receive waits. BP-free receive-wait time is approximately zero because receives are preposted and stages advance independently. Communication is not free: BP-free Stage 0 spends {float(bp_s0['communication_percent']):.1f}% in transfer/runtime, dominated by pending-send budget waits that provide backpressure.

`Input preparation` includes manifest tensor loading, concatenation, CPU-to-GPU copies, and the Stage 0 embedding lookup. It is not pure H2D. Its corrected Stage 0 share is {float(bp_s0['input_percent']):.1f} +/- {float(bp_s0['input_percent_std']):.1f}%, not the 45% shown by the discarded short v1 trace.

## Stability and throughput

| Method | Throughput (records/s) | Stable repetitions | Median fitted period |
|---|---:|---:|---:|
| BP-free | {bp['throughput_mean']:.2f} +/- {bp['throughput_std']:.2f} | {int(bp['stable_count'])}/{int(bp['n'])} | {bp['period_median']:.1f} ms/window |
| Exact BP (1F1B) | {exact['throughput_mean']:.2f} +/- {exact['throughput_std']:.2f} | {int(exact['stable_count'])}/{int(exact['n'])} | {exact['period_median']:.1f} ms/window |

The median paired diagnostic throughput ratio is {float(headline['median_bpfree_throughput_over_exactbp_1f1b']):.3f}x. Exact BP passes the 5% period-spread/lag-drift rule in all repetitions. BP-free passes in two repetitions; the remaining repetition is marginally above the threshold at 5.42%. This is evidence of asynchronous stage lag and backpressure dynamics, not a claim that all BP-free stages are globally lockstep.

## Measurement definitions

- `Forward + loss`: body forward plus local head loss for BP-free; exact stage forward for 1F1B.
- `Receive wait`: blocking wait for a posted CPU/Gloo receive.
- `Transfer/runtime`: D2H, receive post/H2D, send post/runtime, link pacing, and send-budget waits.
- `Idle / other`: untraced gaps plus small control actions.
- These are synchronized action wall times, not CUDA-kernel profiler percentages.

## Boundary

E4.4 is a mechanism diagnostic. Selective synchronization perturbs execution, so E4.1-E4.3 remain the sources for primary throughput claims. The v1/v2 same-window span figure must not be used because it treated BP-free stage-local window IDs as a global transaction.
"""


def chinese_report(
    *,
    run_rows: list[dict[str, Any]],
    stage_rows: list[dict[str, str]],
    lag_rows: list[dict[str, str]],
    headline: dict[str, Any],
) -> str:
    bp = method_stats(run_rows, "bpfree")
    exact = method_stats(run_rows, "exactbp_1f1b")
    rep = int(headline["representative_rep"])
    lags = representative_lags(lag_rows, rep=rep)
    bp_period = float(lags[("bpfree", 0)]["period_ms_per_window"])
    s1_windows = float(lags[("bpfree", 1)]["mean_same_window_lag_ms"]) / bp_period
    s2_windows = float(lags[("bpfree", 2)]["mean_same_window_lag_ms"]) / bp_period
    bp_s0 = stage_row(stage_rows, "bpfree", 0)
    exact_recv = [float(stage_row(stage_rows, "exactbp_1f1b", stage)["recv_wait_percent"]) for stage in range(3)]
    return f"""# E4.4 执行过程与时间开销审计

## 实验问题

当前 BP-free 与 1F1B 的时间分别花在哪里？当异步 buffer 和背压已经建立后，三个 stage 如何推进？

## 实验设置

TinyLlama 切成三段，分别放在三张 L40 上。两种方法均使用 b=8、m=4、有效 batch B=32、bfloat16 和 q_proj/v_proj LoRA。每次训练 4096 条 records，共 128 个 optimizer window；仅同步记录后半段 W64-W95，稳定性拟合使用 W66-W93。三次重复交替方法顺序，六次运行的 source snapshot 完全一致。

## 固定 wall-clock 时间线

![E4.4 fixed wall-clock timeline](figures/e4_4_steady_timeline.png)

图中为两种方法各自独立归零的 1600 ms wall-clock 切片。不能再从 BP-free Stage 0 的 W 起点一直画到 Stage 2 的同名 W 终点，因为 BP-free 的 window 是 stage-local update，不是跨三段的 global transaction。在代表性稳定运行 rep {rep} 中，Stage 1 相对 Stage 0 约落后 {s1_windows:.1f} 个 local window，Stage 2 约落后 {s2_windows:.1f} 个；Exact BP 三段始终处于相同的全局 1F1B window，仅有几十毫秒的 optimizer completion 偏移。

## 每个 stage 的时间分解

![E4.4 stage time breakdown](figures/e4_4_stage_time_breakdown.png)

百分比以各 stage 自己的 trace wall span 为分母，并对三次重复取平均。Exact BP 的 Stage 0/1/2 分别有 {exact_recv[0]:.1f}%、{exact_recv[1]:.1f}%、{exact_recv[2]:.1f}% 花在阻塞 receive wait。BP-free 使用预提交 receive，各 stage 独立推进，因此 receive-wait 接近 0；但通信并没有消失，BP-free Stage 0 仍有 {float(bp_s0['communication_percent']):.1f}% 的 transfer/runtime，主要来自 pending-send budget wait，它正是异步队列施加背压的位置。

`Input preparation` 包含 manifest tensor 读取、拼接、CPU-to-GPU 搬运以及 Stage 0 embedding lookup，并不是纯 H2D。修正后 Stage 0 的占比为 {float(bp_s0['input_percent']):.1f} +/- {float(bp_s0['input_percent_std']):.1f}%，旧 v1 短 trace 中的 45% 不应继续使用。

## 稳定性与诊断吞吐

| 方法 | 吞吐 records/s | 通过稳定性检查 | 拟合周期中位数 |
|---|---:|---:|---:|
| BP-free | {bp['throughput_mean']:.2f} +/- {bp['throughput_std']:.2f} | {int(bp['stable_count'])}/{int(bp['n'])} | {bp['period_median']:.1f} ms/window |
| Exact BP (1F1B) | {exact['throughput_mean']:.2f} +/- {exact['throughput_std']:.2f} | {int(exact['stable_count'])}/{int(exact['n'])} | {exact['period_median']:.1f} ms/window |

配对诊断吞吐倍率的中位数为 {float(headline['median_bpfree_throughput_over_exactbp_1f1b']):.3f}x。Exact BP 三次全部通过 5% 的 stage-period/lag-drift 阈值；BP-free 两次通过，另一次为 5.42%，只是略高于阈值。正确结论是 BP-free 存在异步 stage lag 和背压动态，而不是“三段严格锁步稳态”。

## 分类口径

- `Forward + loss`：BP-free 的 body forward 与 local-head loss；1F1B 的 exact stage forward。
- `Receive wait`：已提交 CPU/Gloo receive 的阻塞等待。
- `Transfer/runtime`：D2H、receive post/H2D、send runtime、链路 pacing 和 send-budget wait。
- `Idle / other`：未被 action 覆盖的间隙与少量控制操作。
- 这些是带 CUDA synchronize 的 action wall time，不是 CUDA kernel profiler 百分比。

## 结论边界

E4.4 用于解释机制。选择性同步会扰动执行，因此论文的主要吞吐结论仍应引用 E4.1-E4.3。v1/v2 的“同 window 端到端 span”图把 BP-free local window 错当成 global transaction，必须废弃。
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    figure_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    raw_root = repo_root / RAW_ROOT
    analysis_root = repo_root / ANALYSIS_ROOT
    source_figure_root = repo_root / FIGURE_ROOT
    steady = read_json(analysis_root / "steady_state_report.json")
    lag_rows = read_csv(analysis_root / "lag_summary.csv")
    stage_rows = read_csv(source_figure_root / "e4_4_stage_time_breakdown.csv")
    headline = read_json(source_figure_root / "e4_4_headline_numbers.json")
    run_rows = stability_rows(raw_root=raw_root, steady=steady)

    snapshots = {row["source_snapshot_sha256"] for row in run_rows}
    if len(snapshots) != 1:
        raise ValueError(f"E4.4 runs use different source snapshots: {snapshots}")
    if any(int(row["returncode"]) != 0 for row in run_rows):
        raise ValueError("E4.4 contains a failed run")

    for stem in ("e4_4_steady_timeline", "e4_4_stage_time_breakdown"):
        for suffix in (".png", ".pdf"):
            shutil.copy2(source_figure_root / f"{stem}{suffix}", figure_dir / f"{stem}{suffix}")
    write_csv(table_dir / "e4_4_runs_and_stability.csv", run_rows)
    shutil.copy2(analysis_root / "lag_summary.csv", table_dir / "e4_4_lag_summary.csv")
    shutil.copy2(source_figure_root / "e4_4_stage_time_breakdown.csv", table_dir / "e4_4_stage_time_breakdown.csv")
    shutil.copy2(analysis_root / "action_summary.csv", table_dir / "e4_4_action_summary.csv")
    (table_dir / "e4_4_stage_time_breakdown.tex").write_text(
        latex_stage_table(stage_rows), encoding="utf-8"
    )

    (output_dir / "E4_4_REPORT.md").write_text(
        english_report(
            run_rows=run_rows,
            stage_rows=stage_rows,
            lag_rows=lag_rows,
            headline=headline,
        ),
        encoding="utf-8",
    )
    (output_dir / "E4_4_REPORT_ZH.md").write_text(
        chinese_report(
            run_rows=run_rows,
            stage_rows=stage_rows,
            lag_rows=lag_rows,
            headline=headline,
        ),
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "# E4.4 audited artifacts\n\n"
        "Use the v3 fixed-wall-clock timeline and late-window decomposition. "
        "Do not use the v1/v2 same-window span figure.\n",
        encoding="utf-8",
    )

    consumed: list[Path] = [Path(__file__).resolve(), repo_root / CONFIG]
    consumed.extend(repo_root / relative for relative in CODE_INPUTS)
    consumed.extend((raw_root / "local" / "throughput_b8_m4").rglob("summary.json"))
    consumed.extend((raw_root / "local" / "throughput_b8_m4").rglob("run_metadata.json"))
    consumed.extend(
        [
            analysis_root / "steady_state_report.json",
            analysis_root / "lag_summary.csv",
            analysis_root / "action_summary.csv",
            source_figure_root / "e4_4_stage_time_breakdown.csv",
            source_figure_root / "e4_4_headline_numbers.json",
        ]
    )
    unique_inputs = sorted({path.resolve() for path in consumed})
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head_at_report_time": git_value(repo_root, "rev-parse", "HEAD"),
        "source_snapshot_sha256": next(iter(snapshots)),
        "inputs": [
            {
                "path": path.relative_to(repo_root.resolve()).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in unique_inputs
        ],
    }
    (output_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"runs={len(run_rows)} snapshot={next(iter(snapshots))[:16]}")
    print(output_dir)


if __name__ == "__main__":
    main()
