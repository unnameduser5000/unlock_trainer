#!/usr/bin/env python3
"""Run E3-B graceful degradation / no-recovery accuracy suite.

This is a reproducibility launcher for:
  - BP-free fault-free
  - BP-free offline-skip
  - 1F1B fault-free
  - 1F1B offline-skip

It intentionally keeps the four cases explicit because BP-free scheduler lab and
1F1B pipeline runner use different execution entry points.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
BPFREE_MODULE = "sg_exe_trainer.runtime.bpfree.orchestrated_runtime"
F1B_RUNNER = REPO_ROOT / "src" / "sg_exe_trainer" / "runtime" / "exactbp" / "distributed_runtime.py"
AGG_REPORT = REPO_ROOT / "tools" / "report" / "aggregate_graceful_degradation_report.py"
ACC_REPORT = REPO_ROOT / "tools" / "report" / "build_e3_no_recovery_accuracy_table.py"


def parse_csv_ints(raw: str) -> list[int]:
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("empty seed list")
    return vals


def run_cmd(cmd: list[str], *, cwd: Path, log_path: Path, dry_run: bool) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(shlex.quote(x) for x in cmd)
    print(f"\n$ {printable}")
    print(f"log: {log_path}")

    if dry_run:
        return 0

    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {printable}\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )

    elapsed = time.perf_counter() - start
    print(f"exit={proc.returncode} elapsed_s={elapsed:.1f}")
    return int(proc.returncode)


def exists_done(case_dir: Path, *, bpfree: bool) -> bool:
    if bpfree:
        return (case_dir / "scheduler_summary.json").exists()
    return (case_dir / "summary.json").exists()


def bpfree_cmd(args: argparse.Namespace, *, seed: int, output_dir: Path, offline: bool) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        BPFREE_MODULE,
        "--model_name", args.model_name,
        "--manifest", str(args.train_manifest),
        "--eval_manifest", str(args.eval_manifest),
        "--output_dir", str(output_dir),
        "--num_chunks", str(args.num_chunks),
        "--workers", args.workers,
        "--topology", "phone_fixed",
        "--limit", str(args.train_limit),
        "--eval_limit", str(args.eval_limit),
        "--max_inflight", str(args.max_inflight),
        "--scheduler_policy", "recovery_first",
        "--recovery_policy", "skip",
        "--failure_mode", "none",
        "--gradient_accumulation_steps", "1",
        "--optimizer", args.optimizer,
        "--label_smoothing", str(args.label_smoothing),
        "--learning_rate", str(args.learning_rate),
        "--grad_clip", str(args.grad_clip),
        "--dtype", args.bpfree_dtype,
        "--trainable_mode", "lora",
        "--lora_rank", str(args.lora_rank),
        "--lora_alpha", str(args.lora_alpha),
        "--lora_targets", args.lora_targets,
        "--lora_init_std", str(args.lora_init_std),
        "--lora_init_seed", str(seed),
        "--seed", str(seed),
        "--progress_interval", str(args.progress_interval),
        "--belief_transport_mode", "terminal",
    ]

    if offline:
        cmd += [
            "--offline_stage", str(args.offline_stage),
            "--offline_start_seq", str(args.offline_start_seq),
            "--offline_end_seq", str(args.offline_end_seq),
        ]

    return cmd


def f1b_cmd(args: argparse.Namespace, *, seed: int, output_dir: Path, offline: bool) -> list[str]:
    cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node", str(args.num_chunks),
        str(F1B_RUNNER),
        "--model_name", args.model_name,
        "--train_manifest", str(args.train_manifest),
        "--eval_manifest", str(args.eval_manifest),
        "--output_dir", str(output_dir),
        "--num_chunks", str(args.num_chunks),
        "--stage_devices", args.stage_devices,
        "--train_limit", str(args.train_limit),
        "--eval_limit", str(args.eval_limit),
        "--train_epochs", "1",
        "--microbatches", "1",
        "--batch_size", "1",
        "--optimizer", args.optimizer,
        "--label_smoothing", str(args.label_smoothing),
        "--learning_rate", str(args.learning_rate),
        "--grad_clip", str(args.grad_clip),
        "--dtype", args.f1b_dtype,
        "--trainable_mode", "lora",
        "--lora_rank", str(args.lora_rank),
        "--lora_alpha", str(args.lora_alpha),
        "--lora_targets", args.lora_targets,
        "--lora_init_std", str(args.lora_init_std),
        "--lora_init_seed", str(seed),
        "--seed", str(seed),
        "--progress_interval", str(args.progress_interval),
        "--skip_eval_before",
    ]

    if offline:
        cmd += [
            "--recovery_policy", "strict_skip",
            "--offline_stage", str(args.offline_stage),
            "--offline_start_seq", str(args.offline_start_seq),
            "--offline_end_seq", str(args.offline_end_seq),
        ]
    else:
        cmd += ["--recovery_policy", "strict_skip"]

    return cmd


def write_manifest(args: argparse.Namespace) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "suite": "graceful_degradation_train512_eval256_3seeds",
        "output_root": str(args.output_root),
        "train_manifest": str(args.train_manifest),
        "eval_manifest": str(args.eval_manifest),
        "seeds": parse_csv_ints(args.seeds),
        "offline_stage": args.offline_stage,
        "offline_start_seq": args.offline_start_seq,
        "offline_end_seq": args.offline_end_seq,
        "bp_free": {
            "effective_optimizer_batch": 1,
            "physical_request_batch": 1,
            "runner": BPFREE_MODULE,
            "transport": "cpu-mp-queue",
            "recovery_policy": "skip",
        },
        "one_f_one_b": {
            "effective_optimizer_batch": 1,
            "batch_size": 1,
            "microbatches": 1,
            "runner": str(F1B_RUNNER),
            "recovery_policy": "strict_skip",
        },
    }
    path = args.output_root / "graceful_degradation_suite_config.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="tinyllama")
    parser.add_argument("--train_manifest", type=Path, default=Path("data/sft_requests/tinyllama_agnews128_label_train512_seed20260531/requests.jsonl"))
    parser.add_argument("--eval_manifest", type=Path, default=Path("data/sft_requests/tinyllama_agnews128_label_eval256_seed20260531/requests.jsonl"))
    parser.add_argument("--output_root", type=Path, default=Path("results/e5_recovery/raw/graceful_degradation_train512_eval256_3seeds"))
    parser.add_argument("--seeds", default="20260531,20260532,20260533")
    parser.add_argument("--num_chunks", type=int, default=3)
    parser.add_argument("--workers", default="0:cuda:0,1:cuda:1,2:cuda:2")
    parser.add_argument("--stage_devices", default="cuda:0,cuda:1,cuda:2")
    parser.add_argument("--train_limit", type=int, default=512)
    parser.add_argument("--eval_limit", type=int, default=256)
    parser.add_argument("--max_inflight", type=int, default=8)
    parser.add_argument("--offline_stage", type=int, default=1)
    parser.add_argument("--offline_start_seq", type=int, default=192)
    parser.add_argument("--offline_end_seq", type=int, default=320)
    parser.add_argument("--optimizer", default="adamw")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--bpfree_dtype", default="float32")
    parser.add_argument("--f1b_dtype", default="bfloat16")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_targets", default="q_proj,v_proj")
    parser.add_argument("--lora_init_std", type=float, default=0.01)
    parser.add_argument("--progress_interval", type=int, default=128)
    parser.add_argument("--only", default="bpfree_fault_free,bpfree_offline_skip,1f1b_fault_free,1f1b_offline_skip")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_report", action="store_true")
    args = parser.parse_args()

    seeds = parse_csv_ints(args.seeds)
    wanted = {x.strip() for x in args.only.split(",") if x.strip()}

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_manifest(args)

    failures: list[tuple[str, int, int]] = []

    for seed in seeds:
        cases = [
            ("bpfree_fault_free", True, False),
            ("bpfree_offline_skip", True, True),
            ("1f1b_fault_free", False, False),
            ("1f1b_offline_skip", False, True),
        ]

        for case, is_bpfree, offline in cases:
            if case not in wanted:
                continue

            case_dir = args.output_root / f"{case}_seed{seed}"
            if exists_done(case_dir, bpfree=is_bpfree) and not args.force:
                print(f"SKIP existing {case_dir}")
                continue

            cmd = (
                bpfree_cmd(args, seed=seed, output_dir=case_dir, offline=offline)
                if is_bpfree
                else f1b_cmd(args, seed=seed, output_dir=case_dir, offline=offline)
            )
            rc = run_cmd(
                cmd,
                cwd=REPO_ROOT,
                log_path=case_dir / "run.log",
                dry_run=args.dry_run,
            )
            if rc != 0:
                failures.append((case, seed, rc))

    if failures:
        print("Failures:")
        for item in failures:
            print(item)
        raise SystemExit(1)

    if not args.skip_report:
        run_cmd(
            [sys.executable, str(AGG_REPORT), "--root", str(args.output_root)],
            cwd=REPO_ROOT,
            log_path=args.output_root / "aggregate_graceful_degradation_report.log",
            dry_run=args.dry_run,
        )
        if ACC_REPORT.exists():
            run_cmd(
                [sys.executable, str(ACC_REPORT), "--root", str(args.output_root)],
                cwd=REPO_ROOT,
                log_path=args.output_root / "build_no_recovery_accuracy_table.log",
                dry_run=args.dry_run,
            )


if __name__ == "__main__":
    main()
