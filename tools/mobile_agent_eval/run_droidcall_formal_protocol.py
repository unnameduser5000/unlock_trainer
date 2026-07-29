#!/usr/bin/env python3
"""Run fixed DroidCall quality protocols across multiple algorithm-style entries."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


FIXED_SEEDS = [20260624, 20260625, 20260626]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--protocol", choices=["json_calls", "code_short"], default="json_calls")
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--model_name_or_path", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--gen_eval_limit", type=int, default=200)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=192)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_init_std", type=float, default=0.01)
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log_interval", type=int, default=250)
    args = parser.parse_args()

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    data = args.data or Path(
        "data/droidcall/droidcall_json_calls_v1.jsonl"
        if args.protocol == "json_calls"
        else "data/droidcall/droidcall_code_short_v1.jsonl"
    )
    manifest = {
        "protocol": f"FORMAL_DROIDCALL_{args.protocol.upper()}_QUALITY_PROTOCOL",
        "fixed_seeds": FIXED_SEEDS,
        "data": str(data),
        "model_name_or_path": args.model_name_or_path,
        "gen_eval_limit": args.gen_eval_limit,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "grad_accum": args.grad_accum,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_init_std": args.lora_init_std,
        "torch_dtype": args.torch_dtype,
        "device": args.device,
        "log_interval": args.log_interval,
    }
    (output_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    status_rows: list[dict[str, Any]] = []
    status_path = output_root / "run_status.csv"
    runner_module = "tools.mobile_agent_eval.run_mobile_actions_lora_sft"
    for seed in FIXED_SEEDS:
        run_dir = output_root / f"seed{seed}"
        command = [
            args.python,
            "-m",
            runner_module,
            "--data",
            str(data),
            "--model_name_or_path",
            args.model_name_or_path,
            "--output_dir",
            str(run_dir),
            "--gen_eval_limit",
            str(args.gen_eval_limit),
            "--max_length",
            str(args.max_length),
            "--max_new_tokens",
            str(args.max_new_tokens),
            "--batch_size",
            str(args.batch_size),
            "--eval_batch_size",
            str(args.eval_batch_size),
            "--grad_accum",
            str(args.grad_accum),
            "--epochs",
            str(args.epochs),
            "--learning_rate",
            str(args.learning_rate),
            "--lora_rank",
            str(args.lora_rank),
            "--lora_alpha",
            str(args.lora_alpha),
            "--lora_init_std",
            str(args.lora_init_std),
            "--seed",
            str(seed),
            "--lora_init_seed",
            str(seed),
            "--torch_dtype",
            args.torch_dtype,
            "--device",
            args.device,
            "--log_interval",
            str(args.log_interval),
        ]
        started = time.time()
        row = {
            "seed": seed,
            "status": "running",
            "run_dir": str(run_dir),
            "protocol": args.protocol,
            "start_ts": int(started),
            "end_ts": "",
            "wall_s": "",
        }
        status_rows.append(row)
        write_csv(status_path, status_rows)
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError:
            row["status"] = "failed"
            row["end_ts"] = int(time.time())
            row["wall_s"] = int(float(row["end_ts"]) - started)
            write_csv(status_path, status_rows)
            raise
        row["status"] = "completed"
        row["end_ts"] = int(time.time())
        row["wall_s"] = int(float(row["end_ts"]) - started)
        write_csv(status_path, status_rows)


if __name__ == "__main__":
    main()
