#!/usr/bin/env python3
"""Run DroidCall E1 across full-BP, 1F1B, and BP-free variants."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PREPARE = REPO_ROOT / "tools" / "data" / "prepare_lora_sft_requests.py"
FULL_BP_MODULE = "tools.mobile_agent_eval.run_mobile_actions_lora_sft"
ONE_F1B = REPO_ROOT / "src" / "sg_exe_trainer" / "runtime" / "exactbp" / "distributed_runtime.py"
BPFREE_MODULE = "sg_exe_trainer.runtime.bpfree.orchestrated_runtime"
EVAL_FROM_STATE_MODULE = "tools.mobile_agent_eval.evaluate_droidcall_lora_from_state"
STATUS_FIELDS = ["method", "seed", "phase", "status", "elapsed_s", "output_dir"]


@dataclass(frozen=True)
class Job:
    method: str
    seed: int
    output_dir: Path
    train_command: list[str]
    eval_command: list[str] | None


def parse_ints(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer.")
    return values


def parse_methods(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one method.")
    return values


def count_split(path: Path, split: str) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("split", "")).strip().lower() == split:
                count += 1
    return count


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STATUS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def torchrun_command(world_size: int) -> list[str]:
    return [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={world_size}"]


def ensure_manifest(
    *,
    python: str,
    dataset: str,
    split: str,
    seq_len: int,
    limit: int,
    output_dir: Path,
    request_prefix: str,
    force: bool,
) -> Path:
    manifest_path = output_dir / "requests.jsonl"
    metadata_path = output_dir / "metadata.json"
    if not force and manifest_path.is_file() and metadata_path.is_file():
        return manifest_path
    command = [
        python,
        str(PREPARE),
        "--model_name",
        "tinyllama",
        "--dataset",
        dataset,
        "--split",
        split,
        "--seq_len",
        str(seq_len),
        "--limit",
        str(limit),
        "--output_dir",
        str(output_dir),
        "--request_prefix",
        request_prefix,
        "--attention_mask",
        "causal",
        "--stage0_input",
        "input_ids",
    ]
    subprocess.run(command, check=True, cwd=REPO_ROOT)
    return manifest_path


def build_full_bp_job(args: argparse.Namespace, *, seed: int, data_path: Path, output_dir: Path) -> Job:
    train_command = [
        args.python,
        "-m",
        FULL_BP_MODULE,
        "--data",
        str(data_path),
        "--model_name_or_path",
        args.model_name_or_path,
        "--output_dir",
        str(output_dir),
        "--train_limit",
        str(args.train_limit),
        "--eval_limit",
        str(args.eval_limit),
        "--gen_eval_limit",
        str(args.gen_eval_limit),
        "--max_length",
        str(args.max_length),
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--batch_size",
        str(args.full_bp_batch_size),
        "--eval_batch_size",
        str(args.eval_batch_size),
        "--grad_accum",
        str(args.full_bp_grad_accum),
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
        args.full_bp_device,
        "--log_interval",
        str(args.log_interval),
    ]
    return Job("full_bp_1gpu", seed, output_dir, train_command, None)


def build_1f1b_job(
    args: argparse.Namespace,
    *,
    seed: int,
    data_path: Path,
    train_manifest: Path,
    eval_manifest: Path,
    output_dir: Path,
) -> Job:
    train_command = torchrun_command(3) + [
        str(ONE_F1B),
        "--train_manifest",
        str(train_manifest),
        "--eval_manifest",
        str(eval_manifest),
        "--output_dir",
        str(output_dir),
        "--num_chunks",
        "3",
        "--stage_devices",
        args.stage_devices,
        "--train_limit",
        str(args.train_limit),
        "--eval_limit",
        str(args.eval_limit),
        "--train_epochs",
        str(args.epochs),
        "--batch_size",
        str(args.pipeline_batch_size),
        "--microbatches",
        str(args.pipeline_microbatches),
        "--learning_rate",
        str(args.learning_rate),
        "--grad_clip",
        str(args.grad_clip),
        "--optimizer",
        "adamw",
        "--label_smoothing",
        "0.0",
        "--trainable_mode",
        "lora",
        "--lora_rank",
        str(args.lora_rank),
        "--lora_alpha",
        str(args.lora_alpha),
        "--lora_targets",
        args.lora_targets,
        "--lora_init_std",
        str(args.lora_init_std),
        "--lora_init_seed",
        str(seed),
        "--dtype",
        args.torch_dtype,
        "--seed",
        str(seed),
        "--progress_interval",
        str(args.progress_interval),
        "--skip_eval_before",
        "--skip_eval_after",
    ]
    eval_command = [
        args.python,
        "-m",
        EVAL_FROM_STATE_MODULE,
        "--data",
        str(data_path),
        "--lora_state",
        str(output_dir / "stage0_lora_state.pt"),
        "--lora_state",
        str(output_dir / "stage1_lora_state.pt"),
        "--lora_state",
        str(output_dir / "stage2_lora_state.pt"),
        "--output_dir",
        str(output_dir / "droidcall_eval"),
        "--model_name_or_path",
        args.model_name_or_path,
        "--eval_limit",
        str(args.eval_limit),
        "--gen_eval_limit",
        str(args.gen_eval_limit),
        "--max_length",
        str(args.max_length),
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--eval_batch_size",
        str(args.eval_batch_size),
        "--device",
        args.eval_device,
        "--torch_dtype",
        args.torch_dtype,
        "--lora_targets",
        args.lora_targets,
        "--lora_rank",
        str(args.lora_rank),
        "--lora_alpha",
        str(args.lora_alpha),
        "--lora_init_std",
        str(args.lora_init_std),
        "--lora_init_seed",
        str(seed),
    ]
    return Job("1f1b_3gpu", seed, output_dir, train_command, eval_command)


def build_bpfree_job(
    args: argparse.Namespace,
    *,
    method: str,
    seed: int,
    data_path: Path,
    train_manifest: Path,
    output_dir: Path,
    belief_transport_mode: str,
    alpha: float,
) -> Job:
    train_command = [
        args.python,
        "-m",
        BPFREE_MODULE,
        "--manifest",
        str(train_manifest),
        "--output_dir",
        str(output_dir),
        "--num_chunks",
        "3",
        "--stage_devices",
        args.stage_devices,
        "--topology",
        "phone_fixed",
        "--max_inflight",
        str(args.max_inflight),
        "--scheduler_policy",
        "fifo",
        "--recovery_policy",
        "replay_after_update",
        "--failure_mode",
        "none",
        "--task_timeout_ms",
        "0",
        "--train_chunks",
        "all",
        "--stage_update_policy",
        "stride",
        "--gradient_accumulation_steps",
        str(args.scheduler_grad_accum),
        "--belief_transport_mode",
        belief_transport_mode,
        "--alpha",
        str(alpha),
        "--label_smoothing",
        "0.0",
        "--limit",
        str(args.train_limit),
        "--model_name",
        "tinyllama",
        "--learning_rate",
        str(args.learning_rate),
        "--optimizer",
        "adamw",
        "--grad_clip",
        str(args.grad_clip),
        "--dtype",
        args.torch_dtype,
        "--trainable_mode",
        "lora",
        "--lora_rank",
        str(args.lora_rank),
        "--lora_alpha",
        str(args.lora_alpha),
        "--lora_targets",
        args.lora_targets,
        "--lora_init_std",
        str(args.lora_init_std),
        "--lora_init_seed",
        str(seed),
        "--seed",
        str(seed),
        "--progress_interval",
        str(args.progress_interval),
    ]
    eval_command = [
        args.python,
        "-m",
        EVAL_FROM_STATE_MODULE,
        "--data",
        str(data_path),
        "--lora_state",
        str(output_dir / "stage0_lora_state.pt"),
        "--lora_state",
        str(output_dir / "stage1_lora_state.pt"),
        "--lora_state",
        str(output_dir / "stage2_lora_state.pt"),
        "--output_dir",
        str(output_dir / "droidcall_eval"),
        "--model_name_or_path",
        args.model_name_or_path,
        "--eval_limit",
        str(args.eval_limit),
        "--gen_eval_limit",
        str(args.gen_eval_limit),
        "--max_length",
        str(args.max_length),
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--eval_batch_size",
        str(args.eval_batch_size),
        "--device",
        args.eval_device,
        "--torch_dtype",
        args.torch_dtype,
        "--lora_targets",
        args.lora_targets,
        "--lora_rank",
        str(args.lora_rank),
        "--lora_alpha",
        str(args.lora_alpha),
        "--lora_init_std",
        str(args.lora_init_std),
        "--lora_init_seed",
        str(seed),
    ]
    return Job(method, seed, output_dir, train_command, eval_command)


def run_command(command: list[str], *, cwd: Path, output_dir: Path, log_name: str) -> float:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
    started = time.time()
    log_path = output_dir / log_name
    with log_path.open("w", encoding="utf-8") as handle:
        subprocess.run(command, check=True, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT)
    return time.time() - started


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--protocol", choices=["json_calls", "code_short"], default="json_calls")
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--manifest_root", type=Path, default=REPO_ROOT / "data" / "sft_requests")
    parser.add_argument("--model_name_or_path", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--seeds", default="20260624")
    parser.add_argument(
        "--methods",
        default="full_bp_1gpu,1f1b_3gpu,bpfree_ce_3gpu,bpfree_belief_3gpu",
    )
    parser.add_argument("--train_limit", type=int, default=None)
    parser.add_argument("--eval_limit", type=int, default=None)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=192)
    parser.add_argument("--gen_eval_limit", type=int, default=200)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_targets", default="q_proj,v_proj")
    parser.add_argument("--lora_init_std", type=float, default=0.01)
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--full_bp_device", default="cuda:0")
    parser.add_argument("--eval_device", default="cuda:0")
    parser.add_argument("--stage_devices", default="cuda:0,cuda:1,cuda:2")
    parser.add_argument("--full_bp_batch_size", type=int, default=1)
    parser.add_argument("--full_bp_grad_accum", type=int, default=8)
    parser.add_argument("--pipeline_batch_size", type=int, default=8)
    parser.add_argument("--pipeline_microbatches", type=int, default=8)
    parser.add_argument("--scheduler_grad_accum", type=int, default=8)
    parser.add_argument("--max_inflight", type=int, default=8)
    parser.add_argument("--progress_interval", type=int, default=256)
    parser.add_argument("--log_interval", type=int, default=16)
    parser.add_argument("--force_prepare", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    protocol_dataset = (
        "droidcall_json_calls_v1" if args.protocol == "json_calls" else "droidcall_code_short_v1"
    )
    data_path = args.data or (
        REPO_ROOT / "data" / "droidcall" / f"{protocol_dataset}.jsonl"
    )
    train_limit = args.train_limit or count_split(data_path, "train")
    eval_limit = args.eval_limit or count_split(data_path, "eval")
    args.train_limit = train_limit
    args.eval_limit = eval_limit

    manifest_stem = f"{protocol_dataset}_len{args.seq_len}_light"
    train_manifest = ensure_manifest(
        python=args.python,
        dataset=protocol_dataset,
        split="train",
        seq_len=args.seq_len,
        limit=train_limit,
        output_dir=args.manifest_root / f"{manifest_stem}_train{train_limit}",
        request_prefix=f"{args.protocol}-train",
        force=args.force_prepare,
    )
    eval_manifest = ensure_manifest(
        python=args.python,
        dataset=protocol_dataset,
        split="eval",
        seq_len=args.seq_len,
        limit=eval_limit,
        output_dir=args.manifest_root / f"{manifest_stem}_eval{eval_limit}",
        request_prefix=f"{args.protocol}-eval",
        force=args.force_prepare,
    )

    run_manifest = {
        "protocol": f"DROIDCALL_E1_MULTI_ALGO_{args.protocol.upper()}",
        "data": str(data_path),
        "train_manifest": str(train_manifest),
        "eval_manifest": str(eval_manifest),
        "train_limit": train_limit,
        "eval_limit": eval_limit,
        "seq_len": args.seq_len,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "gen_eval_limit": args.gen_eval_limit,
        "methods": parse_methods(args.methods),
        "seeds": parse_ints(args.seeds),
        "model_name_or_path": args.model_name_or_path,
        "torch_dtype": args.torch_dtype,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(args.output_root / "run_manifest.json", run_manifest)

    methods = parse_methods(args.methods)
    seeds = parse_ints(args.seeds)
    jobs: list[Job] = []
    for seed in seeds:
        for method in methods:
            output_dir = args.output_root / method / f"seed{seed}"
            if method == "full_bp_1gpu":
                jobs.append(build_full_bp_job(args, seed=seed, data_path=data_path, output_dir=output_dir))
            elif method == "1f1b_3gpu":
                jobs.append(
                    build_1f1b_job(
                        args,
                        seed=seed,
                        data_path=data_path,
                        train_manifest=train_manifest,
                        eval_manifest=eval_manifest,
                        output_dir=output_dir,
                    )
                )
            elif method == "bpfree_ce_3gpu":
                jobs.append(
                    build_bpfree_job(
                        args,
                        method=method,
                        seed=seed,
                        data_path=data_path,
                        train_manifest=train_manifest,
                        output_dir=output_dir,
                        belief_transport_mode="none",
                        alpha=1.0,
                    )
                )
            elif method == "bpfree_belief_3gpu":
                jobs.append(
                    build_bpfree_job(
                        args,
                        method=method,
                        seed=seed,
                        data_path=data_path,
                        train_manifest=train_manifest,
                        output_dir=output_dir,
                        belief_transport_mode="full",
                        alpha=0.5,
                    )
                )
            else:
                raise ValueError(f"Unsupported method: {method}")

    status_rows: list[dict[str, Any]] = []
    status_path = args.output_root / "run_status.csv"
    aggregate_rows: list[dict[str, Any]] = []

    for job in jobs:
        summary_path = job.output_dir / "combined_summary.json"
        if not args.force and summary_path.is_file():
            status_rows.append(
                {
                    "method": job.method,
                    "seed": job.seed,
                    "phase": "done",
                    "status": "skipped_existing",
                    "elapsed_s": "",
                    "output_dir": str(job.output_dir),
                }
            )
            write_csv(status_path, status_rows)
            continue

        status_rows.append(
            {
                "method": job.method,
                "seed": job.seed,
                "phase": "train",
                "status": "running",
                "elapsed_s": "",
                "output_dir": str(job.output_dir),
            }
        )
        write_csv(status_path, status_rows)
        try:
            train_elapsed = run_command(job.train_command, cwd=REPO_ROOT, output_dir=job.output_dir, log_name="train.log")
            status_rows[-1]["phase"] = "train"
            status_rows[-1]["status"] = "completed"
            status_rows[-1]["elapsed_s"] = round(train_elapsed, 2)
            write_csv(status_path, status_rows)
        except subprocess.CalledProcessError:
            status_rows[-1]["phase"] = "train"
            status_rows[-1]["status"] = "failed"
            write_csv(status_path, status_rows)
            raise

        if job.eval_command is not None:
            status_rows.append(
                {
                    "method": job.method,
                    "seed": job.seed,
                    "phase": "eval",
                    "status": "running",
                    "elapsed_s": "",
                    "output_dir": str(job.output_dir),
                }
            )
            write_csv(status_path, status_rows)
            try:
                eval_elapsed = run_command(
                    job.eval_command,
                    cwd=REPO_ROOT,
                    output_dir=job.output_dir / "droidcall_eval",
                    log_name="eval.log",
                )
                status_rows[-1]["phase"] = "eval"
                status_rows[-1]["status"] = "completed"
                status_rows[-1]["elapsed_s"] = round(eval_elapsed, 2)
                write_csv(status_path, status_rows)
            except subprocess.CalledProcessError:
                status_rows[-1]["phase"] = "eval"
                status_rows[-1]["status"] = "failed"
                write_csv(status_path, status_rows)
                raise

        train_summary_file = (
            job.output_dir / "summary.json"
            if (job.output_dir / "summary.json").is_file()
            else job.output_dir / "scheduler_summary.json"
        )
        train_summary = read_json(train_summary_file)
        eval_summary = read_json(job.output_dir / "droidcall_eval" / "summary.json") if job.eval_command else train_summary
        combined = {
            "method": job.method,
            "seed": job.seed,
            "protocol": args.protocol,
            "train_limit": train_limit,
            "eval_limit": eval_limit,
            "seq_len": args.seq_len,
            "train_summary_file": str(train_summary_file),
            "eval_summary_file": str(job.output_dir / "droidcall_eval" / "summary.json") if job.eval_command else "",
            "train_summary": train_summary,
            "eval_summary": eval_summary,
        }
        write_json(summary_path, combined)
        tuned_gen = eval_summary.get("tuned_generation", {})
        aggregate_rows.append(
            {
                "method": job.method,
                "seed": job.seed,
                "protocol": args.protocol,
                "train_limit": train_limit,
                "eval_limit": eval_limit,
                "full_exact": tuned_gen.get("full_exact", ""),
                "tool_exact": tuned_gen.get("tool_exact", ""),
                "args_exact": tuned_gen.get("args_exact", ""),
                "parse_rate": tuned_gen.get("parse_rate", ""),
                "train_summary_file": str(train_summary_file),
                "eval_summary_file": str(job.output_dir / "droidcall_eval" / "summary.json") if job.eval_command else "",
                "output_dir": str(job.output_dir),
            }
        )
        write_json(args.output_root / "aggregate_summary.json", {"rows": aggregate_rows})
        with (args.output_root / "aggregate_summary.csv").open("w", encoding="utf-8", newline="") as handle:
            fields = list(aggregate_rows[0].keys()) if aggregate_rows else []
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(aggregate_rows)


if __name__ == "__main__":
    main()
