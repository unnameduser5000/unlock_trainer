#!/usr/bin/env python3
"""Load LoRA state(s) into a full TinyLlama model and evaluate DroidCall generation metrics."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import torch

from sg_exe_trainer.tasks.label_experiment import (
    configure_lora_trainable,
    inject_lora_adapters,
    lora_parameter_fingerprint,
)
from tools.mobile_agent_eval.run_mobile_actions_lora_sft import (
    evaluate_generation,
    evaluate_loss,
)
from tools.mobile_agent_eval.toolcall_protocol import read_jsonl, rows_to_examples


def load_lora_state(path: Path) -> dict[str, Any]:
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected LoRA state format in {path}")
    return data


def merge_states(paths: list[Path]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for path in paths:
        for name, value in load_lora_state(path).items():
            if name in merged:
                raise ValueError(f"Duplicate LoRA tensor name across files: {name}")
            merged[name] = value
    return merged


def apply_lora_state(model: Any, state: dict[str, Any]) -> None:
    expected = {
        name: param
        for name, param in model.named_parameters()
        if name.endswith("lora_a") or name.endswith("lora_b")
    }
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    if missing or unexpected:
        raise RuntimeError(
            f"LoRA state mismatch: missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    with torch.no_grad():
        for name, param in expected.items():
            param.copy_(state[name].to(dtype=param.dtype, device=param.device))


def write_summary_csv(path: Path, row: dict[str, Any]) -> None:
    fields = list(row.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--lora_state", type=Path, action="append", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_name_or_path", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--eval_limit", type=int, default=None)
    parser.add_argument("--gen_eval_limit", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=192)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--lora_targets", default="q_proj,v_proj")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_init_std", type=float, default=0.01)
    parser.add_argument("--lora_init_seed", type=int, default=None)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.torch_dtype]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(args.data)
    examples = rows_to_examples(rows)
    eval_examples = [ex for ex in examples if ex.split == "eval"]
    if args.eval_limit is not None:
        eval_examples = eval_examples[: args.eval_limit]
    gen_examples = eval_examples[: args.gen_eval_limit]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=dtype)
    model.config.use_cache = False
    inject_lora_adapters(
        module=model,
        target_names={item.strip() for item in args.lora_targets.split(",") if item.strip()},
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        init_std=args.lora_init_std,
        init_seed=args.lora_init_seed,
    )
    configure_lora_trainable(model)
    merged = merge_states(args.lora_state)
    model.to(device)
    apply_lora_state(model, merged)
    applied_fingerprint = lora_parameter_fingerprint(model)

    tuned_loss = evaluate_loss(
        model=model,
        tokenizer=tokenizer,
        examples=eval_examples,
        device=device,
        max_length=args.max_length,
        batch_size=args.eval_batch_size,
    )
    tuned_gen = evaluate_generation(
        model=model,
        tokenizer=tokenizer,
        examples=gen_examples,
        device=device,
        max_input_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        output_jsonl=args.output_dir / "tuned_generations.jsonl",
    )
    summary = {
        "runner": "evaluate_droidcall_lora_from_state",
        "dataset": str(args.data),
        "model_name_or_path": args.model_name_or_path,
        "eval_records": len(eval_examples),
        "generation_eval_records": len(gen_examples),
        "lora_state_files": [str(path) for path in args.lora_state],
        "lora_initialization_fingerprint": applied_fingerprint,
        "tuned_loss": tuned_loss,
        "tuned_generation": tuned_gen,
        "cuda_peak_allocated_mib": (
            torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0) if torch.cuda.is_available() else 0.0
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_summary_csv(
        args.output_dir / "summary.csv",
        {
            "eval_records": len(eval_examples),
            "gen_eval_records": len(gen_examples),
            "tuned_loss": tuned_loss.get("loss"),
            "tuned_full_exact": tuned_gen.get("full_exact"),
            "tuned_tool_exact": tuned_gen.get("tool_exact"),
            "tuned_args_exact": tuned_gen.get("args_exact"),
            "tuned_parse_rate": tuned_gen.get("parse_rate"),
            "cuda_peak_allocated_mib": summary["cuda_peak_allocated_mib"],
        },
    )
    print(f"Wrote {args.output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
