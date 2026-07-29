#!/usr/bin/env python3
"""LoRA-SFT and evaluate a small LM on Google Mobile Actions.

This script intentionally avoids PEFT/TRL so it can run in the existing server
environment.  It uses the local lightweight LoRA module already used by the
BP-free experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any


from sg_exe_trainer.tasks.label_experiment import (
    configure_lora_trainable,
    inject_lora_adapters,
    lora_parameter_fingerprint,
)
from tools.mobile_agent_eval.toolcall_protocol import (
    ToolCallExample,
    parse_prediction_text,
    read_jsonl,
    rows_to_examples,
)


def encode_sft_example(
    tokenizer: Any,
    prompt: str,
    target: str,
    max_length: int,
) -> dict[str, Any] | None:
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    target_ids = tokenizer(target + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    if len(input_ids) > max_length:
        overflow = len(input_ids) - max_length
        if overflow >= len(prompt_ids) - 8:
            return None
        prompt_ids = prompt_ids[overflow:]
        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
    return {"input_ids": input_ids, "labels": labels}


def collate_batch(features: list[dict[str, Any]], tokenizer: Any) -> dict[str, Any]:
    import torch

    max_len = max(len(item["input_ids"]) for item in features)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids = []
    labels = []
    attention_mask = []
    for item in features:
        pad = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_id] * pad)
        labels.append(item["labels"] + [-100] * pad)
        attention_mask.append([1] * len(item["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def iter_batches(items: list[Any], batch_size: int) -> list[list[Any]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def evaluate_loss(
    *,
    model: Any,
    tokenizer: Any,
    examples: list[ToolCallExample],
    device: Any,
    max_length: int,
    batch_size: int,
) -> dict[str, float]:
    import torch

    model.eval()
    encoded = [
        encoded
        for ex in examples
        if (encoded := encode_sft_example(tokenizer, ex.prompt, ex.target, max_length)) is not None
    ]
    total_loss = 0.0
    total_tokens = 0
    started = time.perf_counter()
    with torch.no_grad():
        for batch_items in iter_batches(encoded, batch_size):
            batch = collate_batch(batch_items, tokenizer)
            batch = {key: value.to(device) for key, value in batch.items()}
            out = model(**batch)
            target_tokens = int((batch["labels"] != -100).sum().item())
            total_loss += float(out.loss.item()) * target_tokens
            total_tokens += target_tokens
    elapsed = time.perf_counter() - started
    avg_loss = total_loss / max(1, total_tokens)
    return {
        "records": float(len(encoded)),
        "target_tokens": float(total_tokens),
        "loss": avg_loss,
        "perplexity": math.exp(min(20.0, avg_loss)),
        "records_per_s": len(encoded) / elapsed if elapsed > 0 else 0.0,
        "wall_s": elapsed,
    }


def evaluate_generation(
    *,
    model: Any,
    tokenizer: Any,
    examples: list[ToolCallExample],
    device: Any,
    max_input_length: int,
    max_new_tokens: int,
    output_jsonl: Path,
) -> dict[str, float]:
    import torch

    model.eval()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    parse_ok = 0
    tool_exact = 0
    args_exact = 0
    full_exact = 0
    started = time.perf_counter()
    ttft_proxy_ms: list[float] = []
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for idx, ex in enumerate(examples):
            prompt_ids = tokenizer(
                ex.prompt,
                add_special_tokens=True,
                truncation=True,
                max_length=max_input_length,
                return_tensors="pt",
            )
            prompt_ids = {key: value.to(device) for key, value in prompt_ids.items()}
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)
            item_started = time.perf_counter()
            with torch.no_grad():
                generated = model.generate(
                    **prompt_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            item_ms = (time.perf_counter() - item_started) * 1000.0
            ttft_proxy_ms.append(item_ms)
            new_tokens = generated[0, prompt_ids["input_ids"].shape[1] :]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            parsed, pred_calls, parsed_ok, parsed_format = parse_prediction_text(text, ex.target_format)
            expected_tools = [call["name"] for call in ex.expected_calls]
            pred_tools = [call["name"] for call in pred_calls]
            tools_match = pred_tools == expected_tools
            args_match = pred_calls == ex.expected_calls
            parse_ok += int(parsed_ok)
            tool_exact += int(tools_match)
            args_exact += int(args_match)
            full_exact += int(parsed_ok and args_match)
            handle.write(
                json.dumps(
                    {
                        "index": idx,
                        "user": ex.user,
                        "expected": {"calls": ex.expected_calls},
                        "generated_text": text,
                        "parsed": parsed,
                        "parsed_format": parsed_format,
                        "target_format": ex.target_format,
                        "predicted_calls": pred_calls,
                        "parse_ok": parsed_ok,
                        "tool_exact": tools_match,
                        "args_exact": args_match,
                        "latency_ms": item_ms,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    elapsed = time.perf_counter() - started
    count = len(examples)
    return {
        "records": float(count),
        "parse_rate": parse_ok / max(1, count),
        "tool_exact": tool_exact / max(1, count),
        "args_exact": args_exact / max(1, count),
        "full_exact": full_exact / max(1, count),
        "avg_latency_ms": sum(ttft_proxy_ms) / max(1, len(ttft_proxy_ms)),
        "records_per_s": count / elapsed if elapsed > 0 else 0.0,
        "wall_s": elapsed,
    }


def train_lora(
    *,
    model: Any,
    tokenizer: Any,
    examples: list[ToolCallExample],
    device: Any,
    output_dir: Path,
    max_length: int,
    batch_size: int,
    grad_accum: int,
    epochs: int,
    learning_rate: float,
    max_grad_norm: float,
    log_interval: int,
    seed: int,
) -> dict[str, Any]:
    import torch

    encoded = [
        encoded
        for ex in examples
        if (encoded := encode_sft_example(tokenizer, ex.prompt, ex.target, max_length)) is not None
    ]
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    model.train()
    records_seen = 0
    optimizer_steps = 0
    running_loss = 0.0
    train_csv = output_dir / "train_steps.csv"
    train_csv.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with train_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "batch_index", "records_seen", "optimizer_steps", "loss", "lr", "elapsed_s"],
        )
        writer.writeheader()
        for epoch in range(epochs):
            generator = torch.Generator().manual_seed(seed + epoch)
            order = torch.randperm(len(encoded), generator=generator).tolist()
            shuffled = [encoded[i] for i in order]
            optimizer.zero_grad(set_to_none=True)
            for batch_index, batch_items in enumerate(iter_batches(shuffled, batch_size), start=1):
                batch = collate_batch(batch_items, tokenizer)
                batch = {key: value.to(device) for key, value in batch.items()}
                out = model(**batch)
                loss = out.loss / grad_accum
                loss.backward()
                records_seen += len(batch_items)
                running_loss += float(out.loss.item())
                should_step = batch_index % grad_accum == 0 or batch_index == math.ceil(len(shuffled) / batch_size)
                if should_step:
                    if max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in model.parameters() if p.requires_grad],
                            max_grad_norm,
                        )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1
                if batch_index % log_interval == 0 or batch_index == math.ceil(len(shuffled) / batch_size):
                    avg_loss = running_loss / max(1, log_interval)
                    running_loss = 0.0
                    row = {
                        "epoch": epoch,
                        "batch_index": batch_index,
                        "records_seen": records_seen,
                        "optimizer_steps": optimizer_steps,
                        "loss": avg_loss,
                        "lr": learning_rate,
                        "elapsed_s": time.perf_counter() - started,
                    }
                    writer.writerow(row)
                    handle.flush()
                    print(
                        f"train epoch={epoch} batch={batch_index} records={records_seen} "
                        f"steps={optimizer_steps} loss={avg_loss:.4f}",
                        flush=True,
                    )
    return {
        "train_records": len(encoded),
        "epochs": epochs,
        "optimizer_steps": optimizer_steps,
        "wall_s": time.perf_counter() - started,
    }


def save_lora_state(model: Any, output_dir: Path) -> None:
    import torch

    state = {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if "lora_a" in name or "lora_b" in name
    }
    torch.save(state, output_dir / "lora_state.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model_name_or_path", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--train_limit", type=int, default=None)
    parser.add_argument("--eval_limit", type=int, default=None)
    parser.add_argument("--gen_eval_limit", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=192)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lora_targets", default="q_proj,v_proj")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_init_std", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--lora_init_seed", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--skip_base_generation", action="store_true")
    parser.add_argument("--log_interval", type=int, default=50)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        np = None

    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.torch_dtype]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lora_init_seed = args.lora_init_seed if args.lora_init_seed is not None else args.seed

    random.seed(args.seed)
    if np is not None:
        np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    rows = read_jsonl(args.data)
    examples = rows_to_examples(rows)
    train_examples = [ex for ex in examples if ex.split == "train"]
    eval_examples = [ex for ex in examples if ex.split == "eval"]
    if args.train_limit is not None:
        train_examples = train_examples[: args.train_limit]
    if args.eval_limit is not None:
        eval_examples = eval_examples[: args.eval_limit]
    gen_examples = eval_examples[: args.gen_eval_limit]

    print(
        f"dataset train={len(train_examples)} eval={len(eval_examples)} gen_eval={len(gen_examples)}",
        flush=True,
    )
    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=dtype)
    model.config.use_cache = False
    injected = inject_lora_adapters(
        module=model,
        target_names={item.strip() for item in args.lora_targets.split(",") if item.strip()},
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        init_std=args.lora_init_std,
        init_seed=lora_init_seed,
    )
    trainable, frozen = configure_lora_trainable(model)
    lora_init_fingerprint = lora_parameter_fingerprint(model)
    model.to(device)
    load_s = time.perf_counter() - load_started
    print(
        f"loaded model in {load_s:.2f}s lora_modules={injected} trainable={trainable} "
        f"frozen={frozen} seed={args.seed} lora_init={lora_init_fingerprint[:12]}",
        flush=True,
    )

    base_loss = evaluate_loss(
        model=model,
        tokenizer=tokenizer,
        examples=eval_examples,
        device=device,
        max_length=args.max_length,
        batch_size=args.eval_batch_size,
    )
    print(f"base eval loss={base_loss['loss']:.4f} ppl={base_loss['perplexity']:.2f}", flush=True)
    base_gen = {}
    if not args.skip_base_generation and gen_examples:
        base_gen = evaluate_generation(
            model=model,
            tokenizer=tokenizer,
            examples=gen_examples,
            device=device,
            max_input_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            output_jsonl=args.output_dir / "base_generations.jsonl",
        )
        print(f"base generation full_exact={base_gen['full_exact']:.4f}", flush=True)

    train_summary = train_lora(
        model=model,
        tokenizer=tokenizer,
        examples=train_examples,
        device=device,
        output_dir=args.output_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        log_interval=args.log_interval,
        seed=args.seed,
    )
    save_lora_state(model, args.output_dir)

    tuned_loss = evaluate_loss(
        model=model,
        tokenizer=tokenizer,
        examples=eval_examples,
        device=device,
        max_length=args.max_length,
        batch_size=args.eval_batch_size,
    )
    print(f"tuned eval loss={tuned_loss['loss']:.4f} ppl={tuned_loss['perplexity']:.2f}", flush=True)
    tuned_gen = {}
    if gen_examples:
        tuned_gen = evaluate_generation(
            model=model,
            tokenizer=tokenizer,
            examples=gen_examples,
            device=device,
            max_input_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            output_jsonl=args.output_dir / "tuned_generations.jsonl",
        )
        print(f"tuned generation full_exact={tuned_gen['full_exact']:.4f}", flush=True)

    summary = {
        "runner": "mobile_actions_lora_sft",
        "dataset": str(args.data),
        "model_name_or_path": args.model_name_or_path,
        "train_records": len(train_examples),
        "eval_records": len(eval_examples),
        "generation_eval_records": len(gen_examples),
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "optimizer_batch": args.batch_size * args.grad_accum,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "lora_targets": args.lora_targets,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_init_std": args.lora_init_std,
        "lora_init_seed": lora_init_seed,
        "lora_initialization_fingerprint": lora_init_fingerprint,
        "lora_modules": injected,
        "trainable_params": trainable,
        "frozen_params": frozen,
        "model_load_s": load_s,
        "base_loss": base_loss,
        "base_generation": base_gen,
        "train": train_summary,
        "tuned_loss": tuned_loss,
        "tuned_generation": tuned_gen,
        "cuda_peak_allocated_mib": (
            torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0) if torch.cuda.is_available() else 0.0
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "model",
            "train_records",
            "eval_records",
            "gen_eval_records",
            "base_loss",
            "tuned_loss",
            "base_full_exact",
            "tuned_full_exact",
            "base_tool_exact",
            "tuned_tool_exact",
            "base_args_exact",
            "tuned_args_exact",
            "train_wall_s",
            "cuda_peak_allocated_mib",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "model": args.model_name_or_path,
                "train_records": len(train_examples),
                "eval_records": len(eval_examples),
                "gen_eval_records": len(gen_examples),
                "base_loss": base_loss.get("loss"),
                "tuned_loss": tuned_loss.get("loss"),
                "base_full_exact": base_gen.get("full_exact", ""),
                "tuned_full_exact": tuned_gen.get("full_exact", ""),
                "base_tool_exact": base_gen.get("tool_exact", ""),
                "tuned_tool_exact": tuned_gen.get("tool_exact", ""),
                "base_args_exact": base_gen.get("args_exact", ""),
                "tuned_args_exact": tuned_gen.get("args_exact", ""),
                "train_wall_s": train_summary.get("wall_s"),
                "cuda_peak_allocated_mib": summary["cuda_peak_allocated_mib"],
            }
        )
    print(f"Wrote {args.output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
