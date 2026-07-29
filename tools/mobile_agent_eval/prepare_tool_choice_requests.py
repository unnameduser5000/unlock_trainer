#!/usr/bin/env python3
"""Prepare first-tool multiple-choice manifests from tool-calling datasets."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tools.data.prepare_lora_sft_requests import (
    build_attention_mask,
    build_token_tensors,
    count_valid_labels,
    resolve_model_name,
    tensor_record,
    write_tensor,
)
from tools.mobile_agent_eval.toolcall_protocol import (
    normalize_tool_spec,
    read_jsonl,
    rows_to_examples,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_PRESETS = {
    "mobile_actions": REPO_ROOT / "data" / "mobile_actions" / "mobile-actions.jsonl",
    "droidcall_json_calls_v1": REPO_ROOT / "data" / "droidcall" / "droidcall_json_calls_v1.jsonl",
    "droidcall_code_short_v1": REPO_ROOT / "data" / "droidcall" / "droidcall_code_short_v1.jsonl",
}

CHOICE_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def resolve_dataset_path(raw: str) -> Path:
    preset = DATASET_PRESETS.get(raw)
    if preset is not None:
        return preset
    path = Path(raw)
    if not path.is_absolute():
        path = (REPO_ROOT / raw).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset JSONL not found: {path}")
    return path


def truncate_description(text: str, limit: int) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 3)].rstrip() + "..."


def normalize_candidates(raw_tools: list[dict[str, Any]], *, description_chars: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for raw_tool in raw_tools:
        if isinstance(raw_tool.get("function"), dict) and raw_tool.get("type") is None:
            raw_tool = {"type": "function", "function": raw_tool["function"]}
        spec = normalize_tool_spec(raw_tool)
        name = str(spec.get("name", "")).strip()
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        description = truncate_description(str(spec.get("description", "")).strip(), description_chars)
        required = [str(item) for item in (spec.get("required") or [])]
        candidates.append(
            {
                "name": name,
                "description": description,
                "required": required,
            }
        )
    return candidates


def render_prompt(*, user: str, candidates: list[dict[str, Any]]) -> str:
    lines = [
        "You are choosing the first on-device tool call.",
        "Pick the best first tool for the user request.",
        "If the request needs multiple actions, choose only the first tool to execute.",
        "Answer with one uppercase letter only.",
        "",
        "User request:",
        user.strip(),
        "",
        "Candidate tools:",
    ]
    for idx, candidate in enumerate(candidates):
        letter = CHOICE_LETTERS[idx]
        desc = candidate["description"] or "No description."
        required = candidate["required"]
        required_suffix = f" required={','.join(required)}" if required else ""
        lines.append(f"{letter}. {candidate['name']} - {desc}{required_suffix}")
    lines.extend(["", "Answer:"])
    return "\n".join(lines)


def encode_label_choices(tokenizer: Any, candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    label_choices: list[dict[str, Any]] = []
    letter_to_tool: dict[str, str] = {}
    for idx, candidate in enumerate(candidates):
        if idx >= len(CHOICE_LETTERS):
            raise ValueError(f"Too many candidates for one-token letter labels: {len(candidates)}")
        letter = CHOICE_LETTERS[idx]
        choice_text = f" {letter}"
        token_ids = tokenizer(choice_text, add_special_tokens=False)["input_ids"]
        if len(token_ids) != 1:
            raise ValueError(
                f"Choice label {choice_text!r} tokenized to {token_ids}; expected exactly one token."
            )
        label_choices.append(
            {
                "text": choice_text,
                "token_ids": token_ids,
                "tool_name": candidate["name"],
            }
        )
        letter_to_tool[letter] = candidate["name"]
    return label_choices, letter_to_tool


def should_keep_split(example_split: str, requested_split: str) -> bool:
    normalized = requested_split.strip().lower()
    if not normalized or normalized in {"all", "*"}:
        return True
    return example_split.strip().lower() == normalized


def build_records(
    *,
    rows: list[dict[str, Any]],
    tokenizer: Any,
    dataset_name: str,
    split: str,
    description_chars: int,
    single_call_only: bool,
    offset: int,
    limit: int,
) -> tuple[list[dict[str, Any]], Counter]:
    examples = rows_to_examples(rows)
    if len(examples) != len(rows):
        raise RuntimeError(f"Row/example length mismatch: {len(rows)} vs {len(examples)}")

    stats: Counter[str] = Counter()
    prepared: list[dict[str, Any]] = []
    for dataset_index, (row, example) in enumerate(zip(rows, examples)):
        if not should_keep_split(example.split, split):
            stats["drop_split"] += 1
            continue
        if single_call_only and len(example.expected_calls) != 1:
            stats["drop_multi_action"] += 1
            continue
        if not example.expected_calls:
            stats["drop_no_expected_calls"] += 1
            continue
        candidates = normalize_candidates(list(row.get("tools") or []), description_chars=description_chars)
        if len(candidates) < 2:
            stats["drop_too_few_candidates"] += 1
            continue
        if len(candidates) > len(CHOICE_LETTERS):
            stats["drop_too_many_candidates"] += 1
            continue
        expected_first_tool = str(example.expected_calls[0].get("name", "")).strip()
        answer_index = next((idx for idx, item in enumerate(candidates) if item["name"] == expected_first_tool), None)
        if answer_index is None:
            stats["drop_missing_gold_tool"] += 1
            continue
        label_choices, letter_to_tool = encode_label_choices(tokenizer, candidates)
        answer_letter = CHOICE_LETTERS[answer_index]
        prepared.append(
            {
                "dataset_index": dataset_index,
                "split": example.split,
                "example_id": example.example_id or f"row{dataset_index}",
                "prompt": render_prompt(user=example.user, candidates=candidates),
                "response": f" {answer_letter}",
                "label_choices": label_choices,
                "user": example.user,
                "candidate_tools": candidates,
                "choice_map": letter_to_tool,
                "expected_first_tool": expected_first_tool,
                "expected_calls": example.expected_calls,
                "multi_action": len(example.expected_calls) > 1,
                "source_dataset": dataset_name,
            }
        )
        stats["kept"] += 1

    selected = prepared[offset : offset + limit]
    stats["selected"] = len(selected)
    stats["available_after_filter"] = len(prepared)
    return selected, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="tinyllama")
    parser.add_argument("--dataset", default="mobile_actions")
    parser.add_argument("--split", default="train")
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--description_chars", type=int, default=120)
    parser.add_argument("--single_call_only", action="store_true")
    parser.add_argument("--stage0_input", choices=["input_ids", "hidden_states"], default="input_ids")
    parser.add_argument("--attention_mask", choices=["zero", "causal"], default="causal")
    parser.add_argument("--mask_prompt", action="store_true", default=True)
    parser.add_argument("--no-mask_prompt", dest="mask_prompt", action="store_false")
    parser.add_argument("--max_prompt_tokens", type=int, default=0)
    parser.add_argument("--no_append_eos", action="store_true")
    parser.add_argument("--min_valid_labels", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1.0e-4)
    parser.add_argument("--request_prefix", default="tool-choice")
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    if args.limit <= 0:
        raise ValueError("limit must be positive.")
    if args.offset < 0:
        raise ValueError("offset must be non-negative.")

    dataset_path = resolve_dataset_path(args.dataset)
    resolved_model = resolve_model_name(args.model_name)

    print(f"Loading tokenizer/model: {resolved_model}")
    tokenizer = AutoTokenizer.from_pretrained(resolved_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    embedding = None
    if args.stage0_input == "hidden_states":
        model = AutoModelForCausalLM.from_pretrained(resolved_model, torch_dtype=torch.float32)
        model.eval()
        embedding = model.get_input_embeddings()

    rows = read_jsonl(dataset_path)
    prepared, stats = build_records(
        rows=rows,
        tokenizer=tokenizer,
        dataset_name=args.dataset,
        split=args.split,
        description_chars=args.description_chars,
        single_call_only=args.single_call_only,
        offset=args.offset,
        limit=args.limit,
    )
    if not prepared:
        raise RuntimeError(f"No examples selected from {dataset_path} with split={args.split!r}. stats={dict(stats)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tensor_dir = args.output_dir / "tensors"
    manifest_path = args.output_dir / "requests.jsonl"
    preview_path = args.output_dir / "examples.jsonl"
    metadata_path = args.output_dir / "metadata.json"

    with manifest_path.open("w", encoding="utf-8") as manifest, preview_path.open("w", encoding="utf-8") as preview:
        for row_index, example in enumerate(prepared, start=1):
            input_ids, labels, prompt_token_count = build_token_tensors(
                tokenizer=tokenizer,
                prompt=example["prompt"],
                response=example["response"],
                seq_len=args.seq_len,
                mask_prompt=args.mask_prompt,
                max_prompt_tokens=args.max_prompt_tokens,
                append_eos=not args.no_append_eos,
            )
            valid_label_count = count_valid_labels(labels)
            if valid_label_count < args.min_valid_labels:
                continue

            attention_mask = build_attention_mask(args.seq_len, args.attention_mask).numpy().astype("<f4")
            input_ids_np = input_ids.cpu().numpy().astype("<i8")
            position_ids = torch.arange(args.seq_len, dtype=torch.long).unsqueeze(0).numpy().astype("<i8")
            labels_np = labels.cpu().numpy().astype("<i8")

            request_id = f"{args.request_prefix}-{example['dataset_index']:06d}"
            hidden_path = tensor_dir / f"{request_id}_hidden_states.f32"
            input_ids_path = tensor_dir / f"{request_id}_input_ids.i64"
            mask_path = tensor_dir / f"{request_id}_attention_mask.f32"
            pos_path = tensor_dir / f"{request_id}_position_ids.i64"
            labels_path = tensor_dir / f"{request_id}_labels.i64"

            write_tensor(mask_path, attention_mask)
            write_tensor(pos_path, position_ids)
            write_tensor(labels_path, labels_np)

            if args.stage0_input == "hidden_states":
                assert embedding is not None
                with torch.no_grad():
                    hidden_states = embedding(input_ids).float().cpu().numpy().astype("<f4")
                write_tensor(hidden_path, hidden_states)
                stage0_tensors = {
                    "hidden_states": tensor_record(
                        args.output_dir,
                        hidden_path,
                        "float32",
                        list(hidden_states.shape),
                    )
                }
            else:
                write_tensor(input_ids_path, input_ids_np)
                stage0_tensors = {
                    "input_ids": tensor_record(
                        args.output_dir,
                        input_ids_path,
                        "int64",
                        list(input_ids_np.shape),
                    )
                }

            record = {
                "request_id": request_id,
                "batch_id": 1,
                "chunk_idx": 0,
                "model_name": args.model_name,
                "dataset": f"{args.dataset}_first_tool_choice",
                "dataset_index": example["dataset_index"],
                "seq_len": args.seq_len,
                "prompt_token_count": prompt_token_count,
                "valid_label_count": valid_label_count,
                "learning_rate": args.learning_rate,
                "label_choices": example["label_choices"],
                "stage0_input": args.stage0_input,
                "benchmark_task": "first_tool_choice",
                "source_split": example["split"],
                "example_id": example["example_id"],
                "candidate_tools": example["candidate_tools"],
                "expected_first_tool": example["expected_first_tool"],
                "multi_action": example["multi_action"],
                "tensors": {
                    **stage0_tensors,
                    "attention_mask": tensor_record(args.output_dir, mask_path, "float32", list(attention_mask.shape)),
                    "position_ids": tensor_record(args.output_dir, pos_path, "int64", list(position_ids.shape)),
                    "labels": tensor_record(args.output_dir, labels_path, "int64", list(labels_np.shape)),
                },
                "text": {
                    "prompt": example["prompt"],
                    "response": example["response"],
                    "user": example["user"],
                },
            }
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            preview.write(
                json.dumps(
                    {
                        "row_index": row_index,
                        "request_id": request_id,
                        "dataset_index": example["dataset_index"],
                        "example_id": example["example_id"],
                        "split": example["split"],
                        "multi_action": example["multi_action"],
                        "user": example["user"],
                        "expected_first_tool": example["expected_first_tool"],
                        "choice_map": example["choice_map"],
                        "prompt": example["prompt"],
                        "response": example["response"],
                        "expected_calls": example["expected_calls"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    metadata = {
        "model_name": args.model_name,
        "resolved_model": resolved_model,
        "dataset": args.dataset,
        "dataset_path": str(dataset_path),
        "benchmark_task": "first_tool_choice",
        "split": args.split,
        "single_call_only": args.single_call_only,
        "seq_len": args.seq_len,
        "limit": args.limit,
        "offset": args.offset,
        "attention_mask": args.attention_mask,
        "mask_prompt": args.mask_prompt,
        "append_eos": not args.no_append_eos,
        "max_prompt_tokens": args.max_prompt_tokens,
        "min_valid_labels": args.min_valid_labels,
        "learning_rate": args.learning_rate,
        "description_chars": args.description_chars,
        "stage0_input": args.stage0_input,
        "records_written": sum(1 for _ in manifest_path.open("r", encoding="utf-8")),
        "selection_stats": dict(stats),
        "manifest": manifest_path.name,
        "preview": preview_path.name,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {metadata['records_written']} request record(s) to {manifest_path}")
    print(json.dumps(metadata["selection_stats"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
