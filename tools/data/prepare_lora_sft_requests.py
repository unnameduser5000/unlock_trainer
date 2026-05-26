import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PRESETS = {
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "smollm2_360m": "HuggingFaceTB/SmolLM2-360M",
    "phi2": "microsoft/phi-2",
}

DATASET_PRESETS = {
    "dolly": ("databricks/databricks-dolly-15k", "train"),
    "alpaca": ("tatsu-lab/alpaca", "train"),
    "gsm8k": ("openai/gsm8k", "main/train"),
}


def resolve_model_name(model_name: str) -> str:
    return MODEL_PRESETS.get(model_name, model_name)


def resolve_dataset(raw: str) -> tuple[str, str]:
    if raw in DATASET_PRESETS:
        return DATASET_PRESETS[raw]
    if "/" not in raw:
        raise ValueError(f"Unknown dataset preset: {raw}")
    return raw, "train"


def split_config(raw_split: str) -> tuple[str | None, str]:
    if "/" in raw_split:
        config, split = raw_split.split("/", 1)
        return config, split
    return None, raw_split


def format_example(example: dict[str, Any], dataset_name: str) -> tuple[str, str]:
    if "databricks-dolly-15k" in dataset_name:
        instruction = str(example.get("instruction", "")).strip()
        context = str(example.get("context", "")).strip()
        response = str(example.get("response", "")).strip()
        prompt = f"Instruction:\n{instruction}\n"
        if context:
            prompt += f"\nContext:\n{context}\n"
        prompt += "\nResponse:\n"
        return prompt, response

    if "alpaca" in dataset_name:
        instruction = str(example.get("instruction", "")).strip()
        input_text = str(example.get("input", "")).strip()
        response = str(example.get("output", "")).strip()
        prompt = f"Instruction:\n{instruction}\n"
        if input_text:
            prompt += f"\nInput:\n{input_text}\n"
        prompt += "\nResponse:\n"
        return prompt, response

    if "gsm8k" in dataset_name:
        question = str(example.get("question", "")).strip()
        answer = str(example.get("answer", "")).strip()
        return f"Question:\n{question}\n\nAnswer:\n", answer

    text = str(example.get("text", "")).strip()
    if not text:
        raise ValueError("Generic dataset example must contain a non-empty 'text' field.")
    return "", text


def build_token_tensors(
    tokenizer,
    prompt: str,
    response: str,
    seq_len: int,
    mask_prompt: bool,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        response_ids = response_ids + [eos_id]

    input_ids = (prompt_ids + response_ids)[:seq_len]
    actual_token_count = len(input_ids)
    if actual_token_count < seq_len:
        pad_id = tokenizer.pad_token_id
        input_ids = input_ids + [pad_id] * (seq_len - actual_token_count)

    labels = input_ids.copy()
    prompt_token_count = min(len(prompt_ids), seq_len)
    if mask_prompt:
        labels[:prompt_token_count] = [-100] * prompt_token_count
    if actual_token_count < seq_len:
        labels[actual_token_count:] = [-100] * (seq_len - actual_token_count)

    return (
        torch.tensor(input_ids, dtype=torch.long).unsqueeze(0),
        torch.tensor(labels, dtype=torch.long).unsqueeze(0),
        prompt_token_count,
    )


def count_valid_labels(labels: torch.Tensor) -> int:
    return int((labels != -100).sum().item())


def build_attention_mask(seq_len: int, mode: str) -> torch.Tensor:
    if mode == "zero":
        return torch.zeros((1, 1, seq_len, seq_len), dtype=torch.float32)
    if mode == "causal":
        mask = torch.zeros((seq_len, seq_len), dtype=torch.float32)
        mask = mask.masked_fill(torch.triu(torch.ones_like(mask), diagonal=1).bool(), -1.0e4)
        return mask.unsqueeze(0).unsqueeze(0)
    raise ValueError(f"Unsupported attention mask mode: {mode}")


def write_tensor(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array.tofile(path)


def tensor_record(base_dir: Path, path: Path, dtype: str, shape: list[int]) -> dict[str, Any]:
    return {
        "path": path.relative_to(base_dir).as_posix(),
        "dtype": dtype,
        "shape": shape,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare real SFT samples as SID ForwardChunkRequest tensor manifests. "
            "Stage 0 currently starts from hidden_states, so this script embeds token IDs on the server."
        )
    )
    parser.add_argument("--model_name", default="tinyllama")
    parser.add_argument("--dataset", default="dolly", help="Preset: dolly, alpaca, gsm8k, or a HF dataset name.")
    parser.add_argument("--split", default="", help="Override split. For configs use config/split, e.g. main/train.")
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--output_dir", type=Path, default=Path("data/sft_requests"))
    parser.add_argument("--request_prefix", default="sft")
    parser.add_argument("--attention_mask", choices=["zero", "causal"], default="causal")
    parser.add_argument("--mask_prompt", action="store_true")
    parser.add_argument(
        "--min_valid_labels",
        type=int,
        default=0,
        help="Skip examples with fewer trainable label positions after prompt/pad masking.",
    )
    args = parser.parse_args()

    if args.seq_len < 2:
        raise ValueError("seq_len must be at least 2.")
    if args.limit <= 0:
        raise ValueError("limit must be positive.")

    resolved_model = resolve_model_name(args.model_name)
    dataset_name, default_split = resolve_dataset(args.dataset)
    split_raw = args.split or default_split
    dataset_config, dataset_split = split_config(split_raw)

    print(f"Loading tokenizer/model: {resolved_model}")
    tokenizer = AutoTokenizer.from_pretrained(resolved_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(resolved_model, torch_dtype=torch.float32)
    model.eval()
    embedding = model.get_input_embeddings()

    print(f"Loading dataset: {dataset_name} config={dataset_config} split={dataset_split}")
    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, split=dataset_split)
    else:
        dataset = load_dataset(dataset_name, split=dataset_split)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tensor_dir = args.output_dir / "tensors"
    manifest_path = args.output_dir / "requests.jsonl"
    metadata_path = args.output_dir / "metadata.json"

    records_written = 0
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for dataset_index in range(args.offset, min(args.offset + args.limit, len(dataset))):
            example = dataset[dataset_index]
            prompt, response = format_example(example, dataset_name)
            input_ids, labels, prompt_token_count = build_token_tensors(
                tokenizer=tokenizer,
                prompt=prompt,
                response=response,
                seq_len=args.seq_len,
                mask_prompt=args.mask_prompt,
            )
            valid_label_count = count_valid_labels(labels)
            if valid_label_count < args.min_valid_labels:
                print(
                    f"Skipping dataset_index={dataset_index}: "
                    f"valid_label_count={valid_label_count} < min_valid_labels={args.min_valid_labels}"
                )
                continue
            with torch.no_grad():
                hidden_states = embedding(input_ids).float().cpu().numpy().astype("<f4")

            attention_mask = build_attention_mask(args.seq_len, args.attention_mask).numpy().astype("<f4")
            position_ids = torch.arange(args.seq_len, dtype=torch.long).unsqueeze(0).numpy().astype("<i8")
            labels_np = labels.cpu().numpy().astype("<i8")

            request_id = f"{args.request_prefix}-{dataset_index:06d}"
            hidden_path = tensor_dir / f"{request_id}_hidden_states.f32"
            mask_path = tensor_dir / f"{request_id}_attention_mask.f32"
            pos_path = tensor_dir / f"{request_id}_position_ids.i64"
            labels_path = tensor_dir / f"{request_id}_labels.i64"

            write_tensor(hidden_path, hidden_states)
            write_tensor(mask_path, attention_mask)
            write_tensor(pos_path, position_ids)
            write_tensor(labels_path, labels_np)

            record = {
                "request_id": request_id,
                "batch_id": 1,
                "chunk_idx": 0,
                "model_name": args.model_name,
                "dataset": dataset_name,
                "dataset_index": dataset_index,
                "seq_len": args.seq_len,
                "prompt_token_count": prompt_token_count,
                "valid_label_count": valid_label_count,
                "tensors": {
                    "hidden_states": tensor_record(args.output_dir, hidden_path, "float32", list(hidden_states.shape)),
                    "attention_mask": tensor_record(args.output_dir, mask_path, "float32", list(attention_mask.shape)),
                    "position_ids": tensor_record(args.output_dir, pos_path, "int64", list(position_ids.shape)),
                    "labels": tensor_record(args.output_dir, labels_path, "int64", list(labels_np.shape)),
                },
                "text": {
                    "prompt": prompt,
                    "response": response,
                },
            }
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            records_written += 1

    metadata = {
        "model_name": args.model_name,
        "resolved_model": resolved_model,
        "dataset": dataset_name,
        "dataset_config": dataset_config,
        "dataset_split": dataset_split,
        "seq_len": args.seq_len,
        "limit": records_written,
        "attention_mask": args.attention_mask,
        "mask_prompt": args.mask_prompt,
        "min_valid_labels": args.min_valid_labels,
        "manifest": manifest_path.name,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {records_written} request record(s) to {manifest_path}")


if __name__ == "__main__":
    main()
