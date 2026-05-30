import argparse
import hashlib
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
    "rotten_tomatoes": ("cornell-movie-review-data/rotten_tomatoes", "train"),
    "sst2": ("glue", "sst2/train"),
    "ag_news": ("fancyzhx/ag_news", "train"),
    "sciq": ("allenai/sciq", "train"),
    "gsm8k": ("openai/gsm8k", "main/train"),
    "toy": ("toy", "train"),
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


def format_example(
    example: dict[str, Any],
    dataset_name: str,
    response_style: str,
) -> tuple[str, str]:
    if dataset_name == "toy":
        instruction = str(example.get("instruction", "")).strip()
        response = str(example.get("response", "")).strip()
        prompt = f"Instruction:\n{instruction}\n\nResponse:\n"
        return prompt, response

    if dataset_name == "glue" and "sentence" in example and "label" in example:
        sentence = str(example.get("sentence", "")).strip()
        label = int(example.get("label", 0))
        sentiment = "positive" if label == 1 else "negative"
        if response_style == "label":
            prompt = (
                "Movie review:\n"
                f"{sentence}\n\n"
                "Sentiment (positive or negative):"
            )
            response = f" {sentiment}"
        else:
            prompt = (
                "Movie review:\n"
                f"{sentence}\n\n"
                "Classify the sentiment as positive or negative.\n\n"
                "Answer:\n"
            )
            response = f"The movie review expresses a {sentiment} sentiment."
        return prompt, response

    if "rotten_tomatoes" in dataset_name:
        text = str(example.get("text", "")).strip()
        label = int(example.get("label", 0))
        sentiment = "positive" if label == 1 else "negative"
        if response_style == "label":
            prompt = (
                "Movie review:\n"
                f"{text}\n\n"
                "Sentiment (positive or negative):"
            )
            response = f" {sentiment}"
        else:
            prompt = (
                "Movie review:\n"
                f"{text}\n\n"
                "Classify the sentiment as positive or negative.\n\n"
                "Answer:\n"
            )
            response = f"The movie review expresses a {sentiment} sentiment."
        return prompt, response

    if "ag_news" in dataset_name:
        text = str(example.get("text", "")).strip()
        label = int(example.get("label", 0))
        label_names = {
            0: "World",
            1: "Sports",
            2: "Business",
            3: "Science and Technology",
        }
        topic = label_names.get(label, str(label))
        if response_style == "label":
            prompt = (
                "News article:\n"
                f"{text}\n\n"
                "Topic (World, Sports, Business, or Science and Technology):"
            )
            response = f" {topic}"
        else:
            prompt = (
                "News article:\n"
                f"{text}\n\n"
                "Classify the news topic as World, Sports, Business, or Science and Technology.\n\n"
                "Answer:\n"
            )
            response = f"The topic is {topic} news."
        return prompt, response

    if "sciq" in dataset_name:
        question = str(example.get("question", "")).strip()
        correct_answer = str(example.get("correct_answer", "")).strip()
        distractors = [
            str(example.get("distractor1", "")).strip(),
            str(example.get("distractor2", "")).strip(),
            str(example.get("distractor3", "")).strip(),
        ]
        options = [correct_answer, *distractors]
        digest = hashlib.sha256(question.encode("utf-8", errors="ignore")).digest()
        rotation = digest[0] % len(options)
        options = options[rotation:] + options[:rotation]
        option_lines = "\n".join(f"{chr(65 + idx)}. {option}" for idx, option in enumerate(options))
        prompt = f"Science question:\n{question}\n\nOptions:\n{option_lines}\n\nAnswer:\n"
        response = f"The correct answer is {correct_answer}."
        return prompt, response

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


def label_choice_texts(dataset_name: str, response_style: str) -> list[str]:
    if response_style != "label":
        return []
    if dataset_name == "glue" or "rotten_tomatoes" in dataset_name:
        return [" positive", " negative"]
    if "ag_news" in dataset_name:
        return [" World", " Sports", " Business", " Science and Technology"]
    return []


def build_label_choices(tokenizer, dataset_name: str, response_style: str) -> list[dict[str, Any]]:
    choices = []
    for text in label_choice_texts(dataset_name, response_style):
        choices.append(
            {
                "text": text,
                "token_ids": tokenizer(text, add_special_tokens=False)["input_ids"],
            }
        )
    return choices


def build_candidate_indices(
    dataset,
    offset: int,
    limit: int,
    shuffle_seed: int | None,
    balance_labels: bool,
) -> list[int]:
    indices = list(range(offset, len(dataset)))
    rng = np.random.default_rng(shuffle_seed) if shuffle_seed is not None else None

    if not balance_labels:
        if rng is not None:
            rng.shuffle(indices)
        return indices

    label_to_indices: dict[int, list[int]] = {}
    for dataset_index in indices:
        example = dataset[dataset_index]
        if "label" not in example:
            raise ValueError("--balance_labels requires dataset examples to contain a 'label' field.")
        label = int(example["label"])
        label_to_indices.setdefault(label, []).append(dataset_index)

    if len(label_to_indices) < 2:
        raise ValueError(f"--balance_labels found only {len(label_to_indices)} label class(es).")

    for group in label_to_indices.values():
        if rng is not None:
            rng.shuffle(group)

    labels = sorted(label_to_indices)
    balanced: list[int] = []
    cursor = 0
    while len(balanced) < limit and any(label_to_indices.values()):
        label = labels[cursor % len(labels)]
        if label_to_indices[label]:
            balanced.append(label_to_indices[label].pop(0))
        cursor += 1
        if cursor > len(labels) * (limit + len(labels)):
            break
    return balanced


def build_token_tensors(
    tokenizer,
    prompt: str,
    response: str,
    seq_len: int,
    mask_prompt: bool,
    max_prompt_tokens: int,
    append_eos: bool,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if max_prompt_tokens > 0:
        prompt_ids = prompt_ids[:max_prompt_tokens]
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    eos_id = tokenizer.eos_token_id
    if append_eos and eos_id is not None:
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


def build_toy_dataset(total_count: int) -> list[dict[str, str]]:
    toy_responses = [
        "The correct answer is always option A for this toy run.",
        "The correct answer is always option A for this toy run.",
        "The correct answer is always option A for this toy run.",
        "The correct answer is always option A for this toy run.",
    ]
    dataset: list[dict[str, str]] = []
    for index in range(total_count):
        response = toy_responses[index % len(toy_responses)]
        instruction = (
            f"Toy overfit sample {index:04d}: read the prompt and return the same fixed answer."
        )
        dataset.append({"instruction": instruction, "response": response})
    return dataset


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
    parser.add_argument(
        "--dataset",
        default="dolly",
        help=(
            "Preset: dolly, alpaca, rotten_tomatoes, sst2, ag_news, sciq, "
            "gsm8k, toy, or a HF dataset name."
        ),
    )
    parser.add_argument("--split", default="", help="Override split. For configs use config/split, e.g. main/train.")
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--output_dir", type=Path, default=Path("data/sft_requests"))
    parser.add_argument("--request_prefix", default="sft")
    parser.add_argument("--attention_mask", choices=["zero", "causal"], default="causal")
    parser.add_argument("--mask_prompt", action="store_true")
    parser.add_argument(
        "--response_style",
        choices=["natural", "label"],
        default="natural",
        help=(
            "natural keeps the original instruction-style response. label writes only the class label "
            "for supported classification datasets, so token accuracy is close to task accuracy."
        ),
    )
    parser.add_argument(
        "--no_append_eos",
        action="store_true",
        help="Do not append EOS to the response. Useful for one-token label-only classification probes.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0,
        help=(
            "Optional per-request SGD learning rate stored in the manifest. "
            "0 keeps the Android worker default."
        ),
    )
    parser.add_argument(
        "--max_prompt_tokens",
        type=int,
        default=0,
        help=(
            "Truncate the prompt to this many tokens before appending the response. "
            "Use this with seq_len=64 classification tasks so response labels are not truncated away. "
            "0 keeps the old behavior."
        ),
    )
    parser.add_argument(
        "--min_valid_labels",
        type=int,
        default=0,
        help="Skip examples with fewer trainable label positions after prompt/pad masking.",
    )
    parser.add_argument(
        "--shuffle_seed",
        type=int,
        default=None,
        help="Shuffle candidate dataset indices with this seed before applying limit.",
    )
    parser.add_argument(
        "--balance_labels",
        action="store_true",
        help="Round-robin sample examples by integer 'label' field before applying limit.",
    )
    args = parser.parse_args()

    if args.seq_len < 2:
        raise ValueError("seq_len must be at least 2.")
    if args.limit <= 0:
        raise ValueError("limit must be positive.")
    if args.max_prompt_tokens < 0:
        raise ValueError("max_prompt_tokens must be non-negative.")
    if args.max_prompt_tokens >= args.seq_len:
        raise ValueError("max_prompt_tokens must be smaller than seq_len.")
    if args.learning_rate < 0:
        raise ValueError("learning_rate must be non-negative.")

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
    label_choices = build_label_choices(tokenizer, dataset_name, args.response_style)

    print(f"Loading dataset: {dataset_name} config={dataset_config} split={dataset_split}")
    if dataset_name == "toy":
        dataset = build_toy_dataset(args.offset + args.limit)
    elif dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, split=dataset_split)
    else:
        dataset = load_dataset(dataset_name, split=dataset_split)

    candidate_indices = build_candidate_indices(
        dataset=dataset,
        offset=args.offset,
        limit=args.limit,
        shuffle_seed=args.shuffle_seed,
        balance_labels=args.balance_labels,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tensor_dir = args.output_dir / "tensors"
    manifest_path = args.output_dir / "requests.jsonl"
    metadata_path = args.output_dir / "metadata.json"

    records_written = 0
    last_dataset_index = args.offset - 1
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for dataset_index in candidate_indices:
            if records_written >= args.limit:
                break
            last_dataset_index = dataset_index
            example = dataset[dataset_index]
            prompt, response = format_example(
                example=example,
                dataset_name=dataset_name,
                response_style=args.response_style,
            )
            input_ids, labels, prompt_token_count = build_token_tensors(
                tokenizer=tokenizer,
                prompt=prompt,
                response=response,
                seq_len=args.seq_len,
                mask_prompt=args.mask_prompt,
                max_prompt_tokens=args.max_prompt_tokens,
                append_eos=not args.no_append_eos,
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
                "learning_rate": args.learning_rate if args.learning_rate > 0 else None,
                "label_choices": label_choices or None,
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

    if records_written < args.limit:
        print(
            f"Warning: dataset exhausted after dataset_index={last_dataset_index}; "
            f"wrote {records_written}/{args.limit} requested records."
        )

    metadata = {
        "model_name": args.model_name,
        "resolved_model": resolved_model,
        "dataset": dataset_name,
        "dataset_config": dataset_config,
        "dataset_split": dataset_split,
        "seq_len": args.seq_len,
        "requested_limit": args.limit,
        "limit": records_written,
        "offset": args.offset,
        "last_dataset_index": last_dataset_index,
        "attention_mask": args.attention_mask,
        "mask_prompt": args.mask_prompt,
        "response_style": args.response_style,
        "append_eos": not args.no_append_eos,
        "max_prompt_tokens": args.max_prompt_tokens,
        "min_valid_labels": args.min_valid_labels,
        "shuffle_seed": args.shuffle_seed,
        "balance_labels": args.balance_labels,
        "learning_rate": args.learning_rate if args.learning_rate > 0 else None,
        "label_choices": label_choices or None,
        "manifest": manifest_path.name,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {records_written} request record(s) to {manifest_path}")


if __name__ == "__main__":
    main()
