import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from prepare_lora_sft_requests import (
    format_example,
    resolve_dataset,
    resolve_model_name,
    split_config,
)


def load_examples(dataset_name: str, split_raw: str):
    dataset_config, dataset_split = split_config(split_raw)
    if dataset_config:
        return load_dataset(dataset_name, dataset_config, split=dataset_split)
    return load_dataset(dataset_name, split=dataset_split)


def encode_one_token(tokenizer, text: str, name: str) -> int:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise ValueError(
            f"{name} must tokenize to exactly one token for this probe: "
            f"text={text!r} token_ids={ids}"
        )
    return int(ids[0])


def evaluate_example(
    model,
    tokenizer,
    example: dict[str, Any],
    dataset_name: str,
    choice_texts: list[str],
    choice_token_ids: list[int],
    max_prompt_tokens: int,
    device: torch.device,
) -> dict[str, Any]:
    prompt, response = format_example(
        example=example,
        dataset_name=dataset_name,
        response_style="label",
    )
    target_token_id = encode_one_token(tokenizer, response, "target response")

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if max_prompt_tokens > 0:
        prompt_ids = prompt_ids[:max_prompt_tokens]
    if not prompt_ids:
        raise ValueError("Prompt tokenized to an empty input.")

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[0, -1].float()

    full_log_probs = torch.log_softmax(logits, dim=-1)
    full_pred = int(torch.argmax(logits).item())
    full_loss = float((-full_log_probs[target_token_id]).item())

    choice_logits = logits[torch.tensor(choice_token_ids, dtype=torch.long, device=device)]
    choice_log_probs = torch.log_softmax(choice_logits, dim=-1)
    try:
        target_choice_index = choice_token_ids.index(target_token_id)
    except ValueError as exc:
        raise ValueError(
            f"Target token id {target_token_id} for response={response!r} "
            f"is not in choices {list(zip(choice_texts, choice_token_ids))}."
        ) from exc

    choice_pred_index = int(torch.argmax(choice_logits).item())
    choice_loss = float((-choice_log_probs[target_choice_index]).item())
    choice_pred_token_id = choice_token_ids[choice_pred_index]

    return {
        "prompt_token_count": len(prompt_ids),
        "response": response,
        "target_token_id": target_token_id,
        "full_vocab_loss": full_loss,
        "full_vocab_correct": full_pred == target_token_id,
        "full_vocab_pred_token_id": full_pred,
        "full_vocab_pred_text": tokenizer.decode([full_pred]),
        "choice_loss": choice_loss,
        "choice_correct": choice_pred_token_id == target_token_id,
        "choice_pred_text": choice_texts[choice_pred_index],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Probe a label-only classification prompt on the server before spending "
            "time on Android. This performs no training; it measures the base model's "
            "full-vocab and constrained label-choice loss/accuracy."
        )
    )
    parser.add_argument("--model_name", default="tinyllama")
    parser.add_argument("--dataset", default="rotten_tomatoes")
    parser.add_argument("--split", default="")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--max_prompt_tokens", type=int, default=96)
    parser.add_argument(
        "--choices",
        default="positive,negative",
        help="Comma-separated label words. A leading space is added by --label_prefix.",
    )
    parser.add_argument("--label_prefix", default=" ")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output_json", type=Path, default=None)
    args = parser.parse_args()

    resolved_model = resolve_model_name(args.model_name)
    dataset_name, default_split = resolve_dataset(args.dataset)
    split_raw = args.split or default_split

    tokenizer = AutoTokenizer.from_pretrained(resolved_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(resolved_model, torch_dtype=dtype)
    model.to(device)
    model.eval()

    choice_texts = [args.label_prefix + item.strip() for item in args.choices.split(",") if item.strip()]
    choice_token_ids = [
        encode_one_token(tokenizer, text, f"choice {text!r}")
        for text in choice_texts
    ]

    dataset = load_examples(dataset_name, split_raw)
    rows = []
    last_index = args.offset - 1
    for dataset_index in range(args.offset, min(len(dataset), args.offset + args.limit)):
        last_index = dataset_index
        result = evaluate_example(
            model=model,
            tokenizer=tokenizer,
            example=dataset[dataset_index],
            dataset_name=dataset_name,
            choice_texts=choice_texts,
            choice_token_ids=choice_token_ids,
            max_prompt_tokens=args.max_prompt_tokens,
            device=device,
        )
        result["dataset_index"] = dataset_index
        rows.append(result)

    if not rows:
        raise RuntimeError("No rows evaluated.")

    summary = {
        "model_name": args.model_name,
        "resolved_model": resolved_model,
        "dataset": dataset_name,
        "split": split_raw,
        "offset": args.offset,
        "last_dataset_index": last_index,
        "rows": len(rows),
        "choices": [
            {"text": text, "token_id": token_id}
            for text, token_id in zip(choice_texts, choice_token_ids)
        ],
        "avg_full_vocab_loss": sum(row["full_vocab_loss"] for row in rows) / len(rows),
        "full_vocab_accuracy": sum(row["full_vocab_correct"] for row in rows) / len(rows),
        "avg_choice_loss": sum(row["choice_loss"] for row in rows) / len(rows),
        "choice_accuracy": sum(row["choice_correct"] for row in rows) / len(rows),
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
