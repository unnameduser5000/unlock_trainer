#!/usr/bin/env python3
"""Prepare DroidCall into reusable tool-calling protocol variants."""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import random

from toolcall_protocol import (
    JSON_CALLS_SYSTEM_PROMPT,
    add_raw_tool_distractors,
    calls_to_code_short,
    convert_answer_calls_to_openai,
    convert_raw_tool_to_openai_function,
    make_code_short_prompt,
    read_jsonl,
    write_jsonl,
)


def convert_json_calls_split(
    rows: list[dict[str, object]],
    split_name: str,
    *,
    api_catalog: list[dict[str, object]],
    n_api: int,
    base_seed: int,
) -> list[dict[str, object]]:
    converted: list[dict[str, object]] = []
    for row_idx, row in enumerate(rows):
        rng = random.Random(base_seed + row_idx)
        tool_rows = add_raw_tool_distractors(row.get("tools") or [], api_catalog, n_api=n_api, rng=rng)
        converted.append(
            {
                "id": f"droidcall-{split_name}-{row_idx:05d}",
                "split": split_name,
                "protocol": "droidcall_json_calls_v1",
                "target_format": "json_calls",
                "messages": [
                    {"role": "developer", "content": JSON_CALLS_SYSTEM_PROMPT},
                    {"role": "user", "content": row.get("query", "")},
                    {"role": "assistant", "tool_calls": convert_answer_calls_to_openai(row.get("answers") or [])},
                ],
                "tools": [convert_raw_tool_to_openai_function(tool) for tool in tool_rows],
            }
        )
    return converted


def convert_code_short_split(
    rows: list[dict[str, object]],
    split_name: str,
    *,
    api_catalog: list[dict[str, object]],
    n_api: int,
    base_seed: int,
) -> list[dict[str, object]]:
    converted: list[dict[str, object]] = []
    for row_idx, row in enumerate(rows):
        rng = random.Random(base_seed + row_idx)
        tool_rows = add_raw_tool_distractors(row.get("tools") or [], api_catalog, n_api=n_api, rng=rng)
        calls = [
            {"name": answer.get("name", ""), "arguments": answer.get("arguments") or {}}
            for answer in row.get("answers") or []
        ]
        converted.append(
            {
                "id": f"droidcall-{split_name}-{row_idx:05d}",
                "split": split_name,
                "protocol": "droidcall_code_short_v1",
                "target_format": "code_short",
                "prompt": make_code_short_prompt(user=str(row.get("query", "")), tools=tool_rows),
                "target": calls_to_code_short(calls),
                "user": str(row.get("query", "")),
                "expected_calls": calls,
                "tools": tool_rows,
            }
        )
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--api_catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n_api", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument(
        "--format",
        default="json_calls",
        choices=["json_calls", "code_short"],
        help="Target protocol format for downstream runners.",
    )
    args = parser.parse_args()

    train_rows = read_jsonl(args.train)
    test_rows = read_jsonl(args.test)
    api_catalog = read_jsonl(args.api_catalog)

    converter = convert_json_calls_split if args.format == "json_calls" else convert_code_short_split
    converted = []
    converted.extend(
        converter(
            train_rows,
            "train",
            api_catalog=api_catalog,
            n_api=args.n_api,
            base_seed=args.seed,
        )
    )
    converted.extend(
        converter(
            test_rows,
            "eval",
            api_catalog=api_catalog,
            n_api=args.n_api,
            base_seed=args.seed + 1_000_000,
        )
    )
    write_jsonl(args.output, converted)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "train_rows": len(train_rows),
                "eval_rows": len(test_rows),
                "total_rows": len(converted),
                "n_api": args.n_api,
                "seed": args.seed,
                "format": args.format,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
