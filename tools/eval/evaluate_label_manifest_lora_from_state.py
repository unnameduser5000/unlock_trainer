#!/usr/bin/env python3
"""Load LoRA state(s) into a full causal LM and evaluate label-choice manifests."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from sg_exe_trainer.tasks.label_experiment import (
    configure_lora_trainable,
    inject_lora_adapters,
    label_choice_details,
    load_tensor,
    lora_parameter_fingerprint,
    read_manifest,
    resolve_dtype,
    resolve_model_name,
    stage0_tensor_name,
)


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


def load_batch_tensors(
    records: list[dict[str, Any]],
    manifest_dir: Path,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids_rows: list[torch.Tensor] = []
    attention_mask_rows: list[torch.Tensor] = []
    position_ids_rows: list[torch.Tensor] = []
    labels_rows: list[torch.Tensor] = []
    for record in records:
        if stage0_tensor_name(record) != "input_ids":
            raise ValueError(
                "This evaluator currently supports manifests with tensors.input_ids only."
            )
        tensors = record["tensors"]
        input_ids_rows.append(load_tensor(manifest_dir, tensors["input_ids"]).long())
        attention_mask_rows.append(load_tensor(manifest_dir, tensors["attention_mask"]))
        position_ids_rows.append(load_tensor(manifest_dir, tensors["position_ids"]).long())
        labels_rows.append(load_tensor(manifest_dir, tensors["labels"]).long())
    input_ids = torch.cat(input_ids_rows, dim=0).to(device=device)
    attention_mask = torch.cat(attention_mask_rows, dim=0).to(device=device)
    position_ids = torch.cat(position_ids_rows, dim=0).to(device=device)
    labels = torch.cat(labels_rows, dim=0).to(device=device)
    return input_ids, attention_mask, position_ids, labels


def result_fieldnames() -> list[str]:
    return [
        "seq",
        "request_id",
        "dataset_index",
        "response",
        "predicted_response",
        "predicted_token_id",
        "target_token_id",
        "choice_correct",
        "choice_count",
        "choice_accuracy",
        "choice_loss",
    ]


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def evaluate_manifest(
    *,
    model: Any,
    manifest: Path,
    eval_limit: int | None,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = read_manifest(manifest, eval_limit)
    manifest_dir = manifest.parent
    model_dtype = next(model.parameters()).dtype
    correct_total = 0
    count_total = 0
    loss_sum = 0.0
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    model.eval()
    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        input_ids, attention_mask, position_ids, labels = load_batch_tensors(
            batch_records,
            manifest_dir,
            device,
        )
        attention_mask = attention_mask.to(dtype=model_dtype)
        with torch.inference_mode():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            ).logits
        log_probs = F.log_softmax(logits.detach().float(), dim=-1)
        for offset, record in enumerate(batch_records):
            details = label_choice_details(
                record,
                log_probs[offset : offset + 1],
                labels[offset : offset + 1],
            )
            correct = int(details["choice_correct"])
            count = int(details["choice_count"])
            choice_loss = float(details["choice_loss"])
            correct_total += correct
            count_total += count
            loss_sum += choice_loss * count
            rows.append(
                {
                    "seq": start + offset,
                    "request_id": record.get("request_id", ""),
                    "dataset_index": int(record.get("dataset_index", -1)),
                    "response": (record.get("text") or {}).get("response", "").strip(),
                    "predicted_response": str(details.get("predicted_response", "")).strip(),
                    "predicted_token_id": details.get("predicted_token_id", ""),
                    "target_token_id": details.get("target_token_id", ""),
                    "choice_correct": correct,
                    "choice_count": count,
                    "choice_accuracy": (correct / count) if count else 0.0,
                    "choice_loss": choice_loss,
                }
            )
    wall_ms = (time.perf_counter() - started) * 1000.0
    summary = {
        "records": len(records),
        "choice_correct": int(correct_total),
        "choice_count": int(count_total),
        "choice_accuracy": (correct_total / count_total) if count_total else 0.0,
        "avg_loss": (loss_sum / count_total) if count_total else 0.0,
        "wall_ms": wall_ms,
        "throughput_per_s": len(records) / (wall_ms / 1000.0) if wall_ms > 0 else 0.0,
    }
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lora_state", type=Path, action="append", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_name", default="tinyllama")
    parser.add_argument("--eval_limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--lora_targets", default="q_proj,v_proj")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_init_std", type=float, default=0.01)
    args = parser.parse_args()

    resolved_model = resolve_model_name(args.model_name)
    dtype = resolve_dtype(args.dtype)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(resolved_model, torch_dtype=dtype)
    model.config.use_cache = False
    inject_lora_adapters(
        module=model,
        target_names={item.strip() for item in args.lora_targets.split(",") if item.strip()},
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        init_std=args.lora_init_std,
    )
    configure_lora_trainable(model)
    merged_state = merge_states(args.lora_state)
    model.to(device)
    apply_lora_state(model, merged_state)
    fingerprint = lora_parameter_fingerprint(model)
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    summary, rows = evaluate_manifest(
        model=model,
        manifest=args.manifest,
        eval_limit=args.eval_limit,
        batch_size=args.batch_size,
        device=device,
    )
    summary.update(
        {
            "runner": "evaluate_label_manifest_lora_from_state",
            "manifest": str(args.manifest),
            "model_name": args.model_name,
            "resolved_model": resolved_model,
            "lora_state_files": [str(path) for path in args.lora_state],
            "lora_targets": args.lora_targets,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "dtype": args.dtype,
            "device": str(device),
            "lora_initialization_fingerprint": fingerprint,
            "cuda_peak_allocated_mib": (
                torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
                if torch.cuda.is_available() and device.type == "cuda"
                else 0.0
            ),
        }
    )
    write_csv_rows(args.output_dir / "eval_rows.csv", result_fieldnames(), rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {args.output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
