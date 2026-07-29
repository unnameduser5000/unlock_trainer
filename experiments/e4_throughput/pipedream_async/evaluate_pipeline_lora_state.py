#!/usr/bin/env python3
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

from sg_exe_trainer.runtime.exactbp.distributed_runtime import (
    StageEventRecorder,
    build_stage_module,
)
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


def _merge_states(paths: list[Path]) -> dict[str, torch.Tensor]:
    merged: dict[str, torch.Tensor] = {}
    for path in paths:
        state = torch.load(path, map_location="cpu")
        if not isinstance(state, dict):
            raise ValueError(f"unexpected trainable-state format: {path}")
        for name, value in state.items():
            if name in merged:
                raise ValueError(f"duplicate trainable tensor: {name}")
            merged[name] = value
    return merged


def _apply_lora_state(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
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


def _load_hidden_batch(
    records: list[dict[str, Any]],
    manifest_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden_rows: list[torch.Tensor] = []
    attention_rows: list[torch.Tensor] = []
    position_rows: list[torch.Tensor] = []
    label_rows: list[torch.Tensor] = []
    for record in records:
        if stage0_tensor_name(record) != "hidden_states":
            raise ValueError("pipeline evaluator expects a hidden_states manifest")
        tensors = record["tensors"]
        hidden_rows.append(load_tensor(manifest_dir, tensors["hidden_states"]))
        attention_rows.append(load_tensor(manifest_dir, tensors["attention_mask"]))
        position_rows.append(load_tensor(manifest_dir, tensors["position_ids"]).long())
        label_rows.append(load_tensor(manifest_dir, tensors["labels"]).long())
    return (
        torch.cat(hidden_rows, dim=0).to(device=device, dtype=dtype),
        torch.cat(attention_rows, dim=0).to(device=device, dtype=dtype),
        torch.cat(position_rows, dim=0).to(device=device),
        torch.cat(label_rows, dim=0).to(device=device),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate stage checkpoints on hidden-state label manifests."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lora_state", type=Path, action="append", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_name", default="tinyllama")
    parser.add_argument("--num_chunks", type=int, default=3)
    parser.add_argument("--eval_limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--lora_targets", default="q_proj,v_proj")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_init_std", type=float, default=0.01)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dtype = resolve_dtype(args.dtype)
    resolved_model = resolve_model_name(args.model_name)
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
    _apply_lora_state(model, _merge_states(args.lora_state))
    fingerprint = lora_parameter_fingerprint(model)

    stages: list[torch.nn.Module] = []
    layer_ranges: list[list[int]] = []
    for stage_id in range(args.num_chunks):
        recorder = StageEventRecorder(
            stage_id=stage_id,
            rank=stage_id,
            device_name=str(device),
            enabled=False,
        )
        stage = build_stage_module(
            model=model,
            stage_id=stage_id,
            num_chunks=args.num_chunks,
            recorder=recorder,
        )
        stage.to(device)
        stage.eval()
        stages.append(stage)
        layer_ranges.append([int(stage.layer_start), int(stage.layer_end) - 1])
    del model

    records = read_manifest(args.manifest, args.eval_limit)
    correct_total = 0
    count_total = 0
    loss_sum = 0.0
    rows: list[dict[str, Any]] = []
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for start in range(0, len(records), args.batch_size):
        batch_records = records[start : start + args.batch_size]
        hidden, attention_mask, position_ids, labels = _load_hidden_batch(
            batch_records,
            args.manifest.parent,
            device,
            dtype,
        )
        with torch.inference_mode():
            output = hidden
            for stage in stages:
                output = stage(
                    output,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
        log_probs = F.log_softmax(output.detach().float(), dim=-1)
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
                    "predicted_response": str(details.get("predicted_response", "")).strip(),
                    "target_response": (record.get("text") or {}).get("response", "").strip(),
                    "choice_correct": correct,
                    "choice_count": count,
                    "choice_loss": choice_loss,
                }
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_ms = (time.perf_counter() - started) * 1000.0
    summary = {
        "runner": "evaluate-pipeline-lora-state",
        "manifest": str(args.manifest),
        "records": len(records),
        "choice_correct": correct_total,
        "choice_count": count_total,
        "choice_accuracy": correct_total / count_total if count_total else 0.0,
        "avg_loss": loss_sum / count_total if count_total else 0.0,
        "wall_ms": wall_ms,
        "model_name": args.model_name,
        "resolved_model": resolved_model,
        "num_chunks": args.num_chunks,
        "layer_ranges": layer_ranges,
        "lora_state_files": [str(path) for path in args.lora_state],
        "lora_fingerprint": fingerprint,
        "dtype": args.dtype,
        "device": str(device),
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
    }
    with (args.output_dir / "eval_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["seq"])
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
