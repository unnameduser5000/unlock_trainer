"""Shared manifest/model/label helpers for simulation experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


MODEL_PRESETS = {
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "smollm2_360m": "HuggingFaceTB/SmolLM2-360M",
    "phi2": "microsoft/phi-2",
}

def resolve_model_name(model_name: str) -> str:
    return MODEL_PRESETS.get(model_name, model_name)

def resolve_dtype(raw: str) -> torch.dtype:
    if raw == "float32":
        return torch.float32
    if raw == "float16":
        return torch.float16
    if raw == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {raw}")

def infer_module_compute_dtype(module: nn.Module) -> torch.dtype:
    for submodule in module.modules():
        base = getattr(submodule, "base", None)
        if isinstance(base, nn.Linear) and base.weight.is_floating_point():
            return base.weight.dtype
    for submodule in module.modules():
        if isinstance(submodule, nn.Linear) and submodule.weight.is_floating_point():
            return submodule.weight.dtype
    for param in module.parameters():
        if not param.requires_grad and param.is_floating_point():
            return param.dtype
    for param in module.parameters():
        if param.is_floating_point():
            return param.dtype
    raise RuntimeError("Could not infer module compute dtype.")

def get_model_parts(backbone: nn.Module):
    if not hasattr(backbone, "model"):
        raise ValueError("Expected AutoModelForCausalLM with a .model backbone.")
    body = backbone.model
    if not hasattr(body, "layers"):
        raise ValueError("Expected backbone.model.layers to exist.")
    final_norm = getattr(body, "final_layernorm", None) or getattr(body, "norm", None)
    if final_norm is None:
        raise ValueError("Could not locate final norm.")
    lm_head = getattr(backbone, "lm_head", None)
    if lm_head is None:
        raise ValueError("Could not locate lm_head.")
    rotary_emb = getattr(body, "rotary_emb", None)
    return body.layers, final_norm, lm_head, backbone.config.vocab_size, rotary_emb

def read_manifest(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    if not rows:
        raise RuntimeError(f"No records loaded from {path}")
    return rows

def load_tensor(manifest_dir: Path, spec: dict[str, Any]) -> torch.Tensor:
    path = manifest_dir / spec["path"]
    dtype = spec["dtype"]
    if dtype == "float32":
        np_dtype = "<f4"
    elif dtype == "int64":
        np_dtype = "<i8"
    else:
        raise ValueError(f"Unsupported tensor dtype in manifest: {dtype}")
    array = np.fromfile(path, dtype=np_dtype).reshape(spec["shape"])
    return torch.from_numpy(array.copy())

def stage0_tensor_name(record: dict[str, Any]) -> str:
    tensors = record.get("tensors") or {}
    if "hidden_states" in tensors:
        return "hidden_states"
    if "input_ids" in tensors:
        return "input_ids"
    raise ValueError("Manifest record must contain either tensors.hidden_states or tensors.input_ids.")

def one_token_choice_ids(record: dict[str, Any]) -> list[int]:
    choices = record.get("label_choices") or []
    ids = []
    for choice in choices:
        token_ids = choice.get("token_ids") or []
        if len(token_ids) != 1:
            raise ValueError(f"Only one-token choices are supported, got {choice}")
        ids.append(int(token_ids[0]))
    if not ids:
        raise ValueError("Record does not contain label_choices.")
    return ids

def label_choice_details(
    record: dict[str, Any],
    log_probs: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, Any]:
    choices = record.get("label_choices") or []
    choice_ids = one_token_choice_ids(record)
    shift_log_probs = log_probs[..., :-1, :].float()
    shift_labels = labels[..., 1:].long()
    valid_positions = shift_labels != -100
    if not valid_positions.any():
        return {
            "choice_correct": 0,
            "choice_count": 0,
            "choice_loss": 0.0,
            "predicted_response": "",
            "predicted_token_id": "",
            "target_token_id": "",
        }

    correct = 0
    count = 0
    loss_sum = 0.0
    predicted_response = ""
    predicted_token_id: int | str = ""
    target_token_id: int | str = ""
    choice_index = torch.tensor(choice_ids, dtype=torch.long, device=log_probs.device)
    choice_texts = [str(choice.get("text", "")).strip() for choice in choices]
    for batch_idx, token_idx in valid_positions.nonzero(as_tuple=False):
        target_id = int(shift_labels[batch_idx, token_idx].item())
        if target_id not in choice_ids:
            continue
        scores = shift_log_probs[batch_idx, token_idx, choice_index]
        pred_choice = int(torch.argmax(scores).item())
        target_choice = choice_ids.index(target_id)
        if count == 0:
            predicted_response = choice_texts[pred_choice] if pred_choice < len(choice_texts) else ""
            predicted_token_id = int(choice_ids[pred_choice])
            target_token_id = int(target_id)
        correct += int(pred_choice == target_choice)
        count += 1
        loss_sum += float((-F.log_softmax(scores, dim=-1)[target_choice]).item())
    return {
        "choice_correct": correct,
        "choice_count": count,
        "choice_loss": (loss_sum / count) if count else 0.0,
        "predicted_response": predicted_response,
        "predicted_token_id": predicted_token_id,
        "target_token_id": target_token_id,
    }
