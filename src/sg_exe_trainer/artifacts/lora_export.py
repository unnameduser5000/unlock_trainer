"""Helpers for exporting LoRA-only state from stage chunks and full models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def normalize_lora_name(module: torch.nn.Module, name: str) -> str:
    if not hasattr(module, "layer_start"):
        return name
    if not name.startswith("layers."):
        return name
    parts = name.split(".")
    if len(parts) < 3:
        return name
    try:
        local_index = int(parts[1])
    except ValueError:
        return name
    global_index = int(getattr(module, "layer_start")) + local_index
    return ".".join(["model", "layers", str(global_index), *parts[2:]])


def lora_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        normalize_lora_name(module, name): param.detach().cpu().clone()
        for name, param in module.named_parameters()
        if "lora_a" in name or "lora_b" in name
    }


def save_lora_state(module: torch.nn.Module, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(lora_state_dict(module), path)
    return path


def stage_lora_export_payload(
    *,
    stage_id: int,
    trainable_mode: str,
    lora_init_seed: int | None,
    lora_initialization_fingerprint: str,
    module: torch.nn.Module,
) -> dict[str, Any]:
    return {
        "kind": "stage_lora_export",
        "stage_id": stage_id,
        "trainable_mode": trainable_mode,
        "lora_init_seed": lora_init_seed,
        "lora_initialization_fingerprint": lora_initialization_fingerprint,
        "lora_state": lora_state_dict(module),
    }
