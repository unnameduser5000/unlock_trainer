"""Helpers for switching between LoRA and full-trainable stage experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from sg_exe_trainer.common.lora_layers import configure_lora_trainable, inject_lora_adapters


@dataclass(frozen=True)
class TrainableSetup:
    mode: str
    lora_modules: int
    trainable_params: int
    frozen_params: int


@dataclass(frozen=True)
class ParamStats:
    params: int
    trainable_params: int
    bytes: int
    trainable_bytes: int


def configure_model_trainable(
    module: nn.Module,
    *,
    mode: str,
    lora_targets: str,
    lora_rank: int,
    lora_alpha: float,
    lora_init_std: float,
    lora_init_seed: int | None = None,
) -> TrainableSetup:
    normalized = mode.strip().lower()
    if normalized == "lora":
        injected = inject_lora_adapters(
            module=module,
            target_names={item.strip() for item in lora_targets.split(",") if item.strip()},
            rank=lora_rank,
            alpha=lora_alpha,
            init_std=lora_init_std,
            init_seed=lora_init_seed,
        )
        trainable, frozen = configure_lora_trainable(module)
        return TrainableSetup(
            mode="lora",
            lora_modules=injected,
            trainable_params=trainable,
            frozen_params=frozen,
        )

    if normalized == "full":
        for param in module.parameters():
            param.requires_grad = True
        trainable = sum(param.numel() for param in module.parameters() if param.requires_grad)
        frozen = sum(param.numel() for param in module.parameters() if not param.requires_grad)
        return TrainableSetup(
            mode="full",
            lora_modules=0,
            trainable_params=trainable,
            frozen_params=frozen,
        )

    if normalized == "full_layers":
        for param in module.parameters():
            param.requires_grad = False
        body = getattr(module, "model", None)
        layers = getattr(body, "layers", None) if body is not None else None
        if layers is None:
            raise ValueError("trainable_mode=full_layers requires module.model.layers.")
        for param in layers.parameters():
            param.requires_grad = True
        trainable = sum(param.numel() for param in module.parameters() if param.requires_grad)
        frozen = sum(param.numel() for param in module.parameters() if not param.requires_grad)
        return TrainableSetup(
            mode="full_layers",
            lora_modules=0,
            trainable_params=trainable,
            frozen_params=frozen,
        )

    raise ValueError(f"Unsupported trainable mode: {mode}. Use lora, full, or full_layers.")


def parameter_nbytes(param: torch.Tensor) -> int:
    return int(param.numel() * param.element_size())


def module_param_stats(module: nn.Module) -> ParamStats:
    params = 0
    trainable_params = 0
    total_bytes = 0
    trainable_bytes = 0
    for param in module.parameters():
        nparams = int(param.numel())
        nbytes = parameter_nbytes(param)
        params += nparams
        total_bytes += nbytes
        if param.requires_grad:
            trainable_params += nparams
            trainable_bytes += nbytes
    return ParamStats(
        params=params,
        trainable_params=trainable_params,
        bytes=total_bytes,
        trainable_bytes=trainable_bytes,
    )


def optimizer_state_nbytes(optimizer: torch.optim.Optimizer | None) -> int:
    if optimizer is None:
        return 0
    total = 0
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                total += parameter_nbytes(value)
    return total


def gradient_storage_nbytes(module: nn.Module | None) -> int:
    if module is None:
        return 0
    total = 0
    for param in module.parameters():
        if isinstance(param.grad, torch.Tensor):
            total += parameter_nbytes(param.grad)
    return total
