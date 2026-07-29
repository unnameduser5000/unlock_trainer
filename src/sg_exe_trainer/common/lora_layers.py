"""Shared LoRA layers and utilities for simulation experiments."""

from __future__ import annotations

import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F


def _stable_lora_seed(init_seed: int, module_name: str) -> int:
    """Derive a process-independent RNG seed for one named LoRA adapter."""
    digest = hashlib.sha256(f"bpfree-lora-v1:{init_seed}:{module_name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**63 - 1)


def _initial_lora_a(
    *,
    rank: int,
    in_features: int,
    init_std: float,
    init_seed: int | None,
    module_name: str,
) -> torch.Tensor:
    value = torch.empty((rank, in_features), dtype=torch.float32)
    if init_seed is None:
        return value.normal_(mean=0.0, std=init_std)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_stable_lora_seed(init_seed, module_name))
    return value.normal_(mean=0.0, std=init_std, generator=generator)


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        init_std: float,
        *,
        init_seed: int | None = None,
        module_name: str = "",
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")
        self.base = base
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_a = nn.Parameter(
            _initial_lora_a(
                rank=rank,
                in_features=base.in_features,
                init_std=init_std,
                init_seed=init_seed,
                module_name=module_name,
            )
        )
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        for param in self.base.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_dtype = self.base.weight.dtype
        lora_dtype = self.lora_a.dtype
        base_out = self.base(x.to(dtype=base_dtype))
        lora_input = x.to(dtype=lora_dtype)
        lora_hidden = F.linear(lora_input, self.lora_a)
        lora_out = F.linear(lora_hidden, self.lora_b) * self.scaling
        return (base_out.float() + lora_out.float()).to(dtype=base_dtype)


def inject_lora_adapters(
    module: nn.Module,
    target_names: set[str],
    rank: int,
    alpha: float,
    init_std: float,
    init_seed: int | None = None,
    module_prefix: str = "",
) -> int:
    injected = 0
    for child_name, child in list(module.named_children()):
        full_name = f"{module_prefix}.{child_name}" if module_prefix else child_name
        if isinstance(child, nn.Linear) and child_name in target_names:
            module._modules[child_name] = LoRALinear(
                base=child,
                rank=rank,
                alpha=alpha,
                init_std=init_std,
                init_seed=init_seed,
                module_name=full_name,
            )
            injected += 1
        else:
            injected += inject_lora_adapters(
                module=child,
                target_names=target_names,
                rank=rank,
                alpha=alpha,
                init_std=init_std,
                init_seed=init_seed,
                module_prefix=full_name,
            )
    return injected


def lora_parameter_fingerprint(module: nn.Module) -> str:
    """Fingerprint just the initialized LoRA tensors, not frozen model weights."""
    digest = hashlib.sha256()
    for name, parameter in sorted(module.named_parameters(), key=lambda item: item[0]):
        if not (name.endswith("lora_a") or name.endswith("lora_b")):
            continue
        value = parameter.detach().float().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def configure_lora_trainable(module: nn.Module) -> tuple[int, int]:
    for param in module.parameters():
        param.requires_grad = False
    for submodule in module.modules():
        if isinstance(submodule, LoRALinear):
            submodule.lora_a.requires_grad = True
            submodule.lora_b.requires_grad = True
    trainable = sum(param.numel() for param in module.parameters() if param.requires_grad)
    frozen = sum(param.numel() for param in module.parameters() if not param.requires_grad)
    return trainable, frozen
