"""Model partition and local optimizer construction shared by BPFree runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from sg_exe_trainer.tasks.label_experiment import (
    get_model_parts,
    infer_module_compute_dtype,
)


@dataclass
class StageWorkerConfig:
    model_name: str
    resolved_model: str
    num_chunks: int
    train_chunks: list[int]
    dtype_name: str
    belief_transport_mode: str
    alpha: float
    label_smoothing: float
    lora_rank: int
    lora_alpha: float
    lora_targets: str
    lora_init_std: float
    learning_rate: Optional[float]
    grad_clip: float
    optimizer: str
    sgd_momentum: float
    sgd_dampening: float
    sgd_weight_decay: float
    sgd_nesterov: bool
    seed: int
    progress_interval: int


class ServerBpfreeChunk(nn.Module):
    def __init__(
        self,
        *,
        chunk_idx: int,
        layer_start: int,
        layer_end: int,
        layers: list[nn.Module],
        final_norm: nn.Module,
        lm_head: nn.Module,
        vocab_size: int,
        rotary_emb: Optional[nn.Module],
        is_terminal_chunk: bool,
        belief_transport_mode: str,
        alpha: float,
        label_smoothing: float,
        local_readout_adapter: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.chunk_idx = chunk_idx
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.layers = nn.ModuleList(layers)
        self.final_norm = final_norm
        self.lm_head = lm_head
        self.vocab_size = vocab_size
        self.rotary_emb = rotary_emb
        self.is_terminal_chunk = is_terminal_chunk
        self.belief_transport_mode = normalize_belief_transport_mode(belief_transport_mode)
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.local_readout_adapter = local_readout_adapter
        self.last_choice_metrics: tuple[int, int, float] | None = None
        self.last_choice_details: dict[str, Any] | None = None
        self.last_loss_components: dict[str, float] = {}

    @property
    def consumes_prev_log_probs(self) -> bool:
        return self.chunk_idx > 0 and self.belief_transport_mode == "full"

    @property
    def uses_belief_loss(self) -> bool:
        return self.consumes_prev_log_probs and self.alpha < 1.0

    @property
    def returns_full_log_probs(self) -> bool:
        return self.belief_transport_mode == "full" or (
            self.belief_transport_mode == "terminal" and self.is_terminal_chunk
        )

    def compute_dtype(self) -> torch.dtype:
        return infer_module_compute_dtype(self)

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        labels: torch.Tensor,
        prev_log_probs: Optional[torch.Tensor],
        choice_ids: Optional[list[int]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        self.last_choice_metrics = None
        self.last_choice_details = None
        self.last_loss_components = {}
        dtype = self.compute_dtype()
        hidden_states = hidden_states.to(dtype=dtype)
        attention_mask = attention_mask.to(dtype=dtype)
        position_ids = position_ids.long()
        labels = labels.long()

        position_embeddings = None
        if self.rotary_emb is not None:
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        curr_hidden = hidden_states
        for layer in self.layers:
            layer_out = layer(
                curr_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )
            curr_hidden = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        z = self.final_norm(curr_hidden)
        adapter = getattr(self, "local_readout_adapter", None)
        if adapter is not None:
            z = adapter(z)
        logits = self.lm_head(z)
        shift_logits = logits[..., :-1, :].float()
        shift_labels = labels[..., 1:]
        valid_mask = (shift_labels != -100).float()
        valid_count = valid_mask.sum().clamp_min(1.0)
        safe_labels = torch.where(shift_labels != -100, shift_labels, torch.zeros_like(shift_labels))

        loss_ce_unmasked = F.cross_entropy(
            shift_logits.reshape(-1, self.vocab_size),
            safe_labels.reshape(-1),
            reduction="none",
            label_smoothing=self.label_smoothing,
        ).reshape_as(shift_labels)
        loss_ce = (loss_ce_unmasked * valid_mask).sum() / valid_count

        if not self.uses_belief_loss:
            total_loss = loss_ce
            loss_kl = None
        else:
            if prev_log_probs is None:
                raise RuntimeError("prev_log_probs is required in full belief mode.")
            teacher_log_probs = prev_log_probs[..., :-1, :].float()
            student_log_probs = F.log_softmax(shift_logits, dim=-1)
            loss_kl_unmasked = F.kl_div(
                student_log_probs,
                teacher_log_probs,
                reduction="none",
                log_target=True,
            ).sum(dim=-1)
            loss_kl = (loss_kl_unmasked * valid_mask).sum() / valid_count
            total_loss = self.alpha * loss_ce + (1.0 - self.alpha) * loss_kl

        self.last_loss_components = {
            "ce_loss": float(loss_ce.detach().cpu().item()),
            "belief_kl_loss": float(loss_kl.detach().cpu().item()) if loss_kl is not None else 0.0,
            "total_loss": float(total_loss.detach().cpu().item()),
        }

        if self.is_terminal_chunk and choice_ids:
            self.last_choice_metrics = choice_metrics_from_logits(shift_logits.detach(), shift_labels, choice_ids)
            self.last_choice_details = choice_details_from_logits(shift_logits.detach(), shift_labels, choice_ids)
        needs_output_log_probs = self.returns_full_log_probs and not (
            self.is_terminal_chunk and bool(choice_ids)
        )
        if needs_output_log_probs:
            with torch.no_grad():
                output_log_probs = F.log_softmax(logits.float(), dim=-1)
        else:
            output_log_probs = None
        return total_loss, curr_hidden.detach(), output_log_probs


class LocalReadoutAdapter(nn.Module):
    def __init__(self, hidden_size: int, bottleneck: int) -> None:
        super().__init__()
        if bottleneck <= 0:
            raise ValueError("local readout adapter bottleneck must be positive.")
        self.proj_in = nn.Linear(hidden_size, bottleneck, bias=False)
        self.proj_mid = nn.Linear(bottleneck, bottleneck, bias=False)
        self.proj_out = nn.Linear(bottleneck, hidden_size, bias=False)
        nn.init.zeros_(self.proj_out.weight)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        update = self.proj_in(z)
        update = F.gelu(update)
        update = self.proj_mid(update)
        update = F.gelu(update)
        update = self.proj_out(update)
        return z + update


def _infer_hidden_size(final_norm: nn.Module, model: nn.Module) -> int:
    weight = getattr(final_norm, "weight", None)
    if isinstance(weight, torch.Tensor) and weight.ndim >= 1:
        return int(weight.shape[0])
    config = getattr(model, "config", None)
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is not None:
        return int(hidden_size)
    n_embd = getattr(config, "n_embd", None)
    if n_embd is not None:
        return int(n_embd)
    raise ValueError("Could not infer hidden size for local readout adapter.")


def _build_local_readout_adapter(
    *,
    model: nn.Module,
    final_norm: nn.Module,
    stage_id: int,
    num_chunks: int,
    bottleneck: int,
    stages: str,
) -> Optional[nn.Module]:
    if bottleneck <= 0 or stages == "none":
        return None
    if stages == "all":
        enabled = True
    elif stages == "middle":
        enabled = stage_id < num_chunks - 1
    else:
        raise ValueError(f"Unsupported local readout adapter stages: {stages}")
    if not enabled:
        return None
    hidden_size = _infer_hidden_size(final_norm, model)
    adapter = LocalReadoutAdapter(hidden_size=hidden_size, bottleneck=bottleneck)
    weight = getattr(final_norm, "weight", None)
    if isinstance(weight, torch.Tensor):
        adapter = adapter.to(dtype=weight.dtype)
    for param in adapter.parameters():
        param.requires_grad = True
    return adapter


def choice_metrics_from_logits(
    shift_logits: torch.Tensor,
    shift_labels: torch.Tensor,
    choice_ids: list[int],
) -> tuple[int, int, float]:
    valid_positions = shift_labels != -100
    if not valid_positions.any():
        return 0, 0, 0.0

    correct = 0
    count = 0
    loss_sum = 0.0
    choice_index = torch.tensor(choice_ids, dtype=torch.long, device=shift_logits.device)
    with torch.no_grad():
        for batch_idx, token_idx in valid_positions.nonzero(as_tuple=False):
            target_id = int(shift_labels[batch_idx, token_idx].item())
            if target_id not in choice_ids:
                continue
            scores = shift_logits[batch_idx, token_idx, choice_index].float()
            pred_choice = int(torch.argmax(scores).item())
            target_choice = choice_ids.index(target_id)
            correct += int(pred_choice == target_choice)
            count += 1
            loss_sum += float((-F.log_softmax(scores, dim=-1)[target_choice]).item())
    return correct, count, loss_sum / count if count else 0.0


def choice_details_from_logits(
    shift_logits: torch.Tensor,
    shift_labels: torch.Tensor,
    choice_ids: list[int],
) -> dict[str, Any]:
    valid_positions = shift_labels != -100
    if not valid_positions.any():
        return {
            "choice_correct": 0,
            "choice_count": 0,
            "choice_loss": 0.0,
            "predicted_token_id": "",
            "target_token_id": "",
        }

    correct = 0
    count = 0
    loss_sum = 0.0
    predicted_token_id: int | str = ""
    target_token_id: int | str = ""
    choice_index = torch.tensor(choice_ids, dtype=torch.long, device=shift_logits.device)
    with torch.no_grad():
        for batch_idx, token_idx in valid_positions.nonzero(as_tuple=False):
            target_id = int(shift_labels[batch_idx, token_idx].item())
            if target_id not in choice_ids:
                continue
            scores = shift_logits[batch_idx, token_idx, choice_index].float()
            pred_choice = int(torch.argmax(scores).item())
            target_choice = choice_ids.index(target_id)
            if count == 0:
                predicted_token_id = int(choice_ids[pred_choice])
                target_token_id = int(target_id)
            correct += int(pred_choice == target_choice)
            count += 1
            loss_sum += float((-F.log_softmax(scores, dim=-1)[target_choice]).item())
    return {
        "choice_correct": correct,
        "choice_count": count,
        "choice_loss": (loss_sum / count) if count else 0.0,
        "predicted_token_id": predicted_token_id,
        "target_token_id": target_token_id,
    }


def normalize_belief_transport_mode(raw_mode: str) -> str:
    normalized = raw_mode.strip().lower()
    if normalized in {"", "full", "dense"}:
        return "full"
    if normalized in {"terminal", "terminal_only", "final", "final_only"}:
        return "terminal"
    if normalized in {"none", "off", "disabled", "false"}:
        return "none"
    raise ValueError(f"Unsupported belief transport mode: {raw_mode}. Use full, terminal, or none.")


def build_stage_chunk(
    *,
    model: nn.Module,
    stage_id: int,
    num_chunks: int,
    belief_transport_mode: str,
    alpha: float,
    label_smoothing: float,
    local_readout_adapter_bottleneck: int = 0,
    local_readout_adapter_stages: str = "none",
) -> ServerBpfreeChunk:
    layers, final_norm, lm_head, vocab_size, rotary_emb = get_model_parts(model)
    total_layers = len(layers)
    chunk_size = total_layers // num_chunks
    start = stage_id * chunk_size
    end = (stage_id + 1) * chunk_size if stage_id < num_chunks - 1 else total_layers
    print(f"stage {stage_id}: layers=[{start}, {end - 1}]", flush=True)
    local_readout_adapter = _build_local_readout_adapter(
        model=model,
        final_norm=final_norm,
        stage_id=stage_id,
        num_chunks=num_chunks,
        bottleneck=local_readout_adapter_bottleneck,
        stages=local_readout_adapter_stages,
    )
    return ServerBpfreeChunk(
        chunk_idx=stage_id,
        layer_start=start,
        layer_end=end,
        layers=[layers[i] for i in range(start, end)],
        final_norm=final_norm,
        lm_head=lm_head,
        vocab_size=vocab_size,
        rotary_emb=rotary_emb,
        is_terminal_chunk=stage_id == num_chunks - 1,
        belief_transport_mode=belief_transport_mode,
        alpha=alpha,
        label_smoothing=label_smoothing,
        local_readout_adapter=local_readout_adapter,
    )


def build_optimizer(
    *,
    params: list[nn.Parameter],
    cfg: StageWorkerConfig,
) -> torch.optim.Optimizer:
    learning_rate = cfg.learning_rate or 3e-4
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=learning_rate)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(
            params,
            lr=learning_rate,
            momentum=cfg.sgd_momentum,
            dampening=cfg.sgd_dampening,
            weight_decay=cfg.sgd_weight_decay,
            nesterov=cfg.sgd_nesterov,
        )
    raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")


def tensor_to_cpu(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if value is None:
        return None
    return value.detach().to("cpu")
