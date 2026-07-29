from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional

import torch
import torch.nn.functional as F


_DIAGNOSTIC_LOCAL_OBJECTIVE = os.environ.get(
    "BPFREE_DIAGNOSTIC_LOCAL_OBJECTIVE",
    "full_vocab_ce",
).strip().lower()

if _DIAGNOSTIC_LOCAL_OBJECTIVE not in {"full_vocab_ce", "hidden_l2"}:
    raise RuntimeError(
        "BPFREE_DIAGNOSTIC_LOCAL_OBJECTIVE must be "
        "full_vocab_ce or hidden_l2, got "
        f"{_DIAGNOSTIC_LOCAL_OBJECTIVE!r}"
    )


def _choice_metrics_from_logits(
    shift_logits: torch.Tensor,
    shift_labels: torch.Tensor,
    choice_ids: list[int],
):
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


@dataclass
class BodyForwardResult:
    curr_hidden: torch.Tensor


@dataclass
class LocalHeadLossResult:
    loss: torch.Tensor
    output_log_probs: Optional[torch.Tensor]


def body_forward(
    *,
    chunk: torch.nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
) -> BodyForwardResult:
    """
    The body-only part of ServerBpfreeChunk.forward.

    This runs transformer layers and returns curr_hidden WITH autograd graph.
    Do not detach here. The caller will detach only the send buffer.
    """
    chunk.last_choice_metrics = None
    chunk.last_loss_components = {}

    dtype = chunk.compute_dtype()
    hidden_states = hidden_states.to(dtype=dtype)
    attention_mask = attention_mask.to(dtype=dtype)
    position_ids = position_ids.long()

    position_embeddings = None
    if chunk.rotary_emb is not None:
        position_embeddings = chunk.rotary_emb(hidden_states, position_ids)

    curr_hidden = hidden_states
    for layer in chunk.layers:
        layer_out = layer(
            curr_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )
        curr_hidden = layer_out[0] if isinstance(layer_out, tuple) else layer_out

    return BodyForwardResult(curr_hidden=curr_hidden)


def local_head_loss_from_hidden(
    *,
    chunk: torch.nn.Module,
    curr_hidden: torch.Tensor,
    labels: torch.Tensor,
    prev_log_probs: Optional[torch.Tensor],
    choice_ids: Optional[list[int]] = None,
    record_loss_components: bool = True,
    emit_output_log_probs: bool = True,
) -> LocalHeadLossResult:
    """
    The local-head / local-objective part of ServerBpfreeChunk.forward.

    This intentionally mirrors the original forward loss semantics:
      logits = lm_head(final_norm(curr_hidden))
      CE loss
      optional belief KL
      last_loss_components
      optional last_choice_metrics
      optional transported output_log_probs
    """
    labels = labels.long()

    if _DIAGNOSTIC_LOCAL_OBJECTIVE == "hidden_l2":
        if chunk.uses_belief_loss:
            raise RuntimeError(
                "hidden_l2 diagnostic objective does not support belief KL."
            )

        # Keep gradient support on the same shifted supervised positions as CE,
        # while removing final_norm, full-vocabulary lm_head, and cross-entropy.
        shifted_hidden = curr_hidden[..., :-1, :].float()
        valid_mask = (labels[..., 1:] != -100).to(
            dtype=shifted_hidden.dtype
        ).unsqueeze(-1)
        denominator = (
            valid_mask.sum() * shifted_hidden.shape[-1]
        ).clamp_min(1.0)
        total_loss = (shifted_hidden.square() * valid_mask).sum() / denominator

        if record_loss_components:
            value = float(total_loss.detach().cpu().item())
            chunk.last_loss_components = {
                "ce_loss": 0.0,
                "belief_kl_loss": 0.0,
                "total_loss": value,
                "diagnostic_hidden_l2_loss": value,
            }
        else:
            chunk.last_loss_components = {}

        chunk.last_choice_metrics = None
        return LocalHeadLossResult(
            loss=total_loss,
            output_log_probs=None,
        )

    z = chunk.final_norm(curr_hidden)
    adapter = getattr(chunk, "local_readout_adapter", None)
    if adapter is not None:
        z = adapter(z)
    logits = chunk.lm_head(z)
    shift_logits = logits[..., :-1, :].float()
    shift_labels = labels[..., 1:]

    valid_mask = (shift_labels != -100).float()
    valid_count = valid_mask.sum().clamp_min(1.0)
    safe_labels = torch.where(
        shift_labels != -100,
        shift_labels,
        torch.zeros_like(shift_labels),
    )

    loss_ce_unmasked = F.cross_entropy(
        shift_logits.reshape(-1, chunk.vocab_size),
        safe_labels.reshape(-1),
        reduction="none",
        label_smoothing=chunk.label_smoothing,
    ).reshape_as(shift_labels)

    loss_ce = (loss_ce_unmasked * valid_mask).sum() / valid_count

    if not chunk.uses_belief_loss:
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
        total_loss = chunk.alpha * loss_ce + (1.0 - chunk.alpha) * loss_kl

    if record_loss_components:
        chunk.last_loss_components = {
            "ce_loss": float(loss_ce.detach().cpu().item()),
            "belief_kl_loss": float(loss_kl.detach().cpu().item()) if loss_kl is not None else 0.0,
            "total_loss": float(total_loss.detach().cpu().item()),
        }
    else:
        chunk.last_loss_components = {}

    if chunk.is_terminal_chunk and choice_ids:
        chunk.last_choice_metrics = _choice_metrics_from_logits(
            shift_logits.detach(),
            shift_labels,
            choice_ids,
        )

    needs_output_log_probs = emit_output_log_probs and chunk.returns_full_log_probs and not (
        chunk.is_terminal_chunk and bool(choice_ids)
    )

    if needs_output_log_probs:
        with torch.no_grad():
            output_log_probs = F.log_softmax(logits.float(), dim=-1)
    else:
        output_log_probs = None

    return LocalHeadLossResult(
        loss=total_loss,
        output_log_probs=output_log_probs,
    )
