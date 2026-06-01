import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


MODEL_PRESETS = {
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "smollm2_360m": "HuggingFaceTB/SmolLM2-360M",
    "phi2": "microsoft/phi-2",
}


def resolve_model_name(model_name: str) -> str:
    return MODEL_PRESETS.get(model_name, model_name)


def resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def resolve_dtype(raw: str) -> torch.dtype:
    if raw == "float32":
        return torch.float32
    if raw == "float16":
        return torch.float16
    if raw == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {raw}")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, init_std: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")
        self.base = base
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_a = nn.Parameter(torch.randn(rank, base.in_features) * init_std)
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        for param in self.base.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_hidden = F.linear(x, self.lora_a)
        lora_out = F.linear(lora_hidden, self.lora_b) * self.scaling
        return base_out + lora_out


def inject_lora_adapters(
    module: nn.Module,
    target_names: set[str],
    rank: int,
    alpha: float,
    init_std: float,
) -> int:
    injected = 0
    for child_name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and child_name in target_names:
            module._modules[child_name] = LoRALinear(
                base=child,
                rank=rank,
                alpha=alpha,
                init_std=init_std,
            )
            injected += 1
        else:
            injected += inject_lora_adapters(
                module=child,
                target_names=target_names,
                rank=rank,
                alpha=alpha,
                init_std=init_std,
            )
    return injected


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


class BpfreeChunk(nn.Module):
    def __init__(
        self,
        chunk_idx: int,
        layers: Iterable[nn.Module],
        final_norm: nn.Module,
        lm_head: nn.Module,
        vocab_size: int,
        rotary_emb: nn.Module | None,
        alpha: float,
        label_smoothing: float,
    ) -> None:
        super().__init__()
        self.chunk_idx = chunk_idx
        self.layers = nn.ModuleList(list(layers))
        self.final_norm = final_norm
        self.lm_head = lm_head
        self.vocab_size = vocab_size
        self.rotary_emb = rotary_emb
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def compute_dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        labels: torch.Tensor,
        prev_log_probs: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        logits = self.lm_head(self.final_norm(curr_hidden))
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

        if self.chunk_idx == 0:
            total_loss = loss_ce
        else:
            if prev_log_probs is None:
                raise RuntimeError("prev_log_probs is required for non-zero chunks.")
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

        return total_loss, curr_hidden.detach(), F.log_softmax(logits.float(), dim=-1).detach()


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


def build_chunks(
    backbone: nn.Module,
    num_chunks: int,
    alpha: float,
    label_smoothing: float,
) -> list[BpfreeChunk]:
    layers, final_norm, lm_head, vocab_size, rotary_emb = get_model_parts(backbone)
    total_layers = len(layers)
    chunk_size = total_layers // num_chunks
    chunks = []
    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = (chunk_idx + 1) * chunk_size if chunk_idx < num_chunks - 1 else total_layers
        chunks.append(
            BpfreeChunk(
                chunk_idx=chunk_idx,
                layers=[layers[i] for i in range(start, end)],
                final_norm=final_norm,
                lm_head=lm_head,
                vocab_size=vocab_size,
                rotary_emb=rotary_emb,
                alpha=alpha,
                label_smoothing=label_smoothing,
            )
        )
        print(f"chunk {chunk_idx}: layers=[{start}, {end - 1}]")
    return chunks


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


def parse_train_chunks(raw: str, num_chunks: int) -> set[int]:
    if raw == "all":
        return set(range(num_chunks))
    chunks = {int(item.strip()) for item in raw.split(",") if item.strip()}
    invalid = [idx for idx in chunks if idx < 0 or idx >= num_chunks]
    if invalid:
        raise ValueError(f"Invalid train chunk indices for num_chunks={num_chunks}: {invalid}")
    return chunks


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


def label_choice_metrics(
    log_probs: torch.Tensor,
    labels: torch.Tensor,
    choice_ids: list[int],
) -> tuple[int, int, float]:
    shift_log_probs = log_probs[..., :-1, :].float()
    shift_labels = labels[..., 1:].long()
    valid_positions = shift_labels != -100
    if not valid_positions.any():
        return 0, 0, 0.0

    correct = 0
    count = 0
    loss_sum = 0.0
    choice_index = torch.tensor(choice_ids, dtype=torch.long, device=log_probs.device)
    for batch_idx, token_idx in valid_positions.nonzero(as_tuple=False):
        target_id = int(shift_labels[batch_idx, token_idx].item())
        if target_id not in choice_ids:
            continue
        scores = shift_log_probs[batch_idx, token_idx, choice_index]
        pred_choice = int(torch.argmax(scores).item())
        target_choice = choice_ids.index(target_id)
        correct += int(pred_choice == target_choice)
        count += 1
        loss_sum += float((-F.log_softmax(scores, dim=-1)[target_choice]).item())
    avg_loss = loss_sum / count if count else 0.0
    return correct, count, avg_loss


def load_record_state(
    record: dict[str, Any],
    manifest_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    tensors = record["tensors"]
    return {
        "hidden": load_tensor(manifest_dir, tensors["hidden_states"]).to(device),
        "attention_mask": load_tensor(manifest_dir, tensors["attention_mask"]).to(device),
        "position_ids": load_tensor(manifest_dir, tensors["position_ids"]).to(device),
        "labels": load_tensor(manifest_dir, tensors["labels"]).to(device),
        "prev_log_probs": None,
        "losses": [],
    }


def run_chunk_for_state(
    *,
    record: dict[str, Any],
    state: dict[str, Any],
    chunk_idx: int,
    chunk: BpfreeChunk,
    optimizers: dict[int, torch.optim.Optimizer],
    train_chunks: set[int],
    mode: str,
    learning_rate_override: float | None,
    grad_clip: float,
) -> None:
    train_this_chunk = mode == "train" and chunk_idx in train_chunks
    chunk.train(train_this_chunk)
    if train_this_chunk:
        optimizer = optimizers[chunk_idx]
        lr = learning_rate_override
        if lr is None:
            lr = record.get("learning_rate")
        if lr is not None:
            for group in optimizer.param_groups:
                group["lr"] = float(lr)
        optimizer.zero_grad(set_to_none=True)

    with torch.set_grad_enabled(train_this_chunk):
        loss, next_hidden, next_log_probs = chunk(
            hidden_states=state["hidden"],
            attention_mask=state["attention_mask"],
            position_ids=state["position_ids"],
            labels=state["labels"],
            prev_log_probs=state["prev_log_probs"],
        )
        if train_this_chunk:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(chunk.parameters(), grad_clip)
            optimizers[chunk_idx].step()

    state["losses"].append(float(loss.detach().cpu().item()))
    state["hidden"] = next_hidden.detach()
    state["prev_log_probs"] = next_log_probs.detach()


def finish_record_result(record: dict[str, Any], state: dict[str, Any], mode: str) -> dict[str, Any]:
    prev_log_probs = state["prev_log_probs"]
    if prev_log_probs is None:
        raise RuntimeError("Record finished without final log probabilities.")
    labels = state["labels"]
    losses = state["losses"]
    choice_ids = one_token_choice_ids(record)
    choice_correct, choice_count, choice_loss = label_choice_metrics(prev_log_probs, labels, choice_ids)
    response = (record.get("text") or {}).get("response", "").strip()
    return {
        "request_id": record.get("request_id", ""),
        "dataset_index": int(record.get("dataset_index", -1)),
        "response": response,
        "mode": mode,
        "loss": losses[-1],
        "chunk_losses": losses,
        "choice_correct": choice_correct,
        "choice_count": choice_count,
        "choice_accuracy": (choice_correct / choice_count) if choice_count else 0.0,
        "choice_loss": choice_loss,
    }


def run_record(
    record: dict[str, Any],
    manifest_dir: Path,
    chunks: list[BpfreeChunk],
    optimizers: dict[int, torch.optim.Optimizer],
    train_chunks: set[int],
    device: torch.device,
    mode: str,
    learning_rate_override: float | None,
    grad_clip: float,
) -> dict[str, Any]:
    state = load_record_state(record, manifest_dir, device)
    for chunk_idx, chunk in enumerate(chunks):
        run_chunk_for_state(
            record=record,
            state=state,
            chunk_idx=chunk_idx,
            chunk=chunk,
            optimizers=optimizers,
            train_chunks=train_chunks,
            mode=mode,
            learning_rate_override=learning_rate_override,
            grad_clip=grad_clip,
        )
    return finish_record_result(record, state, mode)


def accumulate_result(
    *,
    result: dict[str, Any],
    correct: int,
    count: int,
    losses: list[float],
    per_class: dict[str, dict[str, float]],
) -> tuple[int, int]:
    correct += result["choice_correct"]
    count += result["choice_count"]
    losses.append(result["loss"])
    label = result["response"] or "unknown"
    stats = per_class.setdefault(label, {"correct": 0.0, "count": 0.0, "loss_sum": 0.0})
    stats["correct"] += result["choice_correct"]
    stats["count"] += result["choice_count"]
    stats["loss_sum"] += result["loss"]
    return correct, count


def write_result_row(writer: csv.DictWriter, seq: int, result: dict[str, Any]) -> None:
    writer.writerow(
        {
            "seq": seq,
            "request_id": result["request_id"],
            "dataset_index": result["dataset_index"],
            "response": result["response"],
            "mode": result["mode"],
            "loss": result["loss"],
            "choice_correct": result["choice_correct"],
            "choice_count": result["choice_count"],
            "choice_accuracy": result["choice_accuracy"],
            "choice_loss": result["choice_loss"],
            "chunk_losses_json": json.dumps(result["chunk_losses"]),
        }
    )


def summarize_phase(
    *,
    name: str,
    records_len: int,
    correct: int,
    count: int,
    losses: list[float],
    per_class: dict[str, dict[str, float]],
    output_csv: Path,
    train_schedule: str,
    pipeline_window: int,
) -> dict[str, Any]:
    class_rows = []
    for label, stats in sorted(per_class.items()):
        class_count = int(stats["count"])
        class_rows.append(
            {
                "label": label,
                "correct": int(stats["correct"]),
                "count": class_count,
                "accuracy": (stats["correct"] / class_count) if class_count else 0.0,
                "avg_loss": stats["loss_sum"] / class_count if class_count else 0.0,
            }
        )
    return {
        "phase": name,
        "rows": records_len,
        "choice_correct": int(correct),
        "choice_count": int(count),
        "choice_accuracy": (correct / count) if count else 0.0,
        "avg_loss": sum(losses) / len(losses),
        "per_class": class_rows,
        "csv": str(output_csv),
        "train_schedule": train_schedule,
        "pipeline_window": pipeline_window,
    }


def result_fieldnames() -> list[str]:
    return [
        "seq",
        "request_id",
        "dataset_index",
        "response",
        "mode",
        "loss",
        "choice_correct",
        "choice_count",
        "choice_accuracy",
        "choice_loss",
        "chunk_losses_json",
    ]


def run_phase_stage_window(
    *,
    name: str,
    records: list[dict[str, Any]],
    manifest_dir: Path,
    chunks: list[BpfreeChunk],
    optimizers: dict[int, torch.optim.Optimizer],
    train_chunks: set[int],
    device: torch.device,
    learning_rate_override: float | None,
    grad_clip: float,
    output_csv: Path,
    pipeline_window: int,
) -> dict[str, Any]:
    if pipeline_window <= 1:
        raise ValueError("stage_window schedule requires pipeline_window > 1.")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    correct = 0
    count = 0
    losses: list[float] = []
    per_class: dict[str, dict[str, float]] = {}
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=result_fieldnames())
        writer.writeheader()
        seq = 0
        for start in range(0, len(records), pipeline_window):
            batch_records = records[start : start + pipeline_window]
            states = [load_record_state(record, manifest_dir, device) for record in batch_records]
            for chunk_idx, chunk in enumerate(chunks):
                for record, state in zip(batch_records, states, strict=False):
                    run_chunk_for_state(
                        record=record,
                        state=state,
                        chunk_idx=chunk_idx,
                        chunk=chunk,
                        optimizers=optimizers,
                        train_chunks=train_chunks,
                        mode="train",
                        learning_rate_override=learning_rate_override,
                        grad_clip=grad_clip,
                    )
            for record, state in zip(batch_records, states, strict=False):
                result = finish_record_result(record, state, "train")
                correct, count = accumulate_result(
                    result=result,
                    correct=correct,
                    count=count,
                    losses=losses,
                    per_class=per_class,
                )
                write_result_row(writer, seq, result)
                seq += 1
                if seq % 16 == 0 or seq == len(records):
                    acc = correct / count if count else 0.0
                    avg_loss = sum(losses) / len(losses)
                    print(f"{name}: {seq}/{len(records)} acc={acc:.4f} loss={avg_loss:.4f}")

    return summarize_phase(
        name=name,
        records_len=len(records),
        correct=correct,
        count=count,
        losses=losses,
        per_class=per_class,
        output_csv=output_csv,
        train_schedule="stage_window",
        pipeline_window=pipeline_window,
    )


def run_phase(
    *,
    name: str,
    records: list[dict[str, Any]],
    manifest_dir: Path,
    chunks: list[BpfreeChunk],
    optimizers: dict[int, torch.optim.Optimizer],
    train_chunks: set[int],
    device: torch.device,
    mode: str,
    learning_rate_override: float | None,
    grad_clip: float,
    output_csv: Path,
    train_schedule: str,
    pipeline_window: int,
) -> dict[str, Any]:
    if mode == "train" and train_schedule == "stage_window" and pipeline_window > 1:
        return run_phase_stage_window(
            name=name,
            records=records,
            manifest_dir=manifest_dir,
            chunks=chunks,
            optimizers=optimizers,
            train_chunks=train_chunks,
            device=device,
            learning_rate_override=learning_rate_override,
            grad_clip=grad_clip,
            output_csv=output_csv,
            pipeline_window=pipeline_window,
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    correct = 0
    count = 0
    losses: list[float] = []
    per_class: dict[str, dict[str, float]] = {}
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=result_fieldnames())
        writer.writeheader()
        for seq, record in enumerate(records):
            result = run_record(
                record=record,
                manifest_dir=manifest_dir,
                chunks=chunks,
                optimizers=optimizers,
                train_chunks=train_chunks,
                device=device,
                mode=mode,
                learning_rate_override=learning_rate_override,
                grad_clip=grad_clip,
            )
            correct, count = accumulate_result(
                result=result,
                correct=correct,
                count=count,
                losses=losses,
                per_class=per_class,
            )
            write_result_row(writer, seq, result)
            if (seq + 1) % 16 == 0 or seq + 1 == len(records):
                acc = correct / count if count else 0.0
                avg_loss = sum(losses) / len(losses)
                print(f"{name}: {seq + 1}/{len(records)} acc={acc:.4f} loss={avg_loss:.4f}")

    return summarize_phase(
        name=name,
        records_len=len(records),
        correct=correct,
        count=count,
        losses=losses,
        per_class=per_class,
        output_csv=output_csv,
        train_schedule=train_schedule if mode == "train" else "eval",
        pipeline_window=pipeline_window if mode == "train" else 1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Server-side BP-free chunked LoRA label experiment. It consumes the same "
            "prepared request manifests used by Android, but runs all chunks in one "
            "PyTorch process for fast quality sweeps."
        )
    )
    parser.add_argument("--model_name", default="tinyllama")
    parser.add_argument("--train_manifest", type=Path, required=True)
    parser.add_argument("--eval_manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_chunks", type=int, default=3)
    parser.add_argument("--train_chunks", default="all")
    parser.add_argument("--train_limit", type=int, default=None)
    parser.add_argument(
        "--train_epochs",
        type=int,
        default=1,
        help="Repeat the selected training manifest records this many times.",
    )
    parser.add_argument("--eval_limit", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument(
        "--optimizer",
        default="adamw",
        choices=["adamw", "sgd"],
        help=(
            "Optimizer for trainable LoRA parameters. adamw is the historical "
            "server default; sgd approximates the current ExecuTorch phone runner."
        ),
    )
    parser.add_argument(
        "--sgd_momentum",
        type=float,
        default=0.0,
        help="Momentum used when --optimizer=sgd.",
    )
    parser.add_argument(
        "--sgd_dampening",
        type=float,
        default=0.0,
        help="Dampening used when --optimizer=sgd.",
    )
    parser.add_argument(
        "--sgd_weight_decay",
        type=float,
        default=0.0,
        help="Weight decay used when --optimizer=sgd.",
    )
    parser.add_argument(
        "--sgd_nesterov",
        action="store_true",
        help="Enable Nesterov momentum when --optimizer=sgd.",
    )
    parser.add_argument(
        "--train_schedule",
        default="fifo",
        choices=["fifo", "stage_window"],
        help=(
            "Training request schedule. fifo is strict record-by-record order. "
            "stage_window processes a small window chunk-by-chunk to approximate "
            "bounded in-flight mobile pipeline scheduling."
        ),
    )
    parser.add_argument(
        "--pipeline_window",
        type=int,
        default=1,
        help="Window size for --train_schedule=stage_window; use 3 to mimic phone maxWindow=3.",
    )
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_targets", default="q_proj,v_proj")
    parser.add_argument("--lora_init_std", type=float, default=0.01)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=20260531)
    args = parser.parse_args()

    if args.num_chunks <= 0:
        raise ValueError("--num_chunks must be positive.")
    if args.train_epochs <= 0:
        raise ValueError("--train_epochs must be positive.")
    if args.pipeline_window <= 0:
        raise ValueError("--pipeline_window must be positive.")
    if args.train_schedule == "stage_window" and args.pipeline_window <= 1:
        raise ValueError("--train_schedule=stage_window requires --pipeline_window > 1.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    resolved_model = resolve_model_name(args.model_name)
    train_chunks = parse_train_chunks(args.train_chunks, args.num_chunks)

    base_train_records = read_manifest(args.train_manifest, args.train_limit)
    train_records = base_train_records * args.train_epochs
    eval_records = read_manifest(args.eval_manifest, args.eval_limit)
    train_dir = args.train_manifest.parent
    eval_dir = args.eval_manifest.parent

    print(f"Loading model: {resolved_model} dtype={dtype} device={device}")
    model = AutoModelForCausalLM.from_pretrained(resolved_model, torch_dtype=dtype)
    injected = inject_lora_adapters(
        module=model,
        target_names={item.strip() for item in args.lora_targets.split(",") if item.strip()},
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        init_std=args.lora_init_std,
    )
    trainable, frozen = configure_lora_trainable(model)
    model.to(device)
    print(f"LoRA modules={injected} trainable_params={trainable} frozen_params={frozen}")

    chunks = build_chunks(
        backbone=model,
        num_chunks=args.num_chunks,
        alpha=args.alpha,
        label_smoothing=args.label_smoothing,
    )
    def build_optimizer(params: list[nn.Parameter]) -> torch.optim.Optimizer:
        learning_rate = args.learning_rate or 3e-4
        if args.optimizer == "adamw":
            return torch.optim.AdamW(params, lr=learning_rate)
        if args.optimizer == "sgd":
            return torch.optim.SGD(
                params,
                lr=learning_rate,
                momentum=args.sgd_momentum,
                dampening=args.sgd_dampening,
                weight_decay=args.sgd_weight_decay,
                nesterov=args.sgd_nesterov,
            )
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    optimizers = {
        idx: build_optimizer([param for param in chunks[idx].parameters() if param.requires_grad])
        for idx in train_chunks
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)

    eval_before = run_phase(
        name="eval_before",
        records=eval_records,
        manifest_dir=eval_dir,
        chunks=chunks,
        optimizers=optimizers,
        train_chunks=train_chunks,
        device=device,
        mode="eval",
        learning_rate_override=args.learning_rate,
        grad_clip=args.grad_clip,
        output_csv=args.output_dir / "eval_before.csv",
        train_schedule=args.train_schedule,
        pipeline_window=args.pipeline_window,
    )
    train = run_phase(
        name="train",
        records=train_records,
        manifest_dir=train_dir,
        chunks=chunks,
        optimizers=optimizers,
        train_chunks=train_chunks,
        device=device,
        mode="train",
        learning_rate_override=args.learning_rate,
        grad_clip=args.grad_clip,
        output_csv=args.output_dir / "train.csv",
        train_schedule=args.train_schedule,
        pipeline_window=args.pipeline_window,
    )
    eval_after = run_phase(
        name="eval_after",
        records=eval_records,
        manifest_dir=eval_dir,
        chunks=chunks,
        optimizers=optimizers,
        train_chunks=train_chunks,
        device=device,
        mode="eval",
        learning_rate_override=args.learning_rate,
        grad_clip=args.grad_clip,
        output_csv=args.output_dir / "eval_after.csv",
        train_schedule=args.train_schedule,
        pipeline_window=args.pipeline_window,
    )

    summary = {
        "model_name": args.model_name,
        "resolved_model": resolved_model,
        "num_chunks": args.num_chunks,
        "train_chunks": sorted(train_chunks),
        "train_epochs": args.train_epochs,
        "unique_train_records": len(base_train_records),
        "train_steps": len(train_records),
        "lora": {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "targets": args.lora_targets,
            "init_std": args.lora_init_std,
            "modules": injected,
            "trainable_params": trainable,
        },
        "learning_rate": args.learning_rate,
        "optimizer": args.optimizer,
        "sgd_momentum": args.sgd_momentum,
        "sgd_dampening": args.sgd_dampening,
        "sgd_weight_decay": args.sgd_weight_decay,
        "sgd_nesterov": args.sgd_nesterov,
        "grad_clip": args.grad_clip,
        "train_schedule": args.train_schedule,
        "pipeline_window": args.pipeline_window,
        "alpha": args.alpha,
        "label_smoothing": args.label_smoothing,
        "seed": args.seed,
        "phases": [eval_before, train, eval_after],
        "delta": {
            "choice_accuracy": eval_after["choice_accuracy"] - eval_before["choice_accuracy"],
            "avg_loss": eval_after["avg_loss"] - eval_before["avg_loss"],
        },
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
