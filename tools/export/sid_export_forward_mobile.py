import argparse
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.export import export
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM

import executorch.exir as exir
from executorch.exir import EdgeCompileConfig


MODEL_PRESETS = {
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "smollm2_360m": "HuggingFaceTB/SmolLM2-360M",
    "phi2": "microsoft/phi-2",
}


def resolve_model_name(model_name: str) -> str:
    return MODEL_PRESETS.get(model_name, model_name)


def transport_cast(tensor: torch.Tensor, transport_dtype: torch.dtype) -> torch.Tensor:
    if tensor.dtype == transport_dtype:
        return tensor
    return tensor.to(transport_dtype)


class ForwardOnlySIDChunk(nn.Module):
    def __init__(
        self,
        chunk_idx: int,
        layers: Iterable[nn.Module],
        final_norm: nn.Module,
        lm_head: nn.Module,
        vocab_size: int,
        rotary_emb: nn.Module | None,
        alpha: float = 0.5,
        label_smoothing: float = 0.1,
        transport_dtype: torch.dtype = torch.float16,
        relay_only: bool = False,
    ) -> None:
        super().__init__()
        self.chunk_idx = chunk_idx
        self.layers = nn.ModuleList(list(layers))
        self.final_norm = final_norm
        self.lm_head = lm_head
        self.rotary_emb = rotary_emb
        self.vocab_size = vocab_size
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.transport_dtype = transport_dtype
        self.relay_only = relay_only

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        labels: torch.Tensor,
        prev_log_probs: torch.Tensor | None = None,
    ):
        hidden_states = hidden_states.float()
        attention_mask = attention_mask.float()
        position_ids = position_ids.long()

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
        if self.relay_only:
            total_loss = curr_hidden.sum() * 0.0
        else:
            labels = labels.long()
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            valid_mask = (shift_labels != -100).float()
            valid_tokens_count = valid_mask.sum().clamp_min(1.0)
            safe_labels = torch.where(shift_labels != -100, shift_labels, torch.zeros_like(shift_labels))

            loss_ce_unmasked = F.cross_entropy(
                shift_logits.reshape(-1, self.vocab_size),
                safe_labels.reshape(-1),
                reduction="none",
                label_smoothing=self.label_smoothing,
            ).reshape_as(shift_labels)
            loss_ce = (loss_ce_unmasked * valid_mask).sum() / valid_tokens_count

            if self.chunk_idx == 0:
                total_loss = loss_ce
            else:
                if prev_log_probs is None:
                    raise RuntimeError("prev_log_probs is required for non-zero chunks.")
                teacher_log_probs = prev_log_probs[..., :-1, :].float()
                student_log_probs = F.log_softmax(shift_logits.float(), dim=-1)
                loss_kl_unmasked = F.kl_div(
                    student_log_probs,
                    teacher_log_probs,
                    reduction="none",
                    log_target=True,
                ).sum(dim=-1)
                loss_kl = (loss_kl_unmasked * valid_mask).sum() / valid_tokens_count
                total_loss = self.alpha * loss_ce + (1.0 - self.alpha) * loss_kl

        next_hidden = transport_cast(curr_hidden.detach(), self.transport_dtype)
        next_log_probs = transport_cast(
            F.log_softmax(logits.float(), dim=-1).detach(),
            self.transport_dtype,
        )
        return total_loss, next_hidden, next_log_probs


def get_model_parts(backbone: nn.Module):
    if not hasattr(backbone, "model"):
        raise ValueError("Expected AutoModelForCausalLM with a .model backbone.")

    body = backbone.model
    if not hasattr(body, "layers"):
        raise ValueError("Expected backbone.model.layers to exist for chunk export.")

    final_norm = getattr(body, "final_layernorm", None) or getattr(body, "norm", None)
    if final_norm is None:
        raise ValueError("Could not locate final norm module on backbone.model.")

    lm_head = getattr(backbone, "lm_head", None)
    if lm_head is None:
        raise ValueError("Could not locate lm_head on the causal LM.")

    rotary_emb = getattr(body, "rotary_emb", None)
    return body.layers, final_norm, lm_head, backbone.config.vocab_size, rotary_emb


def build_chunk_module(
    backbone: nn.Module,
    num_chunks: int,
    target_chunk: int,
    alpha: float,
    label_smoothing: float,
    transport_dtype: torch.dtype,
    relay_only: bool,
) -> tuple[ForwardOnlySIDChunk, int, int]:
    layers, final_norm, lm_head, vocab_size, rotary_emb = get_model_parts(backbone)
    total_layers = len(layers)
    chunk_size = total_layers // num_chunks
    start = target_chunk * chunk_size
    end = (target_chunk + 1) * chunk_size if target_chunk < num_chunks - 1 else total_layers
    chunk_layers = [layers[j] for j in range(start, end)]
    chunk_module = ForwardOnlySIDChunk(
        chunk_idx=target_chunk,
        layers=chunk_layers,
        final_norm=final_norm,
        lm_head=lm_head,
        vocab_size=vocab_size,
        rotary_emb=rotary_emb,
        alpha=alpha,
        label_smoothing=label_smoothing,
        transport_dtype=transport_dtype,
        relay_only=relay_only,
    )
    return chunk_module, start, end


def build_example_args(
    *,
    hidden_dim: int,
    vocab_size: int,
    seq_len: int,
    batch_size: int,
    target_chunk: int,
) -> tuple[torch.Tensor, ...]:
    dummy_hidden = torch.randn((batch_size, seq_len, hidden_dim), dtype=torch.float32)
    dummy_mask = torch.zeros((batch_size, 1, seq_len, seq_len), dtype=torch.float32)
    dummy_pos = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    dummy_labels = torch.ones((batch_size, seq_len), dtype=torch.long)

    if target_chunk == 0:
        return (dummy_hidden, dummy_mask, dummy_pos, dummy_labels)

    dummy_prev_log_probs = torch.randn((batch_size, seq_len, vocab_size), dtype=torch.float32)
    return (dummy_hidden, dummy_mask, dummy_pos, dummy_labels, dummy_prev_log_probs)


def export_chunk(
    *,
    model_name: str,
    num_chunks: int,
    target_chunk: int,
    seq_len: int,
    batch_size: int,
    output_dir: Path,
    artifact_prefix: str,
    artifact_suffix: str,
    alpha: float,
    label_smoothing: float,
    transport_dtype: torch.dtype,
    use_xnnpack: bool,
    relay_only: bool,
) -> Path:
    resolved_model_name = resolve_model_name(model_name)
    print(f"[1/5] Loading model: {resolved_model_name}")
    backbone = AutoModelForCausalLM.from_pretrained(
        resolved_model_name,
        torch_dtype=torch.float32,
    )

    print(f"[2/5] Building forward-only chunk {target_chunk}/{num_chunks - 1}")
    chunk_module, start, end = build_chunk_module(
        backbone=backbone,
        num_chunks=num_chunks,
        target_chunk=target_chunk,
        alpha=alpha,
        label_smoothing=label_smoothing,
        transport_dtype=transport_dtype,
        relay_only=relay_only,
    )
    print(f"      layers=[{start}, {end - 1}]")

    chunk_module.eval()
    for param in chunk_module.parameters():
        param.requires_grad = False

    hidden_dim = backbone.config.hidden_size
    vocab_size = backbone.config.vocab_size
    example_args = build_example_args(
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        seq_len=seq_len,
        batch_size=batch_size,
        target_chunk=target_chunk,
    )
    print(
        f"[3/5] Exporting forward graph with seq_len={seq_len}, "
        f"batch_size={batch_size}, inputs={len(example_args)}"
    )

    with sdpa_kernel([SDPBackend.MATH]):
        exported_program = export(chunk_module, example_args, strict=False)

    print("[4/5] Lowering to ExecuTorch edge program")
    edge_config = EdgeCompileConfig(_check_ir_validity=False)
    if use_xnnpack:
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
        from executorch.exir import to_edge_transform_and_lower

        edge_program = to_edge_transform_and_lower(
            exported_program,
            compile_config=edge_config,
            partitioner=[XnnpackPartitioner(force_fp32_dynamic_linear=True)],
        )
    else:
        edge_program = exir.to_edge(exported_program, compile_config=edge_config)

    executorch_program = edge_program.to_executorch()

    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"{artifact_prefix}_chunk_{target_chunk}{artifact_suffix}.pte"

    print("[5/5] Writing .pte")
    with save_path.open("wb") as f:
        f.write(executorch_program.buffer)

    print(f"Saved: {save_path}")
    return save_path


def parse_transport_dtype(raw: str) -> torch.dtype:
    normalized = raw.strip().lower()
    if normalized == "float16":
        return torch.float16
    if normalized == "float32":
        return torch.float32
    raise ValueError(f"Unsupported transport dtype: {raw}")


def parse_chunk_indices(raw: str, num_chunks: int) -> list[int]:
    normalized = raw.strip()
    if normalized == "-1":
        return list(range(num_chunks))

    chunk_indices = []
    for item in normalized.split(","):
        item = item.strip()
        if not item:
            continue
        chunk_idx = int(item)
        if chunk_idx < 0 or chunk_idx >= num_chunks:
            raise ValueError(f"chunk_idx {chunk_idx} is outside [0, {num_chunks - 1}]")
        chunk_indices.append(chunk_idx)

    if not chunk_indices:
        raise ValueError("No chunk indices were provided.")
    return sorted(dict.fromkeys(chunk_indices))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export forward-only SID chunks for Android Module.execute(). "
            "This avoids TrainingModule.executeForwardBackward() for mobile system tests."
        )
    )
    parser.add_argument("--model_name", type=str, default="tinyllama")
    parser.add_argument("--num_chunks", type=int, default=4)
    parser.add_argument(
        "--chunk_idx",
        type=str,
        default="-1",
        help="Use -1 for all chunks, a single index, or a comma list such as 0,1.",
    )
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--transport_dtype", type=str, default="float16")
    parser.add_argument(
        "--relay_only",
        action="store_true",
        help="Export a simple forward-relay graph without local CE/KD loss. Use this for system validation.",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("./exported_pte"))
    parser.add_argument(
        "--artifact_prefix",
        type=str,
        default="tinyllama",
        help="Output files are named {artifact_prefix}_chunk_{idx}.pte.",
    )
    parser.add_argument(
        "--artifact_suffix",
        type=str,
        default="",
        help="Optional suffix before .pte, for example _inf.",
    )
    parser.add_argument(
        "--enable_xnnpack",
        action="store_true",
        help="Enable XNNPACK lowering. Leave off for the first mobile compatibility run.",
    )
    args = parser.parse_args()

    if args.num_chunks <= 0:
        raise ValueError("num_chunks must be positive.")
    if args.seq_len < 2:
        raise ValueError("seq_len must be at least 2.")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    transport_dtype = parse_transport_dtype(args.transport_dtype)
    chunk_indices = parse_chunk_indices(args.chunk_idx, args.num_chunks)

    for chunk_idx in chunk_indices:
        print("=" * 72)
        export_chunk(
            model_name=args.model_name,
            num_chunks=args.num_chunks,
            target_chunk=chunk_idx,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            artifact_prefix=args.artifact_prefix,
            artifact_suffix=args.artifact_suffix,
            alpha=args.alpha,
            label_smoothing=args.label_smoothing,
            transport_dtype=transport_dtype,
            use_xnnpack=args.enable_xnnpack,
            relay_only=args.relay_only,
        )

    print(
        "Done. Copy the generated .pte files to the coordinator model directory, "
        "then call POST /api/v1/routing/reload and restart the Android workers."
    )


if __name__ == "__main__":
    main()
