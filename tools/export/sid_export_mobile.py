import argparse
from collections import Counter
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.export import export
from torch.export.experimental import _export_forward_backward
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


def normalize_belief_transport_mode(raw_mode: str) -> str:
    normalized = raw_mode.strip().lower()
    if normalized in {"", "full", "dense"}:
        return "full"
    if normalized in {"terminal", "terminal_only", "final", "final_only"}:
        return "terminal"
    if normalized in {"none", "off", "disabled", "false"}:
        return "none"
    raise ValueError(f"Unsupported belief transport mode: {raw_mode}. Use full, terminal, or none.")


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        init_std: float,
    ) -> None:
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
    prefix: str = "",
) -> int:
    injected = 0
    for child_name, child in list(module.named_children()):
        qualified_name = f"{prefix}.{child_name}" if prefix else child_name
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
                prefix=qualified_name,
            )
    return injected


def configure_trainable_parameters(module: nn.Module, use_lora: bool) -> tuple[int, int]:
    if use_lora:
        for param in module.parameters():
            param.requires_grad = False
        for submodule in module.modules():
            if isinstance(submodule, LoRALinear):
                submodule.lora_a.requires_grad = True
                submodule.lora_b.requires_grad = True
    else:
        for param in module.parameters():
            param.requires_grad = True

    trainable = sum(param.numel() for param in module.parameters() if param.requires_grad)
    frozen = sum(param.numel() for param in module.parameters() if not param.requires_grad)
    return trainable, frozen


class ExportableSIDChunk(nn.Module):
    def __init__(
        self,
        chunk_idx: int,
        layers: Iterable[nn.Module],
        final_norm: nn.Module,
        lm_head: nn.Module,
        vocab_size: int,
        rotary_emb: nn.Module | None,
        is_terminal_chunk: bool,
        belief_transport_mode: str,
        alpha: float = 0.5,
        label_smoothing: float = 0.1,
        transport_dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        self.chunk_idx = chunk_idx
        self.layers = nn.ModuleList(list(layers))
        self.final_norm = final_norm
        self.lm_head = lm_head
        self.rotary_emb = rotary_emb
        self.vocab_size = vocab_size
        self.is_terminal_chunk = is_terminal_chunk
        self.belief_transport_mode = normalize_belief_transport_mode(belief_transport_mode)
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.transport_dtype = transport_dtype

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

        if not self.uses_belief_loss:
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
        if self.returns_full_log_probs:
            next_log_probs = transport_cast(
                F.log_softmax(logits.float(), dim=-1).detach(),
                self.transport_dtype,
            )
        else:
            next_log_probs = transport_cast(curr_hidden.detach().flatten()[:1] * 0.0, self.transport_dtype)
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
    belief_transport_mode: str,
) -> tuple[ExportableSIDChunk, int, int]:
    layers, final_norm, lm_head, vocab_size, rotary_emb = get_model_parts(backbone)
    total_layers = len(layers)
    chunk_size = total_layers // num_chunks
    start = target_chunk * chunk_size
    end = (target_chunk + 1) * chunk_size if target_chunk < num_chunks - 1 else total_layers
    chunk_layers = [layers[j] for j in range(start, end)]
    chunk_module = ExportableSIDChunk(
        chunk_idx=target_chunk,
        layers=chunk_layers,
        final_norm=final_norm,
        lm_head=lm_head,
        vocab_size=vocab_size,
        rotary_emb=rotary_emb,
        is_terminal_chunk=target_chunk == num_chunks - 1,
        belief_transport_mode=belief_transport_mode,
        alpha=alpha,
        label_smoothing=label_smoothing,
        transport_dtype=transport_dtype,
    )
    return chunk_module, start, end


def build_example_args(
    *,
    hidden_dim: int,
    vocab_size: int,
    seq_len: int,
    batch_size: int,
    target_chunk: int,
    consumes_prev_log_probs: bool,
) -> tuple[torch.Tensor, ...]:
    dummy_hidden = torch.randn((batch_size, seq_len, hidden_dim), dtype=torch.float32)
    dummy_mask = torch.zeros((batch_size, 1, seq_len, seq_len), dtype=torch.float32)
    dummy_pos = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    dummy_labels = torch.ones((batch_size, seq_len), dtype=torch.long)

    if not consumes_prev_log_probs:
        return (dummy_hidden, dummy_mask, dummy_pos, dummy_labels)

    dummy_prev_log_probs = torch.randn((batch_size, seq_len, vocab_size), dtype=torch.float32)
    return (dummy_hidden, dummy_mask, dummy_pos, dummy_labels, dummy_prev_log_probs)


def rewrite_empty_permuted(joint_graph) -> int:
    """Rewrite aten.empty_permuted, which ExecuTorch edge/mobile does not implement."""
    rewritten = 0
    graph = joint_graph.graph
    for node in list(graph.nodes):
        if node.op != "call_function":
            continue
        if node.target not in (
            torch.ops.aten.empty_permuted.out,
            torch.ops.aten.empty_permuted.default,
        ):
            continue

        with graph.inserting_before(node):
            empty_node = graph.create_node(
                "call_function",
                torch.ops.aten.empty.memory_format,
                (node.args[0],),
                node.kwargs,
            )
            permute_target = (
                torch.ops.aten.permute
                if node.target == torch.ops.aten.empty_permuted.out
                else torch.ops.aten.permute.default
            )
            permute_node = graph.create_node(
                "call_function",
                permute_target,
                (empty_node, node.args[1]),
            )
            permute_node.meta.update(node.meta)

        node.replace_all_uses_with(permute_node)
        graph.erase_node(node)
        rewritten += 1

    if rewritten:
        graph.lint()
        joint_graph.graph_module.recompile()

    return rewritten


def dump_joint_graph_diagnostics(
    joint_graph,
    *,
    output_path: Path | None = None,
) -> None:
    graph_signature = joint_graph.graph_signature
    output_specs = list(getattr(graph_signature, "output_specs", []))

    lines: list[str] = []
    lines.append("== Joint Graph Signature ==")
    lines.append(str(graph_signature))
    lines.append("")
    lines.append("== Output Specs ==")
    gradient_targets: list[str] = []
    user_input_gradient_targets: list[str] = []
    for index, spec in enumerate(output_specs):
        kind = getattr(spec.kind, "name", str(spec.kind))
        target = getattr(spec, "target", None)
        arg = getattr(spec, "arg", None)
        lines.append(f"{index:04d}: kind={kind} target={target} arg={arg}")
        if kind == "GRADIENT_TO_PARAMETER":
            gradient_targets.append(str(target))
        elif "GRADIENT" in kind:
            user_input_gradient_targets.append(str(target))

    lines.append("")
    lines.append("== Gradient Summary ==")
    lines.append(f"parameter_gradient_count={len(gradient_targets)}")
    for target in gradient_targets:
        lines.append(f"  parameter_grad={target}")
    lines.append(f"non_parameter_gradient_count={len(user_input_gradient_targets)}")
    for target in user_input_gradient_targets:
        lines.append(f"  non_parameter_grad={target}")

    op_counter = Counter(
        str(node.target)
        for node in joint_graph.graph.nodes
        if node.op == "call_function"
    )
    lines.append("")
    lines.append("== Operator Counts ==")
    for target, count in sorted(op_counter.items()):
        lines.append(f"{count:04d} {target}")

    interesting_terms = (
        "log_softmax",
        "_softmax",
        "softmax",
        "cross_entropy",
        "nll_loss",
        "kl_div",
        "mm",
        "matmul",
        "linear",
    )
    lines.append("")
    lines.append("== Interesting Operator Counts ==")
    for target, count in sorted(op_counter.items()):
        if any(term in target for term in interesting_terms):
            lines.append(f"{count:04d} {target}")

    text = "\n".join(lines)
    print(text)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")


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
    belief_transport_mode: str,
    use_xnnpack: bool,
    lora_rank: int,
    lora_alpha: float,
    lora_targets: set[str],
    lora_init_std: float,
    dump_joint_graph: bool,
) -> Path:
    resolved_model_name = resolve_model_name(model_name)
    normalized_belief_transport_mode = normalize_belief_transport_mode(belief_transport_mode)
    print(f"[1/5] Loading model: {resolved_model_name}")
    backbone = AutoModelForCausalLM.from_pretrained(
        resolved_model_name,
        torch_dtype=torch.float32,
    )

    print(f"[2/5] Building chunk {target_chunk}/{num_chunks - 1}")
    chunk_module, start, end = build_chunk_module(
        backbone=backbone,
        num_chunks=num_chunks,
        target_chunk=target_chunk,
        alpha=alpha,
        label_smoothing=label_smoothing,
        transport_dtype=transport_dtype,
        belief_transport_mode=normalized_belief_transport_mode,
    )
    print(f"      layers=[{start}, {end - 1}]")
    print(
        "      belief_transport_mode="
        f"{normalized_belief_transport_mode} consumes_prev_log_probs={chunk_module.consumes_prev_log_probs} "
        f"returns_full_log_probs={chunk_module.returns_full_log_probs}"
    )

    chunk_module.train()
    injected_lora = 0
    if lora_rank > 0:
        injected_lora = inject_lora_adapters(
            module=chunk_module,
            target_names=lora_targets,
            rank=lora_rank,
            alpha=lora_alpha,
            init_std=lora_init_std,
        )
        if injected_lora == 0:
            raise RuntimeError(
                "LoRA was enabled but no target Linear modules were replaced. "
                f"Requested targets: {sorted(lora_targets)}"
            )
    trainable_params, frozen_params = configure_trainable_parameters(
        module=chunk_module,
        use_lora=lora_rank > 0,
    )
    print(
        "      trainable_params="
        f"{trainable_params} frozen_params={frozen_params} lora_modules={injected_lora}"
    )

    hidden_dim = backbone.config.hidden_size
    vocab_size = backbone.config.vocab_size
    example_args = build_example_args(
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        seq_len=seq_len,
        batch_size=batch_size,
        target_chunk=target_chunk,
        consumes_prev_log_probs=chunk_module.consumes_prev_log_probs,
    )
    print(
        f"[3/5] Tracing joint graph with seq_len={seq_len}, "
        f"batch_size={batch_size}, inputs={len(example_args)}"
    )

    with sdpa_kernel([SDPBackend.MATH]):
        ep = export(chunk_module, example_args, strict=False)
        joint_ep = _export_forward_backward(ep)
        if dump_joint_graph:
            diagnostic_path = (
                output_dir
                / f"{artifact_prefix}_chunk_{target_chunk}{artifact_suffix}.joint_graph.txt"
            )
            dump_joint_graph_diagnostics(joint_ep, output_path=diagnostic_path)
        rewritten_empty_permuted = rewrite_empty_permuted(joint_ep)
        if rewritten_empty_permuted:
            print(
                "      rewrote "
                f"{rewritten_empty_permuted} aten.empty_permuted node(s) "
                "to aten.empty + aten.permute"
            )

    print("[4/5] Lowering to ExecuTorch edge program")
    edge_config = EdgeCompileConfig(_check_ir_validity=False)
    if use_xnnpack:
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
        from executorch.exir import to_edge_transform_and_lower

        edge_program = to_edge_transform_and_lower(
            joint_ep,
            compile_config=edge_config,
            partitioner=[XnnpackPartitioner(force_fp32_dynamic_linear=True)],
        )
    else:
        edge_program = exir.to_edge(joint_ep, compile_config=edge_config)

    executorch_program = edge_program.to_executorch()

    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"{artifact_prefix}_chunk_{target_chunk}{artifact_suffix}.pte"

    if b"aten::empty_permuted" in executorch_program.buffer:
        raise RuntimeError(
            "Exported PTE still contains aten::empty_permuted. "
            "Do not deploy this artifact to Android; the current ExecuTorch "
            "mobile runtime does not implement that op."
        )

    print("[5/5] Writing .pte")
    with save_path.open("wb") as f:
        f.write(executorch_program.buffer)

    print(f"Saved: {save_path}")
    print(
        "Notes: if you replace a coordinator-served artifact in-place, "
        "call POST /api/v1/routing/reload and restart the workers."
    )
    return save_path


def parse_transport_dtype(raw: str) -> torch.dtype:
    normalized = raw.strip().lower()
    if normalized == "float16":
        return torch.float16
    if normalized == "float32":
        return torch.float32
    raise ValueError(f"Unsupported transport dtype: {raw}")


def parse_csv_set(raw: str) -> set[str]:
    values = {item.strip() for item in raw.split(",") if item.strip()}
    if not values:
        raise ValueError("Expected at least one comma-separated value.")
    return values


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
            "Export joint forward/backward SID chunks for Android TrainingModule.executeForwardBackward(). "
            "This is the chunk-local backward path; it does not add cross-stage gradient RPC."
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
        "--belief_transport_mode",
        type=str,
        default="full",
        choices=("full", "terminal", "none"),
        help=(
            "Controls full-vocab log-prob IO. full keeps the previous belief/KL path; "
            "terminal omits intermediate full-vocab outputs but returns final log-probs for metrics; "
            "none omits all full-vocab log-prob outputs."
        ),
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
        help="Optional suffix before .pte.",
    )
    parser.add_argument(
        "--enable_xnnpack",
        action="store_true",
        help="Enable XNNPACK lowering. Leave off for the first joint-graph training export.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=0,
        help="Enable LoRA adapter training with this rank. 0 keeps the current full-parameter chunk export.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=16.0,
        help="LoRA scaling alpha. Effective scale is alpha/rank.",
    )
    parser.add_argument(
        "--lora_targets",
        type=str,
        default="q_proj,v_proj",
        help="Comma-separated Linear child module names to wrap, for example q_proj,v_proj,o_proj.",
    )
    parser.add_argument(
        "--lora_init_std",
        type=float,
        default=0.01,
        help="Stddev for LoRA A initialization. LoRA B starts at zero.",
    )
    parser.add_argument(
        "--dump_joint_graph",
        action="store_true",
        help=(
            "Print and save joint forward/backward graph diagnostics before lowering. "
            "Use this to verify gradient output targets and BP-free boundary behavior."
        ),
    )
    args = parser.parse_args()

    if args.num_chunks <= 0:
        raise ValueError("num_chunks must be positive.")
    if args.seq_len < 2:
        raise ValueError("seq_len must be at least 2.")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if args.lora_rank < 0:
        raise ValueError("lora_rank must be non-negative.")
    if args.lora_rank > 0 and args.lora_alpha <= 0:
        raise ValueError("lora_alpha must be positive when LoRA is enabled.")
    if args.lora_init_std <= 0:
        raise ValueError("lora_init_std must be positive.")

    transport_dtype = parse_transport_dtype(args.transport_dtype)
    belief_transport_mode = normalize_belief_transport_mode(args.belief_transport_mode)
    chunk_indices = parse_chunk_indices(args.chunk_idx, args.num_chunks)
    lora_targets = parse_csv_set(args.lora_targets)

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
            belief_transport_mode=belief_transport_mode,
            use_xnnpack=args.enable_xnnpack,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_targets=lora_targets,
            lora_init_std=args.lora_init_std,
            dump_joint_graph=args.dump_joint_graph,
        )


if __name__ == "__main__":
    main()
