import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from torch.nn.attention import sdpa_kernel, SDPBackend

from torch.export import export
from torch.export.experimental import _export_forward_backward
import executorch.exir as exir
from executorch.exir import EdgeCompileConfig
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

import argparse


class ExportableSIDChunk(nn.Module):
    def __init__(self, chunk_idx, layers, final_norm, lm_head, vocab_size, rotary_emb, alpha=0.5, label_smoothing=0.1):
        super().__init__()
        self.chunk_idx = chunk_idx
        self.layers = nn.ModuleList(layers)

        self.final_norm = final_norm
        self.lm_head = lm_head
        self.rotary_emb = rotary_emb

        self.vocab_size = vocab_size
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, hidden_states, attention_mask, position_ids, labels, prev_log_probs):
        position_embeddings = None
        if self.rotary_emb is not None:
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        curr_hidden = hidden_states
        for layer in self.layers:
            layer_out = layer(
                curr_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings
            )
            curr_hidden = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        logits = self.lm_head(self.final_norm(curr_hidden))
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_prev_log_probs = prev_log_probs[..., :-1, :].contiguous()
        valid_mask = (shift_labels != -100).float()
        valid_tokens_count = valid_mask.sum() + 1e-6
        safe_labels = torch.where(shift_labels != -100, shift_labels, torch.zeros_like(shift_labels))

        loss_ce_unmasked = F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            safe_labels.view(-1),
            reduction='none',
            label_smoothing=self.label_smoothing
        ).view(shift_labels.shape)

        loss_ce = (loss_ce_unmasked * valid_mask).sum() / valid_tokens_count
        student_log_p = F.log_softmax(shift_logits, dim=-1)
        teacher_p = torch.exp(shift_prev_log_probs)

        loss_kl_unmasked = (teacher_p * (shift_prev_log_probs - student_log_p)).sum(dim=-1)
        loss_kl = (loss_kl_unmasked * valid_mask).sum() / valid_tokens_count

        if self.chunk_idx == 0:
            total_loss = loss_ce
        else:
            total_loss = self.alpha * loss_ce + (1.0 - self.alpha) * loss_kl

        next_log_probs = F.log_softmax(logits, dim=-1)

        return total_loss, curr_hidden.detach(), next_log_probs.detach()


def export_sid_to_executorch(model_name="microsoft/phi-2", num_chunks=4, target_chunk=0):
    print(f"\n[1/5] 正在加载模型架构 {model_name}...")
    backbone = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)

    layers = backbone.model.layers
    final_norm = backbone.model.final_layernorm if hasattr(backbone.model, "final_layernorm") else backbone.model.norm
    lm_head = backbone.lm_head
    vocab_size = backbone.config.vocab_size
    rotary_emb = getattr(backbone.model, "rotary_emb", None)

    total_layers = len(layers)
    chunk_size = total_layers // num_chunks

    start = target_chunk * chunk_size
    end = (target_chunk + 1) * chunk_size if target_chunk < num_chunks - 1 else total_layers
    chunk_layers = [layers[j] for j in range(start, end)]

    print(f"[2/5] 正在组装 Chunk {target_chunk} (包含第 {start} 到 {end-1} 层)...")
    chunk_module = ExportableSIDChunk(
        chunk_idx=target_chunk,
        layers=chunk_layers,
        final_norm=final_norm,
        lm_head=lm_head,
        vocab_size=vocab_size,
        rotary_emb=rotary_emb
    )

    chunk_module.train()
    for param in chunk_module.parameters():
        param.requires_grad = True

    print("[3/5] 正在构造静态图 Tracing 所需的 Dummy Inputs...")
    batch_size = 1
    seq_len = 128
    hidden_dim = backbone.config.hidden_size

    dummy_hidden = torch.randn((batch_size, seq_len, hidden_dim), dtype=torch.float32)
    dummy_mask = torch.zeros((batch_size, 1, seq_len, seq_len), dtype=torch.float32)
    dummy_pos = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    dummy_labels = torch.ones((batch_size, seq_len), dtype=torch.long)
    dummy_prev_log_probs = torch.randn((batch_size, seq_len, vocab_size), dtype=torch.float32)

    example_args = (dummy_hidden, dummy_mask, dummy_pos, dummy_labels, dummy_prev_log_probs)

    print("[4/5] 正在生成 前向+全参数反向求导 的联合静态图...")
    with sdpa_kernel([SDPBackend.MATH]):
        ep = export(chunk_module, example_args, strict=False)
        joint_ep = _export_forward_backward(ep)

    print("[5/5] 正在降维并编译为 ExecuTorch .pte 固件 (支持 Android XNNPACK)...")
    edge_config = EdgeCompileConfig(_check_ir_validity=False)
    from executorch.exir import to_edge_transform_and_lower

    edge_program = to_edge_transform_and_lower(
        joint_ep,
        compile_config=edge_config,
        partitioner=[XnnpackPartitioner(force_fp32_dynamic_linear=True)]
    )
    executorch_program = edge_program.to_executorch()
    os.makedirs("./exported_pte", exist_ok=True)
    save_path = f"./exported_pte/phi2_sid_full_ft_chunk_{target_chunk}.pte"

    with open(save_path, "wb") as f:
        f.write(executorch_program.buffer)

    print(f"\n全参数微调固件已保存至: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="导出 SID 模型的 ExecuTorch 固件")
    parser.add_argument("--model_name", type=str, default="microsoft/phi-2")
    parser.add_argument("--num_chunks", type=int, default=4, help="总共切分的 Chunk 数量")
    parser.add_argument("--chunk_idx", type=int, default=-1, help="指定导出的 Chunk 索引。如果是 -1，则一次性导出所有 Chunk")

    args = parser.parse_args()

    if args.chunk_idx == -1:
        print(f"[*] 检测到 chunk_idx=-1，准备连续导出全部 {args.num_chunks} 个 Chunk...")
        for i in range(args.num_chunks):
            print(f"\n" + "=" * 50)
            print(f" 开始导出 Chunk {i} / {args.num_chunks - 1} ".center(50, "="))
            print("=" * 50)
            export_sid_to_executorch(
                model_name=args.model_name,
                num_chunks=args.num_chunks,
                target_chunk=i
            )
        print("\n[*] 恭喜！所有 Chunk 的固件均已成功导出！")
    else:
        export_sid_to_executorch(
            model_name=args.model_name,
            num_chunks=args.num_chunks,
            target_chunk=args.chunk_idx
        )
