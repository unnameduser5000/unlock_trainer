from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from sg_exe_trainer.runtime.bpfree.model_runtime import (
    build_stage_chunk,
    normalize_belief_transport_mode,
)
from sg_exe_trainer.tasks.label_experiment import resolve_dtype, resolve_model_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="tinyllama")
    parser.add_argument("--num_chunks", type=int, default=3)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--belief_transport_mode", default="terminal")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    resolved_model = resolve_model_name(args.model_name)
    dtype = resolve_dtype(args.dtype)

    print(f"resolved_model={resolved_model}")
    print(f"dtype={dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        resolved_model,
        torch_dtype=dtype,
    )

    for stage_id in range(args.num_chunks):
        print("\n" + "=" * 100)
        print(f"stage_id={stage_id}")

        chunk = build_stage_chunk(
            model=model,
            stage_id=stage_id,
            num_chunks=args.num_chunks,
            belief_transport_mode=normalize_belief_transport_mode(
                args.belief_transport_mode
            ),
            alpha=args.alpha,
            label_smoothing=args.label_smoothing,
        )
        chunk.to(torch.device(args.device))

        print(f"class={chunk.__class__.__module__}.{chunk.__class__.__name__}")

        interesting_attrs = [
            "stage_id",
            "layer_start",
            "layer_end",
            "layers",
            "final_norm",
            "lm_head",
            "rotary_emb",
            "belief_transport_mode",
            "alpha",
            "label_smoothing",
            "vocab_size",
            "last_choice_metrics",
            "last_loss_components",
        ]

        for name in interesting_attrs:
            has = hasattr(chunk, name)
            print(f"hasattr({name})={has}")
            if has:
                value = getattr(chunk, name)
                if name == "layers":
                    try:
                        print(f"  layers_len={len(value)}")
                    except Exception as exc:
                        print(f"  layers_len_error={exc!r}")
                elif name in {"final_norm", "lm_head", "rotary_emb"}:
                    print(f"  type={type(value)}")
                    print(f"  is_none={value is None}")
                else:
                    print(f"  value={value!r}")

        print("\nforward signature:")
        print(inspect.signature(chunk.forward))

        print("\nforward source:")
        try:
            print(inspect.getsource(chunk.forward))
        except Exception as exc:
            print(f"FAILED_TO_GET_SOURCE: {exc!r}")

        methods = [
            name
            for name in dir(chunk)
            if any(token in name.lower() for token in ["loss", "head", "body", "forward", "hidden"])
        ]
        print("\ninteresting methods/attrs:")
        for name in methods:
            if name.startswith("__"):
                continue
            obj = getattr(chunk, name)
            print(f"  {name}: {type(obj)}")


if __name__ == "__main__":
    main()
