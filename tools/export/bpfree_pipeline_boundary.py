from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


OP_NAMESPACE = "sid_bpfree_pipeline"
OP_NAME = "pipeline_boundary"
OUT_OPERATOR = f"{OP_NAMESPACE}::{OP_NAME}.out"

_LIBRARIES: list[torch.library.Library] = []
_REGISTERED = False


def register_pipeline_boundary_ops() -> None:
    """Register the export-time definition of the mobile pause marker."""
    global _REGISTERED
    if _REGISTERED:
        return

    definition = torch.library.Library(OP_NAMESPACE, "DEF")
    definition.define(
        "pipeline_boundary(Tensor loss, Tensor hidden, Tensor belief) -> Tensor"
    )
    definition.define(
        "pipeline_boundary.out(Tensor loss, Tensor hidden, Tensor belief, "
        "*, Tensor(a!) out) -> Tensor(a!)"
    )

    cpu_impl = torch.library.Library(OP_NAMESPACE, "IMPL", "CPU")
    meta_impl = torch.library.Library(OP_NAMESPACE, "IMPL", "Meta")

    def boundary(
        loss: torch.Tensor,
        hidden: torch.Tensor,
        belief: torch.Tensor,
    ) -> torch.Tensor:
        del hidden, belief
        return loss.clone()

    def boundary_meta(
        loss: torch.Tensor,
        hidden: torch.Tensor,
        belief: torch.Tensor,
    ) -> torch.Tensor:
        del hidden, belief
        return torch.empty_like(loss)

    def boundary_out(
        loss: torch.Tensor,
        hidden: torch.Tensor,
        belief: torch.Tensor,
        *,
        out: torch.Tensor,
    ) -> torch.Tensor:
        del hidden, belief
        out.copy_(loss)
        return out

    cpu_impl.impl(OP_NAME, boundary)
    cpu_impl.impl(f"{OP_NAME}.out", boundary_out)
    meta_impl.impl(OP_NAME, boundary_meta)
    meta_impl.impl(f"{OP_NAME}.out", boundary_out)

    def setup_context(ctx: Any, inputs: Any, output: Any) -> None:
        del ctx, inputs, output

    def backward(ctx: Any, grad_loss: torch.Tensor):
        del ctx
        return grad_loss, None, None

    torch.library.register_autograd(
        f"{OP_NAMESPACE}::{OP_NAME}",
        backward,
        setup_context=setup_context,
    )

    _LIBRARIES.extend((definition, cpu_impl, meta_impl))
    _REGISTERED = True


class PipelineBoundaryChunk(nn.Module):
    """Marks the point where hidden outputs are safe to send downstream."""

    def __init__(self, chunk: nn.Module) -> None:
        super().__init__()
        self.chunk = chunk

    def forward(self, *args: torch.Tensor):
        loss, hidden, belief = self.chunk(*args)
        marked_loss = torch.ops.sid_bpfree_pipeline.pipeline_boundary.default(
            loss,
            hidden,
            belief,
        )
        return marked_loss, hidden, belief


def wrap_with_pipeline_boundary(chunk: nn.Module) -> PipelineBoundaryChunk:
    register_pipeline_boundary_ops()
    return PipelineBoundaryChunk(chunk)


def audit_joint_graph(joint_program: Any) -> dict[str, Any]:
    """Verify that the marker precedes every trainable gradient output."""
    nodes = list(joint_program.graph_module.graph.nodes)
    indices = {node.name: index for index, node in enumerate(nodes)}
    markers = [
        node
        for node in nodes
        if "sid_bpfree_pipeline.pipeline_boundary" in str(node.target)
    ]
    if len(markers) != 1:
        raise RuntimeError(f"Expected exactly one BP-free boundary, found {len(markers)}.")

    marker = markers[0]
    marker_index = indices[marker.name]
    backward_users = [user for user in marker.users if user.op != "output"]
    if not backward_users:
        raise RuntimeError("BP-free boundary loss has no backward seed user.")

    first_backward = min(backward_users, key=lambda node: indices[node.name])
    first_backward_index = indices[first_backward.name]
    if marker_index >= first_backward_index:
        raise RuntimeError("BP-free boundary was not preserved before backward.")

    gradient_indices = [
        indices[spec.arg.name]
        for spec in joint_program.graph_signature.output_specs
        if spec.kind.name == "GRADIENT_TO_PARAMETER"
    ]
    if not gradient_indices:
        raise RuntimeError("Joint graph has no trainable parameter gradients.")
    if min(gradient_indices) <= marker_index:
        raise RuntimeError("A trainable gradient output appears before the boundary.")

    return {
        "joint_node_count": len(nodes),
        "boundary_node": marker.name,
        "boundary_index": marker_index,
        "first_backward_seed_node": first_backward.name,
        "first_backward_seed_index": first_backward_index,
        "gradient_output_count": len(gradient_indices),
        "first_gradient_output_index": min(gradient_indices),
    }


def verify_runtime_operator(executorch_program: Any) -> None:
    operators = executorch_program.executorch_program.execution_plan[0].operators
    names = {
        f"{operator.name}.{operator.overload}" if operator.overload else operator.name
        for operator in operators
    }
    if OUT_OPERATOR not in names:
        raise RuntimeError(
            f"Boundary operator {OUT_OPERATOR} is missing from the PTE operator table."
        )
