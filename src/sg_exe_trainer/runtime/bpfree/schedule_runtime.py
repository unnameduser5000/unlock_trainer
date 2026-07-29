from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import os
from typing import Any, Optional

import torch

from sg_exe_trainer.metrics.activation_memory import SavedTensorTracker
from sg_exe_trainer.runtime.bpfree.schedule import BPFreeMicrobatch, BPFreeUpdateWindow
from sg_exe_trainer.runtime.bpfree.gpu_stage import (
    BPFreePipelineStageV0,
    BackwardOutput,
    CommonInputs,
    ForwardInput,
    ForwardOutput,
    OptimizerStepOutput,
    SendOutput,
)


@dataclass
class BPFreeMicrobatchRun:
    mb: BPFreeMicrobatch
    fwd_input: ForwardInput
    common: CommonInputs
    fwd_output: ForwardOutput
    send_output: SendOutput
    backward_output: BackwardOutput
    opt_output: OptimizerStepOutput
    activation_stats: dict[str, Any]
    loss_value: float
    optimizer_step_index_after: int


class ScheduleBPFree1F1BLikeV0:
    """
    BP-free schedule runtime v0.

    This mirrors the *outer runtime shape* of PyTorch PipelineScheduleSingle:

      step_window(local_update_window)
        -> iterate microbatches
        -> execute scheduled stage actions
        -> local optimizer step at window boundary

    It intentionally does NOT implement exact-BP backward P2P:
      no BWD_RECV
      no BWD_SEND

    Current v0 microbatch order:
      FWD_RECV / LOAD
      LOAD_COMMON_INPUTS
      FWD_COMPUTE_INCLUDES_LOCAL_HEAD
      FWD_SEND_POST
      LOCAL_BACKWARD
      LOCAL_OPTIMIZER_STEP only on last microbatch of window

    Later v1/v2 optimization will split:
      FWD_COMPUTE_INCLUDES_LOCAL_HEAD
        -> BODY_FORWARD
        -> FWD_SEND_POST
        -> LOCAL_HEAD_LOSS
        -> LOCAL_BACKWARD
    """

    def __init__(
        self,
        *,
        stage: BPFreePipelineStageV0,
        train_this_rank: bool,
        track_activation_memory: bool,
        activation_tracker: SavedTensorTracker,
        vocab_size: int,
        learning_rate_override: Optional[float],
        optimizer_steps_start: int = 0,
        perf_minimal_metrics: bool = False,
        window_input_staging: bool = False,
    ) -> None:
        self.stage = stage
        self.train_this_rank = train_this_rank
        self.track_activation_memory = track_activation_memory
        self.activation_tracker = activation_tracker
        self.vocab_size = vocab_size
        self.learning_rate_override = learning_rate_override
        self.optimizer_steps = optimizer_steps_start
        self.perf_minimal_metrics = perf_minimal_metrics
        self.window_input_staging = bool(window_input_staging)

    def step_window(self, window: BPFreeUpdateWindow) -> list[BPFreeMicrobatchRun]:
        self.stage.begin_window(
            window=window,
            learning_rate_override=self.learning_rate_override,
        )

        runs: list[BPFreeMicrobatchRun] = []

        for mb in window.microbatches:
            fwd_input = self.stage.load_or_recv_forward_input(mb)
            common = self.stage.load_common_inputs(mb)

            if self.track_activation_memory:
                self.activation_tracker.configure(
                    hidden_size=int(fwd_input.hidden.shape[-1]),
                    vocab_size=self.vocab_size,
                )
                self.activation_tracker.reset()

            hook_context = (
                torch.autograd.graph.saved_tensors_hooks(
                    self.activation_tracker.pack,
                    self.activation_tracker.unpack,
                )
                if self.train_this_rank and self.track_activation_memory
                else nullcontext()
            )

            fwd_output = self.stage.forward_compute_includes_local_head(
                mb=mb,
                fwd_input=fwd_input,
                common=common,
                hook_context=hook_context,
            )

            send_output = self.stage.post_forward_send(
                mb=mb,
                fwd_output=fwd_output,
            )

            backward_output = self.stage.local_backward(
                mb=mb,
                window=window,
                loss=fwd_output.loss,
            )

            opt_output = self.stage.maybe_optimizer_step(
                mb=mb,
                window=window,
            )

            if opt_output.applied:
                self.optimizer_steps += 1

            activation_stats = (
                self.activation_tracker.snapshot()
                if self.track_activation_memory
                else {}
            )

            loss_value = float(fwd_output.loss.detach().cpu().item())

            runs.append(
                BPFreeMicrobatchRun(
                    mb=mb,
                    fwd_input=fwd_input,
                    common=common,
                    fwd_output=fwd_output,
                    send_output=send_output,
                    backward_output=backward_output,
                    opt_output=opt_output,
                    activation_stats=activation_stats,
                    loss_value=loss_value,
                    optimizer_step_index_after=self.optimizer_steps,
                )
            )

        return runs

    def drain(self) -> float:
        return self.stage.drain_pending_sends()


class ScheduleBPFreeBodySendHeadV1(ScheduleBPFree1F1BLikeV0):
    """
    BP-free schedule runtime with true body-send-head split.

    Microbatch order:
      FWD_RECV / LOAD
      LOAD_COMMON_INPUTS
      BODY_FORWARD
      FWD_SEND_POST
      LOCAL_HEAD_LOSS
      LOCAL_BACKWARD
      LOCAL_OPTIMIZER_STEP only on last microbatch of window

    No backward P2P:
      no BWD_RECV
      no BWD_SEND
    """

    def step_window(self, window: BPFreeUpdateWindow) -> list[BPFreeMicrobatchRun]:
        raw_group_size = os.environ.get(
            "BPFREE_DEFERRED_BACKWARD_GROUP_SIZE",
            "1",
        )
        try:
            requested_group_size = int(raw_group_size)
        except ValueError as exc:
            raise ValueError(
                "BPFREE_DEFERRED_BACKWARD_GROUP_SIZE must be an integer, "
                f"got {raw_group_size!r}"
            ) from exc

        if requested_group_size <= 0:
            raise ValueError(
                "BPFREE_DEFERRED_BACKWARD_GROUP_SIZE must be positive, "
                f"got {requested_group_size}"
            )

        # Evaluation and non-training ranks preserve the original immediate path.
        group_size = (
            min(requested_group_size, window.num_microbatches)
            if self.train_this_rank
            else 1
        )

        # The current SavedTensorTracker is reset per microbatch and does not
        # represent several simultaneously retained autograd graphs correctly.
        if group_size > 1 and self.track_activation_memory:
            raise RuntimeError(
                "deferred backward currently requires "
                "--no-track_activation_memory"
            )

        first_mb = window.microbatches[0]
        with self.stage._span(first_mb, "BEGIN_WINDOW"):
            self.stage.begin_window(
                window=window,
                learning_rate_override=self.learning_rate_override,
            )

        window_input_staging = bool(
            getattr(self, "window_input_staging", False)
        )
        if window_input_staging:
            self.stage.prepare_window_common_inputs(window)

        runs: list[BPFreeMicrobatchRun] = []

        try:
            microbatches = window.microbatches

            for group_start in range(
                0,
                len(microbatches),
                group_size,
            ):
                group = microbatches[
                    group_start : group_start + group_size
                ]
                deferred = []

                # Forward/head wave.
                for mb in group:
                    with self.stage._span(
                        mb,
                        "MAINTAIN_RECV_INFLIGHT",
                    ):
                        if hasattr(
                            self.stage,
                            "maintain_forward_recv_inflight",
                        ):
                            self.stage.maintain_forward_recv_inflight(mb)

                    fwd_input = self.stage.load_or_recv_forward_input(mb)

                    if window_input_staging:
                        common = (
                            self.stage
                            .window_common_inputs_for_microbatch(mb)
                        )
                    else:
                        common = self.stage.load_common_inputs(mb)

                    if self.track_activation_memory:
                        self.activation_tracker.configure(
                            hidden_size=int(
                                fwd_input.hidden.shape[-1]
                            ),
                            vocab_size=self.vocab_size,
                        )
                        self.activation_tracker.reset()

                    body_hook_context = (
                        torch.autograd.graph.saved_tensors_hooks(
                            self.activation_tracker.pack,
                            self.activation_tracker.unpack,
                        )
                        if (
                            self.train_this_rank
                            and self.track_activation_memory
                        )
                        else nullcontext()
                    )

                    body_output = self.stage.body_forward_one_chunk(
                        mb=mb,
                        fwd_input=fwd_input,
                        common=common,
                        hook_context=body_hook_context,
                    )

                    send_output = (
                        self.stage.post_body_forward_send(
                            mb=mb,
                            body_output=body_output,
                        )
                    )

                    head_hook_context = (
                        torch.autograd.graph.saved_tensors_hooks(
                            self.activation_tracker.pack,
                            self.activation_tracker.unpack,
                        )
                        if (
                            self.train_this_rank
                            and self.track_activation_memory
                        )
                        else nullcontext()
                    )

                    head_output = (
                        self.stage.local_head_loss_one_chunk(
                            mb=mb,
                            body_output=body_output,
                            fwd_input=fwd_input,
                            common=common,
                            hook_context=head_hook_context,
                        )
                    )

                    deferred.append(
                        (
                            mb,
                            fwd_input,
                            common,
                            body_output,
                            send_output,
                            head_output,
                        )
                    )

                # Backward wave, preserving original microbatch order.
                for (
                    mb,
                    fwd_input,
                    common,
                    body_output,
                    send_output,
                    head_output,
                ) in deferred:
                    backward_output = self.stage.local_backward(
                        mb=mb,
                        window=window,
                        loss=head_output.loss,
                    )

                    opt_output = self.stage.maybe_optimizer_step(
                        mb=mb,
                        window=window,
                    )

                    if opt_output.applied:
                        self.optimizer_steps += 1

                    if self.perf_minimal_metrics:
                        activation_stats = {}
                        loss_value = float("nan")
                    else:
                        activation_stats = (
                            self.activation_tracker.snapshot()
                            if self.track_activation_memory
                            else {}
                        )
                        with self.stage._span(
                            mb,
                            "LOSS_ITEM_CPU",
                        ):
                            loss_value = float(
                                head_output.loss
                                .detach()
                                .cpu()
                                .item()
                            )
                            self.stage._sync_cuda(mb)

                    compat_fwd_output = ForwardOutput(
                        loss=head_output.loss,
                        next_hidden=(
                            body_output.next_hidden.detach()
                        ),
                        next_log_probs=(
                            head_output.next_log_probs
                        ),
                        forward_ms=(
                            body_output.body_forward_ms
                            + head_output.local_head_loss_ms
                        ),
                    )

                    runs.append(
                        BPFreeMicrobatchRun(
                            mb=mb,
                            fwd_input=fwd_input,
                            common=common,
                            fwd_output=compat_fwd_output,
                            send_output=send_output,
                            backward_output=backward_output,
                            opt_output=opt_output,
                            activation_stats=activation_stats,
                            loss_value=loss_value,
                            optimizer_step_index_after=(
                                self.optimizer_steps
                            ),
                        )
                    )
        finally:
            if window_input_staging:
                self.stage.clear_window_common_inputs(window)

        return runs
