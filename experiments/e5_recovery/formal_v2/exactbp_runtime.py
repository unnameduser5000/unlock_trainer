from __future__ import annotations

import gc
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import torch

from sg_exe_trainer.runtime.bpfree.schedule import BPFreeUpdateWindow
from sg_exe_trainer.runtime.exactbp import distributed_runtime as f1b

from sg_exe_trainer.runtime.recovery.checkpoint_store import (
    StageCheckpointMetadata,
    StageCheckpointStore,
)
from sg_exe_trainer.runtime.recovery.runtime_adapter import request_ids_for_window
from sg_exe_trainer.runtime.recovery.state_contract import StageCommit, StageCommitLedger


@dataclass(frozen=True)
class ExactBPWindowResult:
    phase: str
    window_id: int
    records: int
    microbatches: int
    h2d_ms: float
    schedule_ms: float
    optimizer_ms: float
    durability_ms: float
    elapsed_ms: float
    optimizer_step: int
    loss_count: int
    avg_loss: Optional[float]
    checkpoint: Optional[dict[str, Any]]


class ExactBPWindowRuntime:
    """Small E5 adapter around PyTorch's real Schedule1F1B runtime."""

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        device: torch.device,
        manifest_dir: Path,
        module: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        pipeline_stage: Any,
        train_schedule: Any,
        stage0_input_embedding: Optional[torch.nn.Module],
        dtype: torch.dtype,
        learning_rate: float,
        grad_clip: float,
        microbatches_per_window: int,
        trainable_mode: str,
    ) -> None:
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.manifest_dir = manifest_dir
        self.module = module
        self.optimizer = optimizer
        self.pipeline_stage = pipeline_stage
        self.train_schedule = train_schedule
        self.stage0_input_embedding = stage0_input_embedding
        self.dtype = dtype
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.microbatches_per_window = microbatches_per_window
        self.trainable_mode = trainable_mode
        self.optimizer_steps = 0

    @classmethod
    def build(
        cls,
        *,
        rank: int,
        world_size: int,
        device: torch.device,
        device_name: str,
        manifest_dir: Path,
        cfg: dict[str, Any],
    ) -> "ExactBPWindowRuntime":
        dtype = f1b.resolve_dtype(str(cfg["dtype"]))
        model = f1b.AutoModelForCausalLM.from_pretrained(
            str(cfg["resolved_model"]),
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

        recorder = f1b.StageEventRecorder(
            stage_id=rank,
            rank=rank,
            device_name=device_name,
            enabled=False,
        )
        trainable_setup = f1b.configure_model_trainable(
            module=model,
            mode=str(cfg["trainable_mode"]),
            lora_targets=str(cfg["lora_targets"]),
            lora_rank=int(cfg["lora_rank"]),
            lora_alpha=float(cfg["lora_alpha"]),
            lora_init_std=float(cfg["lora_init_std"]),
            lora_init_seed=int(cfg["lora_init_seed"]),
        )
        stage0_input_embedding = model.get_input_embeddings() if rank == 0 else None
        module = f1b.build_stage_module(
            model=model,
            stage_id=rank,
            num_chunks=world_size,
            recorder=recorder,
        )
        module.to(device)
        if stage0_input_embedding is not None:
            stage0_input_embedding.to(device)
            stage0_input_embedding.eval()
            for parameter in stage0_input_embedding.parameters():
                parameter.requires_grad = False

        params = [parameter for parameter in module.parameters() if parameter.requires_grad]
        optimizer = f1b.build_optimizer(
            params=params,
            optimizer_name=str(cfg["optimizer"]),
            learning_rate=float(cfg["learning_rate"]),
            sgd_momentum=0.0,
            sgd_dampening=0.0,
            sgd_weight_decay=0.0,
            sgd_nesterov=False,
        )
        pipeline_stage = f1b.PipelineStage(module, rank, world_size, device)
        loss_fn = lambda logits, labels: f1b.causal_lm_loss(  # noqa: E731
            logits,
            labels,
            label_smoothing=float(cfg["label_smoothing"]),
        )
        train_schedule = f1b.Schedule1F1B(
            pipeline_stage,
            n_microbatches=int(cfg["microbatches_per_window"]),
            loss_fn=loss_fn,
        )

        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        return cls(
            rank=rank,
            world_size=world_size,
            device=device,
            manifest_dir=manifest_dir,
            module=module,
            optimizer=optimizer,
            pipeline_stage=pipeline_stage,
            train_schedule=train_schedule,
            stage0_input_embedding=stage0_input_embedding,
            dtype=dtype,
            learning_rate=float(cfg["learning_rate"]),
            grad_clip=float(cfg["grad_clip"]),
            microbatches_per_window=int(cfg["microbatches_per_window"]),
            trainable_mode=trainable_setup.mode,
        )

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def run_window(
        self,
        *,
        phase: str,
        window: BPFreeUpdateWindow,
        checkpoint_store: Optional[StageCheckpointStore] = None,
        commit_ledger: Optional[StageCommitLedger] = None,
    ) -> ExactBPWindowResult:
        if (checkpoint_store is None) != (commit_ledger is None):
            raise ValueError("checkpoint_store and commit_ledger must be provided together")
        records = [record for microbatch in window.microbatches for record in microbatch.records]
        if len(window.microbatches) != self.microbatches_per_window:
            raise ValueError("window microbatch count differs from Schedule1F1B configuration")

        self.module.train(True)
        self.module.recorder.start_batch(phase=phase, batch_seq=window.window_id)
        self._sync()
        total_started = time.monotonic_ns()

        h2d_started = time.monotonic_ns()
        loaded = f1b.load_batch_tensors(
            records=records,
            manifest_dir=self.manifest_dir,
            device=self.device,
            dtype=self.dtype,
            load_hidden=self.rank == 0,
            load_labels=self.rank == self.world_size - 1,
            input_embedding=self.stage0_input_embedding if self.rank == 0 else None,
        )
        self._sync()
        h2d_ms = (time.monotonic_ns() - h2d_started) / 1_000_000.0

        f1b.set_optimizer_lr(self.optimizer, self.learning_rate)
        self.optimizer.zero_grad(set_to_none=True)
        losses: list[torch.Tensor] = []
        target = loaded["labels"] if self.rank == self.world_size - 1 else None
        kwargs = {
            "attention_mask": loaded["attention_mask"],
            "position_ids": loaded["position_ids"],
        }
        if self.pipeline_stage.inputs_meta is None:
            self.module.recorder.skip_next_forward_event()

        schedule_started = time.monotonic_ns()
        with torch.enable_grad():
            if self.rank == 0:
                hidden = loaded["hidden"]
                if hidden is None:
                    raise RuntimeError("stage 0 did not load an input tensor")
                self.train_schedule.step(
                    hidden,
                    target=target,
                    losses=losses if self.rank == self.world_size - 1 else None,
                    return_outputs=False,
                    **kwargs,
                )
            else:
                self.train_schedule.step(
                    target=target,
                    losses=losses if self.rank == self.world_size - 1 else None,
                    return_outputs=False,
                    **kwargs,
                )
        self._sync()
        schedule_ms = (time.monotonic_ns() - schedule_started) / 1_000_000.0

        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.module.parameters(), self.grad_clip)
        optimizer_started = time.monotonic_ns()
        self.optimizer.step()
        self._sync()
        optimizer_ms = (time.monotonic_ns() - optimizer_started) / 1_000_000.0
        self.optimizer_steps += 1

        checkpoint: Optional[StageCheckpointMetadata] = None
        durability_started = time.monotonic_ns()
        if checkpoint_store is not None and commit_ledger is not None:
            checkpoint = checkpoint_store.save(
                module=self.module,
                optimizer=self.optimizer,
                stage_id=self.rank,
                window_id=window.window_id,
                optimizer_step=self.optimizer_steps,
                device=self.device,
            )
            commit_ledger.record(
                StageCommit(
                    run_id=commit_ledger.run_id,
                    stage_id=self.rank,
                    window_id=window.window_id,
                    optimizer_step=self.optimizer_steps,
                    request_ids=request_ids_for_window(window),
                    input_producer_versions=(),
                    checkpoint_id=checkpoint.checkpoint_id,
                )
            )
        durability_ms = (time.monotonic_ns() - durability_started) / 1_000_000.0

        loss_values = [float(loss.detach().cpu().item()) for loss in losses]
        elapsed_ms = (time.monotonic_ns() - total_started) / 1_000_000.0
        del loaded
        return ExactBPWindowResult(
            phase=phase,
            window_id=window.window_id,
            records=len(records),
            microbatches=len(window.microbatches),
            h2d_ms=h2d_ms,
            schedule_ms=schedule_ms,
            optimizer_ms=optimizer_ms,
            durability_ms=durability_ms,
            elapsed_ms=elapsed_ms,
            optimizer_step=self.optimizer_steps,
            loss_count=len(loss_values),
            avg_loss=(sum(loss_values) / len(loss_values)) if loss_values else None,
            checkpoint=asdict(checkpoint) if checkpoint is not None else None,
        )
