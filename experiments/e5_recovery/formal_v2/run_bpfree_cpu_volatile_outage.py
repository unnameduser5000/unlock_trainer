#!/usr/bin/env python3
from __future__ import annotations

from .cpu_bpfree_runtime import CpuVolatileCaptureBPFreePipelineStage
from .run_bpfree_outage import main


if __name__ == "__main__":
    main(
        default_catchup_policy="window_streamed",
        default_recovery_state_mode="volatile",
        default_backend="gloo",
        default_recv_inflight_depth=0,
        transport_name="gloo-cpu-hidden-pinned-budgeted",
        transport_details=(
            "GPU compute with pinned CPU/Gloo hidden transport; outage hidden is "
            "retained in transport-ready CPU memory and replayed without H2D staging"
        ),
        volatile_stage_cls=CpuVolatileCaptureBPFreePipelineStage,
    )
