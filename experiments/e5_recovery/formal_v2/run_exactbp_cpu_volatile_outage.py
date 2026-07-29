#!/usr/bin/env python3
from __future__ import annotations

from .cpu_exactbp_runtime import ExactBPCpuWindowRuntime
from .run_exactbp_outage import main


if __name__ == "__main__":
    main(
        default_recovery_state_mode="volatile",
        default_backend="gloo",
        transport_name="gloo-cpu-hidden-and-grad-pinned-budgeted",
        transport_details=(
            "E4 fair 1F1B runtime with forward hidden and backward hidden-gradient "
            "messages sharing pinned CPU/Gloo byte budgets"
        ),
        runtime_cls=ExactBPCpuWindowRuntime,
    )
