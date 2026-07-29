#!/usr/bin/env python3
from __future__ import annotations

from .run_bpfree_outage import main


if __name__ == "__main__":
    main(
        default_catchup_policy="window_streamed",
        default_recovery_state_mode="volatile",
    )
