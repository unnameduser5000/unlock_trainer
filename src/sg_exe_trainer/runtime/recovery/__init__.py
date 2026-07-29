"""Durable recovery mechanisms shared by runtime and experiments.

The package owns checkpoint, boundary-outbox, commit-ledger, window-journal,
catch-up, and event-recording behavior. Fault schedules and evaluation policy
belong under ``experiments/e5_recovery`` instead.
"""

__all__: list[str] = []
