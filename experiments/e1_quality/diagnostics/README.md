# E1-adjacent diagnostics

These runners provide additional readout and pipeline comparisons alongside the
four-method E1 protocol.

- `run_agnews_readout_adapter_formal.py` evaluates a local-readout adapter
  ablation.
- `run_gpipe_curve.py` and `run_pipedream_curve.py` compare additional pipeline
  learning curves under legacy AG News settings.

They use the shared runtime modules from `src/` and separate configurations from
the E1 protocol in `../configs/agnews_quality_v1.json`.
