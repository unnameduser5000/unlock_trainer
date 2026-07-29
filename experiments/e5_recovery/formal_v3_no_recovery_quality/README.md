# E5 no-recovery quality formal v3

This isolated suite measures whether BP-free prefix-local updates remain useful
when Stage 1 is unavailable for samples `[768,1280)`.

It adds the missing BP-free balanced-skip control and evaluates both immediately
after the outage and after normal training resumes.

```bash
python experiments/e5_recovery/formal_v3_no_recovery_quality/run_formal_v3.py \
  --output_root results/e5_recovery/quality/formal_v3_stage1_middle
```

The runner is resumable. Each completed run writes `normalized_result.json`, and
the suite continuously rebuilds `results.csv` and `PROGRESS.md`.

After the matrix finishes, build the paired statistics and figures with:

```bash
python experiments/e5_recovery/formal_v3_no_recovery_quality/analyze_formal_v3.py \
  results/e5_recovery/quality/formal_v3_stage1_middle
```

The analysis writes `aggregate.csv`, per-seed and aggregate paired deltas,
`REPORT_ZH.md`, and two figures. The causal comparison for the local-head claim
is `bpfree_local_retain - bpfree_balanced_skip`, because those methods see the
same successful samples and differ only in the 64 Stage-0 updates retained
during the outage.
