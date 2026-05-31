# Notes From `results_report_2026_5.pdf`

Source PDF:

- `D:/kk/xwechat_files/wxid_t8g57661dlio22_4e92/msg/file/2026-05/results_report_2026_5.pdf`
- Reviewed on 2026-05-31.

This note exists so future context resets do not restart the algorithm story from scratch.

## What The Report Already Establishes

The report studies belief-style chunk-local training before the mobile systems prototype.

Main ViT setting:

- Model/task: scratch ViT-small on ImageNet-100.
- Split: 4 chunks.
- Seeds: 42, 43, 44.
- Baselines:
  - Full BP.
  - CE-only chunk-local training.
  - Prev-only belief KD.
  - Prev+Next belief KD.

Main numbers:

- Full BP: `79.52 +/- 0.12` top-1.
- CE-only: `72.65 +/- 0.51` top-1.
- Prev-only: `71.61 +/- 0.37` top-1.
- Prev+Next: `71.93 +/- 0.42` top-1.

Interpretation:

- CE-only is the strongest stable chunk-local baseline in the main ViT experiment.
- Belief KD does not reliably beat CE-only.
- Full BP remains about `6.87 pp` above CE-only, so chunk-local training is not algorithmically equivalent to full BP.

## Gradient Probe Takeaway

The gradient probe compares boundary hidden-state directions:

- `cos(BP, CE)`: local CE gradient vs full-BP boundary gradient.
- `cos(BP, KD)`: belief-KD gradient vs full-BP boundary gradient.
- `cos(BP, CE+KD)`: combined local gradient vs full-BP boundary gradient.
- `cos(CE, KD)`: CE and KD agreement.

Key observation:

- KD gradients are near-orthogonal to full-BP boundary gradients, and sometimes slightly negative in middle chunks.
- CE+KD is almost the same direction as CE alone.

Implication:

- Current belief KD is not a convincing credit-assignment signal.
- For the mobile paper/prototype, belief should not be the central accuracy-improvement claim unless new evidence changes this.

## Target-Space Search Takeaway

The report tested alternatives:

- Logit KD temperature sweep.
- Prototype target.
- Hidden-state target.
- Hidden-delta target.
- Gated KD.
- BP-teacher control.

Key observation:

- Most target spaces remain near zero or negative alignment with full-BP boundary gradient.
- Prototype with small weight is less bad, but not strong enough to be a reliable positive signal.

Implication:

- Do not spend mobile systems time optimizing top-k belief transport as if belief quality is already validated.
- Belief transport can still be treated as a modular auxiliary channel, but the current strong system story should not depend on it improving quality.

## LLM Sanity Takeaway

LLM setting:

- Dolly-15k.
- Llama-2-7B LoRA.
- 4 chunks.
- 12k train / 2k validation.
- Seeds 42, 43, 44.

Main numbers:

- BP LoRA validation ppl: `5.1816 +/- 0.0288`.
- CE-only async: `6.6042 +/- 0.0192`.
- Prev-only: `6.6109 +/- 0.0208`.
- Prev+Next: `6.5968 +/- 0.0556`.

Interpretation:

- Prev+Next is only a near-tie with CE-only, not a stable LLM gain.
- Full BP LoRA is still much better than async CE-only.

## How This Should Shape The Mobile Prototype Story

Do not frame the current project as:

- "belief makes training much more accurate."
- "BP-free reaches full BP quality."
- "top-k belief is the core algorithmic contribution."

Stronger framing:

- The algorithmic substrate is BP-free / chunk-local LoRA training.
- The systems contribution is making that substrate actually run across heterogeneous Android devices.
- The key operator-level/system distinction from full BP is boundary-gradient elision:
  - no cross-device hidden-gradient return,
  - no cross-stage backward dependency,
  - local backward only serves local LoRA parameters,
  - activation and state requirements are stage-local.
- This enables:
  - bounded in-flight pipeline execution,
  - weaker synchronization requirements than full pipeline BP,
  - easier failure isolation and retry around request/stage boundaries,
  - mobile telemetry and scheduling decisions based on stage health.

## How To Use This In The Current Report

Keep the old report as algorithm motivation and negative evidence:

- It justifies why CE-only is a serious baseline.
- It justifies why belief should be optional in the mobile system.
- It motivates a systems-first paper: quality is measured as "not catastrophically broken" on-device, while the novelty is deployment, scheduling, telemetry, and boundary-gradient-elided execution.

The mobile report should include:

- One table showing prior algorithm evidence:
  - Full BP > CE-only.
  - Belief variants near or below CE-only.
- One paragraph saying the prototype therefore uses BP-free chunk-local LoRA as the deployable training primitive, while treating belief as an optional auxiliary signal.
- The main experimental burden should move to:
  - real Android multi-device execution,
  - serial vs bounded in-flight pipeline comparison,
  - transient failure recovery,
  - memory/battery/temperature telemetry,
  - operator-level explanation of why boundary-gradient elision matters.

## Immediate Diagnostic Rule For Current Rotten-Tomatoes Label Experiments

If mobile/server label quality is weak, do not blindly keep tuning LR.

Run controls:

- `NUM_CHUNKS=1`: server-side full-LoRA upper bound on the same prepared label data.
- `NUM_CHUNKS=3 TRAIN_CHUNKS=2 ALPHA=1.0`: terminal-chunk-only CE control.
- `NUM_CHUNKS=3 TRAIN_CHUNKS=all ALPHA=1.0`: BP-free CE-only control.
- Compare against `ALPHA=0.5` belief/KL.

Interpretation:

- Full-LoRA weak means the label task/prompt/LoRA capacity is weak.
- Full-LoRA good but BP-free weak means the detached-boundary local objective is the bottleneck.
- CE-only good but belief/KL weak means belief is hurting this task.
- Server good but Android weak means check export/PTE/runtime/optimizer/checkpoint mismatch.
