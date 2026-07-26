---
name: metric-extension
description: Implement or review experiment metrics under experiments/metrics. Use when adding metric semantics, aggregation, thresholds, uncertainty estimates, missing-value handling, metric records, or metric-specific fixtures.
---

# Metric Extension

## Workflow

1. Define metric inputs using protocol artifacts and records.
2. Document the unit of analysis, direction, range, aggregation, exclusions, missing-value policy, and uncertainty treatment.
3. Separate threshold selection or calibration from held-out evaluation.
4. Return metric results without writing governed records or selecting preferred experiment outcomes.
5. Register persisted metric fields and add small deterministic tests.

## Blocking Rules

- Metrics may depend on `experiments.protocol` but not runtime, methods, attacks, runners, or governance.
- Do not inspect method implementations or attack internals.
- Do not change aggregation semantics without a schema/protocol change.
- Do not support claims with an undocumented headline metric alone.

## Required Validation

- Run metric unit tests, protocol and field audits, default tests, and all harness audits.
