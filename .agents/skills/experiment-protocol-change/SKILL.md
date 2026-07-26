---
name: experiment-protocol-change
description: Define or revise CEG-WM internal design-validation or external comparison cases, records, splits, thresholds, attacks, exclusions, and protocol interfaces under experiments/protocol.
---

# Experiment Protocol Change

## Workflow

1. Define the unit of analysis, case identity, artifact types, split, sample scope, and exclusions.
2. Classify the protocol as internal component validation or external method comparison.
3. For internal validation, cover LF-only, HF-only, routing, combination, wrong-key, geometry, rectification and rescue-gate ablations as applicable.
4. Specify fixed-FPR calibration, `tau`, `tau_rescue`, geometry reliability and evaluation boundaries without importing implementations.
5. Require per-sample traces for raw/rectified detector identity, branch scores, routing, geometry attempt, reliability, transform estimate, threshold identity and final decision.
6. Register every persisted field and version schema change that affects existing records.
7. Keep protocol independent of runtime, methods, attacks, metrics, runners, and governance.
8. Add small synthetic fixtures and schema/constraint tests.
9. For external comparison, define a `ComparisonProtocol` that fixes sample and split manifests, generation conditions, seed policy, output specification, attack and metric sets, calibration/evaluation separation, tuning and compute budgets, and failure/exclusion policies.
10. Pin each participating method's role, implementation revision, configuration digest and declared deviation before preflight approval.
11. Document migration or incompatibility for changed record semantics.

## Blocking Rules

- Do not encode a specific method or backend into the shared protocol.
- Do not silently reinterpret existing fields or splits.
- Do not let notebooks define the only protocol implementation.
- Do not force internal component validation to claim an external baseline.
- Do not approve an external comparison without both a project method and an external baseline.
- Do not reuse calibration data for evaluation or fit a separate threshold for rectified images.
- Do not omit failed geometry, rejected rescue or wrong-key cases from records.

## Required Validation

- Run protocol schema tests, field audits, dependency audits, default tests, and all audits.
