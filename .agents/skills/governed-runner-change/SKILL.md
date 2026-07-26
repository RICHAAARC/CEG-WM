---
name: governed-runner-change
description: Implement or review experiment runners that combine protocol, methods, attacks, and metrics and write governed records. Use for experiment orchestration, run manifests, record writing, resumability, or execution-matrix changes under experiments/runners.
---

# Governed Runner Change

## Workflow

1. Accept governed configuration, protocol cases, and a `PreflightApproval` as inputs for external comparisons.
2. Recompute the loaded `ComparisonProtocol` digest and refuse execution if it differs from the approval.
3. Compose methods, attacks, and metrics only in `experiments/runners/`.
4. Capture protocol digest, method role, code revision, configuration digest, input manifest, split, model revision, random traces, exclusions, and failure states.
5. Validate record schemas before the runner writes governed records.
6. Make interrupted and resumed execution traceable without duplicating completed cases.
7. Add lightweight orchestration tests with fakes; place real matrices under integration or formal tests.

## Blocking Rules

- Methods, attacks, and metrics must not write governed records.
- Do not discard failed, skipped, or excluded cases without governed status.
- Do not derive formal claims directly from logs or partial records.
- Do not write formal outputs into checked-in `outputs/`.
- Do not start an external-comparison matrix without a matching preflight approval.

## Required Validation

- Run runner tests, record-schema checks, dependency audits, default tests, and all audits.
