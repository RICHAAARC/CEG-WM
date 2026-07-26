---
name: attack-extension
description: Implement or review attack and distortion transformations under experiments/attacks. Use when adding attack families, parameters, strength schedules, compositions, artifact transformations, or attack-specific fixtures.
---

# Attack Extension

## Workflow

1. Define the input and output artifact types through `experiments.protocol`.
2. Specify attack semantics, parameter units, valid ranges, composition order, applicability, and failure states.
3. Keep the transformation independent of method identity and detector behavior.
4. Return transformed artifacts and structured metadata without writing governed records.
5. Add deterministic unit tests and register persisted parameters or random traces.

## Blocking Rules

- Attacks may depend on `experiments.protocol` but not methods, metrics, runtime, runners, or governance.
- Do not invoke the detector or optimize parameters against the evaluated method inside the attack implementation.
- Do not silently clip, skip, or reinterpret invalid attack parameters.
- Do not put large attack matrices in the default test suite.

## Required Validation

- Run attack unit tests, dependency and field audits, default tests, and all harness audits.
