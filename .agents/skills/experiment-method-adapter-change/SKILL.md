---
name: experiment-method-adapter-change
description: Implement or review adapters that connect a core watermark method and runtime execution to experiments/protocol. Use for changes under experiments/methods, method registration, protocol conversion, or experiment-facing method configuration.
---

# Experiment Method Adapter Change

## Workflow

1. Identify the `main/` API, runtime adapter, and protocol case/artifact types being connected.
2. Keep the adapter thin: translate configuration and objects without reimplementing the watermark method.
3. Return protocol-compatible artifacts and structured statuses to the runner.
4. Register persisted adapter fields and random traces before serialization.
5. Add lightweight tests with fake method and runtime implementations.

## Blocking Rules

- This layer may depend only on `main`, `runtime`, and `experiments.protocol` among project layers.
- Do not import attacks, metrics, runners, paper artifacts, workflows, tests, or governance.
- Do not write governed records or hide attack/metric computation in the adapter.
- Do not duplicate core embedding or detection logic.

## Required Validation

- Run adapter tests, dependency audits, field audits, default tests, and all harness audits.
