---
name: runtime-adapter-change
description: Implement or review CEG-WM runtime adapters for generative models, HF carrier/direct-score execution, Q/K observation, inference backends, devices, checkpoints, or runtime provenance under runtime.
---

# Runtime Adapter Change

## Workflow

1. Define the adapter boundary between the external model/backend and `main/`.
2. Keep model-specific loading, device, dtype, batching, and execution logic in `runtime/`.
3. Expose only the model observations requested by the `main/` public API, including frozen HF carrier/direct-score inputs or Q/K features.
4. Record model identity, revision, preprocessing, environment, configuration digest, and random traces needed for provenance.
5. Return runtime-owned execution artifacts for conversion by experiment method adapters; do not import experiment implementations.
6. Add mocked or synthetic lightweight tests and place real-backend checks under smoke or integration.

## Blocking Rules

- Runtime may depend on `main/` but not experiments, paper artifacts, workflows, tests, or governance.
- Runtime must import the `main` public surface, not bypass it through `main.content_chain` or `main.geometry_chain`.
- Do not implement near-threshold gating, geometry reliability policy, rectified re-evaluation or final decision in runtime.
- Do not expose original reference images, embed records or private embedding state as formal detector inputs.
- Do not commit credentials, private paths, model weights, or large generated outputs.
- Do not place real model execution in the default pytest suite.

## Required Validation

- Run lightweight adapter tests, dependency audits, and any explicitly requested smoke test.
