---
name: watermark-method-change
description: Implement or review CEG-WM content-chain, geometry-chain, joint-decision, key, detector, or method-statistic changes under main. Use for LF, HF carrier/direct score, routing, Q/K synchronization, rectification, rescue gating, or core API work.
---

# Watermark Method Change

## Workflow

1. Read every design path in `.codex/research_state/research_definition.yaml`.
2. Confirm the project is at `method_construction_authorized` or a later implementation stage and that the authorization transition occurred in an earlier change without method implementation.
3. Classify the change as shared key/type, content chain, geometry chain, joint decision or public API work.
4. Preserve the CEG-WM HF direct-score/content-detector identity unless an explicit governed research-definition change authorizes replacement.
5. Keep content routing, LF/HF carrier directions, combined embedding, LF/HF
   blind statistics and content-score combination in their policy-fixed independent
   responsibilities; keep LF, HF and combined scores independently observable.
6. Keep Q/K synchronization, transform estimation, geometry reliability and image
   rectification independent of each other, content-positive semantics and embedding-private state.
7. Keep near-threshold gating and same-detector/same-threshold re-evaluation in `main.joint_decision`.
8. Keep model loading, device, dtype and Q/K extraction execution in `runtime/`.
9. Register cross-boundary fields and add synthetic unit/functional tests for the affected invariant.
10. Connect all 13 required responsibility components to their policy-fixed module
    paths, exact candidate IDs, unique implementation symbols, responsibilities and
    candidate-specific behavior nodes in `.codex/research_state/method_readiness.yaml`.
11. Before claiming `method_implemented`, obtain an independent semantic review bound to the candidate digest and reviewed repository revision; keep all reviewed candidate, implementation and test paths unchanged afterward.
12. Update method documentation without making unsupported effectiveness claims.

## Blocking Rules

- `main/` must not import runtime, experiments, paper artifacts, workflows, tests, or governance.
- `main.content_chain` and `main.geometry_chain` must not import each other.
- Do not hide model adapters, attack logic, or record writing inside the core algorithm path.
- Do not duplicate schema validation and error construction throughout business logic.
- Do not introduce fixed LF/HF weights without governed calibration and independent evaluation.
- Do not allow geometry reliability to add content score or directly cause a positive.
- Do not use a reference image, embed record, private latent or cached embed Q/K in formal detection.
- Do not use a different detector, key semantics or threshold after rectification.
- Do not declare the method implemented while any required component or behavior check is missing.
- Do not begin implementation at `research_defined`, and do not combine construction authorization with method implementation.
- Do not reuse one generic implementation symbol for multiple required components.
- Do not let content or geometry modules redefine root-key encoding, KDF/PRG,
  wrong-key or public-noise semantics owned by `main.shared.key_schedule`.
- Do not collapse distinct content, geometry and joint-decision responsibilities into a centralized proxy module.
- Do not fold combined embedding into a carrier/detector, LF blind scoring into a
  carrier/combiner, or geometry reliability into the transform estimator.
- Do not use constant or input-independent implementations, repeated test nodes or structurally isomorphic tests as readiness evidence.
- Do not treat AST/readiness mechanics as a substitute for candidate-specific tests and independent semantic review.
- Do not read `.codex/` or `governance/` from method code; readiness metadata is removable construction guidance, not a method input.

## Required Validation

- Run affected component tests, dependency audits, method-readiness audit, default tests and all harness audits.
