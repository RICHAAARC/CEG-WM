---
name: test-case-governance
description: Design, place, mark, and validate project tests by purpose and execution cost. Use when adding or changing tests, fixtures, pytest configuration, formal gates, model-backed checks, or test helpers.
---

# Test Case Governance

## Workflow

1. Classify the test as governance self-test, unit, constraint, functional, integration, smoke, or formal.
2. Put governance implementation, copy, extraction, and detachability tests under `governance/tests/`; put only research-code tests under `tests/`.
3. Apply the correct pytest marker and use only small synthetic fixtures in the default path.
4. Write outputs through `tmp_path` or `tmp_path_factory`.
5. Keep helpers free of `test_` prefixes.
6. Bind method-readiness checks to direct calls into the policy-fixed component path and exact candidate IDs.
7. Keep root/domain separation, counter/quantile golden, wrong/public derivation,
   HF sparse support/template normalization/unit-L2, HF score-time centering,
   LF carrier and independent blind score, routing mask partition/range and disabled
   uniform control, router-mask consumption through carrier directions, combined-
   embedding frozen mixing coefficients/non-orthogonal cross term/target total
   budget/zero-direction behavior,
   LF/HF/combined score observability, wrong-key non-masking, Q/K
   relation/identifiability, independent reliability, rectification coordinates,
   near-threshold, no-direct-positive and same-threshold checks distinct. Keep
   actual-dtype realized combined total norm/relative-L2 evidence in the later
   runtime gate rather than a CPU fixture; never model mixing coefficients as
   additive direction shares.
8. Treat AST collection and binding checks as structural only; they do not replace independent semantic review.

## Blocking Rules

- Do not put tests directly under `tests/`.
- Do not run real models, networks, GPUs, large attacks, or formal matrices in the default suite.
- Do not write test outputs into checked-in `outputs/`.
- Do not add `governance/tests/` to the root project pytest collection; use `governance/pytest.ini` for outer self-tests.
- Do not reuse one pytest node for multiple required readiness behaviors.
- Do not replace model-backed evidence with a constant or API-shape assertion.

## Required Validation

- Run the smallest affected test group, then project tests with `.venv/bin/python -m pytest -q -s`, outer self-tests with `.venv/bin/python -m pytest -q -s -c governance/pytest.ini`, and the test-governance audit.
