---
name: seed-determinism-auditor
description: Audit random seed control and determinism boundaries across pipeline, attacks, and evaluation outputs. Use when users ask for reproducibility drift checks, nondeterminism investigation, or seed policy enforcement.
---
# seed-determinism-auditor

Audit where determinism is guaranteed, where it is not, and whether runs remain reproducible.

## Execute

1. Inspect seed inputs in workflow scripts, attack runners, and evaluation entrypoints.
2. Verify seeds are recorded in artifacts and linked to run metadata.
3. Check known nondeterminism sources and whether they are declared.
4. Verify fixed-seed reruns produce stable digests where required.
5. Report missing controls and ambiguous reproducibility claims.

## Output Contract

- `issue`
- `determinism_impact`
- `evidence`
- `static_or_runtime_required`
- `recommended_fix`
