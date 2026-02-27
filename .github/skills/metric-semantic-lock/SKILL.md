---
name: metric-semantic-lock
description: Lock metric semantics and reporting definitions to prevent drift across experiments and paper revisions. Use when users ask for metric definition audits, FPR FNR consistency checks, or evaluation semantics freeze.
---
# metric-semantic-lock

Ensure metric meaning stays stable across scripts, reports, and claims.

## Execute

1. Locate metric definitions and aggregation logic.
2. Verify naming and formula semantics are consistent across pipeline stages.
3. Verify report fields map to metric definitions without silent remapping.
4. Check for threshold or denominator drift that changes interpretation.
5. Report semantic inconsistencies with exact correction actions.

## Output Contract

- `issue`
- `metric_semantic_risk`
- `evidence`
- `severity`
- `recommended_fix`
