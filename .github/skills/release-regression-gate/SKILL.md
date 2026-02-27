---
name: release-regression-gate
description: Define and enforce regression gates before release or submission to prevent behavior drift. Use when users ask for pre-release audit gates, critical test set enforcement, or signoff blocking policy.
---
# release-regression-gate

Create a strict but focused regression gate for release confidence.

## Execute

1. Define critical test suites and audit scripts required for release.
2. Verify gate criteria are explicit and machine-checkable.
3. Ensure failure output identifies blockers and ownership clearly.
4. Ensure gate does not allow silent skips on mandatory checks.
5. Report gate gaps and upgrade path for stricter coverage.

## Output Contract

- `issue`
- `release_risk`
- `evidence`
- `severity`
- `recommended_fix`
