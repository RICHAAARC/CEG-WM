---
name: dataset-provenance-guard
description: Audit dataset provenance, split integrity, licensing constraints, and leakage risks for research reproducibility. Use when users ask for dataset source validation, train test leakage checks, data governance review, or reproducibility data audits.
---
# dataset-provenance-guard

Audit whether dataset usage is traceable, lawful, and leakage-safe.

## Execute

1. Locate dataset source declarations and version pins.
2. Verify split definitions are explicit and stable across runs.
3. Verify licenses and usage constraints are documented and compatible with project goals.
4. Check for leakage paths between train, calibration, and evaluation sets.
5. Report unresolved provenance or compliance risks with evidence anchors.

## Output Contract

- `issue`
- `data_risk`
- `evidence`
- `severity`
- `recommended_fix`
