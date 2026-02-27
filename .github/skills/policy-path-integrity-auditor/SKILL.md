---
name: policy-path-integrity-auditor
description: Audit policy path integrity so decision routing is unique, frozen, and non-bypassable. Use when users ask for policy_path checks, path semantics audit, decision layer freeze validation, or hidden fallback risk review.
---
# policy-path-integrity-auditor

Audit whether policy routing remains explicit and immutable.

## Execute

1. Inspect policy path definitions and runtime selection logic.
2. Verify only approved policy paths can be selected.
3. Verify no evidence chain can silently switch or bypass policy routing.
4. Verify outputs record active policy path and source of selection.
5. Report any mutable or implicit fallback behavior.

## Repository Anchors

- `configs/policy_path_semantics.yaml`
- `main/policy/path_policy.py`
- `main/policy/freeze_gate.py`
- `tests/test_path_audit_unbound_binding.py`
- `tests/test_paper_path_closure_regressions.py`

## Output Contract

- `issue`
- `policy_risk`
- `evidence`
- `severity`
- `recommended_fix`
