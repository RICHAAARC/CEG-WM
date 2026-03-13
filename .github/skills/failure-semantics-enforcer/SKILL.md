---
name: failure-semantics-enforcer
description: Enforce explicit failure semantics across evidence chains and fusion outputs. Use when users ask for status failed or absent behavior audits, failure reason completeness checks, or anti pseudo-robustness verification.
---
# failure-semantics-enforcer

Ensure failures remain explicit, informative, and non-score-producing where required.

## Execute

1. Inspect status transitions for content, geometry, and fusion stages.
2. Verify failed or absent states carry non-empty reasons.
3. Verify failure states do not produce misleading confidence scores.
4. Verify failure propagation does not flip primary decision improperly.
5. Report semantic drift between contracts, code, and tests.

## Repository Anchors

- `main/core/status.py`
- `main/watermarking/content_chain/unified_content_extractor.py`
- `main/watermarking/geometry_chain/interfaces.py`
- `main/watermarking/fusion/decision.py`
- `tests/test_detect_failure_reasons_must_be_non_empty.py`
- `tests/test_geo_failure_must_not_flip_primary_decision.py`
- `tests/test_disabled_module_outputs_absent_and_no_score.py`

## Output Contract

- `issue`
- `failure_semantics_impact`
- `evidence`
- `static_or_runtime_required`
- `recommended_fix`
