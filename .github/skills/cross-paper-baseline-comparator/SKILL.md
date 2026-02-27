---
name: cross-paper-baseline-comparator
description: Standardize baseline comparison protocols across methods and papers with aligned attacks, metrics, and reporting semantics. Use when users ask for fair baseline comparison, T2SMark style comparison setup, or cross-method evaluation consistency.
---
# cross-paper-baseline-comparator

Audit whether baseline comparisons are fair, aligned, and reproducible.

## Execute

1. Align protocol definitions, attack strength grids, and metric semantics across methods.
2. Verify calibration and decision policy are not mixed inconsistently across baselines.
3. Verify reports include comparable anchors and group keys.
4. Flag mismatched assumptions that can bias comparison claims.
5. Provide a normalized comparison checklist and remediation plan.

## Repository Anchors

- `configs/attack_protocol.yaml`
- `main/evaluation/experiment_matrix.py`
- `main/evaluation/report_builder.py`
- `tests/test_evaluation_protocol_engineering.py`
- `tests/test_aggregate_report_has_all_anchors.py`
- `tests/test_metrics_group_key_canonicalized.py`

## Output Contract

- `issue`
- `comparison_bias_risk`
- `evidence`
- `severity`
- `recommended_fix`
