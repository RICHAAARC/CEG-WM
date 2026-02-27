---
name: ablation-readiness-auditor
description: Audit whether the project is ready for controlled ablation studies with reproducible, policy-compliant, and statistically comparable outputs. Use when users ask for ablation readiness, experiment matrix audit, or ablation risk review.
---
# ablation-readiness-auditor

Assess whether ablation experiments are controllable, reproducible, and auditable.

## Execute

1. Enumerate ablation knobs from configs and CLI/script entrypoints.
2. Verify each knob is explicit, traceable, and digest-bound in outputs.
3. Verify experiment matrix execution does not bypass policy/freeze constraints.
4. Verify summary artifacts include per-group counts and stable metric keys.
5. Report readiness gaps with concrete blockers and fixes.

## Required Checks

- Ablation switches are exposed via config/CLI, not hidden hardcoded branches.
- Run metadata records seeds, plan/config digests, and active policy path.
- Aggregated reports preserve group identity and anchor fields for reproducibility.
- Threshold calibration/evaluation boundaries remain read-only where required.
- Write paths stay inside run roots and maintain append-only evidence discipline.

## Suggested Test Targets

- `tests/test_ablation_flag_changes_cfg_digest.py`
- `tests/test_ablation_override_is_blocked_when_not_whitelisted.py`
- `tests/test_aggregate_report_includes_ablation_group_counts.py`
- `tests/test_experiment_matrix_runner_writes_summary_under_run_root.py`
- `tests/test_experiment_matrix_detect_stage_input_binding.py`

## Output Contract

For each finding provide:
- `issue`
- `ablation_impact`
- `evidence`
- `static_or_real_workflow`
- `recommended_fix`
