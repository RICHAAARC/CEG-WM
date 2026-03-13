---
name: attack-protocol-steward
description: Audit and maintain attack protocol definitions for implementability, coverage, reproducibility, and fair comparison. Use when users ask for attack protocol review, attack family validation, protocol schema checks, or multi-attack evaluation integrity.
---
# attack-protocol-steward

Ensure attack protocols are valid, executable, and comparable across experiments.

## Execute

1. Validate protocol schema and family whitelist.
2. Verify parameter domains are explicit and implementable by attack runners.
3. Verify coverage reports include all declared protocol anchors.
4. Verify seeds and deterministic controls are recorded for reproducibility.
5. Flag hidden hardcoding or protocol bypass in evaluation paths.

## Repository Anchors

- `configs/attack_protocol.yaml`
- `main/evaluation/attack_protocol_guard.py`
- `main/evaluation/attack_runner.py`
- `main/evaluation/attack_plan.py`
- `tests/test_attack_runner_protocol_only.py`
- `tests/test_attack_protocol_guard_blocks_unknown_family.py`
- `tests/test_audit_attack_protocol_report_coverage.py`

## Output Contract

For each finding include:
- `issue`
- `comparison_risk`
- `evidence`
- `severity`
- `recommended_fix`
