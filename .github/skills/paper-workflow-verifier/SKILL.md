---
name: paper-workflow-verifier
description: Verify that Colab and one-file paper workflow are consistent, executable, and output-complete for the research protocol. Use when users ask to check Colab_Workflow.ipynb, paper_faithfulness_spec.yaml, run_onefile_workflow.py, output contracts, or paper workflow closure.
---
# paper-workflow-verifier

Verify workflow closure for paper-faithful execution.

## Execute

1. Check configuration-binding consistency:
- `notebook/Colab_Workflow.ipynb`
- `configs/paper_faithfulness_spec.yaml`
- `scripts/run_onefile_workflow.py`
2. Confirm profile and path policy are explicit and frozen where required.
3. Confirm outputs include auditable evidence-chain fields (status, quality, reason, bundle metadata).
4. Run workflow-focused tests when available.
5. Separate findings into:
- statically provable workflow defects
- defects requiring real SD runtime to decide

## Minimum Static Checks

- Notebook parameters pass through to one-file script without silent overrides.
- Script reads intended config and does not fall back to unintended defaults.
- Output locations are deterministic and under expected run roots.
- Required report/evidence artifacts are produced or explicitly marked absent with reason.
- Policy path and decision-path semantics are not bypassed.

## Suggested Test Targets

- `tests/test_onefile_workflow.py`
- `tests/test_onefile_workflow_paper_full_profiles.py`
- `tests/test_publish_workflow_uses_paper_profile_by_default.py`
- `tests/test_paper_faithfulness_gate.py`

## Real-Workflow Boundary

Mark as runtime-required when the claim depends on true model generation quality, real robustness under attack, or behavior observable only with full SD inference.
