---
name: wm-gap-auditor
description: Audit implementation gaps and risks between the current watermarking project and stated research goals. Use when users ask for gap audit, risk audit, goal alignment, static code audit, or a split report of static-confirmable gaps vs gaps that require real SD workflow.
---
# wm-gap-auditor

Produce a defensible gap audit tied to repository evidence.

## Execute

1. Extract target claims from user text and project docs.
2. Build a claim-to-anchor map with `configs/` and `main/` as the core release truth source; use `scripts/` and `tests/` only as auxiliary evidence.
3. Validate each claim with static evidence first:
- Read implementation paths.
- Read contract/config paths.
- Run only relevant tests or audit scripts when they can decide the claim, but do not let helper script behavior override `main/`.
4. Classify each gap item into exactly one bucket:
- `static-audit-confirmable`
- `requires-real-sd-workflow`
5. If the user mandates a fixed phrase for undecidable items, output that phrase exactly.

## Required Output Shape

For every item include:
- `gap_item`
- `risk_level` (`high`/`medium`/`low`)
- `evidence` (file path or test name)
- `reasoning`
- `fix_or_validation_action`

## Repository Anchors

Prioritize these anchors before expanding search:
- `configs/paper_faithfulness_spec.yaml`
- `notebook/Colab_Workflow.ipynb`
- `main/watermarking/content_chain/`
- `main/watermarking/geometry_chain/`
- `main/watermarking/fusion/`
- `main/policy/`
- `tests/test_onefile_workflow.py`
- `tests/test_publish_workflow_uses_paper_profile_by_default.py`
- `tests/test_paper_faithfulness_gate.py`

Use these only as auxiliary workflow anchors after core files:
- `scripts/run_onefile_workflow.py`
- `notebook/Colab_Workflow.ipynb`

## Classification Rules

- Put into static bucket if code/contracts/tests already prove presence, absence, or violation.
- Put into real-workflow bucket if conclusion depends on SD model runtime behavior, generated image quality, attack robustness curves, or end-to-end geometry recovery under real inference.
