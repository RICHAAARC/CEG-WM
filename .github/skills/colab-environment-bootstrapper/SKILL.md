---
name: colab-environment-bootstrapper
description: Prepare and verify Colab runtime prerequisites for end-to-end workflow execution. Use when users ask for Colab setup checks, dependency bootstrap, GPU readiness validation, or notebook environment troubleshooting.
---
# colab-environment-bootstrapper

Define and validate a stable Colab boot process for workflow runs.

## Execute

1. Identify required packages, model configuration files, and runtime assumptions.
2. Verify notebook setup cells align with script and config expectations.
3. Verify GPU and memory preflight checks are explicit.
4. Verify output paths and artifact persistence are deterministic.
5. Produce a minimal startup checklist with failure recovery steps.

## Repository Anchors

- `notebook/Colab_Workflow.ipynb`
- `notebook/README.md`
- `scripts/run_onefile_workflow.py`
- `configs/paper_faithfulness_spec.yaml`
- `requirements.txt`

## Output Contract

- `check_item`
- `status`
- `evidence`
- `failure_symptom`
- `fix_action`
