---
name: sd-runtime-readiness-checker
description: Assess whether unresolved risks require real Stable Diffusion runtime workflow and define executable validation plans. Use when users ask what cannot be decided statically, how to validate with real SD runs, or Colab/runtime readiness checks.
---
# sd-runtime-readiness-checker

Separate static conclusions from real-runtime-only conclusions and provide a testable runtime plan.

## Execute

1. Inspect workflow entrypoints, model loader path, and runtime dependency gates.
2. Identify claims that depend on real generation behavior or robustness under attack.
3. Mark each such item as runtime-required and provide a minimal executable validation protocol.
4. Verify local scripts and notebook can drive the required end-to-end run.
5. Report expected artifacts and pass/fail criteria for each runtime-required item.

## Repository Anchors

- `notebook/Colab_Workflow.ipynb`
- `scripts/run_onefile_workflow.py`
- `main/diffusion/sd3/diffusers_loader.py`
- `main/diffusion/sd3/infer_runtime.py`
- `tests/test_pipeline_shell_synthetic_runtime.py`
- `tests/test_pipeline_preflight_observability.py`

## Output Contract

For each runtime-required item include:
- `risk_item`
- `why_static_is_insufficient`
- `required_runtime_workflow`
- `required_artifacts`
- `pass_fail_rule`
