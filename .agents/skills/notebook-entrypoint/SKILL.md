---
name: notebook-entrypoint
description: Keep Jupyter and Colab notebooks as thin orchestration entrypoints. Use when adding or modifying notebooks, notebook support code, Colab setup, interactive experiment workflows, or notebook-produced artifacts.
---

# Notebook Entrypoint

## Workflow

1. Identify logic that belongs in `main/`, `runtime/`, a specific `experiments/protocol/` or `experiments/runners/` entrypoint, `paper_artifacts/`, or `scripts/`.
2. Keep notebook cells limited to environment setup, configuration, invocation, inspection, and presentation.
3. Store Colab notebooks in `notebooks/colab/`; add narrowly named helpers under `notebooks/support/` only after real reuse appears.
4. Route formal records and artifacts through repository modules.
5. Clear cell outputs and execution counts before committing a notebook.

## Blocking Rules

- Do not make a notebook the only implementation of protocol or method logic.
- Do not hand-write formal records, thresholds, tables, figures, reports, or manifests in cells.
- Do not treat cell output as governed evidence.
- Do not split generic `notebook_utils` and `colab_utils` without a concrete non-overlapping responsibility.

## Required Validation

- Test the invoked repository modules outside the notebook.
- Run the notebook-boundary audit, default tests, and all harness audits.
