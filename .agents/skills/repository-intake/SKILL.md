---
name: repository-intake
description: Inspect the CEG-WM contract, research-definition manifest, registered roots, dependency boundaries, current stage, and existing changes before repository work. Use for any governance, method, runtime, experiment, migration, refactor, or release task.
---

# Repository Intake

## Workflow

1. Read `.codex/project_contract.md` completely.
2. Read `.codex/research_state/research_definition.yaml` and every design path it registers when the task can affect research semantics.
3. Read `governance/contracts/architecture.md` and the relevant policies under `governance/policies/`.
4. Inspect the target paths, adjacent tests, documentation, and current repository state.
5. Classify the request as governance, method, runtime, experiment, artifact, notebook, infrastructure, or release work.
6. Confirm that every target root and planned dependency direction are registered before editing.
7. Identify the other project skills required for the change.
8. Preserve the detachable outer boundary: `.agents/`, `.codex/`, and `governance/` may inspect the research project, while research and delivery code must remain runnable without them.

## Blocking Rules

- Do not modify an unregistered root.
- Do not infer a later project stage from directory presence alone.
- Do not overwrite unrelated user changes.
- Do not treat placeholder structure as implemented research capability.
- At `research_defined`, do not add substantive code under `main/`; advance the semantic stage only after the declared gate is satisfied.
- Substantive `main/` work is legal only at `method_construction_authorized` or a later registered implementation stage, after a separate user-authorized stage transition.
- Do not treat the four historical project snapshots as current authority.
- Without a user-authorized CEG-WM revision, historical repositories may be recorded only as provisional read-only source revisions and file digests. Do not claim migration provenance, copy source, or begin migration.

## Required Validation

- Run the smallest tests appropriate to the changed layer first.
- Complete governance-only work with
  `conda run -n CEG-WM python governance/tools/run_validation_profile.py governance`.
- Complete research-code-only work with
  `conda run -n CEG-WM python governance/tools/run_validation_profile.py method`.
- Use the `full` profile for cross-plane work, stage/research-state or registered
  design changes, pytest-selection changes, and ambiguous scope.
- Run every completion profile in the registered Conda environment; use `.venv`
  only for targeted checks known not to import research code.
