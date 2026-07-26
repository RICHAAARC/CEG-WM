---
name: naming-governance
description: Review and govern semantic names across Python identifiers, comments, docstrings, paths, configuration keys, notebook code, persisted fields, policies, reports, and artifacts. Use when adding or renaming project terminology, code, configs, notebooks, schemas, or publication assets.
---

# Naming Governance

## Workflow

1. Identify every new or renamed term in Python identifiers, comments, docstrings, repository paths, config keys, Notebook code and persisted interfaces.
2. Use semantic `snake_case` for repository paths, modules, Python identifiers, config keys, and fields unless an external format requires otherwise.
3. Check names against the detachable outer rule at `governance/docs/naming_governance.md` and the research field registry at `docs/reference/field_registry.md`.
4. Use explicit version semantics such as `schema_version`, `model_revision`, or `upstream_commit` only when the version role is real.
5. Update references, tests, policies, and manifests atomically with a rename.

## Blocking Rules

- Reject `v1`, `v1v2`, `p1`, numbered stages, `proxy`, `new`, `old`, `best`, or `final` as standalone weak labels in identifiers, comments, docstrings, governed config keys, Notebook code, paths, fields, policies, reports, or artifacts.
- Do not keep aliases that create two authoritative names.
- Do not rename external assets solely to satisfy project naming rules.

## Required Validation

- Run naming and field audits plus the tests covering renamed interfaces.
