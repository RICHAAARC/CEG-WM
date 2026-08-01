---
name: naming-governance
description: Review and govern semantic names across Python identifiers, comments, docstrings, paths, configuration keys, notebook code, persisted fields, policies, reports, and artifacts. Use when adding or renaming project terminology, code, configs, notebooks, schemas, or publication assets.
---

# Naming Governance

## Workflow

1. Identify every new or renamed term in Python identifiers, comments, docstrings, every governed path basename, config keys and formal identity/name/label/path values, Notebook code/Markdown and persisted interfaces.
2. Use semantic `snake_case` for repository paths, modules, Python identifiers, config keys, and fields unless an external format requires otherwise.
3. Check names against the detachable outer rule at `governance/docs/naming_governance.md` and the research field registry at `docs/reference/field_registry.md`.
4. Use explicit version semantics such as `schema_version`, `model_revision`, or `upstream_commit` only when the version role is real.
5. Update references, tests, policies, and manifests atomically with a rename.

## Blocking Rules

- Reject `v1`, `v1v2`, `p1`, numbered stages, `proxy`, `new`, `old`, `best`, or `final` as standalone weak labels in identifiers, comments, docstrings, governed config keys and formal string values, Notebook code, paths, fields, policies, reports, or artifacts.
- Reject ordinal work identities such as `A1`, `A-2`, `A3a`, `A3b`, `a3b_metric`, `C0`, `C1-P`, `C1-M`, `C1-E`, `c1_*`, `R1`, `R2`, `S1`, `S2`, `P1`, `P_1`, `P-2`, numbered batches and numbered stages. Lettered ordinal identities include compact, underscore and hyphen variants with leading or trailing semantic tokens; Batch/Runtime Batch/Stage identities cover complete numbered work labels and their space, underscore, hyphen or compact separators without truncating a longer semantic fixture token. Apply this rule to each governed path basename, Python comments/docstrings, formal Python/config identities, Notebook cells, Markdown, SVG, Drawio and registered fields. Only a Python `test_*` function definition physically below a directory named `tests` may retain an ordinal-looking test-node name; its file basename, body identifiers, strings and configuration remain governed.
- Keep only the narrow scientific/platform literals `relative_l2`, `F32`, `RGB8`, `P95`, `x86_64`, `L4`, `SHA-256` and `SHA256`, plus immediately defined or backticked local mathematical notation such as `C_0`, `C_1(w)` and `S_0`.
- Do not keep aliases that create two authoritative names.
- Do not rename external assets solely to satisfy project naming rules.

## Required Validation

- Run naming and field audits plus independent negative fixtures for every newly closed surface. Assert the exact violation reason and path rather than only the overall decision.
