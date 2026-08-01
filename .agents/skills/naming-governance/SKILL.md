---
name: naming-governance
description: Review and govern semantic names across Python identifiers, comments, docstrings, paths, configuration keys, notebook code, persisted fields, policies, reports, and artifacts. Use when adding or renaming project terminology, code, configs, notebooks, schemas, or publication assets.
---

# Naming Governance

## Workflow

1. Identify every new or renamed term in Python identifiers, comments, docstrings, every governed path basename, config keys and formal identity/name/label/path values, Notebook code/Markdown and persisted interfaces.
2. Use semantic `snake_case` for repository paths, modules, Python identifiers, config keys, and fields unless an external format requires otherwise.
3. Check names against the detachable outer rule at `governance/docs/naming_governance.md` and the research field registry at `docs/reference/field_registry.md`. Values bound to every registered `method_identity` or `runtime_identity` field receive both weak-identity and ordinal-identity checks; explicit identity/name/label/path tokens remain the fallback for unregistered formal identities.
4. Use explicit version semantics such as `schema_version`, `model_revision`, or `upstream_commit` only when the version role is real.
5. Update references, tests, policies, and manifests atomically with a rename.

## Blocking Rules

- Reject unknown or temporary words such as `tmp`, `temp`, `misc`, `other`, `todo`, `tbd`, `dummy`, `fake`, `mock`, `proxy`, `new`, `old`, `latest`, `best`, `final`, `backup`, `copy`, `foo`, and `bar` when they carry a persisted, public, method, runtime, artifact, path, or other formal identity. Synthetic test fixtures may name an explicitly non-formal `fake_gpu` or `mock_backend`; there is no blanket `tests/` exemption, and the same values are forbidden when bound to a formal identity field.
- Reject every internal single-letter-plus-number work identity, not a fixed list of letters. This includes compact, underscore, and hyphen forms and semantic affixes, such as `B1`, `D-2`, `M_3`, or `candidate_x1_gate`. Reject numbered responsibility identities formed from `phase`, `step`, `stage`, `batch`, `tier`, `level`, `group`, `track`, `route`, `gate`, `case`, `option`, `variant`, `module`, `component`, `method`, `model`, `baseline`, `run`, `experiment`, or `trial`, with compact, underscore, hyphen, or semantic-affix variants. Apply these rules to every governed path basename, production Python identifier/comment/docstring, formal Python/config identity, Notebook cell, Markdown/SVG/Drawio identity, and registered field. A real `test_*` node is governed exactly like every other Python function. Only non-formal local test-fixture variables may describe fixture dimensions or synthetic objects; they cannot become persisted/public/registered identities, test nodes, or compatibility aliases.
- Reject every unexplained mechanical numeric identity suffix, not only a noun list; examples include `detector2`, `metric_3`, `config_2`, `result4`, `router2`, `artifact_3`, `candidate4`, `protocol_7`, and `method_v2`. Keep an explicit version word only where its role is real, such as `schema_version`, `model_revision`, or `upstream_commit`; explicit dtype literals are scientific roles, not identity suffixes.
- Apply the mechanical-suffix rule to ordinary identifiers in business production paths and to code-shaped tokens in their comments/docstrings, not only functions, classes, or hand-written formal contexts. Preserve only exact scientific, statistical, coordinate, and version responsibilities.
- Fail closed when the field registry is missing, empty, unreliable, duplicates an identity field, removes an identity field, or downgrades any currently registered `method_identity`/`runtime_identity` category. Naming and field audits keep this detachable outer non-degradation contract; project schemas and runtime objects must not import it. Reject unreadable JSON/YAML/TOML with an explicit `config_unreadable` violation.
- Keep only the narrow scientific/platform literals `relative_l2`, `F32`, `RGB8`, `P95`, `x86_64`, `L4`, `SHA-256`, `SHA256`, and the actually used external model literal `SD3.5`, plus explicit semantic L2-norm identifier roles such as `realized_total_l2` and immediately defined or backticked local mathematical notation such as `C_0`, `C_1(w)`, and `S_0`.
- Do not keep aliases that create two authoritative names.
- Do not rename external assets solely to satisfy project naming rules.

## Required Validation

- Run naming and field audits plus independent negative fixtures for every newly closed surface. Assert the exact violation reason and path rather than only the overall decision.
