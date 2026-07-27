# CEG-WM Repository Agent Contract

1. Before any modification, read `.codex/project_contract.md`.
2. Read `.codex/research_state/research_definition.yaml` and every registered design path when research semantics can be affected.
3. Before any modification, read each relevant `.agents/skills/<skill-name>/SKILL.md`.
4. Respect the current semantic stage; at `research_defined`, do not add substantive `main/` implementation.
5. Do not bypass governance tests or harness audits.
6. Keep content evidence as the only positive authority; geometry cannot directly produce a watermark positive.
7. Do not introduce reference-image, embed-record or private embedding-state dependencies into formal detection.
8. Preserve the same detector identity, key semantics and threshold before and after rectification.
9. Do not inherit fixed LF/HF weights or treat historical project snapshots as current authority.
10. Do not place runtime-heavy, real-model or GPU tests in the default pytest path.
11. Do not write formal outputs into checked-in `outputs/`.
12. Placeholder fields must end with `_placeholder`; random traces must end with `_random` or `_digest_random`.
13. Supported claims must map to governed records, manifests and rebuildable artifacts.
14. Avoid duplicated defensive validation and error-message construction in business paths.
15. Complete authorized changes with the smallest affected tests and exactly one registered validation profile: `governance` for governance-only or non-semantic documentation changes, `method` for research-code-only changes, and `full` for cross-plane changes, stage/research-state changes, registered design changes, or pytest-selection changes. Run all formal profiles in the registered `CEG-WM` Conda environment through `governance/tools/run_validation_profile.py`; when classification is ambiguous, use `full`. A `.venv` without PyTorch may run targeted lightweight checks only, not a completion profile.
16. Docs, governance, policies, skills and tests cannot substitute for method, runtime or experiment implementation when the task explicitly targets those layers.
