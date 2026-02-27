---
name: impl-whitelist-governor
description: Govern implementation whitelist and identity version discipline for reproducible research paths. Use when users ask for impl whitelist audits, resolver integrity checks, version bump enforcement, or unauthorized implementation drift detection.
---
# impl-whitelist-governor

Ensure implementation selection is locked, versioned, and auditable.

## Execute

1. Inspect runtime resolver, registry identity, and whitelist config.
2. Verify new implementations require whitelist updates and identity version discipline.
3. Verify selected implementations are written into artifacts and digests.
4. Verify no bypass path can inject non-whitelisted implementations.
5. Report governance gaps that break reproducibility.

## Repository Anchors

- `main/registries/runtime_resolver.py`
- `main/registries/impl_identity.py`
- `configs/runtime_whitelist.yaml`
- `tests/test_impl_identity_version_bumped_or_new_impl_whitelisted.py`
- `tests/test_registry_whitelist_required_for_planner_impl.py`
- `tests/test_align_impl_whitelist_enforcement.py`

## Output Contract

- `issue`
- `governance_risk`
- `evidence`
- `severity`
- `recommended_fix`
