---
name: placeholder-random-field-governance
description: Govern placeholder, random trace, and persisted intermediate fields across configs, records, manifests, fixtures, examples, and notebook boundaries. Use when adding, renaming, serializing, grading, or removing such fields.
---

# Placeholder And Random Field Governance

## Workflow

1. Classify each cross-boundary field as formal, placeholder, random trace, intermediate, temporary, or cache state.
2. Assign `internal_state`, `cross_boundary`, `persisted_protocol`, or `evidence_bearing` as its semantic governance level.
3. Register persisted or cross-boundary fields with a non-empty description in `docs/reference/field_registry.md` before use.
4. Apply the required suffix and declare whether replacement or cleanup is required.
5. Keep placeholder and non-authoritative state out of supported claims.

## Blocking Rules

- Placeholder fields must end with `_placeholder`.
- Random trace fields must end with `_random` or `_digest_random`.
- Intermediate, temporary, and cache fields must use their governed suffixes.
- Do not use numbered or weak names for governance levels.
- Do not treat placeholder, intermediate, temporary, or cache values as claim evidence.

## Required Validation

- Run field registry tests, naming and field audits, default tests, and all harness audits.
