---
name: evidence-schema-guardian
description: Protect evidence and records schemas from drift and implicit semantics changes. Use when users ask for EvidenceBundle integrity checks, schema append-only audit, field compatibility review, or auditability hardening.
---
# evidence-schema-guardian

Guard schema evolution so evidence stays auditable and comparable across runs.

## Execute

1. Locate canonical schema and extension definitions.
2. Check append-only discipline for required fields and interpretation fields.
3. Verify all evidence status, quality, and failure-reason fields remain explicit.
4. Verify new fields are digest-bound where required and covered by tests.
5. Flag any implicit fallback or silent field repurposing.

## Repository Anchors

- `main/core/schema.py`
- `main/core/schema_extensions.py`
- `main/core/records_bundle.py`
- `configs/records_schema_extensions.yaml`
- `tests/test_records_schema_append_only_fields.py`
- `tests/test_new_fields_must_be_registered_append_only.py`
- `tests/test_schema_requires_interpretation.py`

## Output Contract

For each finding include:
- `issue`
- `schema_impact`
- `evidence`
- `static_or_runtime_required`
- `recommended_fix`
