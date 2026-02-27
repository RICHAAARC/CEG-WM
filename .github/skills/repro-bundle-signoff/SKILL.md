---
name: repro-bundle-signoff
description: Build and audit reproducibility signoff bundles with config digests, environment fingerprints, artifact manifests, and policy anchors. Use when users ask for reproducibility package checks, signoff readiness, or release-grade audit bundles.
---
# repro-bundle-signoff

Prepare and verify signoff-grade reproducibility bundles.

## Execute

1. Collect required provenance anchors: config digest, environment fingerprint, policy path, and run metadata.
2. Verify artifact manifests and pointers are complete and consistent.
3. Verify reports contain required anchor fields and stable naming.
4. Run targeted reproducibility and signoff tests.
5. Produce a release gate decision with blockers and remediation actions.

## Repository Anchors

- `scripts/run_repro_pipeline.py`
- `scripts/run_freeze_signoff.py`
- `main/core/env_fingerprint.py`
- `main/core/digests.py`
- `tests/test_repro_bundle_manifest_and_pointers.py`
- `tests/test_signoff_profile_paper_includes_repro_audits.py`
- `tests/test_repro_pipeline_minimal.py`

## Output Contract

For each blocker include:
- `blocker`
- `artifact_or_anchor`
- `evidence`
- `release_risk`
- `required_fix`
