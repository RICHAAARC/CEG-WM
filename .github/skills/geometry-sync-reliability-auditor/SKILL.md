---
name: geometry-sync-reliability-auditor
description: Audit geometry synchronization reliability and gating logic in latent space. Use when users ask for crop resize rotate robustness path checks, sync quality gate review, or attention anchor ordering validation.
---
# geometry-sync-reliability-auditor

Audit whether geometry evidence is emitted only after valid synchronization.

## Execute

1. Inspect sync template, geometric fitting, and attention anchor integration.
2. Verify geometry evidence appears only when sync confidence is sufficient.
3. Verify attention anchors are auxiliary and not used as primary sync substitutes.
4. Verify failed geometric fit yields explicit failure without content pollution.
5. Report unstable or pseudo-stable geometry evidence risks.

## Repository Anchors

- `main/watermarking/geometry_chain/sync/latent_sync_template.py`
- `main/watermarking/geometry_chain/align_invariance_extractor.py`
- `main/watermarking/geometry_chain/attention_anchor_extractor.py`
- `tests/test_attention_consistency_only_after_align_success.py`
- `tests/test_geo_score_absent_when_fit_fails.py`
- `tests/test_geometry_chain_absent_fail_no_content_pollution.py`

## Output Contract

- `issue`
- `geometry_reliability_risk`
- `evidence`
- `severity`
- `recommended_fix`
