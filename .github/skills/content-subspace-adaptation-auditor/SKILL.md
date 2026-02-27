---
name: content-subspace-adaptation-auditor
description: Audit content adaptive subspace behavior driven by semantic masks and frequency split roles. Use when users ask for high frequency low frequency role separation checks, subspace planner validity, or semantic mask binding audits.
---
# content-subspace-adaptation-auditor

Audit whether adaptive content subspace logic is truly active and traceable.

## Execute

1. Inspect semantic mask provider and subspace planner bindings.
2. Verify high-frequency and low-frequency channels follow distinct mechanisms.
3. Verify plan digests bind mask and configuration inputs.
4. Verify fallback behavior is explicit and auditable.
5. Report hardcoded or non-adaptive behavior that weakens claims.

## Repository Anchors

- `main/watermarking/content_chain/semantic_mask_provider.py`
- `main/watermarking/content_chain/subspace/subspace_planner_impl.py`
- `main/watermarking/content_chain/channel_hf.py`
- `main/watermarking/content_chain/channel_lf.py`
- `tests/test_subspace_planner.py`
- `tests/test_subspace_plan_digest_binds_mask_digest.py`
- `tests/test_semantic_mask_fallback_is_explicit.py`

## Output Contract

- `issue`
- `claim_impact`
- `evidence`
- `severity`
- `recommended_fix`
