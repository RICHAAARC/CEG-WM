# Content V5 scalar evidence package

This directory preserves the exact JSON members of the authenticated terminal
artifact for `content-v5-c5a0c4bf7d6d-805bc21e173a`.

- `receipt.json` is the exact umbrella artifact receipt.
- `result.json` contains the two independent cohort results: 16 ordered
  records and 8 unit aggregates for each cohort, with 51 LF/HF/joint scalar
  scores per record.
- `audit-state.json` is the exact terminal audit-only state member. It repeats
  public identities and committed scalar records; it is not a resume authority.
- `artifact_manifest.json` binds all three files to the original ZIP and
  sidecar and records an independent recomputation of the strict decisions.

The package contains no image, raw prompt text, raw key, token, latent, delta,
route, mask, tensor, or private embed state. The source ZIP remains external;
its exact hash, size, sidecar binding, and member hashes are recorded here.

For recomputation, use each entry under `result.json["cohort_results"]`. The
reference and primary cohorts remain separate 8-unit denominators. They must
not be pooled, and the diagnostic LF/HF counts must not be replaced with a raw
cross-branch score maximum.
