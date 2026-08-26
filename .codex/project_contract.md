# CEG-WM Minimal Project Contract

## Current authorization

- Project stage: `content_chain_method_complete`.
- `Content-V9@d9cd6932c3e9532453511203c5a2f5fcbefe8428` is the validated and accepted content-chain method promoted to `main`.
- The promotion is bound to `Content-V9-Evidence@cb3d30d4c3ee498ab4696291124d899aeacc685c` and terminal archive SHA-256 `d7a850db67398aab66aab74a48e84ba38ba48866d2f9eeb1f74a239e80382177`.
- The V9 calibrated weighted-joint method and its independent 8/8/32/32 stability strata are accepted as completing the content chain.
- This stage does not claim geometry completion, fixed-FPR calibration, attack robustness, paper readiness, or completion of the full CEG-WM system.

## Research invariants

1. The protected property is attribution to a detection key, not payload recovery or platform attestation.
2. Content statistics are the only positive authority.
3. Formal detection is blind: image + detection key + frozen public assets + calibrated threshold.
4. Wrong-key attribution and unwatermarked primary-null FPR are distinct questions and remain separately reported.
5. LF must first show independent keyed attribution before it can be combined with HF.
6. LF/HF roles and allocation are empirical questions; no historical fixed weight is inherited.
7. Geometry cannot directly create a positive and cannot recover deleted crop content.
8. Any future rectified decision must reuse the same detector, key semantics, preprocessing, and threshold.
9. Failures remain in the fixed denominator unless an exclusion was defined before execution and is independent of method outcome.

## Historical boundary

`../CEG-WM-Archive` preserves the former repository, revisions, harness, implementations, and results. Nothing there is current authority unless a future change explicitly identifies the exact source revision, migrates the smallest required code, and revalidates it under this repository's method identity.

## Governance boundary

`.codex/` and `governance/` are outer construction aids. Deleting them must not prevent importing the research package or running research-owned tests. Harness success is engineering evidence only.
