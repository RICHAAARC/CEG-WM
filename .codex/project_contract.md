# CEG-WM Minimal Project Contract

## Current authorization

- Project stage: `stage_a_method_feasibility_authorized`.
- Substantive Stage-A method, runtime, protocol, runner, configuration, and lightweight test work is allowed.
- This stage does not claim that LF, HF, routing, geometry, fixed-FPR detection, or the complete CEG-WM method has been scientifically validated.

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
