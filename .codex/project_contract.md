# CEG-WM Minimal Project Contract

## Current authorization

- On 2026-11-08, after reviewing the measured reconstruction-direction mismatch, the user adopted advancement of the fixed blind reader / host-conditioned two-channel one-shot precompensation candidate. Implement the joint interval solver and the matched-information gradient-ray control, a four-new-source two-bit user-run entry, and CPU/static/fake checks. This adoption supersedes the earlier stage-A-only coding restriction for this separate candidate. The user subsequently explicitly requested pushing this delivery and providing a directly runnable notebook link; commit and publish this candidate branch accordingly. Real-model execution remains user-run, and merging is not authorized. The user explicitly selected quality reporting only: no PSNR eligibility gate or quality-based source exclusion.
- Candidate branch `dev/reconstruction-channel-v1`: on 2026-11-08 the user adopted advancement of the stage-A measurement prototype. Implement controlled full same-seed replay, ordinary-RGB feature reading, and channel measurements, with CPU/static/fake validation. Model execution needs an explicitly selected run configuration and is not performed by engineering checks. Shared-map fitting and watermark coding remain later stages.
- The user subsequently selected the fixed four-source Colab delivery and authorized committing and pushing this candidate branch. The notebook obtains published candidate code automatically; real model/GPU/Colab execution remains user-run. This does not authorize merging into main or expanding the measurement roster.
- The LF/HF status below describes the retained historical method, not the new candidate.
- Project stage: `content_chain_method_complete`.
- The calibrated weighted-joint method and its independent 8/8/32/32 stability strata complete the content chain.
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

## Governance boundary

`.codex/` and `governance/` are outer construction aids. Deleting them must not prevent importing the research package or running research-owned tests. Harness success is engineering evidence only.
