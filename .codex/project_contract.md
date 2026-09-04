# CEG-WM Minimal Project Contract

## Current authorization

- Project stage: `content_chain_method_complete`.
- The calibrated weighted-joint method and its independent 8/8/32/32 stability strata complete the content chain.
- Formal experiment state: `FORMAL_EXPERIMENT_CONTRACT_FROZEN / EXECUTION_READY / EXECUTION_NOT_AUTHORIZED`.
- `EXECUTION_READY` means only that the frozen formal entries, remotely available producer exacts, and engineering-canary prerequisites are closed. It does not authorize Colab/GPU/Drive execution, any formal denominator, a paper result, or merge to `main`.
- No paper calibration or evaluation result exists yet. The content-chain stage does not itself claim geometry completion, fixed-FPR performance, attack robustness, paper conclusions, or completion of the full CEG-WM system.

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
10. Each formal method calibrates independently on exactly 2,000 clean unwatermarked negatives at target alpha 0.001: nearest-rank `k=1998`, `tau=c_(1998)`, and strict `score > tau`. Attacked negatives and all test outcomes are excluded from threshold selection.
11. Geometry-Direct/SyncSeal supplies only reference-frame or coordinate recovery. It is not a positive vote or reliability gate; the same content statistic and threshold remain the sole watermark authority before and after any rectification.
12. The proposed-method-only N=100 reconstruction experiment is a supplementary stress test at approximately 1% negative-sample resolution. It is excluded from the five-method main table and cannot support comprehensive generative-attack robustness or baseline-fairness claims.
13. Rotation plus scale is excluded from the formal table.

## Governance boundary

`.codex/` and `governance/` are outer construction aids. Deleting them must not prevent importing the research package or running research-owned tests. Harness success is engineering evidence only.

Formal execution and merge remain separate user-authorized actions. Until the user explicitly authorizes formal execution, no `N_cal`, `N_clean_test`, `N_pair`, or reconstruction denominator may run.
