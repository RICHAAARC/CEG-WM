# Content V9 evidence boundary

Formal terminal artifact identity:

- Execution branch and exact: `Content-V9-Evidence@647e88862f2e8f8594b14a9e9e731161765729f8`.
- Run: `content-v9-stability-9bc8a94c1d02-63c17e8200a9-805bc21e173a`.
- Protocol digest: `9bc8a94c1d022cfaaf3c36018422b245e42764571314ee048d612e58a19ca031`.
- Public-key digest: `805bc21e173a83898f3b7034d75e6ed02f65894a6885377d9659ee3091b4dd77`.
- Terminal ZIP SHA-256: `d7a850db67398aab66aab74a48e84ba38ba48866d2f9eeb1f74a239e80382177`.

The artifact is complete (`rc=0`, 80 committed units and 160 records). Its
four independent physical strata have weighted-joint Gate A/B counts of
8/8, 8/8, 32/32, and 32/32 against required counts 7, 7, 28, and 28.
No strata are pooled and no cross-section conjunction or combined result is
present.

The manifest records the legacy `content_v9_calibration` Drive directory,
its two public files, their Drive IDs, sizes, hashes, sidecar binding, and the
verified byte equality to the calibration asset retained in this checkout.
The legacy directory is not required by the formal runtime.

`result.json` is preserved exactly with the runner-produced
`scientific_status=not_adjudicated`; the raw artifact is not rewritten by a
later decision. On 2026-08-26 the project owner accepted this exact V9 result
as the validated content-chain method. `Content-V9` was promoted through main
merge `2f5e7e5f80616eefb22c586ca7430038be45807b`, and main recorded
`content_chain_method_complete` at
`e9308f0611d472568124a3b61b87a2cff36f28ee`. This adjudication does not claim
calibrated FPR, attack robustness, generalization, geometry, full-system
completion, or paper readiness.

## Branch role

`Content-V9` is the canonical method branch. `Content-V9-Evidence` is its
complete executable evidence snapshot: it retains the canonical project,
configuration, runtime, protocol, and stability runner together with the
formal Notebook and portable scalar evidence. The Notebook clones this
Evidence branch directly. The Evidence branch is not a method-development base
and must not be merged into `main`.
