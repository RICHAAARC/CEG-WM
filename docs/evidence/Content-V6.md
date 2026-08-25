# Content V6 evidence

## Method identity

Content V6 keeps the Content V3 unweighted LF write and adaptive HF path, uses
the Content V4 clean-null-whitened LF scorer, and adds detector-domain ISS. A
callback-free first pass is the sole primary null; the independently reset
same-seed second pass is the sole embedded candidate. The finite ISS multiplier
is clamped to `[1, 2]`, applies only to the LF preprojection delta, and the
shared actual-dtype projector remains bounded by `0.012`. Blind scoring uses
only the final image, detection key, and frozen public assets; the formal joint
score remains `min(LF, HF)`.

The accepted ISS gain-target asset has SHA-256
`d66ff88640a3d1a020646cfde3face7502282bf835c9d3fb746b518dfb02c231`.

## Current-roster formal result

- Source exact: `49bb03ed697a47048f5730ecfc85a9f29cb0b58a`
- Run ID: `content-v6-855fb511afa2-805bc21e173a`
- Protocol ID: `cegwm-stage-a-content-v6-detector-domain-iss-clean-v1`
- Protocol digest: `855fb511afa23548c30a5fcad17525589b340aac7067ae3491941fc8fc99427d`
- Public-key digest: `805bc21e173a83898f3b7034d75e6ed02f65894a6885377d9659ee3091b4dd77`
- Roster SHA-256: `20058788bfe7d75878e7263efda2b8de94c6fdcd3a963f64368f2ba4d594868f`
- Formal archive SHA-256: `bb187802b8e376f0c1f00d740cdb7010068f49f6a46fea578bdf64aefa71fbb1`

The authenticated RC0 artifact completed all 8 units and 16 records without a
failed unit. Mechanical requirements passed 8/8. Independent recomputation
found LF Gate A/B `7/8`, `8/8`; HF Gate A/B `8/8`, `8/8`; and joint Gate A/B
`8/8`, `8/8`, with no ties. Its preregistered conjunction therefore passed.

## Old-roster reference result

- Source exact: `39720994cc3316af8c2cac586689d0811232b4c7`
- Run ID: `content-v6-reference-oldroster-c98175252406-805bc21e173a`
- Protocol ID: `cegwm-stage-a-content-v6-detector-domain-iss-reference-oldroster-v1`
- Protocol digest: `c98175252406cce147b329b016fe3f6acb62b2ed1ba5bba66ca9fea5ae37fa80`
- Public-key digest: `805bc21e173a83898f3b7034d75e6ed02f65894a6885377d9659ee3091b4dd77`
- Roster SHA-256: `dd30c719ae5a48b2a9a652420a3237adb74ffd26af8bac90e25c1d03fe845b88`
- Formal archive SHA-256: `0d7bc51b34a39b4acf9cbe80a9b10adaf58f9405f494dcdaaa19dfba20b03e50`

This independent RC0 reference also completed 8/8 units and 16/16 records and
passed every mechanical requirement. LF Gate A/B was `6/8`, `8/8`; HF Gate
A/B was `8/8`, `8/8`; and joint Gate A/B was `6/8`, `8/8`. The reference
conjunction failed because both LF and joint Gate A were below `7/8`.

## Personal adjudication

Under the user's explicit V6/V7/V8 adjudication authorization, the exact
current-roster result is recorded as `CONTENT_CHAIN_MECHANISM_COMPLETE` only
for its exact source, protocol, key, roster, asset, and archive. The independent
old-roster reference is recorded as a narrow `SCIENTIFIC_NEGATIVE` only for its
exact bound reference protocol and roster.

The two denominators remain separate. Together they show a roster-specific
mixed outcome and do not establish cross-roster generalization. Both immutable
artifacts retain their original `scientific_status=not_adjudicated`; the later
adjudication is recorded only on this Evidence branch. Neither result supports
a calibrated threshold, fixed-FPR, attack, robustness, geometry, Stage/main,
paper, retry, tuning, replacement, or promotion claim.

## Branch role

`Content-V6` is the canonical method branch. `Content-V6-Evidence` is its
evidence leaf and must not be used for development or merged into `main`. The
current tree retains the latest old-roster formal Notebook plus two portable
scalar evidence packages. Earlier fit, canary, and current-roster handoff
commits remain reachable through branch history but are omitted from the
current tree.
