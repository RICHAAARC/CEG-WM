# Content-Curve evidence

## Method identity

- Canonical method branch: `Content-Curve`
- Protocol: `standalone-lf-hf-frequency-response-v1`
- Purpose: independently measure descriptive LF and HF score response under the frozen ordinary-RGB identity, JPEG, Gaussian blur, and Gaussian noise conditions.
- The LF and HF methods run independently at their full actual-callback-dtype relative-L2 budget. They are never co-injected or fused by this route.

## Preserved implementation provenance

- Reviewed resumable implementation: `b1a806a34a16435c4242e45eafa3818b3a37b8a6`
- Earlier divergent resumable implementation: `35143fa0ba6a40f69152f2aacef458902f2c188b`
- Their common historical contract correction: `94deb0489d0a765f9cd76f6872642ff4f9f72af9`
- The canonical `Content-Curve` branch replays the reviewed implementation on the method-first mechanism baseline without the attack-complementarity route.

Both historical tips are parents of this evidence branch so their commits remain reachable after obsolete branch names are removed.

## Evidence status

- Repository implementation and lightweight tests are preserved.
- The authenticated RC0 terminal artifact is registered as a portable scalar
  evidence package at
  `evidence/content-curve/slhfr-196e620c349c2604baeb824d/`.
- The exact artifact completed all 8 units, 10 ordered conditions per unit, and
  320 records. Its `result.json` preserves all 5,440 registered/wrong-key
  scalar scores and the original descriptive summaries.
- Original ZIP SHA-256:
  `c947320b3e175c3a1b11563cc01e02f90cacb4d1a746bd70c720b49591991900`.
- Exact `receipt.json` SHA-256:
  `76297134b08ed24226dd71270f6fa5da87f29cb12ea6324bf428786fbde313aa`.
- Exact `result.json` SHA-256:
  `514e7afd77a5fec0faa0e69df91d2846033b63fe52f06664234cfa28aee5ad5c`.
- Scientific status: `not_adjudicated`.
- This branch does not establish a winner, complementarity, joint detection, calibrated threshold, fixed FPR, robustness, or promotion claim.

The committed scalar records permit later read-only recomputation under a new,
explicitly declared statistic. No image or private runtime state is archived,
and no posthoc 50/50 analysis is recorded here.

`Content-Curve` and `Content-Curve-Evidence` are independent experiment
branches and must not be merged into `main`. The historical `Curve` names were
replaced without changing the frozen method or artifact identities.
