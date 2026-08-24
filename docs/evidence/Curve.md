# Curve Evidence

## Method identity

- Canonical method branch: `Curve`
- Protocol: `standalone-lf-hf-frequency-response-v1`
- Purpose: independently measure descriptive LF and HF score response under the frozen ordinary-RGB identity, JPEG, Gaussian blur, and Gaussian noise conditions.
- The LF and HF methods run independently at their full actual-callback-dtype relative-L2 budget. They are never co-injected or fused by this route.

## Preserved implementation provenance

- Reviewed resumable implementation: `b1a806a34a16435c4242e45eafa3818b3a37b8a6`
- Earlier divergent resumable implementation: `35143fa0ba6a40f69152f2aacef458902f2c188b`
- Their common historical contract correction: `94deb0489d0a765f9cd76f6872642ff4f9f72af9`
- The canonical `Curve` branch replays the reviewed implementation on the method-first mechanism baseline without the attack-complementarity route.

Both historical tips are parents of this evidence branch so their commits remain reachable after obsolete branch names are removed.

## Evidence status

- Repository implementation and lightweight tests are preserved.
- No authenticated terminal Curve artifact is registered in this branch at the time of consolidation.
- Scientific status: `not_adjudicated`.
- This branch does not establish a winner, complementarity, joint detection, calibrated threshold, fixed FPR, robustness, or promotion claim.

`Curve` and `Curve-Evidence` are independent experiment branches and must not be merged into `main`.
