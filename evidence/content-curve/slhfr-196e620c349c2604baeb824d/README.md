# Content-Curve scalar evidence package

This directory preserves the exact scalar JSON members of the authenticated
terminal artifact for `slhfr-196e620c349c2604baeb824d`.

- `receipt.json` is the exact 1,399-byte artifact member.
- `result.json` is the exact 369,852-byte artifact member, including all 320
  ordered records and all 17 registered/wrong-key scalar scores per record.
- `artifact_manifest.json` binds those files to the original ZIP and sidecar
  hashes without retaining the redundant ZIP container.

The package is sufficient to recompute response summaries or test a newly
specified scalar decision rule. It contains no image, model secret, detection
key, token, latent, delta, mask, or private embed state. It does not preserve
the earlier 50/50 posthoc analysis and does not itself authorize a scientific
claim.

After cloning this branch, start from `result.json["records"]`. Records are
ordered by unit, condition, and the four declared arms. Each record's `scores`
object contains `registered` and `wrong_00` through `wrong_15`.
