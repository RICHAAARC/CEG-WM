# Content V3 scalar evidence package

This directory preserves the exact scalar JSON members of the authenticated
terminal artifact for `content-v3-6b812bbef380-805bc21e173a`.

- `receipt.json` is the exact artifact receipt.
- `result.json` contains all 16 ordered records, all 51 LF/HF/joint
  registered/wrong-key scores per record, the 8 unit aggregates, operational
  status, and the original frozen Gate evidence.
- `artifact_manifest.json` binds both files to the original ZIP/sidecar hashes
  and records an independent recomputation of the original strict Gates.

This is the raw scalar evidence needed to recompute a newly specified public
decision statistic without Google Drive. It contains no image, prompt, raw
key, token, latent, delta, route, mask, or private embed state. The earlier
50/50 posthoc analysis is intentionally not included.

For recomputation, use `result.json["records"]`. Each record exposes
`lf__*`, `hf__*`, and `joint__*` scores for `registered` and `wrong_00` through
`wrong_15`; candidate and primary-null records are distinguished by `arm`.
