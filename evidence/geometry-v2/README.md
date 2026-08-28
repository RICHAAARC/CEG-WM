# Geometry-V2 Evidence

This independent evidence branch freezes the final operational record for `geometry_v2_keyed_neural_corner_sync` without changing the method branch.

## Bound result

- source branch: `Geometry-V2@bc147f8985d5e54477d0bbd47f7d44a73f70a6e6`;
- execution: `d82efc292db8a16f60d272635e577a4186ed866a`;
- run: `geometry-v2-neural-corner-sync-n0-d82efc292db8`;
- protocol: `geometry-v2-keyed-neural-corner-sync-n0-v1`;
- artifact: complete;
- N0 status: `N0_UNRESOLVED`;
- final semantic status: `OPERATIONAL_UNRESOLVED`;
- evidence ceiling: operational evidence only, `science_denominator=0`.

The fixed run trained on 128 procedural images, observed 32 validation images under four attacks, and retained 32 independent confirmation images under the same four attacks. All 128 confirmation units were calculated and none failed. The public candidate gate was not met because the median error, p95 error, and reliable fraction failed their frozen bounds; the complete-unit and residual-bound terms passed.

## Files

- `index.json`: package identity and bounded-file map.
- `n0/provenance.json`: Drive folder/file IDs, URLs, bytes, and independently computed SHA-256 values.
- `n0/receipt_summary.json`: bounded public method/run/training/gate facts extracted from the three source sidecars.
- `n0/metrics_public.jsonl`: exact 28,698-byte public metrics payload, 128 ordered records.
- `n0/comparison_summary.json`: independently recomputed overall and per-attack statistics for later comparisons.

No source image, secret material, access token, trained parameter, checkpoint, prompt, latent, or non-public local path is stored here. This branch is evidence-only and has policy `do_not_merge_into_Geometry-V2`.
