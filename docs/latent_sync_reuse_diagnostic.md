# Reuse-image geometry/content diagnostic

Run `python -m experiments.run_latent_sync_reuse_diagnostic` for its plan and
CPU coordinate audit. Add `--execute --output <new-directory>` for explicit
model execution. The default input is the existing Drive directory
`/content/drive/MyDrive/CEG-WM/development/latent-sync-v1/diagnostic-20260909-152107-462244`;
`--input-dir` can point to a copy containing `plain.png`, `content_only.png`,
`latent_anchor.png`, and `v2_rgb_sync.png`. Input images are read only.

There are zero new generations. Each source has eight fixed content-score rows:

1. Clean RGB without correction.
2. Clean RGB with an actual identity `rectify_once(I)` call.
3. Clean RGB with the image-only reader's predicted correction.
4. Exact V2 +10 attacked RGB without correction.
5. Attacked RGB corrected with the actual renderer's known H.
6. Attacked RGB corrected with the image-only reader's predicted H.
7. Attacked RGB corrected with true angle minus one degree.
8. Attacked RGB corrected with true angle plus one degree.

All corrections are one resampling directly from the clean or attacked image,
never from a previously corrected result. Rows 5, 7 and 8 use truth only for
diagnosis. They do not select a detector transform or enter the complete-path
score. The latter is explicitly `max(raw pre, predicted post)`; individual raw
scores remain separate. The fixed +/-1 degree probes are coarse observations,
not precise tolerance limits. No threshold or FPR is estimated.

The named V2 +10 renderer has forward screen angle -10 degrees around integer
pixel centre (255,255). Its actual H is recovered from the existing Pillow
sampler. The current reader uses latent centre (31.5,31.5), mapping to RGB
(255.5,255.5); exact-angle correction still differs by about 0.123257 pixels
in translation. The diagnostic records this and does not alter either method.
Truth +/-1 uses the real centre (255,255), giving forward angles -11 and -9.

Eight reader calls and 32 content scorer calls are planned, with no caching:
40 VAE encodes when complete. The scorer uses the unchanged V2 shared-observation
branch function and registered-minus-max16wrong arithmetic. Each row retains all
17 key values for LF, HF and the weighted joint, plus the registered joint,
wrong-key maximum and raw final scalar. Only the public VAE and scorer assets
load; no diffusion transformer, text encoder, DINO, SyncSeal or generation is
needed. Version/code revision are recorded. Completed reader/content rows flush
incrementally to JSONL; failed or unavailable inputs remain in the 8/32 rows.

This diagnoses final-RGB geometry/interpolation and VAE re-encoding. It cannot
separately identify loss at step 18 versus remaining step 19 from final PNGs.
The historical injection and sampling are not replayed, and no private latent
is passed to detection. Existing negative/plain and no-anchor/content-only
controls remain in the full matrix.

Local CPU validation: 21 targeted tests passed. On the four copied real PNGs,
identity resampling was pixel-identical for all four. Full-RGB MSE after true-H
correction was 0.00736--0.00739 versus 0.11885--0.11909 when deliberately passing
inverse H; these are pixel-direction diagnostics, not content detection results.
No VAE or content scoring was run locally.
