# Geometry-V4 G1R: independently validated blind geometry

V4-G1R continues `geometry_v4_keyed_multiscale_sync_anchor_v1`; it is not V5.
The frozen G1 artifact remains `G1_METHOD_PARTIAL — NOT_PASSED`: 20/20 units
completed, its old gate passed 2/20, and its records SHA-256 is
`946eb98e2cc2d6ce867056ffc3f77764ec9ba63d386c364c911ec18aa5f333f5`.
Those seeds, files, artifact, thresholds, and verdict are historical read-only
evidence and are neither modified nor rerun by G1R.

Using only the predeclared normalized-diagonal .02 safety tolerance, the frozen
outputs reclassify as correct: 19 `RELIABLE` (0 safe, 19 unsafe) and 1
`UNRELIABLE`; wrong: 18 `RELIABLE` (0 harmless, 18 unsafe) and 2 `UNRELIABLE`.
All 20 records completed without exception. Final observability passed only two
of four unique sources: seed 6102 had correct anchor below wrong anchor and seed
6103 had content drift .099 above .05. The artifact has no s0, s1, or tau, so
no content decision or flip can be inferred.

Geometry remains coordinate-only. Its complete output is
`(H_hat, corners_hat, support, reliability, status)`. `RELIABLE` means only
that applying the attacked-to-canonical H is safe. Geometry cannot vote for a
watermark, alter content scores, or supplement the unchanged content detector,
key, preprocessing, or threshold. A wrong key is an independent stress arm,
not a classification negative; a harmless wrong-key `RELIABLE` result is not
itself failure.

## Independent domains and fixed budget

The existing normalized detection key enters the unchanged Geometry-V4 HKDF
root. G1R derives `k_search`, `k_fit`, and `k_validate` using the three fixed
labels in the canonical contract. The content key is unchanged and is never
derived from Geometry. Search, fit, and validation consume respectively .40,
.36, and .24 of one unchanged anchor budget. The 4x4 canonical grid is split by
checkerboard into eight fit and eight validation tiles; each partition covers
all four 2x2 macro regions. The global search constellation and the two
spatially disjoint local partitions are constructed independently: each field
reads only its own derived domain key, so changing one key cannot change either
of the other two fields. Tile identity and coordinates never depend on RGB.

The versioned G1R writer is one forward hook on the real `AutoencoderKL.decoder`
output, immediately before ordinary RGB postprocess. The clean arm has no hook;
the marked arm registers it only for its final decode, requires exactly one
invocation, and removes it in a `finally` block. There is no latent-adjoint
writer, final-RGB feedback, search, retry, fallback, or budget increase. The
update may not exceed luma RMS 2/255 or peak 8/255. CPU tests use only a fake
decoder module; real final-RGB observability requires separate GPU authorization.

## Blind detector

The detector accepts only current attacked ordinary RGB and normalized key.
Truth, original or clean RGB, writer residuals, latents, and attack names are
forbidden. Search uses normalized narrow-band and normalized cross-power phase
evidence over fixed physical bounds and retains exactly the fixed top K. Raw
pixel sums are not core evidence. No candidate means `UNRELIABLE`; search can
never emit `RELIABLE` and has no fallback.

Each fit tile contributes at most one masked normalized-correlation match with
fixed cubic-polynomial local detrending followed by the frozen narrow band,
correlation at least .42, and margin at least .025. A deterministic robust
similarity fit must retain support at least 6, spatial coverage at least .75,
at least three macro regions, condition number at most 1e4, reprojection RMS at
most .02, inlier ratio at least .5, and strictly convex H-consistent corners.
Public H always maps attacked to canonical. Any translation or holdout PSR used
by a `RELIABLE` decision must be at least 8.

`k_validate` bytes are absent from search, rank, fit, and tie-breaking. Only
after H is frozen are the eight validation tiles evaluated for correlation,
margin, PSR, cross-scale consistency, coverage, and corner validity. Fit and
holdout gates must both pass. Changing validation bytes may therefore change
only `RELIABLE`/`UNRELIABLE`, never the selected candidate or H.

The public detector returns exactly `(H_hat, corners_hat, support, reliability,
status)`. Each blind arm's fixed top-K search summary, selected fit gates, and
holdout gates are captured once from that same detector execution as private,
JSON-safe engineering records. They contain no key material, field patterns,
truth, clean/original RGB, or writer residual and cannot enlarge that public
geometry interface.

## Rosters and evidence ceiling

Development is the fixed 4x5 roster at seeds 6201--6204; confirmation is the
separate fixed 4x5 roster at seeds 6301--6304. Their prompts are distinct from
each other and from legacy G1. Seeds 6101--6104 are rejected. A runner must
execute exactly one full split in order with no subset, mixing, retry,
replacement, or dropped failure.

The real CLI permits only `development`. It generates each of the four unique
sources once as a clean/marked G1R pair and applies all five attacks afterward,
retaining correct-key, wrong-key, and unwatermarked-negative blind arms for all
20 units. Confirmation is rejected by this CLI. Its create-only artifact
directory contains full records, summary, and manifest JSON files, each with
an independent SHA-256 sidecar. Source observability measures all three G1R
domain scores only on the final-RGB luma residual `(marked-clean)`; this paired
writer evaluator is forbidden from the blind detector. It requires every
correct domain score to beat its corresponding wrong-key score, PSNR above 40,
SSIM above .98, luma RMS and peak within 2/255 and 8/255, and unchanged-content
score drift below .05. The real gate requires 4/4 observable sources, 20/20
safe correct-key recoveries, zero unsafe result in every arm, and zero retained
failure.

Truth is attached only after all arm outputs are frozen. A `RELIABLE` result is
unsafe if maximum mapped-corner error or center reprojection error exceeds .02,
rotation error exceeds 2 degrees, or absolute log-scale error exceeds .03.
`UNRELIABLE` is safe fail-closed; `RELIABLE` within every tolerance is harmless.

The CPU engineering exit is synthetic-only with formal denominator zero: four
fixed ordinary-RGB carriers crossed with five attacks, at least 18/20 safe
correct recoveries and at least 3/4 per attack, zero correct unsafe, and zero
wrong unsafe. The later real GPU exit additionally requires 4/4 unique-source
final-RGB observability, 20 safe correct recoveries with all five attack types,
and zero unsafe across all arms. This window cannot execute or claim that GPU
exit, content flips, robustness, or a scientific result.
