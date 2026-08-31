# Geometry-V4 G1R: independently validated blind geometry

V4-G1R continues `geometry_v4_keyed_multiscale_sync_anchor_v1`; it is not V5.
The frozen G1 artifact remains `G1_METHOD_PARTIAL — NOT_PASSED`: 20/20 units
completed, its old gate passed 2/20, and its records SHA-256 is
`946eb98e2cc2d6ce867056ffc3f77764ec9ba63d386c364c911ec18aa5f333f5`.
Those seeds, files, artifact, thresholds, and verdict are historical read-only
evidence and are neither modified nor rerun by G1R.

The later development artifact at source
`a3877d6491308057547caf5993ec9bb1629e791c` is also retained as failed method
evidence; its records SHA-256 is
`df2f9957d73b0eac9172fba2ec5f5be76168c750bc436905041455292cdab20f`.
All 20 units completed with zero unsafe arm, but correct safe `RELIABLE` was
0/20 and final observability was 2/4. Its post-freeze truth probes found no
valid truth-H fit or holdout; fit support was fixed by source at 0, 0, 0, and
2 across each source's five attacks. Paired residual scores separated the
correct domains from wrong keys, while blind evidence remained host-dominated.
The opponent-carrier development artifact at source
`40261d26580e439d76ff93d344ac806a8e4744fc` is likewise retained with records
SHA-256 `006384488cc20646a26493185e94e0ab4c1a272c403c2b58d599853c2516a88c`.
It completed 20/20 units with zero unsafe/failure, but final observability was
1/4 and correct safe `RELIABLE` was 0/20. Truth-H fit and holdout passed 0/20;
18/20 truth-H fits had support zero and all 20 best translations at truth R/S
missed by more than .02. Both the diffuse-luma and opponent implementation
families are therefore `METHOD_UNRESOLVED`; neither artifact is rerun,
discarded, or treated as a positive result. The bounded change below targets
carrier sparsity and reference support, not color-axis or rank tuning.

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

The versioned v4 sparse-luma writer is one forward hook on the real `AutoencoderKL.decoder`
output, immediately before ordinary RGB postprocess. The clean arm has no hook;
the marked arm registers it only for its final decode, requires exactly one
invocation, and removes it in a `finally` block. There is no latent-adjoint
writer, final-RGB feedback, search, retry, fallback, or budget increase. The
fixed scalar update targets .25 of the luma RMS cap and is added equally to all
three RGB channels, so the fixed REC709-luma extractor receives exactly that
same scalar atlas. No image, key, content detector, or residual can choose or
adapt a color axis.

The original luma RMS 2/255 and peak 8/255 limits remain. In addition, every
actual RGB channel must have RMS no greater than the unchanged scalar target
`.25 * 2/255`, and no channel peak may exceed 8/255. PSNR above 40, SSIM above .98, and content
drift below .05 are unchanged; the writer is never content-adaptive. CPU tests use only a fake
decoder module; real final-RGB observability requires separate GPU authorization.

## Blind detector

The detector accepts only current attacked ordinary RGB and normalized key.
Search translation, local fit, and holdout correlation/PSR all read the same
fixed REC709-luma projection used by the writer; none chooses a channel from
the attacked image.
Truth, original or clean RGB, writer residuals, latents, and attack names are
forbidden. Every fixed coarse and fine rotation/scale hypothesis is ranked by
the keyed search reference's normalized complex cross-power phase-correlation
under the same valid mask and Hann window. Magnitude-only FFT evidence is not a
candidate control quantity. The search field is a deterministic keyed sparse
PRN atlas with signed Gaussian chips on fixed 8, 12, and 16 grids, four
independent groups per scale, fixed one-half cell duty, and scale-normalized
chip radius .20 of a cell. Coarse search uses all eight grid-8/grid-12 groups
and a fixed trimmed component consensus; fine search jointly uses all twelve
grid groups. No single low-energy component can substitute for the joint
atlas. Each component's normalized cross-power is restricted to its keyed
reference-derived near-exact support. For each hypothesis, three deterministic translated
peaks are selected inside the frozen physical bounds with two-pixel NMS; each
peak has its own PSR, phase consistency, and normalized narrow-band correlation.
The joint candidates use the frozen lexicographic rank and retain exactly the
fixed top K. Raw pixel sums are not core evidence. No candidate means
`UNRELIABLE`; search can never emit `RELIABLE` and has no fallback.

Each fit tile contributes at most one masked normalized-correlation match with
fixed cubic-polynomial local detrending followed by the frozen narrow band,
correlation at least .42, and margin at least .025. A deterministic robust
similarity fit must retain support at least 6, spatial coverage at least .75,
at least three macro regions, condition number at most 1e4, reprojection RMS at
most .02, inlier ratio at least .5, and strictly convex H-consistent corners.
Public H always maps attacked to canonical. Any translation or holdout PSR used
by a `RELIABLE` decision must be at least 8.

Fit and validation tile identities, coordinates, and checkerboard split remain
unchanged. Each tile uses its own fixed keyed signed-Gaussian PRN atlas on an
8x8 local grid with one-half cell duty. Local whitening is fixed cubic detrend
plus the same narrow band. Matched correlation reads only support whose
canonical keyed reference magnitude is at least .18 of its maximum; that mask
comes solely from key and canonical coordinates and is warped by the candidate
H, never selected from RGB. Every tile still supplies at most one match. Total
energy shares remain .40/.36/.24.

Fit uses the fixed divisor-20 local window so translated edge tiles remain
eligible without changing their canonical identity or any gate. Holdout uses
fixed divisor-20 and divisor-24 windows and computes its fixed-H correlation
only on radius 12--31 frequencies whose keyed-reference magnitude is at least
.10 of its maximum; this is fixed preprocessing, not per-image selection.

`k_validate` bytes are absent from search, rank, fit, and tie-breaking. Only
after H is frozen are the eight validation tiles evaluated for correlation,
margin, PSR, cross-scale consistency, coverage, and corner validity. Fit and
holdout gates must both pass. Changing validation bytes may therefore change
only `RELIABLE`/`UNRELIABLE`, never the selected candidate or H.

The public detector returns exactly `(H_hat, corners_hat, support, reliability,
status)`. Each blind arm's fixed top-K search summary, selected fit gates, and
holdout gates are captured once from that same detector execution as private,
JSON-safe engineering records. Fit diagnostics also preserve each tile's
pre-threshold best correlation and margin plus deterministic rejection counts.
They contain no key material, field patterns, truth, clean/original RGB, or
writer residual and cannot enlarge that public geometry interface.

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
domain scores only on the fixed REC709-luma projection of the paired final-RGB
residual `(marked-clean)`; this paired
writer evaluator is forbidden from the blind detector. It requires every
correct domain score to beat its corresponding wrong-key score, PSNR above 40,
SSIM above .98, luma RMS and peak within 2/255 and 8/255, per-channel RGB RMS
within `.25 * 2/255`, RGB peak within 8/255, and unchanged-content
score drift below .05. The real gate requires 4/4 observable sources, 20/20
safe correct-key recoveries, zero unsafe result in every arm, and zero retained
failure.

Truth is attached only after all arm outputs are frozen. A `RELIABLE` result is
unsafe if maximum mapped-corner error or center reprojection error exceeds .02,
rotation error exceeds 2 degrees, or absolute log-scale error exceeds .03.
`UNRELIABLE` is safe fail-closed; `RELIABLE` within every tolerance is harmless.
After that freeze and truth creation, the development runner records a
`truth_probe` for the correct key at the declared H: exact-R/S search and
translation evidence, fit pre-threshold tile evidence/support, and holdout
metrics. The probe is record-only, cannot rerun or replace any arm, and is
absent from candidate rank, H fitting, reliability, and summary calculation.
It serializes no key, phase, field, pattern, RGB, latent, or secret material.

The CPU engineering exit is synthetic-only with formal denominator zero: four
fixed ordinary-RGB carriers crossed with five attacks, at least 18/20 safe
correct recoveries and at least 3/4 per attack, zero correct unsafe, and zero
wrong unsafe. The later real GPU exit additionally requires 4/4 unique-source
final-RGB observability, 20 safe correct recoveries with all five attack types,
and zero unsafe across all arms. This window cannot execute or claim that GPU
exit, content flips, robustness, or a scientific result.
