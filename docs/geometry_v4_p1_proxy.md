# Geometry-V4 P1 RGB proxy

P1D is a local NumPy ordinary-RGB mechanism test. It is not a generated-image,
robustness, fixed-FPR, or scientific result. P1C uses a disjoint frozen seed
split and is not executed by the P1D canary.

## Writer

The unchanged normalized detection-key bytes are the root input to the
Geometry-only HKDF domain. Twelve global components cover the Cartesian
directions 0/45/90/135 degrees at 8/16/32 cycles per image. Every
direction/scale component has its own keyed phase and sign. Sixteen fixed 4x4
canonical tiles likewise have keyed local identities without changing their
public coordinates.

Global and local fields are separately mean-centred and RMS-normalized. The
local field is projected off the global field, then the two orthogonal fields
are combined with squared-L2 fractions 0.40 and 0.60 and jointly normalized
once. The final clipped ordinary RGB is measured with Rec.709 luma; its RMS is
bounded by 2/255 and its peak by 8/255. No raw key, derived key, phase, sign, or
pattern is emitted in a writer or runner record.

## Blind detector

The detector accepts only current attacked ordinary RGB and the supplied
detection key. It obtains a coarse rotation/scale estimate by normalized
cross-power phase correlation of log-polar spectral magnitudes, refines that
estimate on a fixed public neighbourhood, rectifies rotation/scale, then uses
Cartesian normalized cross-power phase correlation for translation. Fixed
canonical tile templates are matched in the rectified attacked image. Those
measured correspondences, not a fixed count, enter a deterministic robust
similarity fit. It enumerates every two-point hypothesis, ranks by inlier count,
weighted inlier RMS, and lexicographic tile IDs, then refits only the selected
inliers. Support, macro regions, and normalized convex-hull spatial coverage are
computed from those inliers. Each scale obtains its own keyed-band log-polar
normalized-phase R/S estimate and fixed-neighbourhood refinement. A
quality-weighted circular/log-scale consensus uses bounded search/refinement
estimates only to form the coarse rectification. Separately recorded, unclipped
raw log-polar rotations and raw log-scales alone enter the frozen spread gates;
rotation spread uses 180-degree periodic distances. The rectified valid-overlap
mask excludes fill-only samples. The top-level PSR supplied to the frozen
reliability gate is the valid-masked Cartesian phase-correlation translation
PSR. Tile matching uses zero-mean normalized correlation and records a real
per-tile sidelobe PSR as a local-match diagnostic only: it currently does not
participate in match acceptance, aggregate reliability, or any frozen
reliability gate. Only valid measured matches may contribute to deterministic
macro-balanced inliers, support, or coverage. These measured diagnostics are
passed to the frozen P0
`reliability_is_reliable` gate and a
`GeometryV4Observation`; geometry never produces positive watermark evidence.
There is no original/residual/truth input, oracle initialization, retry, or
fallback in this path.


The public `H_hat` direction is attacked-to-canonical. `corners_hat` maps the
attacked-image TL/TR/BR/BL unit-square corners into canonical coordinates. A
consumer rectifies by inverse sampling with that public transform. Internal
canonical-to-attacked correspondences are inverted before the public
observation is constructed. The fixed scale search is 0.65 through 1.55, which
contains every frozen attack, including the 1/0.7 crop-rescale construction.

## Runner and evidence boundary

The runner validates the canonical P1 config digest and creates every source
with the sole `geometry_v4_procedural_rgb_v1` generator. Its public source ID,
seed, shape, and ordinary-image SHA-256 identity are bound into each record.
P1D seeds are 4101..4108 and P1C seeds are 4201..4208. A full mode accepts no
external image mapping or attack subset and internally constructs exactly all
8x16 units. `engineering_canary` is a separate non-formal identity requiring an
explicit subset; it never claims P1D/P1C full membership or their denominator.
The current development-canary API accepts only seeds 4101..4108. Every P1C
seed, including a mixed P1D/P1C request, is rejected before source generation;
a future P1C canary requires separate authorization after P1D freeze.

Every planned physical unit retains a marked correct-key arm, a matching
attacked-unwatermarked negative, a same-unit external wrong-key control, and
any failure. Upstream failures materialize all three arms as `STOPPED` records.
Correct and wrong keys must normalize to different bytes before execution.
Attack results retain opaque truth objects while the marked correct-key,
unwatermarked-negative, and wrong-key blind detector outputs are frozen. Only
then may truth consistency and attacked-to-canonical coordinate error be read;
a mismatch cannot initialize, alter, or prematurely stop those detector arms.
The present local canaries do
not execute the full 8x16 P1D or any P1C outcome.

## Frozen development canary

This iteration's non-formal engineering canary is frozen in the canonical P1
config before execution: P1D seeds 4101 and 4102 crossed with identity,
rotation -5/+5, scale 0.9/1.1, horizontal translation -0.10/+0.10, and
crop-rescale 0.9. It is a fixed 16-unit diagnostic roster, never a replacement
for the 128-unit P1D denominator, and must not be changed in response to its
outcomes. P1C is neither selected nor inspected.
