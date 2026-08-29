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
measured correspondences, not a fixed count, determine support and a
deterministic least-squares similarity fit. Its measured diagnostics are passed
to the frozen P0 `reliability_is_reliable` gate and a
`GeometryV4Observation`; geometry never produces positive watermark evidence.
There is no original/residual/truth input, oracle initialization, retry, or
fallback in this path.

## Runner and evidence boundary

The runner validates the canonical P1 config digest before enumerating the 16
frozen attacks. P1D seeds are 4101..4108 and P1C seeds are 4201..4208. Every
physical image-by-attack unit retains a marked correct-key arm, a matching
attacked-unwatermarked negative, a same-unit external wrong-key control, and
any failure. Attack truth is used only after all detector calls to compute
coordinate error. Missing or failed units stay in the enumerated denominator.
The present local canaries do not execute the full 8x16 P1D or any P1C outcome.
