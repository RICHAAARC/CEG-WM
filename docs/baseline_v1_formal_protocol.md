# Baseline-V1 formal protocol (user-frozen)

This protocol covers Tree-Ring, Gaussian Shading, Shallow Diffuse, and T2SMark
only. It excludes CEG and Geometry-V4 rows, results, thresholds, and attacks.
It freezes a measurement plan, not a completed scientific result.

## Threshold and clean confirmation

Each method freezes its own method-native threshold from **2,000 independent
clean unwatermarked-negative** `threshold_freeze` units. The threshold freezer
cannot read or receive feedback from the separate confirmation partition.

The primary FPR gate uses **3,000 independent clean unwatermarked-negative**
`clean_confirmation` units. It computes the exact one-sided 95% Clopper-Pearson
upper limit `BetaInv(0.95; FP + 1, n - FP)`, with the exact zero-count form
`1 - 0.05^(1/n)`. The formal clean-confirmation admission condition is
`upper_bound <= 0.001` (0.1%): at `n=3000`, `FP=0` passes (about 0.000998) and
`FP>=1` does not pass. `operating_point_violation` reports the complementary
condition, `one_sided_clopper_pearson_upper(FP, n) > 0.001`; the observed rate
`FP/n` is not the formal gate.

Not passing this predeclared admission condition preserves and reports every
result, the fixed denominator, every failure, and the interval. It forbids
threshold retuning, sample replacement, and result deletion. This condition is
not a claim that a real FPR has been absolutely proved; TPR is reported as **TPR
at a threshold calibrated for target FPR=0.1%**. Attacked negatives report their
own per-condition FPR and interval and never enter this clean-confirmation
denominator.

## Evaluation and attacks

Evaluation has 1,000 independent prompt × base-latent × sampling-seed physical
units. Each creates one clean/watermarked pair that shares prompt, base latent,
and seed. Every condition applies identically and deterministically to positive
and unwatermarked-negative images: clean; JPEG quality 50; resize to 50% then
bicubic restore; center crop retaining 80% area then restore; Gaussian blur
sigma 1.0 px; rotation 10 degrees.

Rotation uses attack ID `rotation_10_bicubic_reflect_center_crop_v1`: ordinary
uint8 sRGB H×W×3 input (`H,W>=3`) whose computed `p_x<W,p_y<H`, +10.0 degrees in Pillow's visual
counter-clockwise convention, pixel-center center `(W-1)/2,(H-1)/2`, bicubic
margin 2, NumPy `reflect` RGB padding (edge not repeated), Pillow bicubic rotate
on the padded canvas, and an exact original-size center crop. The valid-support
mask is padded constant-zero and rotated nearest-neighbor; reflected RGB outside
original support is therefore mask=0. Runtime records bind all formulas, actual
padding/crop, library versions, RGB/mask digests, and implementation exact/digest.
Inputs outside this aspect-safe NumPy-reflect domain fail closed. The runtime
resolves a clean Git checkout exact and verifies the implementation module blob
before emitting provenance, so callers cannot supply an arbitrary exact.

## Counts and failures

Per method: 2,000 threshold-freeze detections + 3,000 clean-confirmation
detections + 12,000 evaluation detections = **17,000 detections**. Planned
artifacts are 7,000 source-generation images, 10,000 non-clean attack
derivatives, and 6,000 paired quality comparisons. Across four methods this is
68,000 detections. Every observation has a 20-minute wall-clock cap; failures
and timeouts are retained in the planned physical denominator without retry or
replacement and reported per method/condition.

Wrong-key remains an optional diagnostic only. It is excluded from threshold
calibration, confirmation, and the main-table FPR.
