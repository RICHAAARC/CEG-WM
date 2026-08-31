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
`1 - 0.05^(1/n)`. This interval is explanatory uncertainty, not a result-admission
or experiment-success gate. At `n=3000`, `FP=0` gives about 0.000998; `FP=1`
has a larger upper bound, but both complete results remain reportable.

The predeclared operating-point flag is instead the observed rate: set
`operating_point_violation` when `FP / 3000 > 0.001` (thus four or more false
positives). A violation preserves every result, forbids threshold retuning or
sample replacement, and is reported alongside the threshold statement: **TPR at
a threshold calibrated for target FPR=0.1%**. It must never be worded as a
confirmed true FPR bound. Attacked negatives report their own per-condition FPR
and interval and never enter this clean-confirmation denominator.

## Evaluation and attacks

Evaluation has 1,000 independent prompt × base-latent × sampling-seed physical
units. Each creates one clean/watermarked pair that shares prompt, base latent,
and seed. Every condition applies identically and deterministically to positive
and unwatermarked-negative images: clean; JPEG quality 50; resize to 50% then
bicubic restore; center crop retaining 80% area then restore; Gaussian blur
sigma 1.0 px; rotation 10 degrees.

Rotation uses attack ID `rotation_10_bicubic_reflect_center_crop_v1`: ordinary
uint8 sRGB H×W×3 input (`H,W>=3`), +10.0 degrees in Pillow's visual
counter-clockwise convention, pixel-center center `(W-1)/2,(H-1)/2`, bicubic
margin 2, NumPy `reflect` RGB padding (edge not repeated), Pillow bicubic rotate
on the padded canvas, and an exact original-size center crop. The valid-support
mask is padded constant-zero and rotated nearest-neighbor; reflected RGB outside
original support is therefore mask=0. Runtime records bind all formulas, actual
padding/crop, library versions, RGB/mask digests, and implementation exact/digest.

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
