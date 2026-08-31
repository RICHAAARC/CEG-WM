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
`1 - 0.05^(1/n)`. The gate passes only when that bound is at most 0.001. At
`n=3000`, `FP=0` gives about 0.000998 and passes; `FP>=1` fails. Confirmation
outcomes never adjust a frozen threshold. Attacked negatives report their own
per-condition FPR and interval and never enter this clean-confirmation gate.

## Evaluation and attacks

Evaluation has 1,000 independent prompt × base-latent × sampling-seed physical
units. Each creates one clean/watermarked pair that shares prompt, base latent,
and seed. Every condition applies identically and deterministically to positive
and unwatermarked-negative images: clean; JPEG quality 50; resize to 50% then
bicubic restore; center crop retaining 80% area then restore; Gaussian blur
sigma 1.0 px; rotation 10 degrees.

Rotation fill/crop remains the one execution blocker: it is recorded as
`pending_user_freeze`, and an execution must fail closed until the user chooses
the policy. No implicit library default is allowed.

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
