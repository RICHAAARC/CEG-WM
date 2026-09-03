# Baseline-V1 formal protocol (user-frozen)

This protocol covers Tree-Ring, Gaussian Shading, Shallow Diffuse, and T2SMark
only. It excludes CEG and Geometry-V4 rows, results, thresholds, and attacks.
It freezes a measurement plan, not a completed scientific result.

## Threshold and clean confirmation

Each method freezes its own method-native threshold from **2,000 independent
clean unwatermarked-negative** `threshold_freeze` units. The threshold freezer
cannot read or receive feedback from the separate confirmation partition.

The clean-negative test uses **3,000 independent clean unwatermarked-negative**
`clean_confirmation` units. It reports `FP/3000`, observed FPR, and the exact
two-sided 95% Clopper-Pearson interval. It may also compute the one-sided 95% Clopper-Pearson
upper limit `BetaInv(0.95; FP + 1, n - FP)`, with the exact zero-count form
`1 - 0.05^(1/n)`, as a diagnostic. `operating_point_deviation` records whether
that diagnostic exceeds 0.001. Neither the observed FPR, interval, nor UCB is an
admission gate.

Every operating point preserves and reports every result, the fixed denominator,
every failure, and the interval. No value authorizes threshold retuning, sample
replacement, result deletion, or result-package suppression. TPR is reported as **TPR
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

`jpeg_q50` is a Pillow RGB JPEG round trip with `format=JPEG`, `quality=50`,
`subsampling=2` (4:2:0), `optimize=False`, and `progressive=False`.
`resize_50_bicubic_restore` computes `max(1, round(W*0.50))` and
`max(1, round(H*0.50))` using Python's ties-to-even `round`, downsamples with
Pillow BICUBIC, then restores to `(W,H)` with Pillow BICUBIC.
`center_crop_80_restore` uses linear scale `sqrt(0.80)`, computes each crop size
as `max(1, round(dimension*sqrt(0.80)))` with the same rounding, takes
`(left, top, left+crop_w, top+crop_h)` where `left=(W-crop_w)//2` and
`top=(H-crop_h)//2`, then restores with Pillow BICUBIC. `gaussian_blur_sigma_1px`
uses `ImageFilter.GaussianBlur(radius=1.0)`; this protocol names that radius
`sigma_px=1.0`. Each attack records its actual sizes/crop and frozen parameters.
Pillow and JPEG codec output bytes may vary by library version; this protocol
does not claim cross-version byte identity. Attack execution records the frozen
scientific parameters and actual geometry. A surrounding runner may optionally
record library versions, digests, or implementation exacts, but they are not
attack execution prerequisites or validator hard gates.

Rotation uses attack ID `rotation_10_bicubic_reflect_center_crop_v1`: ordinary
uint8 sRGB H×W×3 input (`H,W>=3`) whose computed `p_x<W,p_y<H`, +10.0 degrees in Pillow's visual
counter-clockwise convention, pixel-center center `(W-1)/2,(H-1)/2`, bicubic
margin 2, NumPy `reflect` RGB padding (edge not repeated), Pillow bicubic rotate
on the padded canvas, and an exact original-size center crop. The valid-support
mask is padded constant-zero and rotated nearest-neighbor; reflected RGB outside
original support is therefore mask=0. Runtime records bind the frozen formulas
and actual padding/crop. Inputs outside this aspect-safe NumPy-reflect domain
fail closed. A surrounding runner may optionally record versions, digests, or
implementation identity, but attacks execute in dirty, notebook, and non-Git
environments without those records.

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

## Final baseline robustness-table contract

The machine-readable final artifact uses this ordered long-table contract:
`baseline_id`, `threshold_identity`, `attack_family`, `attack_condition`,
`planned_positive_units`,
`observed_positive_units`, `failed_positive_units`, `planned_negative_units`,
`observed_negative_units`, `failed_negative_units`, `true_positive`,
`false_negative`, `false_positive`, `true_negative`, `tpr`, `tpr_ci95_lower`,
`tpr_ci95_upper`, `fpr`, `fpr_ci95_lower`, `fpr_ci95_upper`,
`clean_confirmation_false_positives`, `clean_confirmation_negatives`,
`clean_confirmation_failure_count`, `clean_confirmation_ucb95`,
`operating_point_deviation`, `status`.

The paper primary table has one `baseline_id` row and six condition columns in
the frozen order: clean, JPEG Q50, 50% bicubic restore, 80% center-crop restore,
Gaussian blur sigma 1.0 px, and rotation
`rotation_10_bicubic_reflect_center_crop_v1`. Each condition cell reports TPR,
its exact two-sided 95% Clopper-Pearson interval, unwatermarked-negative FPR,
its exact two-sided 95% Clopper-Pearson interval, positive/negative failure
counts, and their fixed planned denominators. All intervals are descriptive and
nonblocking.

The clean-confirmation presentation reports `FP/3000`, its exact two-sided 95%
interval, and optionally the one-sided UCB plus nonblocking
`operating_point_deviation`. If a condition has failures, it reports planned,
scored, failed, and missing counts, coverage, the scored-only conditional rate
and interval, and planned-denominator best/worst bounds. For negatives these
bounds are `FP/N_planned` through `(FP+failed+missing)/N_planned`; positives use
the analogous TP bounds. The status is `INCOMPLETE_OPERATIONAL`, but valid rows
remain usable. Failures are never converted to TN/FN, deleted, or removed from
the planned denominator. Wrong-key remains supplementary. Quality is exactly
PSNR, SSIM, and LPIPS on the existing clean pairs and is not a gate.

## Recovery and publication

Formal work uses stable JOB_ID/RUN_ID, create-only per-unit terminal records,
append-only numbered checkpoints at every 25-unit shard end and at least every
two hours, and a create-only final result published last. A completed score is
never rerun. Only typed `CUDA_OOM_TRANSIENT` and `MODEL_RUNTIME_TRANSIENT`
failures may retry the identical unit once, for two total attempts; all attempts
remain in the unit record. There is no lock, lease, heartbeat, force-rerun-all,
replacement unit, alternate RUN_ID, checksum, receipt, signature, or byte-size
gate. One runtime per JOB_ID is an operator constraint.
