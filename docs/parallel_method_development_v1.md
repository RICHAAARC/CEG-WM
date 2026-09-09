# Parallel method development V1

This development supersedes historical stage descriptions only within the newly
authorized A/B worktrees. V2 results supplied by the control task are the starting
point; this work does not claim to have repeated its raw-artifact audit.

A fixes V2 content embedding and scoring while replacing the old RGB sync carrier
and estimator with a step-18 latent anchor and a final RGB VAE observation. B fixes
V2 RGB sync and blind scoring while replacing only the allocator. Both routes keep
full-support keyed LF/HF carriers and content as the sole positive authority.

## Coordinates and observation

The development attack module uses H from reference RGB integer pixel centres to
observed RGB integer pixel centres. x points right, y down; a positive angle is
clockwise. Rotation and isotropic scale are about ((W-1)/2,(H-1)/2), followed by
translation in observed pixels. The attack samples inverse H; rectification samples
H once. Pillow uses edge coordinates, so the sampler conjugates by a half-pixel
translation. Latent-to-RGB mapping is x_rgb=(x_latent+.5)*W/w-.5, analogously for y;
convert transforms through this scale matrix rather than assuming equal units.

Historical V2 geometry H maps observed to canonical in normalized coordinates.
It MUST NOT be passed directly to these pixel-coordinate helpers. New A adapters
must explicitly convert their own native estimator coordinates and direction.
Render combined attacks in one operation, never sequential rotate then resize.
Fixed-canvas content scaling is not image downsampling followed by restoration.

Valid regions may be recorded for geometric diagnostics. They do not alter the
frozen content score or normalization. Any later use of validity in scoring is an
explicit method change. Formal detection consumes only current RGB, key, frozen
public assets and threshold; no prompt, original, oracle H, private embedding
latent/mask or cached Q/K. Oracle rectification is a development diagnostic only.

## Development conditions and allocation labels

`python -m experiments.parallel_method_protocol_v1` prints the proposed counts and
nine conditions without generating images. `CORE_ATTACKS` and `render_attack`
provide clean, ±10°, fixed-canvas scales .75/1.25, AWGN sigma .02/.05 and the
+10°/.75 and -10°/1.25 combinations. AWGN is independent per RGB channel in [0,1],
then clipped and rounded to uint8; it is not Gaussian blur. Use the same explicit
noise seed for matched variants of a pair and condition. Bilinear interpolation,
black fill, and canvas dimensions are shared across variants.

Suggested first run: 8 fit pairs plus 24 independent validation pairs; unmarked,
V2, A geometry-target, A content-tolerance-target, B simplified and B survival
variants total 192 base images, plus 72 B continuations if still needed. These are
adjustable development counts, not a launch prerequisite. Label-path and oracle
counts should reflect actual calls, not repeated calls to reach a quota.

B labels must include perturbation through remaining sampler steps, attacks, old
RGB sync and current blind registered-minus-max16wrong scoring. Record evidence
increment relative to matched uniform baseline and final perceptual costs. Simple
VAE encode/decode response is a feature candidate, not a complete generation
Jacobian. Fit only on fit pairs and freeze the small shared allocator before
validation. Do not feed allocation masks to the detector.

Compare final RGB PSNR/SSIM/LPIPS plus local distortion, including content LF/HF
and synchronization interactions; latent L2 alone is insufficient. LF/HF roles
remain empirical. A content tolerance curves should use residual angle, scale,
and translations with the unchanged score; compare content-target and geometry-
target fitting on held-out samples. Failures remain visible.

## Evidence and next experiments

These CPU protocol checks verify coordinates and rendering only. They do not
establish anchor observability, allocation benefit, robustness, or final quality.
Real-model entrypoints must state actual pair/continuation counts before external
execution is requested. This work authorizes no GPU/Colab/Drive writes or push.
24 validation negatives cannot establish 0.1% FPR. Any later threshold must
calibrate the whole pre/post and candidate-selection path; describe separately
mixed-distribution average versus condition-wise FPR objectives and report the
independent achieved FPR/intervals without blocking result output.

`experiments/conditional_threshold_candidate.py` provides the separate later-stage
candidate: take each condition's empirical ceil((1-alpha)N) order statistic, then
the maximum across conditions and use strict greater-than. Inputs must score the
complete detection path. It is fit-only and gives no population FPR guarantee.

Local T2SMark matched-sync/matched-quality controls follow mechanism development.
JPEG, blur, crop-rescale and generative reconstruction remain staged research
targets; their omission from the first nine conditions does not remove them.
