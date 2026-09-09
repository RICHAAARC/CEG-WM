# Parallel method development V1

This development supersedes historical stage descriptions only within the newly
authorized A/B worktrees. V2 results supplied by the control task are the starting
point; this work does not claim to have repeated its raw-artifact audit.

A fixes V2 content embedding and scoring while replacing the old RGB sync carrier
and estimator with one step-18 latent writer-reader candidate and a final RGB VAE
observation, compared with a no-anchor reference. No multi-candidate selection or
broad RST scan is part of the active work. B changes only HF spatial weights while
preserving each image's original LF allocation, LF/HF shares, ISS, V2 RGB sync and
blind scoring. Both routes keep
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

`python -m experiments.parallel_method_protocol_v1` prints the current narrow scope
without generating images. `ACTIVE_A_ATTACKS` contains clean and the exact current
V2 +10 rotation renderer; `ACTIVE_B_ATTACKS` contains clean, AWGN sigma .02 and
JPEG50 (4:2:0 subsampling, nonprogressive, no optimization). The V2 rotation label
must not be interpreted as the historical protocol's clockwise pixel-space +10:
the helper preserves the V2 renderer exactly and derives truth H from its actual
Pillow sampling matrix. `CORE_ATTACKS` retains the previous nine conditions only
as an explicitly selected historical utility; it is not the active default.
AWGN is independent per RGB channel in [0,1],
then clipped and rounded to uint8; it is not Gaussian blur. Use the same explicit
noise seed for matched variants of a pair and condition. Bilinear interpolation,
black fill, and canvas dimensions are shared across variants.

The previous 264-image / 3528-path suggested envelope has been withdrawn. Actual
pair, continuation, score and oracle counts come from each CLI's selected options.
There is no requirement to complete the former five-variant matrix.

B labels must include perturbation through remaining sampler steps, attacks, old
RGB sync and current blind registered-minus-max16wrong scoring. Record evidence
increment relative to the matched uniform-HF baseline with original per-image LF
configuration and branch shares held fixed, and final perceptual costs. Uniform,
probe and predicted HF maps use the same bilinear interpolation to the latent
grid. Original HF remains a separate comparison on its unchanged nearest-neighbor
map. The unchanged ISS joint budget projection can
still change the actual LF perturbation when HF weights change; record actual
LF/HF perturbation norms rather than claiming identical final LF perturbations. Simple
VAE encode/decode response is a feature candidate, not a complete generation
Jacobian. Fit only on fit pairs and freeze the small shared allocator before
validation. Do not feed allocation masks to the detector.

Compare final RGB PSNR/SSIM/LPIPS plus local distortion, including content LF/HF
and synchronization interactions; latent L2 alone is insufficient. LF/HF roles
remain empirical. A currently tests one writer-reader candidate and its no-anchor
reference on clean/+10 only. Oracle is auxiliary; candidate selection by its own
oracle-minus-post loss is not active. That loss can favor a candidate whose oracle
and post scores are both poor and cannot establish recovered detection benefit.
Failures remain visible.

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
Blur, crop-rescale and generative reconstruction remain staged research targets;
the narrowed first experiment does not remove them from the research scope.
