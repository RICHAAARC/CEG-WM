# Geometry-V5 M0 — global R/S/T development mechanism

M0 independently adapts a global initial-`z_T` X-template route for frozen
SD2.1-facing construction. It is not a real SD2.1 execution, detector,
rectifier, crop method, content integration, or scientific result.

The source bindings are `Mao718/MaXsive@a9554024aed176e705cc15ca1cbd31b9c7f75bfb`
for X-template geometry/parameters and
`YuxinWenRick/tree-ring-watermark@3015283d9cf82e90b628f02ad2121bd37408ca9a`
for DDIM inversion equations/flow. License status is recorded only. M0 does not
vendor or wholesale-copy either project. In particular, inspected MaXsive
`Modified_DPMSolver` step-wise model-output peak injection is deliberately not
copied: V5 writes initial `z_T` only, using a Hermitian real-latent adaptation,
and makes no byte-for-byte official-parity claim. Phase-correlation translation
is a minimal V5 extension, not an attribution to inspected MaXsive code.

The fixed conceptual adapter is 512×512 RGB, 4×64×64 latent, normal frozen
generation, deterministic attacked-RGB preprocessing, VAE mode encoding,
empty-prompt inversion, 50 steps, eta 0, and guidance 7.5. M0-R0 binds the
public mirror model revision in its execution contract. Template identity
is channel 3, scale 5, radial lengths `[0.2, 0.3, 0.4, 0.5]`.

M0 covers only global rotation, scale, and translation. It excludes local tiles,
holdout finalization, content integration, crops, SD3.5, attention,
affine/projective models, learning/optimization, retries, and fallback. The
production-facing boundary is attacked ordinary RGB plus frozen
model/scheduler/inversion identities; it cannot use prompts, original latent,
clean/pre-attack RGB, true geometry/attack values, writer residuals, content
keys/scores, or evaluation truth. Truth belongs only to a later evaluator after
a raw record is frozen.

Raw M0 records are `ESTIMATE_AVAILABLE` or `FAILED`; they cannot emit
`GeometryV5Observation.RELIABLE`, rectify an image, or add content evidence.
The fixed four-seed, eleven-attack roster has denominator 44 with no replacement
or retry. Its engineering exit is only
`M0_SD21_global_RST_development_engineering_only_science_denominator_0`; it
does not establish reliable geometry, crop handling, dual-chain behavior, SD3.5,
fixed-FPR, regeneration, or science success.

## M0-R0 SD2.1 Colab-facing execution contract

M0-R0 binds `sd2-community/stable-diffusion-2-1-base` at revision
`4e63672c03103b6c636b8fb4119ba982469b2955`, using its bound scheduler
configuration as `DDIMScheduler`. This is a public community mirror; no
byte-equivalence claim is made for an unresolved `stabilityai` source. Generation
uses each manifest prompt, 50 DDIM steps, eta 0, guidance 7.5, CUDA float16.
Inversion uses empty prompt, guidance 1, 50 steps, eta 0, VAE
`latent_dist.mode()`, and the bound VAE scaling factor.

The concrete adapter is lazy and has not been executed locally: importing it
does not import model packages, load weights, contact a network, or execute a
model. Its one concrete combined entry accepts only a bound pipeline, attacked
ordinary RGB, and the frozen runtime identity. It has no parameter for truth,
clean RGB, original latent, prompt, or attack values.

For spatial forward `A=sR(theta)`, the spectral relation is
`k_observed=A^-T k_canonical=(1/s)R(theta)k_canonical`. The blind spectral
candidate `cR(phi)` therefore produces the public attacked-to-canonical spatial
estimate `R(-phi)` with scale `c`: the reciprocal relation is already in the
frequency candidate. M0-R0 freezes a -15…15 degree, 1-degree spectral grid and
0.85…1.15, 0.01 spatial-scale grid. This does not use truth or per-unit tuning.

After selecting `B=cR(-phi)`, the recovered channel-3 plane is resampled into
canonical orientation and scale as `g(B^-1 q)` using bilinear `grid_sample`,
centered normalized coordinates, `align_corners=True`, and zero padding. Phase
correlation compares this normalized observed plane with the canonical
reference. Its signed relative shift is negated to form `u`, so the reported
transform is exactly `H=B x+u`; for a forward translation `t_forward`, the
compound fixture relation is `u=-B*t_forward`, not `-t`. The runtime records
phase peak, PSR, and zero-padding overlap as diagnostics; these never establish
RELIABLE. Flat, non-finite, insufficiently separated spectral candidates,
inadequate overlap, or degenerate phase surfaces return `FAILED` rather than an
equal-score tie-break estimate.

The real runner is create-only and retains 44 raw/evaluation records, including
seed-wide generation failures and attack/inversion failures. Truth is read only
after raw detector records are frozen. No runner path emits RELIABLE, rectifies
RGB, or votes content. Local static/fake checks remain engineering construction
evidence only; real Colab execution requires a detached exact checkout and a
separate execution authorization.
