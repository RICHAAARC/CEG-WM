# Geometry-V5 M0 — global R/S/T development mechanism

M0 is a training-free, initial-`z_T`-only geometry mechanism for a frozen
SD2.1-facing route. It is not a detector decision, rectifier, crop method,
content integration, or scientific result. The X-template geometry/parameters
are independently adapted from
`Mao718/MaXsive@a9554024aed176e705cc15ca1cbd31b9c7f75bfb`; DDIM inversion
equations/flow are independently adapted from
`YuxinWenRick/tree-ring-watermark@3015283d9cf82e90b628f02ad2121bd37408ca9a`.
Neither project is vendored or claimed byte-for-byte equivalent.

V5 writes the Hermitian X-template into initial `z_T` once, channel 3 only;
it does not copy inspected step-wise model-output peak injection. The template
uses scale 5 and radial lengths `[0.2, 0.3, 0.4, 0.5]`. The method covers only
global rotation, scale, and translation—not local tiles, crops, attention,
affine/projective transforms, learning, retries, fallback, content evidence, or
SD3.5.

The concrete method chain is: attacked ordinary RGB → deterministic VAE mode
encoding → empty-prompt, guidance-1 DDIM inversion → recovered `z_T` → blind
R/S spectral search → R/S-normalized phase translation → strict
attacked-to-canonical `H` → `ESTIMATE_AVAILABLE` or `FAILED`. The detector
accepts only attacked ordinary RGB with frozen runtime identities; it cannot
read a prompt, original latent, clean/pre-attack RGB, true geometry, attack
parameters, residuals, content material, or evaluation truth.

For forward `A=sR(theta)`, the spectrum follows
`k_observed=A^-T k_canonical=(1/s)R(theta)k_canonical`. A blind candidate
`cR(phi)` gives the attacked-to-canonical spatial linear map
`B=cR(-phi)`: the reciprocal scale is already carried by the Fourier relation.
Before translation, the recovered channel is resampled as `g(B^-1 q)` onto the
canonical grid. Phase correlation uses normalized observed relative to the
canonical reference and negates its signed shift, returning `u` in `H=B x+u`.
Flat, non-finite, ambiguous, inadequate-overlap, and degenerate-phase inputs
fail closed rather than producing a tie-break estimate.

Public translations use centered unit-image coordinates: one image width is
1.0, and a `p`-pixel latent displacement is `p/64`. The runtime's single
conversion helper maps these translations to `grid_sample(align_corners=True)`
endpoint coordinates. Similarity linear terms are unchanged across the two
bases because they differ by a common scalar.

The runtime binds the community mirror
`sd2-community/stable-diffusion-2-1-base` at revision
`4e63672c03103b6c636b8fb4119ba982469b2955` where a runtime needs it. This is
not a claim of byte equivalence with an unresolved `stabilityai` source. Local
work has not loaded or run a model. GPU execution, if later authorized, belongs
to a future Colab stage and is not established by the local static smoke.

Raw M0 output cannot emit `GeometryV5Observation.RELIABLE`, rectify RGB, or
vote content. The current engineering claim ceiling remains
`M0_SD21_global_RST_development_engineering_only_science_denominator_0`.
