# Geometry-V5 method route — P0 only

`geometry_v5_training_free_initial_noise_sync` is a training-free, latent-space
geometry route. P0 freezes only its governance and validation plane. It has no
writer, synchronization, inversion, detector runtime, rectification runtime,
experiment, or performance result.

The conceptual writer flow is geometry root key → three HKDF-SHA256 domains
(`k_search`, `k_fit`, `k_validate`) → writes to initial `z_T` for a global
X-shaped multiscale spectrum, canonical local fit tiles, and disjoint holdout
tiles → normal frozen diffusion → final RGB. The route uses fixed pretrained
diffusion, a fixed VAE, and fixed detector components. Training, tuning,
fine-tuning, learned synchronization, parameter updates, per-image optimization,
and search-budget adaptation are outside the method.

The intended progression is V5-M0 (faithful global RST reproduction on SD2.1),
V5-M1 (key-domain separation and holdout safety), V5-C0 (keyed local latent
tiles for crop/crop-rescale), V5-I0 (unchanged content detector integration),
and only then V5-SD35, after the method is frozen and fixed repeatable SD3.5
inversion is separately proven. None of those stages is implemented by P0.

## Blind detector boundary

A later detector may consume only current attacked ordinary RGB, the geometry
root key, and frozen model/scheduler/inversion identities. It must obtain a
recovered `z_T` through a fixed inversion defined in a later contract. It may
not consume clean or pre-attack RGB, original `z_T`, writer tensors/residuals,
true transform/crop/attack parameters, content scores/keys, evaluation truth,
or retry/fallback channels.

`k_search` may propose only fixed R/S/T candidates and cannot make an output
RELIABLE. `k_fit` may choose at most one correspondence per local tile and fit
one deterministic similarity model; missing tiles are allowed. `k_validate`
uses disjoint holdout tiles only for safety validation, never candidate proposal,
correspondence, estimation, tie-breaks, threshold tuning, or fallback.

## Public output and content boundary

The complete public output is exactly `(H_hat, corners_hat, support,
reliability, status)`. `H_hat`, when exported, is a finite normalized 3×3
attacked-to-canonical, orientation-preserving similarity matrix; `corners_hat`
is strict-convex TL/TR/BR/BL normalized coordinates consistent with it. Status
is `RELIABLE`, `UNRELIABLE`, or `STOPPED`.

RELIABLE means safe rectification only, never watermark presence. Structurally
it requires a search candidate, fit support, macro-region coverage, residual,
holdout correlation/PSR, cross-scale R/S consistency, valid corners, and legal
conditioning. P0 binds no numerical gates, tile layout, X-template parameters,
inversion schedule, seeds, attack roster, or denominators. Missing/nonfinite
data, identity mismatch, holdout failure, or regeneration destruction must end
in predeclared `UNRELIABLE` or `STOPPED` behavior, without fabricated geometry,
identity fallback, retry, replacement, alternate inversion, geometry voting, or
positive geometry evidence.

Content and geometry keys remain separate. The content detector, preprocessing,
score, and tau are unchanged; geometry correlation, PSR, support, and
reliability cannot add positive evidence. A later rectification can be considered
only when geometry is RELIABLE and original `s0` is inside a later-frozen
boundary interval; `s1` must reuse the exact same content path and tau. P0 does
not choose tau or delta.

Evidence ceiling: `P0_local_static_engineering_only_science_denominator_0`.
This is neither a real-method, robustness, crop-success, regeneration-success,
fixed-FPR, nor scientific conclusion. Geometry-V5 is a direct child of clean
main and remains parallel to—not derived from—Geometry-V4 method code.
