# Geometry-V7 P0 and staged evaluation contract

## Scope and evidence authority

Geometry-V7 starts from main exact `5af1fadd6a8025c6b2664703d519606238a89143` and the complete step-18 content chain. SyncSeal is a frozen final-RGB postprocessor and coordinate estimator; it is not trained or fine-tuned here. Existing embedding, keyed LF/HF scoring, calibrated weighted-joint scoring, key normalization, preprocessing, and threshold semantics remain unchanged. Geometry produces coordinates plus `RELIABLE`, `UNRELIABLE`, `UNSUPPORTED`, or `ERROR`; only content statistics may create a positive watermark decision.

This P0 + R0/F1 implementation is local engineering infrastructure. It does not run a real model, GPU, Colab, or Drive job and does not adjudicate geometry, robustness, FPR, or science.

## P0 public interface

- Input is an ordinary RGB image at its public raw size of exactly 512x512. The adapter rejects implicit resize, non-RGB input, proxy RGB, oracle RGB, and non-finite tensors.
- Pixel centers are integer lattice locations `0..511`. Public normalized coordinates use `n = 2*p/(size-1)-1`, so the first and last pixel centers are exactly -1 and +1. SyncSeal's documented internal detector resize remains its own bilinear, antialiased, `align_corners=False` implementation. Its raw 256-grid points are retained verbatim, then converted exactly as official unwarp does: `p256 = round(raw*128+128)`, followed by public normalization `2*p256/255-1` (the intermediate 256-grid-to-512 scaling cancels algebraically).
- Detected corners contain exactly eight finite values ordered TL, TR, BR, BL. They identify, in the observed/current normalized frame, the locations corresponding to canonical TL, TR, BR, BL.
- The homography direction is frozen as column-vector `x_canonical ~ H_current_to_canonical * x_current`. The canonical target is the normalized 512 square `(-1,-1),(1,-1),(1,1),(-1,1)`.
- A strict convex quadrilateral and a finite invertible homography are the minimum legal/basic-observable conditions. The first SyncSeal raw output is recorded only as `uncalibrated_sync_logit`; P0/R0 has no approved calibration that could promote a legal output to `RELIABLE`, so it remains `UNRELIABLE`. Invalid geometry is `UNSUPPORTED`; malformed/non-finite model output is `ERROR`.
- The official TorchScript source is `https://dl.fbaipublicfiles.com/wmar/syncseal/paper/syncmodel.jit.pt`. The adapter requires official `embed -> {preds_w,imgs_w}` and `detect -> {preds,preds_pts}`, with detector shapes 1x9 and 1x8. No source SHA-256 is frozen here; checkpoint bytes, environment lock, and official逐值 reproduction are record-only gaps rather than fabricated gates.

## D4 and attacks

D4 acts in canonical normalized coordinates after raw recovery. For every predeclared D4 element, composition is exactly `H_candidate = D_canonical * H_raw_current_to_canonical`; no result-dependent D4 roster change is allowed. The eight fixed elements are identity, image-coordinate rotations by 90/180/270 degrees counterclockwise, left-right mirror, and that mirror followed by each rotation.

Core nontrivial attack families are frozen by name as rotation about the 512 canvas center, axis-aligned crop followed by resize to 512, four-corner perspective to 512, and the fixed-order rotation -> crop/resize -> perspective composition. Bilinear resampling, antialiasing where the resize API supports it, integer pixel centers, and fixed output 512x512 are common conventions. Numeric ranges, fill policy, roster, seeds, and the exact compound parameter table are pending user confirmation and may not be invented after results.

## R0/F1 four-arm isolation

Each predeclared unit has exactly four paired arms:

- `U`: no content watermark, no SyncSeal;
- `G`: no content watermark, SyncSeal applied once to final U RGB;
- `C`: complete unchanged content watermark, no SyncSeal;
- `CG`: SyncSeal applied once to final C RGB.

The official model has `base_syncseal_alpha = 0.20`, and official `embed().imgs_w` already includes it. R0 varies only the ordered residual multipliers `(0.25, 0.50, 0.75, 1.00)` via `clamp(I + multiplier*(I_official-I),0,1)` before final uint8 storage; alpha is recorded but never multiplied a second time.

Development uses the first four units of the fixed ordered reference roster. It runs multipliers in the frozen order and selects the first one for which every gate below passes. Test uses the fixed ordered eight-unit evaluation roster and runs that selected multiplier exactly once, under the same gates, without reselection, retry, fallback, denominator change, or successful-subset filtering.

The real call chain accepts prebound callables for the unchanged content detector/key/preprocessing/tau, official SyncSeal embedding/detection, and PSNR/SSIM/LPIPS. It has no shipped proxy implementation of SyncSeal or the content chain. Every arm is attempted once. Stage exceptions, malformed results, and detector `ERROR` records remain attached to their arm; absent CG does not remove C or shrink a denominator.

Per unit the record keeps all-arm raw LF/HF/weighted-joint content scores, content margin, and frozen decisions; CG-C raw LF/HF/weighted-joint/margin deltas and decision flip; G content false-positive state; G-U and CG-C PSNR/SSIM/LPIPS; and each arm's SyncSeal raw logit, corners, H, legality, and coordinate diagnostics. Denominators are fixed at two negative arms, two positive arms, and four failure-eligible arms.

All R0 numeric gates live only in `R0NumericGates`:

- G-U and CG-C are separate pair families. Each family uses its entire fixed roster denominator and must independently satisfy arithmetic `mean PSNR >= 40.0`, `mean SSIM >= 0.98`, and `mean LPIPS <= 0.05`. Per-image values and family min/max are record-only. A missing, invalid, non-finite, or failed pair makes that family fail; no successful-subset mean exists.
- `max_cg_c_decision_flip_rate = 0.0` compares only C to CG, and `max_g_content_false_positive_rate = 0.0`. Passing means only that no event was observed in the fixed four- or eight-unit roster. Raw score and margin deltas have no additional change gate.
- `min_identity_coordinate_valid_rate = 1.0` counts only G and CG, on a fixed denominator of twice the roster size. A valid estimate has `legal = True`, a null error, and finite 4x2 `corners_current_normalized`; those corners must be strictly convex and nondegenerate in the declared TL/TR/BR/BL order, use the frozen coordinate convention, and have maximum same-order per-coordinate L-infinity error from `CANONICAL_CORNERS_NORMALIZED` at most `2/255`. This direct-corner tolerance absorbs only one official 256-grid pixel-center rounding step and supports only identity-interface and carrier-compatibility checking, not synchronization observability or pilot readability. U/C diagnostics do not enter this gate.

R0 is identity-only. Passing supports only final-RGB carrier compatibility and validity of the identity coordinate interface; it does not show that the pilot is observable/readable or that synchronization or geometric recovery works. Nonidentity synchronization utilization remains for R1A truth-error and identity-baseline-improvement evaluation.

If every frozen multiplier fails on the fixed development roster, the only conclusion is: no carrier-compatibility window was found on the preregistered strength grid and fixed R0 roster, so Geometry-V7 stops by contract. It is not a general impossibility claim.

## Predeclared later-stage contracts (no executor in this change)

R1A is a sanity control only. Identity, pure resize under the frozen convention, and normalization-equivalent identity check coordinate direction and implementation equivalence; passing them is not nontrivial geometry evidence.

For a nontrivial attack `a`, raw prediction stability is required only when the truth-measured identity-baseline error exceeds the frozen `delta_nontrivial`. Stability means the raw prediction reduces the predeclared coordinate error against truth; it does not use truth at detection time and it does not promote content evidence.

R1B eligibility is frozen before attacked recovery. `E_a` contains units whose clean, pre-attack C image is content-positive and whose attacked image is either below the unchanged tau or lies in the predeclared boundary band. Every predeclared eligible unit and every failure stays in the denominator. `D_a` contains attacked units that remain positive; rectification there checks only resampling harm and cannot count as recovery success.

If `E_a` is empty, the attack result is `NOT_APPLICABLE/INSUFFICIENT_ELIGIBLE`, never success or failure. If every core attack has empty `E_a`, the stage has no evidence that geometry was necessary. Rectification must reuse the identical content detector, detection key, preprocessing, and tau.

## Immediate stop conditions and claim ceiling

`METHOD_DEVIATION` stops the route if detection consumes original RGB, prompt, embed records, private latents, embedding-side routes, cached features, truth, or attack parameters; if any proxy/oracle/hardcoded H or pseudo RGB is used; if roster, attacks, denominators, thresholds, D4 candidates, failures, retries, fallback, or successful subsets change after results; or if rectification changes detector, key, preprocessing, or tau.

The current ceiling is `LOCAL_ENGINEERING_P0_R0_F1_ONLY`: interface, routing, static notebook, and deterministic injected-callable tests only. It is not real SyncSeal execution, content-chain execution, robustness evidence, geometry completion, fixed-FPR evidence, or scientific adjudication.
