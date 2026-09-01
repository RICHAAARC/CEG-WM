# Geometry-V7 P0 and staged evaluation contract

## Scope and evidence authority

Geometry-V7 starts from main exact `5af1fadd6a8025c6b2664703d519606238a89143` and the complete step-18 content chain. SyncSeal is a frozen final-RGB postprocessor and coordinate estimator; it is not trained or fine-tuned here. Existing embedding, keyed LF/HF scoring, calibrated weighted-joint scoring, key normalization, preprocessing, and threshold semantics remain unchanged. Geometry produces coordinates plus `RELIABLE`, `UNRELIABLE`, `UNSUPPORTED`, or `ERROR`; only content statistics may create a positive watermark decision.

This P0 + R0/F1 + R1A/F2 implementation is local engineering infrastructure. It does not itself run a real model, GPU, Colab, or Drive job and does not adjudicate robustness, FPR, or science.

## P0 public interface

- Input is an ordinary RGB image at its public raw size of exactly 512x512. The adapter rejects implicit resize, non-RGB input, proxy RGB, oracle RGB, and non-finite tensors.
- Pixel centers are integer lattice locations `0..511`. Public normalized coordinates use `n = 2*p/(size-1)-1`, so the first and last pixel centers are exactly -1 and +1. SyncSeal's documented internal detector resize remains its own bilinear, antialiased, `align_corners=False` implementation. Its raw 256-grid points are retained verbatim, then converted exactly as official unwarp does: `p256 = round(raw*128+128)`, followed by public normalization `2*p256/255-1` (the intermediate 256-grid-to-512 scaling cancels algebraically).
- The observed/output-canvas points `q` are fixed as the normalized 512 square `(-1,-1),(1,-1),(1,1),(-1,1)` in TL, TR, BR, BL order. SyncSeal's eight predicted values are `p_hat`: the corresponding `q` points expressed in original/canonical CG-image coordinates. They are recorded as `observed_corners_in_canonical_normalized`; they are not locations in the current frame.
- The homography direction is frozen as column-vector `p_hat ~ H_observed_to_canonical * q`. `homography_observed_to_canonical` is solved from the fixed `q` source to predicted `p_hat` target. Read-only deprecated aliases retain the old API names but use this corrected direction and semantics.
- A strict convex `p_hat` quadrilateral and a finite invertible `H_observed_to_canonical` are the minimum legal/basic-observable conditions. Finite parsed raw points and converted `p_hat` are retained when convexity or homography validation yields `UNSUPPORTED`; only H is absent. Malformed/non-finite point output is `ERROR`. The first SyncSeal raw output is recorded only as `uncalibrated_sync_logit`; P0/R0 has no approved calibration that could promote a legal output to `RELIABLE`, so it remains `UNRELIABLE`.
- The official TorchScript source is `https://dl.fbaipublicfiles.com/wmar/syncseal/paper/syncmodel.jit.pt`. The adapter requires official `embed -> {preds_w,imgs_w}` and `detect -> {preds,preds_pts}`, with detector shapes 1x9 and 1x8. No source SHA-256 is frozen here; checkpoint bytes, environment lock, and official逐值 reproduction are record-only gaps rather than fabricated gates.

## D4 and attacks

D4 acts in canonical normalized coordinates after raw recovery. For every predeclared D4 element, composition is exactly `H_candidate = D_canonical * H_raw_observed_to_canonical`; no result-dependent D4 roster change is allowed. The eight fixed elements are identity, image-coordinate rotations by 90/180/270 degrees counterclockwise, left-right mirror, and that mirror followed by each rotation.

R1A freezes thirteen independent conditions. The three sanity controls are identity, 512 -> 384 -> 512, and 512 -> 768 -> 512. The ten core conditions are rotations -15 and +15 degrees; fixed-canvas zooms 0.8 and 1.2; translations (+32,0), (-32,0), (0,+32), and (0,-32) pixels; one offset crop-rescale whose truth rectangle is TL(-.875,-.625), TR(.625,-.625), BR(.625,.875), BL(-.875,.875); and one composite `F = C_0.85 @ T(+16,-16) @ R(+10 degrees)`. Here `C_0.85` means a centered 85% linear crop rescaled to 512, so its canonical-to-observed scale is `1/0.85` and its observed-to-canonical truth scale is `0.85`. No plain 80% crop, perspective, JPEG, or generative condition is part of R1A.

All core images have fixed 512x512 output, black fill, and bilinear sampling. Rotation is about pixel center 255.5; positive display rotation is counterclockwise in y-down image coordinates, so a right-side landmark moves upward. Translation uses normalized offset `2*d/511`. `F` maps canonical/source to observed/output and `H_truth = inverse(F)` maps observed `q` back to canonical truth. Pillow receives the output-to-source `H_truth` mapping. Each core condition is composed in normalized coordinates and resampled exactly once. Resize sanity controls use their declared two resize steps; downsampling may use bilinear antialiasing.

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
- `min_identity_coordinate_valid_rate = 1.0` counts only G and CG, on a fixed denominator of twice the roster size. A valid estimate has `legal = True`, a null error, and finite 4x2 `observed_corners_in_canonical_normalized`; those correspondences must be strictly convex and nondegenerate in the declared TL/TR/BR/BL order, use the frozen coordinate convention, and have maximum same-order per-coordinate L-infinity error from `CANONICAL_CORNERS_NORMALIZED` at most `2/255`. This direct-corner tolerance absorbs only one official 256-grid pixel-center rounding step and supports only identity-interface and carrier-compatibility checking, not synchronization observability or pilot readability. U/C diagnostics do not enter this gate.

R0 is identity-only. Passing supports only final-RGB carrier compatibility and validity of the identity coordinate interface; it does not show that the pilot is observable/readable or that synchronization or geometric recovery works. Nonidentity synchronization utilization remains for R1A truth-error and identity-baseline-improvement evaluation.

If every frozen multiplier fails on the fixed development roster, the only conclusion is: no carrier-compatibility window was found on the preregistered strength grid and fixed R0 roster, so Geometry-V7 stops by contract. It is not a general impossibility claim.

## R1A/F2 fixed blocking method canary

R1A consumes only the eight CG PNGs from the fixed R0 evaluation roster in the passed artifact produced by exact `4f0bf1560805672f786dc86dd50d793aec18aae7`, with selected residual multiplier 0.75. It does not regenerate SD3.5 images, recompute content scores, use a key, or use R0 hashes or sidecars as an input gate. For every condition and unit, official SyncSeal receives exactly one attacked RGB and no original, truth, matrix, or attack parameter.

Normalization is `n = 2*p/511 - 1`. The fixed observed points are `q = CANONICAL_CORNERS_NORMALIZED`. Corner error is exactly `RMSE(a,b) = sqrt(sum over 8 coordinates of (a-b)^2 / 8)`. For core truth `p_truth = H_truth*q`, define `e_pred = RMSE(p_hat,p_truth)`, `e_id = RMSE(q,p_truth)`, and paired `d = e_pred-e_id`.

Before detection, a truth-only CPU preflight requires every one of the ten core conditions to have `e_id > delta_nontrivial`, where `delta_nontrivial = 2/255`. If any condition fails this check, the whole status is `ATTACK_SPEC_REQUEST_CHANGES`; no method pass/fail is recorded and conditions cannot be reclassified after results. The sanity controls each require 8/8 finite, legal, strictly convex, nondegenerate, correctly ordered estimates whose direct identity-corner L-infinity error is at most `2/255`. They are interface controls only, not pilot observability or readability evidence.

Every condition has fixed `N = 8`. Core truth eligibility is fixed before detector output and is expected to be eight. Missing, failed, illegal, malformed, or non-finite predictions remain in the denominator and prevent a successful-subset median. A core condition passes only if all eight eligible records have finite paired `d`, at least `ceil(.75*8) = 6` have `d < 0`, and the median of all eight paired `d` values is strictly below zero. All ten core conditions and all three sanity controls must independently pass for `R1A_BLOCKING_METHOD_CANARY_PASSED`; opposite conditions and attack families cannot offset one another. Truth and attack matrices are confined to the renderer/evaluator. Geometry remains coordinate-only and cannot vote content-positive.

The real runner writes all 104 attacked PNGs, all 104 fixed-denominator records, condition aggregates, failures, matrices, raw SyncSeal logit/points, predicted canonical correspondences, `H_observed_to_canonical`, truth, `e_pred`, `e_id`, `d`, eligibility, improvement, and record-only provenance to a create-only package. Setup and detector failures remain in all affected denominators with no retry, fallback, replacement, or subset filtering. If any fixed record carries `syncseal_runtime_setup:*` or `geometry_detect:*`, the top-level status is `OPERATIONAL_FAILURE_RETAINED_FIXED_DENOMINATOR` and `blocking_method_canary_passed` is null while the 104 records and underlying fixed-denominator evaluation remain present. A complete set of finite legal predictions that simply misses the frozen gates remains a blocking method-canary failure instead.

## Predeclared R1B contract (no executor in this change)

R1B eligibility is frozen before attacked recovery. `E_a` contains units whose clean, pre-attack C image is content-positive and whose attacked image is either below the unchanged tau or lies in the predeclared boundary band. Every predeclared eligible unit and every failure stays in the denominator. `D_a` contains attacked units that remain positive; rectification there checks only resampling harm and cannot count as recovery success.

If `E_a` is empty, the attack result is `NOT_APPLICABLE/INSUFFICIENT_ELIGIBLE`, never success or failure. If every core attack has empty `E_a`, the stage has no evidence that geometry was necessary. Rectification must reuse the identical content detector, detection key, preprocessing, and tau.

## Immediate stop conditions and claim ceiling

`METHOD_DEVIATION` stops the route if detection consumes original RGB, prompt, embed records, private latents, embedding-side routes, cached features, truth, or attack parameters; if any proxy/oracle/hardcoded H or pseudo RGB is used; if roster, attacks, denominators, thresholds, D4 candidates, failures, retries, fallback, or successful subsets change after results; or if rectification changes detector, key, preprocessing, or tau.

The current code ceiling is `LOCAL_ENGINEERING_P0_R0_F1_R1A_ONLY`: interface, deterministic CPU truth/renderer tests, artifact-bound routing, static notebook, and injected-callable tests only. Until the bound notebook is executed and independently audited, it is not real R1A evidence. Even a real eight-image result is only a small-sample blocking geometry method canary, not attack robustness, content recovery, fixed-FPR evidence, full geometry completion, or scientific adjudication.
