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

For residual strength `a`, the adapter freezes `clamp(I + a*(SyncSeal(I)-I),0,1)` and stores final uint8 RGB. Development may run only a predeclared ordered strength sequence. The frozen selection rule is the first sequence entry satisfying every frozen gate; after selection, test units use that strength without reselection, retry, fallback, or successful-subset filtering.

The real call chain accepts prebound callables for the unchanged content detector/key/preprocessing/tau, official SyncSeal embedding/detection, and PSNR/SSIM/LPIPS. It has no shipped proxy implementation of SyncSeal or the content chain. Every arm is attempted once. Stage exceptions, malformed results, and detector `ERROR` records remain attached to their arm; absent CG does not remove C or shrink a denominator.

Per unit the record keeps all-arm raw LF/HF/weighted-joint content scores and frozen decisions, CG-C raw score deltas and flip, G content false-positive state, G-U and CG-C PSNR/SSIM/LPIPS, and each arm's SyncSeal raw logit, corners, H, legality, and basic observability. Denominators are fixed at two negative arms, two positive arms, and four failure-eligible arms.

All R0 numeric gates live only in `R0NumericGates`. The ordered residual strengths, minimum PSNR, minimum SSIM, maximum LPIPS, maximum CG-C flip rate, maximum G content false-positive rate, and minimum SyncSeal basic-observability rate are currently pending confirmation. Missing values fail closed and cannot be interpreted as pass.

## Predeclared later-stage contracts (no executor in this change)

R1A is a sanity control only. Identity, pure resize under the frozen convention, and normalization-equivalent identity check coordinate direction and implementation equivalence; passing them is not nontrivial geometry evidence.

For a nontrivial attack `a`, raw prediction stability is required only when the truth-measured identity-baseline error exceeds the frozen `delta_nontrivial`. Stability means the raw prediction reduces the predeclared coordinate error against truth; it does not use truth at detection time and it does not promote content evidence.

R1B eligibility is frozen before attacked recovery. `E_a` contains units whose clean, pre-attack C image is content-positive and whose attacked image is either below the unchanged tau or lies in the predeclared boundary band. Every predeclared eligible unit and every failure stays in the denominator. `D_a` contains attacked units that remain positive; rectification there checks only resampling harm and cannot count as recovery success.

If `E_a` is empty, the attack result is `NOT_APPLICABLE/INSUFFICIENT_ELIGIBLE`, never success or failure. If every core attack has empty `E_a`, the stage has no evidence that geometry was necessary. Rectification must reuse the identical content detector, detection key, preprocessing, and tau.

## Immediate stop conditions and claim ceiling

`METHOD_DEVIATION` stops the route if detection consumes original RGB, prompt, embed records, private latents, embedding-side routes, cached features, truth, or attack parameters; if any proxy/oracle/hardcoded H or pseudo RGB is used; if roster, attacks, denominators, thresholds, D4 candidates, failures, retries, fallback, or successful subsets change after results; or if rectification changes detector, key, preprocessing, or tau.

The current ceiling is `LOCAL_ENGINEERING_P0_R0_F1_ONLY`: interface, routing, static notebook, and deterministic injected-callable tests only. It is not real SyncSeal execution, content-chain execution, robustness evidence, geometry completion, fixed-FPR evidence, or scientific adjudication.
