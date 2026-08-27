# Geometry-V2 Method Route

## Frozen identity and scope

- Branch: `Geometry-V2`.
- Method identity: `geometry_v2_keyed_neural_corner_sync`.
- Protocol identity for the present contract scaffold: `geometry-v2-keyed-neural-corner-sync-contract-v1`.
- Stage: local method-route and pure-CPU contract scaffold only.
- Evidence ceiling: engineering readiness. There is no neural training result, model inference result, operational artifact, detector result, watermark result, or scientific conclusion.

Geometry-V2 is an active deep synchronization method. A weak geometric synchronization signal is written by an embedder constrained by a domain-separated `geometry_key`. An extractor observes only the current attacked RGB image and predicts the four ordered image corners, a constrained homography, support, and an independent reliability measurement. A reliable estimate may request inverse rectification. The rectified RGB is then evaluated by the same content detector with the same detection key identity, preprocessing identity, and calibrated `tau`.

Geometry has coordinate authority only. It cannot create a positive watermark conclusion. Positive attribution remains exclusively with the frozen content detector.

## Method flow

1. Normalize a public per-image context that is independent of detector outcome.
2. Derive a bounded bipolar synchronization target with HMAC-SHA256 under the domain `CEG-WM/geometry-v2/keyed-neural-corner-sync/v1`. The contract exposes the target and public-context digest, never the raw `geometry_key`.
3. A future trainable embedder writes a weak signal constrained by that target and by a separately frozen total distortion/interference budget.
4. Apply the declared geometric attack to the final RGB image.
5. A future extractor consumes only the current attacked RGB and predicts corners in canonical order: top-left, top-right, bottom-right, bottom-left. It also predicts the canonical-to-attacked 3x3 homography, confidence, and support.
6. Validate that corners and homography are finite, bounded, strictly convex, non-degenerate, mutually consistent, and within the frozen coefficient limits.
7. Apply the independent fail-closed reliability policy. Missing geometry, non-finite measurements, or values below either reliability threshold cannot produce a rectification request.
8. If reliable, invert the validated homography and rectify coordinates only.
9. Re-run the unchanged content detector. Detector identity, detection-key identity, preprocessing identity, and `tau` must match the pre-rectification identity exactly.

There is no automatic retry, layer switch, alternate geometry method, detector change, threshold change, or fallback in this route.

## Frozen contract surface

The local scaffold freezes:

- method, protocol, geometry-key domain, and coordinates-only authority constants;
- deterministic key-separated target derivation with bounded key, context, and code lengths;
- normalized canonical corner ordering;
- finite, convex, bounded, non-degenerate corner validation;
- finite, normalized, bounded, invertible 3x3 homography validation;
- corner-to-homography consistency on all four canonical corners;
- an independent confidence-and-support reliability policy that fails closed;
- a reliability binding to the exact geometry estimate;
- a coordinate-only inverse-rectification request;
- an immutable content detector identity binding detector, detection key, preprocessing, and `tau` before and after rectification.

The scaffold intentionally contains no neural architecture, optimizer, dataset, trained weights, SD3.5 integration, GPU path, notebook, attack benchmark, or detector decision.

## Future implementation gates

Any executable neural implementation requires a separately exact-bound protocol before training or real execution. That protocol must predeclare at least:

- embedder and extractor architecture and the exact final-RGB observation boundary;
- geometry-key context construction and sample independence rules;
- total weak-signal distortion budget and content-chain interference measurements;
- training, validation, confirmation, and null splits with no reuse for selection and confirmation;
- attack roster, corner/H truth convention, fixed denominator, retained failures, and stopping rules;
- reliability statistic and calibration data independent of content-detector outcomes;
- rectifier interpolation, boundary, and invalid-coordinate behavior;
- preservation of the same content detector, detection key, preprocessing, and `tau`;
- bounded public artifacts without raw keys, prompts, latents, model weights, private paths, image bytes, or embedding-side records;
- operational and scientific evidence ceilings.

Minimum targeted tests for that future implementation include key/domain separation, distortion-budget accounting, actual RGB attack-to-corner correspondence, H convention, failure retention, reliability false/true behavior, blind extractor inputs, and exact content-detector identity preservation. Model, GPU, Colab, Drive, and scientific evaluation remain separately authorized actions.

## Relationship to prior methods

SyncSeal and SynTag may inform design choices or be evaluated as independently named baselines. They are not the Geometry-V2 method identity, cannot silently supply its implementation or evidence, and cannot be invoked as an automatic fallback.

## Current completion statement

The method route and pure-CPU boundary contracts are frozen locally. The contracts demonstrate deterministic domain separation and fail-closed coordinate handoff behavior only. They do not demonstrate that a weak synchronization signal can be embedded, recovered after attack, or used to improve the unchanged content detector.
