# Geometry-V2 Method Route

## Frozen identity and authority

- Branch: `Geometry-V2`.
- Method: `geometry_v2_keyed_neural_corner_sync`.
- Executable protocol: `geometry-v2-keyed-neural-corner-sync-n0-v1`.
- N0 is an operational neural geometry candidate run with `science_denominator=0`.

Geometry-V2 actively writes a weak synchronization signal under a domain-separated `geometry_key`. Its extractor observes only the current attacked RGB and predicts ordered normalized corners `(TL, TR, BR, BL)`, confidence, and support. A constrained homography is determined from those corners and validated before any reliability or rectification request.

Geometry has coordinate authority only. It cannot create positive watermark evidence. If a later separately authorized stage rectifies a reliable estimate, the same content detector, detection key, preprocessing, and calibrated `tau` must be used again unchanged. N0 itself does not run or adjudicate the content detector.

SyncSeal and SynTag may inform an independently named baseline. They are not this method and are never an automatic fallback.

## N0 architecture and training

N0 jointly optimizes two real PyTorch modules in one process:

1. `KeyedResidualEmbedder` receives ordinary RGB and a per-sample 64-bit bipolar target derived with HMAC-SHA256 under the frozen Geometry-V2 domain. It returns RGB in `[0,1]`; the actual post-clamp residual is hard-bounded by `L_inf <= 4/255`.
2. `BlindCornerExtractor` receives only attacked RGB. It returns four corner slots in the fixed order plus bounded confidence and complete-estimate support. It has no geometry key, clean image, embed record, prompt, latent, detector score, or original-image input.

The input data are procedural RGB at `128x128`; no external dataset is used. Splits are disjoint and fixed:

- training seeds `1000..1127` (128);
- validation seeds `2000..2031` (32);
- independent confirmation seeds `3000..3031` (32).

Training uses seed 73, batch size 8, eight epochs, and Adam with learning rate `1e-3`. Each epoch assigns the four fixed attack classes equally across the training split; attack parameters are not tuned. The immutable loss is:

`SmoothL1(corners) + 0.25 * sync_reconstruction + 0.05 * residual_L2`.

No checkpoint or model weight is saved. Validation and confirmation inference use the same in-memory trained modules.

## Frozen RGB attacks and truth

Every validation and confirmation image is evaluated under all four actual Pillow transforms:

- identity;
- `rotate90`, implemented as `PIL.Image.Transpose.ROTATE_90`;
- similarity with angle `7 deg`, scale `0.93`, and 512-coordinate translation `(13,17)` scaled to 128;
- crop-rescale with 512-coordinate crop `[32,44,476,468]`, scaled to `[8,11,119,117]`, followed by BICUBIC resize to 128.

Each transform returns the frozen source-to-attacked normalized H and its four mapped corner truths. Actual Pillow RGB correspondence is a required CPU regression. Invalid or non-finite predicted corners/H remain explicit failed units.

## Reliability and confirmation gate

Validation observes the predeclared reliability rule before confirmation:

- `minimum_support = 1.0`;
- score `clamp(1 - mean_corner_error / 0.25, 0,1)`;
- threshold `0.5`.

The threshold is fixed in source and is not fitted to validation results. Confirmation cannot change architecture, attack parameters, reliability, or gates. It has exactly `32 images x 4 attacks = 128` retained units.

Allowed statuses are only:

- `N0_STOPPED`: any retained confirmation unit failed;
- `N0_UNRESOLVED`: all units calculated but one or more candidate gates failed;
- `N0_GEOMETRY_CANDIDATE`: all 128 calculated, median corner error `<0.05`, p95 corner error `<0.10`, reliable fraction `>=0.75`, and observed residual maximum `<=4/255`.

PSNR-equivalent residual summaries, key-separation, no-sync behavior, extractor confidence, validation observations, and per-attack records are audit-only. They do not alter the gate.

## Runtime and artifact boundary

The runner validates a bounded public plan, exact clean checkout, runtime-only geometry key, and CPU/GPU device before training. The Colab handoff uses one detached exact checkout and one child runner, bounded control receipt, suppressed child stdout/stderr, and a create-only Drive directory under `/content/drive/MyDrive/CEG-WM/Geometry-V2/N0/`. It has no retry, fallback, alternate route, or dynamic threshold.

The geometry secret is read from Colab userdata or generated once with `secrets`; only an in-memory domain-separated digest reaches the child environment. The raw secret and derived runtime key are cleared and are never printed or written.

Artifacts contain only bounded `receipt.json`, `manifest.json`, `terminal.json`, and 128 public `metrics.jsonl` records. They exclude images, raw key/key material, model/checkpoint weights, prompts, latents, tokens, private paths, raw Q/K, and detector decisions.

## Evidence ceiling

Local CPU tests establish engineering behavior only. A future controlled N0 run may establish operational candidate evidence with `science_denominator=0`; it cannot establish content detection, positive watermark attribution, attack robustness, method success, or a scientific conclusion. Training, inference, artifact completion, and a candidate status do not change that ceiling.
