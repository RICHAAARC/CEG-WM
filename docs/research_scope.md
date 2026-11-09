# Research Scope

## Single-source method validation (2026-11-09)

The user adopted direct method validation after NG1=NG2=G matched internally but
all differed from the historical zero image. Use the first existing source and
its captured pre-write state. A fresh suffix zero, new clean/design RGB
sensitivities, latent VJPs, and four marked suffix outputs share the same
runtime map. Historical pixel mismatch is diagnostic only. Predicted versus
observed readout changes are assessed on those actual writes without extra
probes or retuning. This is a scoped execution of the existing candidate, not a
new reader or new carrier. The other three sources remain unmeasured here.
See `channel_precompensation_single_source_v1.md`.

## Adopted one-shot precompensation candidate (2026-11-08)

After the real single-direction measurement showed reconstruction-direction
mismatch, the user adopted a separate fixed-reader, host-conditioned write-side
candidate. It uses clean/design-reconstruction proxy sensitivities to solve two
fixed QIM inner-cell constraints once, with a matched-information gradient-ray
control. Four new sources, both bits, and a separate reconstruction seed are the
minimal falsification roster. The ordinary-RGB reader remains blind. Quality is
reported without an eligibility gate, as explicitly selected by the user.
Implementation and CPU/static/fake checks do not establish actual one-bit
transmission, attribution, FPR, or general regeneration robustness. See
`channel_precompensation_v1.md`. The stage-A code and results below are retained.

## Reconstruction-channel candidate (2026-11-08)

The adopted next work on `dev/reconstruction-channel-v1` is a stage-A measurement prototype: compare no-write and signed probes from the same pre-injection state using complete same-seed generation replays, then read only ordinary RGB images before and after reconstruction. Measure actual-dtype perturbations, finite-amplitude response, nonlinear residual, host drift, and image quality. This does not establish a shared subspace, bit transmission, watermark presence, attribution, or FPR. See `reconstruction_channel_v1.md` for the candidate boundary and invocation.

The historical LF/HF content-chain specification below remains the baseline. The new path does not require its carrier decomposition, allocation, fusion score, or threshold. The blind image-only reading boundary remains in force. No geometry module or full watermark encoder is added in this stage.

## Research question

Can a generated image carry a key-specific content watermark whose evidence remains detectable from the ordinary image through the frozen content-adaptive simultaneous LF+HF embedding and its clean joint statistic?

The content-adaptive simultaneous LF+HF embedding and calibrated weighted-joint statistic are the validated content-chain method.

The four positive neutral-counterfactual effects establish only that each frozen allocation input changed the allocation. The two public branch-share population standard deviations can support only that allocation was not identical across the fixed eight-image roster, and only when both independently computed RC0 values are finite and strictly positive. They do not expose or recover masks, tile weights, attention, latents, deltas, or probe state.

Local CPU, fake, and runtime checks remain engineering evidence only. The accepted formal result contains four independent physical strata. Geometry, fixed-FPR calibration, attack robustness, and paper promotion remain outside this completion.

## Content-chain decision

The content-chain method progression is complete:

1. the keyed HF attribution anchor and LF content statistic remain distinct inputs;
2. correct-key evidence, external wrong keys, and primary-null controls remain separately represented;
3. the public calibration asset freezes the weighted-joint statistic;
4. the formal stability evaluation retains four independent 8/8/32/32 physical strata without pooling;
5. all predeclared units and failures remain in their physical denominators;
6. the accepted formal result completes the content chain without adjudicating deferred geometry or fixed-FPR claims.

The accepted status is `content_chain_method_complete`. This is a content-method conclusion, not a fixed-FPR detector, geometry, robustness, full-system, or paper-readiness conclusion.

## Deferred work

- LF/HF combined detector calibration;
- crop oracle fault decomposition;
- Q/K synchronization and geometric estimation;
- conditional rectification and same-threshold recheck;
- formal FPR evaluation and paper artifacts.

## Access boundary

Detection may use the current image, detection key, and frozen public assets. It may not use the original image, prompt, embed record, embedding latent, embed-side routing maps, or cached embedding Q/K tensors.
