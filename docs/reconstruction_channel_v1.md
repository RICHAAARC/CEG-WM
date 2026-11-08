# Reconstruction-channel measurement, stage A

This candidate is implemented on `dev/reconstruction-channel-v1`. The adopted
first measurement uses **one fixed HF-family input direction and full VAE-mode
features read from ordinary RGB images**. It is a channel diagnostic, not a
watermark encoder or detector. No real-model run is part of the CPU checks.

The concrete four-source user-run Colab configuration is
`configs/reconstruction_channel_v1_colab.json`. Its complete fixed roster, probe
amplitude, model revisions, generation parameters and reconstruction seed are
explicit; there is no additional preflight source outside this roster.

## Mechanism

For each source prompt and generation seed, run the complete SD3.5 pipeline three
times: no write, positive probe, negative probe. Every call starts with a fresh
generator using the same seed. Inject at the end of the selected denoising step
(the adopted existing position is zero-based step 18 of 20). Compare the actual
pre-injection latent, conditioning, schedule, and callback state across arms.
An unmatched state invalidates the paired measurement; it is retained as a
failure. There is no captured-latent suffix restart or nested pipeline call.

The fixed probe uses the historical HF family: real Rademacher coefficients in
the normalized radial shell `[0.58, 1]`, followed by `irfft2` and unit-L2
normalization. Its public probe seed, new measurement domain, and tensor shape
determine the direction, independently of source prompt/seed. It is not the old
key's bitstream. Only this direction is reused: no old LF allocation, whitening,
ISS adjustment, HF score, fusion, threshold, or SyncSeal module is loaded.
Probe strength is an explicit relative-L2 input, not an inherited `0.012` budget.

Measure the actual post-cast changes in callback dtype, including projected
signed amplitude, off-direction energy, and asymmetry. Quantization can erase
or alter a nominal probe. Finite responses must use actual signed amplitudes;
they must not divide blindly by the requested strength. Off-direction changes
are reported and mean the endpoints are not an exact one-dimensional path.

The reader accepts only an RGB PIL image or uint8 RGB array and public VAE and
processor. It returns the complete flattened deterministic VAE posterior mode,
with the VAE's shift/scaling convention. It never accepts the private generation
latent, original image, prompt, or injection record as reader input.

Reconstruct all three RGB outputs using the same explicit SDXL img2img settings
and reconstruction seed, with a fresh generator for each arm. Read reconstructed
RGB through the same fixed VAE. Signed finite response, endpoint deviation,
and the reconstruction drift of the no-write host are separate measurements.
The endpoint deviation is not by itself evidence of nonlinearity: noncollinear
post-cast perturbations can produce it even for a linear feature map.
Original/paired images are permitted for this offline diagnostic, not a future
blind detector. Pixel RMSE/PSNR describe distortion, not perceptual equivalence.

## Invocation and configuration

From this checkout, with its dependencies available:

```bash
PYTHONPATH=src python experiments/run_reconstruction_channel_v1.py --config measurement.json
```

The default validates the configuration and prints the planned calls without
loading models or creating an output directory. A later explicitly selected
model run uses `--execute --output <new-directory>`. The directory must be new;
the runner does not overwrite results or silently retry failed sources.

Real execution generates the entire batch first, removes the SD3 pipeline hooks
and releases its large components, and only then loads SDXL. The same SD3 VAE
and processor remain the reader. `memory_mode: "model_cpu_offload"` is selected
in the Colab config for both pipelines; the general API also supports `direct`.
`collection_complete` records successful data acquisition separately from the
validity of a finite-response estimate. A collapsed actual probe can make the
estimate invalid while collection is complete; it does not justify selecting
only sources with a favorable response.

Configuration fields are explicit. The following is a schema illustration;
the null fields must be selected before it is a runnable configuration, and the
empty roster is not a scientific sample:

```json
{
  "generation": {
    "model_id": "stabilityai/stable-diffusion-3.5-medium",
    "revision": null,
    "dtype": "float16",
    "device": "cuda",
    "height": 512,
    "width": 512,
    "num_inference_steps": 20,
    "guidance_scale": null,
    "injection_step": 18
  },
  "probe": {"seed": null, "relative_l2": null},
  "feature": "full_vae_mode",
  "reconstruction": {
    "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
    "revision": null,
    "dtype": "float16",
    "device": "cuda",
    "strength": null,
    "num_inference_steps": 20,
    "guidance_scale": 1.0,
    "prompt": "",
    "seeds": []
  },
  "sources": []
}
```

Each source has `source_id`, `prompt`, and integer `seed`. All sources share
dimensions, probe direction/strength convention, readout, and reconstruction
settings. With `n` sources and `r` reconstruction seeds, the planned cost is
`3n` complete generations, `3nr` reconstructions, and `3n(1+r)` feature reads.
One reconstruction seed identifies a conditional response; it does not estimate
reconstruction variability. No extra capture generation is hidden in this count.

Saved outputs include the configuration, fixed probe, RGB images, raw readout
features, actual perturbations/state diagnostics, and source measurements.
Failed/missing arms and sources remain in the planned denominator. A failed
triple does not produce an eligible response estimate. No aggregate scientific
PASS is emitted.

The focused engineering checks use CPU arrays and fake pipelines, including a
complete runner-to-files test:

```bash
python -m pytest -o addopts= tests/unit/test_reconstruction_channel.py tests/unit/test_reconstruction_channel_runner.py tests/integration/test_reconstruction_channel_runtime.py -q
```

## What this can establish

CPU/fake tests establish state bookkeeping, ordinary-RGB reading, arithmetic,
and failure retention only. Real execution can measure finite response of this
one direction at the selected amplitude and conditions. It cannot yet establish
a derivative/Jacobian, transferable shared subspace, blind separation from host
content, bit transmission, presence detection, attribution, or FPR. In particular,
pairwise subtraction removes host terms in an offline diagnostic; a future blind
reader cannot use that subtraction. Failure for this HF-family direction does
not exclude other latent directions.

The next scientific decision, after real measurements exist, is whether a
source-independent readout can retain signed response against host variation on
held-out sources. No sourcewise sign alignment or response-selected exclusions
are permitted to manufacture such transfer. Shared-map fitting and coding are
later decisions, not implemented in this prototype.
