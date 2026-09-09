# Survival allocator development V1

This branch implements an embedding-only allocator change. V2 content carriers,
ISS host beta, 0.012 joint projection, RGB SyncSeal embedding and continuous RGB
recovery, and registered-minus-max16wrong content scoring are reused unchanged.
No geometry score contributes positive evidence. Labels use the same max(pre,post)
path for baseline and probes; there is no newly calibrated decision threshold.

The shared allocator uses 2x2 macroblock DINO saliency, RGB texture and latent
energy features. Latent energy is a simplified feature, not a perturbation response,
stability estimate, full generation Jacobian or semantic proof. Its learned linear
utility is passed through tanh, centered, bilinearly expanded to the existing 4x4
allocation, and converted into positive unit-mean weights inside (0.5,1.5). LF and
HF share weights and retain equal branch shares before the existing ISS scaling.
LF/HF evidence roles remain unproven. No embed mask or features enter detection.

For each fit pair, a full same-seed independent SD3.5 call reaches callback 18;
four separate macroblock-biased allocator variants are injected there and continue
through step 19 to final RGB. This is real remaining sampling, not decode/encode
label substitution. It replays the prefix for simplicity instead of recursively
calling the active pipeline or saving scheduler internals. Plain, uniform, original
and probes share seed and old final RGB sync; label increments compare each probe
to uniform, both with synchronization. All nine attacks use identical noise seeds
per pair/condition. Candidate original-rule allocation still uses its original
64 decode/encode probes; these are never called generation-Jacobian labels.

Labels retain per-attack final content score increments, final incremental LPIPS
and utility = mean(score increments) - cost_penalty * LPIPS_increment.
The penalty (default 1) is a development hyperparameter with score/LPIPS units;
it is not evidence of matched quality. PSNR, SSIM, LPIPS and four macroblock MSEs
are reported against the same-seed plain final image, including content/sync
interference. Final quality is not inferred from latent L2. Fitted assets store
feature standardization, shared coefficients and fit ids/seeds. Validation refuses
fit id or seed reuse, never refits, and needs a separate prompt/seed roster.

A roster is a JSON list: [{"id":"fit-000","prompt":"a mountain lake","seed":2030000000}].
Choose new ids/seeds and independent fit/validation units before external execution.

```
PYTHONPATH=src python -m experiments.run_survival_allocator_dev --mode plan --roster fit.json --output unused
PYTHONPATH=src python -m experiments.run_survival_allocator_dev --mode fit --roster fit.json --output fit-output --runtime-root runtime
PYTHONPATH=src python -m experiments.run_survival_allocator_dev --mode validation --roster validation.json --output validation-output --runtime-root runtime --allocator fit-output/allocator.json
```

Plan loads no models. Fit/validation load the existing v2 production runtime and
need its public model assets plus HF_TOKEN and CEG_WM_ROOT_KEY. They are prepared
entrypoints, not authorization to run GPU/Colab, push or change Drive. Output paths
must be new; failure rows and generated images remain. Failed fit labels are not
silently discarded into a smaller fit population. Reports are written even if
setup fails. A poor utility or score never suppresses the report or model fitting.

Fit8 gives 56 images and 504 attack score paths; validation24 gives 96 images and
864 paths, total 152 images and 1368 logical scoring paths. Each path internally
uses the existing content keys and pre/post selection. This deliberately reduces
the earlier suggested budget: four local allocation perturbations per fit pair,
not nine separate output generations per probe. Original allocator decode/encode
probes, quality computation and ISS host scores are additional internal work.
This small development split supports neither .1% FPR nor robustness conclusions.
Real model quality, survival signal and allocation benefit are still unverified.
Crop-rescale, JPEG/blur and generative reconstruction remain later research tasks.
