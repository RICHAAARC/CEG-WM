# Self-reference relation v1: frozen engineering pilot

This is an **unvalidated, isolated candidate**, not a replacement for the completed
CEG-WM LF/HF content chain. No true-model result is included. Existing observation,
detector, thresholds, historical failures and default runtime are unchanged.
Self-reference demodulation and latent writers with RGB extractors have precedents;
the basic formula is not claimed as novel. Old oracle 7/8 versus direct/predicted H
0/8 motivates exploration only. The stopped cross-spectrum oracle result (correct
key top1 1/8 versus baseline 8/8) is not reinterpreted. Old images contain one true
writing key, not 17 independent writes, and cannot validate this new carrier.

## Frozen implementation

`configs/self_reference_v1.json` is checked against `Config()` exactly. The v1 config
hash is `6c74bd18231e18f56b4dd6e717804624e5f78929511645e108a4d8971c866495`.
The implementation is `src/cegwm/self_reference/`; its public constants and this
document are part of the protocol, not a hyperparameter search interface.

* Frozen SD3.5 denoiser produces final **scaled** `z0`, shape `16×64×64` float32.
  The frozen VAE decodes `zw = z0 + delta`; this is not the previous step-18 path.
  This implementation trains from a validated local cache and does not load the
  denoiser. Its identity and generation provenance must accompany the cache.
* Writer and reader each use three `Conv2d(3×3,padding=1)` + SiLU layers, width 64,
  then independent `1×1` heads. Explicit neural radius is 3 latent sites, same-zero
  padding. Writer residual is `0.05*tanh(head)` in scaled latent coordinates.
* B0 writer sees `(z,broadcast(b))`; B1 sees `(z,p,broadcast(b))`; C sees
  `(z,p,p*b1,...,p*b32)`. B0 has no auxiliary reference task. B1 has exactly C's
  reference supervision but ordinary bit extraction. These are matched-path
  baselines, **not official DistSeal** implementations.
* Reader consumes only an image's frozen-VAE observation. It has a shared local
  trunk and separate local payload/reference heads, with no key, nonce, original,
  reference, attention, global pooling before evidence, reference-to-payload
  feedback, adaptive mask, alignment or per-image selection.
* B0/B1 use a fixed spatial mean of logits, BCE during training, and `mean(b*logit)`
  for ranking. C uses float32 spatial sums:
  `rj=2*sum(p_hat*q_hat_j)/(sum(p_hat²)+sum(q_hat_j²)+1e-8)` and `S=mean(b*r)`.
  The only correlation stabilizer is `1e-8`. Query keys enter only this final
  score. Score margins are not compared across ordinary and relation readers.
* The common scoring/supervision region removes 8 latent sites from every edge.
  This includes the explicit reader radius but **does not isolate VAE attention**.
  The VAE may have global attention. An implicit classify-then-synthesize shortcut
  remains possible; the architecture and four-write diagnostic do not exclude it.
* Reference `p` is source/message-independent: 15 Gaussian coefficients on public
  separable DCT bases `(u,v)` in lexicographic u-then-v order, u,v=0..3 excluding
  (0,0), then unit RMS on the full field. Generation nonce is writer/supervision
  metadata only. Reference draws are independent of source/key sampling; finite
  Gaussian sampling is deterministic given a nonce. CPU torch RNG/version is
  pinned by the checkpoint, not a cross-library Gaussian standard.
* Key bits are SHA256 of the compact JSON array
  `["cegwm-self-reference-v1","message-bits",key]`, first four bytes, MSB first,
  bit 1→+1 and bit 0→−1. Wrong query number t is `wrong-v1:` plus SHA256 hex of
  the corresponding `wrong-key-query,key,t` domain array. Start t=0 and skip key
  duplicates or 32-bit collisions until 16 wrong queries exist. The same K17 is
  used for all arms/attacks; strict ties fail. UTF-8 JSON uses ensure_ascii=True.

Parameter counts (VAE excluded):

| Arm | Writer | Reader | Total |
|---|---:|---:|---:|
| B0 | 102608 | 85216 | 187824 |
| B1 | 103184 | 85281 | 188465 |
| C | 103184 | 85281 | 188465 |

The 641 parameter difference is approximately 0.34%, reflecting useful auxiliary
inputs/heads. No dummy task or unused parameters are imposed on B0.

## Training and RGB boundary

Each arm runs 10000 updates, batch 4, AdamW lr=1e-4, weight_decay=1e-4, seed=20261118.
All arms use the same sorted unit roster and stateless update/slot schedule for
source/message and attacks. B1/C reference nonces also match step by step.
Separate models, optimizers and RNG snapshots are saved. Training samples units
uniformly with replacement; a Cartesian source/key roster is allowed and determines
the source/key weighting. Do not change that roster between arms.

Bit loss is BCE on spatial mean logits for B0/B1 and `mean(1-b*r)` for C.
B1/C reference MSE weight is 0.1. All arms have ordinary RGB reconstruction MSE
weight 10 and scaled latent residual MSE weight 0.01. RGB loss uses quantized STE
pixels before attack against verified same-z clean RGB8. This initial budget is
not a guarantee that a writable channel exists or that optimization will find it.

For each sample, uniformly select identity or (-2,0),(2,0),(0,-2),(0,2) RGB pixel
translation. Tensor slicing gives exact integer displacement with zero RGB padding,
no interpolation. All arms share this attack schedule. Reference supervision is
explicitly a **bilinear latent proxy** translated by `(dx/8,dy/8)` via
`grid_sample(...,padding_mode="zeros",align_corners=False)`. It is not an assertion
that encoding a shifted RGB image equals shifting its latent or its reference.
For dx>0 the target moves right (inverse-grid offset is negative). The fixed
central mask is applied to prediction and target. Identity supervises unshifted p.

Training uses a separate differentiable adapter: decode `z/scale+shift`, convert
[-1,1]→[0,1], clamp, round-to-nearest `RGB8/255` with straight-through gradient,
then normalize to [-1,1], VAE posterior **mode**, `(mode-shift)*scale`.
All VAE parameters are frozen; gradients still flow through its operations.
Formal evaluation always uses PIL/uint8 and existing
`runtime.observation.encode_final_rgb_image`, which detaches and runs under no_grad.
The processor must implement the exact frozen RGB/normalization convention;
configuration checks plus a full-size pixel probe reject unsupported behavior.
Finite values and expected shapes are checked before quantization and encoding.
This is an adapter convention check, not a proof of an arbitrary processor's code.

## Local cache manifest contract

No generator is included; the cache is the supported input interface. A cache
exporter must save the final SD3.5 scaled z0 **before decode**, not RGB-reencoded
latents, intermediate noisy latents or unscaled VAE codes. Keep the original model
revision and denoiser digest, sampler settings, prompt digest, generation seed,
source identities and augmentation-family grouping. All paths below are relative
to the manifest directory and may not escape it; put/copy the local VAE export
inside that directory. The export should be an `AutoencoderKL.save_pretrained`
directory containing `config.json` and all local weights, with no extra files.

Example shape (replace placeholders with real hashes and provide all five splits):

```json
{
  "schema": "self-reference-cache-v1",
  "config_hash": "6c74bd18231e18f56b4dd6e717804624e5f78929511645e108a4d8971c866495",
  "latent_convention": "final_z0_scaled_before_decode",
  "model": {
    "family": "SD3.5", "id": "exact-model-id", "revision": "immutable-revision",
    "denoiser_sha256": "64-hex-content-digest-of-denoiser-artifact",
    "vae_export": "vae",
    "vae_files": {"config.json": "sha256", "diffusion_pytorch_model.safetensors": "sha256"}
  },
  "keys": {"train": ["train-k1"], "dev": ["dev-k1", "dev-k2"],
           "model_selection": ["selection-k1"], "calibration": ["cal-k1"], "test": ["test-k1"]},
  "units": [{
    "unit_id": "train-source001-key001", "split": "train",
    "source_id": "source001", "source_group": "original-source001-family",
    "source_sha256": "digest-of-original-source-record",
    "key": "train-k1", "reference_nonce": "independently-sampled-256-bit-hex",
    "latent": "z/source001.npy", "latent_sha256": "sha256",
    "clean": "clean/source001.png", "clean_sha256": "sha256",
    "provenance": {"generation_seed": 123, "prompt_sha256": "sha256",
                   "sampler": "exact-name-and-config", "steps": 28, "guidance_scale": 7.0}
  }]
}
```

Use `secrets.token_hex(32)` independently for each evaluation unit's reference
nonce; never derive it from the image, source, key or message. Training nonces are
domain-separated draws from seed/update/slot, independently of source sampling.
Each split's true writing-key set is disjoint, with no 32-bit collisions even
between different keys. Multiple images may use the same key **within** a split.
Source identity/group/content hashes and exact latent/clean hashes cannot cross
splits. Augmented copies must retain source_group. Different keys on the same
source may share z/clean assets within their split but have unique unit IDs/nonces.
The validator cannot independently prove a declared source family or model origin;
honest immutable provenance is required. Checksums bind declared artifacts, not
the historical generation event.

`.npy` is finite float32 CHW 16×64×64 without pickle. Clean PNG is RGB512 decoded
from exactly the same z and frozen VAE using the stated clamp/round convention.
`validate` checks all shapes, file hashes, VAE file inventory and split rules
without loading a VAE. Execution additionally re-decodes same-z clean and requires
pixel equality (first source use in training, every evaluation unit). Mismatches
are engineering errors, not low-quality watermark results. VAE runs in float32;
cache clean images made with other precision must be regenerated from saved z
with this convention before freezing the manifest.

Suggested source-group counts: train256, dev64, model_selection64, calibration128,
test128. Tiny fixtures are accepted for engineering but never called formal data.
Train is for optimizer updates; model_selection is only for checkpoint ranking;
dev is a predeclared A investment gate and optional B pilot; calibration is unused
and reserved for future independent calibration; test is the locked final report.
Do not move sources/keys between roles or use dev/test to choose checkpoints.

## Commands (explicit execution; local assets only)

Run from the repository with its environment and `PYTHONPATH=src`:

```bash
python -m cegwm.self_reference plan --config configs/self_reference_v1.json
python -m cegwm.self_reference validate --manifest /cache/manifest.json --execute
# Required before starting any CUDA training Python process:
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m cegwm.self_reference train --manifest /cache/manifest.json --run-dir /runs/sr-v1 --arm all --device cuda --execute
# Resume just one interrupted arm; --stop-after N is an interruption, not a new budget.
python -m cegwm.self_reference train --manifest /cache/manifest.json --run-dir /runs/sr-v1 --arm C --resume /runs/sr-v1/C/update_004000.pt --device cuda --execute
python -m cegwm.self_reference select --manifest /cache/manifest.json --run-dir /runs/sr-v1 --output /runs/sr-v1/selection.json --device cuda --execute
python -m cegwm.self_reference evaluate --manifest /cache/manifest.json --selection /runs/sr-v1/selection.json --split dev --output /runs/sr-v1/dev.json --lpips-weights /weights/lpips-alex-full.pt --device cuda --execute
python -m cegwm.self_reference evaluate --manifest /cache/manifest.json --selection /runs/sr-v1/selection.json --split test --dev-report /runs/sr-v1/dev.json --output /runs/sr-v1/test.json --lpips-weights /weights/lpips-alex-full.pt --device cuda --execute
python -m cegwm.self_reference diagnose --manifest /cache/manifest.json --selection /runs/sr-v1/selection.json --output /runs/sr-v1/four-write.json --device cuda --execute
```

Without `--execute`, every command only prints the plan; import/plan never loads
a model. There are no network calls to fetch a model. VAE loading requires a
local export and `local_files_only=True`. LPIPS is optional: to measure it, install
its package separately and supply a trusted **complete** AlexNet LPIPS state_dict
(backbone and linear calibration weights). `pnet_rand=True,pretrained=False`
prevents implicit backbone/pretrained downloads, followed by strict full-state
loading. Missing metric/weights yields not_evaluable quality, never research PASS.
Do not substitute random LPIPS weights. The checkpoint's hash is reported.

At each 1000 updates save writer, reader, optimizer, update, Python/NumPy/torch/
CUDA RNG, next-input signature, identity hashes, environment and elapsed time.
An interruption can save an additional resumable snapshot, which is excluded
from selection. All arms must complete the same 10000 budget and all ten frozen
checkpoint opportunities. Selection ranks only model_selection top1, then bit
accuracy, then earlier update. Frozen selection locks file hashes. Resume requires
identical manifest/config, torch version and device; deterministic algorithms are
enabled. CPU fixture verifies bitwise-identical weights, optimizer, RNG and loss
after interruption. This does not guarantee cross-hardware GPU determinism.
For deterministic CUDA matrix operations, the frozen environment setting is
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, exported **before** Python/CUDA initialization.
CLI training rejects a missing/different value before loading any model or
initializing CUDA. The direct training function also checks it. No GPU validation
was performed; other unsupported deterministic kernels can still fail explicitly.

## Gates, reporting and limits

A reports zero-shift strict K17 correct-key top1, bit accuracy and whole-message
exact separately, for all arms and all units. Pilot continuation requires C top1
>=95% and no >2pp shortfall relative to either baseline. Baseline pass/fail is
reported independently. Failed C with any usable baseline stops C v1 and skips B;
failure of all arms is implementation/budget/optimization unresolved, not proof of
nonexistence. Incomplete units block continuation. Wilson 95% intervals accompany
fixed-denominator unit rates; within-source dependence limits their interpretation.

B executes only after the **dev** A gate passes; test never opens or adjusts gates.
The primary B unit is `source_id` (one source image): all predeclared writing keys
and all four directions must have strict correct-key top1. Report directional rates,
paired image-level rescue/harm and 2000 `source_group` cluster-bootstrap draws
(seed fixed). Repeated images from a thematic family are resampled together, not
conjoined into a new family-level all-images success criterion. C must improve
>=10pp over **both** baselines for the pilot investment target. That is not a
statistical significance criterion. Failed/no-evidence/low-energy units remain in
the denominator; per-unit engineering exceptions are explicit, not skipped.

Quality uses unshifted ordinary RGB8 paired with same-z clean RGB8 for **all**
predeclared units, without dropping low quality. Each arm needs mean PSNR>=30dB
and mean LPIPS<=0.05. C-minus-each-baseline mean PSNR must be >=−0.5dB and mean
LPIPS <=+0.005. Report p05 PSNR/p95 LPIPS, all values and paired differences. These
are engineering pilot budgets, not universal perceptual equivalence. Missing
measurements or an infinite-PSNR pair that prevents finite subtraction conservatively
yields not_evaluable (exact-RGB cases are explicitly counted). A can pass and B can
be measured without quality, but research continuation cannot pass until complete
quality matches. Total evaluation cost is separated from synchronized
image→VAE→reader→K17 timing; the latter includes PIL/CPU transfers and key mapping.

The four-write diagnostic fixes the first dev host and first two dev keys, using
the first two independent dev nonces for (p1,b1),(p1,b2),(p2,b1),(p2,b2). It outputs
ordinary RGB8 images and blind scores/reference MSE. B0 deliberately repeats the
same write when only p changes. This can expose constant-reference/control errors
but cannot rule out an ideal classify-then-synthesize mechanism.

`A(F)=(A p)[1,b...]` motivates the hypothesis for a shared linear spatial channel;
it is not proven for the learned nonlinear writer, RGB8 boundary, VAE or attacks.
The relation-residual identity is algebra, not a second independent evidence source.
No calibrated TAR/FPR is reported. Wrong-key top1 is not natural-image FPR; even
0/100 false positives only bounds a binomial rate at about 2.95% (one-sided 95%).
Stage C (fixed-reference ablation, independent seeds, small rotation, real sync
interface, FPR calibration) is deferred. ±0.5° is not ±2px; this pilot makes no
arbitrary crop, large-angle, SyncSeal compatibility or official-baseline claim.

## Engineering audit checklist

* [x] Legacy LF/HF and observation semantics unchanged; module imports no governance.
* [x] Key restricted to final reader scoring; local reference/payload heads independent.
* [x] Reference sampling independent; domain-separated bits/queries with collisions skipped.
* [x] Exact RGB shifts and fixed common mask; latent supervision proxy explicitly labeled.
* [x] Split/key/nonce/hash checks; same-z clean check; shared schedule; no test selection.
* [x] CPU fake-VAE train→interrupt→resume→select→dev/test evaluation/diagnostic exercised.
* [x] Resume model/optimizer/RNG/input equality, gate stop, failed denominator tested.
* [ ] Real SD3.5 cache/processor compatibility and GPU feasibility measured.
* [ ] Real A writing/key distinction, B gain and full PSNR/LPIPS quality measured.
* [ ] Scientific hypothesis or novelty established. **No such claim is delivered.**

Lightweight checks: `python -m pytest tests/unit/test_self_reference_v1.py -q`.
The tests use a deliberately small fake VAE/config and do not download models,
instantiate SD3.5, run real generation or validate the proposed carrier.

The separately versioned Colab execution/generation delivery is documented in
`docs/self_reference_v1_colab.md`. It adds microbatch accumulation and explicit
nonreentrant VAE checkpointing without changing this scientific config hash,
effective batch or update/selection budgets.
