# Colab Run all delivery: self-reference v1

Open `notebooks/CEG_WM_SelfReference_V1_Colab.ipynb` through the published research
branch in Colab and choose **Run all**. All parameters and paths are supplied.
Necessary interactions are Google Drive mount and Hugging Face gated-model access
(Colab Secret `HF_TOKEN`, or hidden getpass). A completed cache can be reused without
an HF token. Tokens stay in memory/environment, never command arguments, Drive
files or logs. Subprocess output is redacted before console/Drive tee; no login
token is saved with `huggingface_hub.login`.

**The capacity-policy update has not been rerun in real Colab.** The user's first Run all
stopped at venv's ensurepip invocation, before any model loading or training.
Format, cell compilation and CPU/fake tests are engineering evidence only.
Model access, full dependency installation on the user's
current Colab image, true-model numerics, GPU memory/time and scientific results
remain unverified. A single Colab session may not complete the frozen 3×10000
updates; reconnect and rerun all to resume verified artifacts.

## Frozen identities and actual source review

* Scientific config remains
  `6c74bd18231e18f56b4dd6e717804624e5f78929511645e108a4d8971c866495`.
* Generator config hash:
  `5e882f0b0aeed62aa2cce06232e5018f5a0f4b602a3f4ae51c5f2cab17ee6b00`.
* Execution profile hash:
  `c1497138ecccadf756f05231c32ec6437958fe283d45ae4e78e362ca0f1c56b7`.
* Authored roster file SHA256:
  `fc232cd7249fc725017432cb323c9c8e5bff738b68841355da05b482c9d345c0`.
* Model: `stabilityai/stable-diffusion-3.5-medium`, immutable HF revision
  `b940f670f0eda2d07fbb75229e779da1ad11eb80`, verified through the
  [official metadata API](https://huggingface.co/api/models/stabilityai/stable-diffusion-3.5-medium).
  The [model card](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium)
  requires accepting its license gate.
* [Diffusers v0.35.1 SD3 source](https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py):
  the latent-output branch returns the post-scheduler latent directly. Only the
  RGB-output branch applies division by VAE scale plus shift before decoding.
  The generator checks the pinned implementation and saves `output_type=latent`
  directly as final scaled z0; it never substitutes an RGB-reencoded latent.

The notebook is published in a second commit and pins the already-created code
commit. It checks out that exact SHA in its own local directory. Run ID binds code
commit, scientific/generator/execution hashes, model revision and roster hash.
The same code/run is reused on rerun; a changed code/profile cannot silently resume
the old experiment. Original `README.md` and untracked skill changes are excluded.

## Data, model and storage

`configs/self_reference_v1_sources.json` contains 640 concrete scene prompts,
authored by `scripts/build_self_reference_sources.py`; no external images/data are
downloaded. There are 80 thematic families with eight different scenes each.
Whole families are assigned to one split: train 256/32 families, dev 64/8,
model_selection 64/8, calibration 128/16, test 128/16. `source_id` is an image
sampling unit, `source_group` and `template_family` are the thematic family.
Generic photographic wording is shared; family grouping reduces obvious template
leakage but is not proof of semantic or generated-pixel independence.

Each split has 16 disjoint deterministic true writing keys. Generation seeds,
writing-key assignment and reference nonces use separate hash domains; changing
source content does not select a nonce/message based on that content. No prompt is
repeated across splits by merely changing its seed. A records image-unit Wilson
intervals with dependence caveats. B's rate and rescue/harm stay **image-level**:
one source's keys and four directions must all succeed. Bootstrap resamples whole
families, aggregating their image outcomes. Reports show both image and family
denominators (dev 64/8, test 128/16); few clusters limit inference.

Generation uses 512×512, 40 inference steps, guidance 4.5, max sequence length 256,
BF16 denoising and all three original text encoders (including T5). No quantization
or encoder removal. Model CPU offload is enabled. The pipeline loads once in one
generation subprocess and handles all pending sources, with per-source atomic
latent/marker writes and visible progress. It exits before a separate clean
subprocess loads the original VAE in float32 and decodes all pending clean RGB8.
The clean marker binds the actual saved latent SHA256.

Preflight requires CUDA availability and the frozen BF16 support dependency.
Actual GPU VRAM and host RAM are displayed and recorded as informational values;
there is no capacity admission threshold or GPU-model-class requirement. Actual
model stages can still exhaust memory. No automatic smaller model, precision,
resolution, batch, data or update budget fallback is allowed.

Drive root is `/content/drive/MyDrive/CEG-WM/self-reference-v1/`. Model snapshots
are persistent under `models/<revision>`; runs live under `runs/<identity-prefix>`.
Expect **tens of GiB** of storage; a typical 15GB Drive is insufficient. Before
large downloads, official HF per-file sizes are requested, missing bytes summed,
and an additional 8GiB reserved for duplicate VAE, cache, checkpoints and quality
assets. Insufficient reported DriveFS space stops before large download. DriveFS
free space can differ from shared-account quota; that remains a practical limit.
Downloaded weight shards are verified against official LFS SHA256 and config files
against official Git blob IDs. The denoiser identity is SHA256 of canonical JSON
mapping every transformer shard/config filename to its SHA256. Full VAE file
inventory is separately recorded and checked.

## Execution, recovery and safe stopping

The parent notebook never imports torch or retains a model. Download/quality
preparation, hardware checks, generation, clean decoding, each arm's training,
selection, dev evaluation and test evaluation are separate subprocesses. Their
stdout/stderr is visible and tee'd to Drive. Each sample/update reports progress.
The summary keeps A metrics, quality status, compact B rates/CIs and paired gains,
plus stop/error reasons; it does not print raw record arrays.

Before any download/generation, `cache_status` validates existing per-source
metadata/hashes, paired clean-latent identity, local VAE, and an existing manifest.
A complete cache skips full model preparation, generation and decoding, even if
the full snapshot has been removed. Partial cache restoration uses source markers
even when manifest creation was interrupted. Orphan files without completion
markers are quarantined, not treated as complete. Completed hash or identity
conflicts fail explicitly. Files are flushed and atomically renamed before the
completion marker; the pair is considered complete only when both validate.

Training keeps effective batch4 and 10000 AdamW updates per arm. The Colab profile
uses microbatch1, four loss/4 backward passes, then exactly one optimizer step.
Nonreentrant `torch.utils.checkpoint` wraps the actual decode/encode functions;
VAE eval mode alone is not called gradient checkpointing. VAE and reader stay
float32. Generation/clean/training/selection/evaluation share deterministic
algorithms, disabled cuDNN benchmark and TF32, deterministic cuDNN, highest float32
matmul precision, and `CUBLAS_WORKSPACE_CONFIG=:4096:8` set before process startup.
Runtime versions, CUDA/GPU identity and execution hash are frozen. Unsupported
deterministic GPU kernels can still fail; CPU tests do not prove GPU compatibility.

Each 100 updates writes a recovery snapshot; each 1000 writes one of exactly ten
selection candidates. Weights, optimizer, RNG, update and common input signatures
are restored. Latest corruption falls back to the latest verified checkpoint.
If a fixed selection candidate is missing, replay starts before that missing
candidate, and all later snapshots/sidecars are quarantined to avoid overwrite
collisions. A completed arm is skipped only when all ten valid candidates including
10000 exist. Recovery files never add selection opportunities. Identity/profile
conflicts abort rather than starting a different experiment in place.

Selection needs all arms complete and only reads model_selection. Dev A failure
persists its report and stops shifted evaluation/test dispatch. Training's fixed
shift augmentation is unchanged by this evaluation gate. Subsequent notebook cells
safe-skip after errors, interrupts or gate stops. `execution_completed` is separate
from scientific status; missing or failed quality never means research PASS.

## Dependencies and LPIPS

The isolated venv uses `--system-site-packages --without-pip` to reuse Colab
torch/torchvision without invoking unavailable/broken ensurepip. Every rerun safely
recreates/completes the venv even when a failed prior attempt left its Python
executable. Host pip>=22.3 installs through `host-python -m pip --python target-python`;
the notebook checks target sys.prefix and purelib/platlib belong to the venv and
the host prefix/site identity stays unchanged. There is no system-install fallback.
Every pip install uses `--no-user --no-deps`, system torch version constraints and binary-only
dependency wheels; the repository is installed editable with `--no-deps` and no
build isolation. Dependencies include sentencepiece≥0.2.1 (no forced old source
build). No torch upgrade is attempted. After pip, a child process imports actual
torch/torchvision, StableDiffusion3Pipeline, AutoencoderKL, T5EncoderModel,
CLIPTextModel/WithProjection, sentencepiece and LPIPS without loading weights.
Incompatibility stops visibly. The exact Colab image was not tested locally.

In Colab only, quality preparation downloads the official
[torchvision AlexNet weights](https://github.com/pytorch/vision/blob/v0.23.0/torchvision/models/alexnet.py)
with the published checksum prefix and the immutable
[LPIPS v0.1 AlexNet calibration file](https://github.com/richzhang/PerceptualSimilarity/blob/082bb24f84c091ea94de2867d34c4544f68e0963/lpips/weights/v0.1/alex.pth)
(Git blob `1df9dfe62abb1fc89cc7f82b4e5fe886c979708e`). It explicitly copies the
pretrained backbone feature layers and calibrated linear weights into LPIPS,
exports the **complete** state_dict, and records full asset SHA256 values. No random
backbone or linear-only file is used as the final metric. These weights were not
downloaded or loaded on the implementation machine.

## Reproducible local engineering validation

Use `/home/richar/projects/CEG-WM/alive/CEG-WM/.venv/bin/python` in WSL Ubuntu-24.04:

```bash
.venv/bin/python -m pytest tests/unit/test_self_reference_colab.py tests/unit/test_self_reference_v1.py tests/unit/test_final_image_observation.py -q
.venv/bin/python scripts/build_self_reference_notebook.py --code-commit <already-created-code-commit>
```

At implementation review: **49 passed** (2.93s), including 13 new Colab tests.
The final lightweight unit sweep was **187 passed, 2 pre-existing golden-hash
failures, 16 deselected** (6.12s), using `.venv/bin/python -m pytest tests/unit -q`.
Tests cover accumulation versus original batch4 (weights and Adam gradient moments
within stated float tolerances), bitwise exact resume within one profile, partial
cache markers before manifest, corrupt latest and missing-candidate replay, family
CI with image-level rate/rescue, full fake runner reruns, A-fail non-dispatch,
token redaction, code/profile mismatch rejection, notebook stage-cell execution,
all cell compilation and nbformat validity. CPU stub spies verify shared numerical
settings precede clean/train/dev adapter loading; they are not real GPU tests.
The notebook is saved with empty outputs/execution counts. Full Colab execution
and rendered-result inspection remain unperformed; perform Run all with the stated
hardware, access and storage to validate those runtime outcomes.

The original repository's two golden-hash failures and global collection/import
limitations documented in `self_reference_v1_delivery.md` remain unchanged. They
are not repaired or hidden in this delivery. No merge to main is authorized.

### Ensurepip bootstrap repair

The earlier bootstrap-only repair changed the notebook/builder, validation test
and this document while retaining execution code
`984955b1049af48f2f93bd32d91c0bd99c37736e`; scientific configuration, training,
generator/profile hashes and Drive run identity were unchanged in that repair. All required
bootstrap helpers are embedded in the notebook rather than imported from new code
missing in that fixed execution commit.

The real CPU subprocess test constructs a standard tiny wheel with Python's
`zipfile`, with no network/build dependencies. It starts with an existing
`--without-pip` venv, invokes the notebook's actual helper, installs via host pip's
`--python`, and repeats environment creation and installation. Both passes verify
the target prefix/site and installed module path, no local pip bootstrap, preserved
existing state, host `find_spec` returning None, and unchanged host package list,
prefix/site and environment. No model, GPU or external package install occurs.

```bash
.venv/bin/python -m pytest tests/unit/test_self_reference_colab.py::test_notebook_without_ensurepip_repairs_partial_venv_and_targets_local_wheel -q
```

Observed: **1 passed in 2.99s**. The prior immutable notebook link at `819ed99`
does not update; open the newly delivered immutable link or the research-branch
Colab link for this repair.

### Capacity reporting update

GPU/RAM capacity rejection was subsequently removed at the user's request. The
generator policy is now `capacity_policy=report_only`; CUDA and BF16 checks,
model access, disk/integrity checks and all scientific budgets remain intact.
This execution-code change is committed first, then the notebook pins that new
code SHA in a second commit. The code/generator identity therefore creates a new
Drive run directory. Old directories are retained, not reset or mixed into the
new run. The `--without-pip` plus host pip `--python` bootstrap remains embedded.

CPU mocks verify an 8GiB GPU / 12GiB RAM passes hardware preflight and records both
actual capacities when CUDA/BF16 are available. Missing CUDA or BF16 still raises
the corresponding dependency error. No weights or real GPU are loaded. The three
related test files passed **53 tests in 4.94s**, including the real offline tiny-wheel
bootstrap recovery test and the existing scientific/observation checks.
