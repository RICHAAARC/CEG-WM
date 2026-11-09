# V2 local implementation delivery — 2026-11-09

P1 is implemented and locally reviewable. This is an engineering delivery,
not a real-model P2 result, scientific PASS or innovation verdict. No model
weights were loaded, no GPU/Colab experiment was started, and nothing was
pushed or merged by this implementation chat.

Checkout: `/home/richar/projects/CEG-WM/worktrees/Paper-Main-Reconstruction-V2`.
Branch: `dev/reconstruction-channel-v1`.
Base HEAD: `805846f13dfc46752643fb1471aa85f4c2e1643d`.
The checkout was clean at intake; this delivery remains an uncommitted local
working-tree patch for source-session review.

## Review entry points

- Method/config/counting map: [stage_protection_v2.md](stage_protection_v2.md).
- Colab delivery: [stage_protection_v2_colab.ipynb](../notebooks/stage_protection_v2_colab.ipynb).
- Frozen configuration: `configs/stage_protection_v2_colab.json`.
- CPU solver and grouped verdict: `src/cegwm/stage_protection_v2.py`.
- Capture/suffix/model adapter: `src/cegwm/runtime/stage_protection_v2.py`.
- Operation saving/recovery: `src/cegwm/runtime/stage_protection_store.py`.
- Credential reuse and persistent logs: `src/cegwm/runtime/stage_protection_colab.py`.
- Four-trial execution and result summary: `experiments/run_stage_protection_v2.py`.
- Notebook builder: `scripts/build_stage_protection_v2_notebook.py`.

The notebook embeds the research source snapshot (68 Python/config/requirements
files) and works independently of publishing this patch. Its first cell contains
only Drive mounting. Under the user's subsequent publication authorization,
default Run all installs the pinned dependencies and directly executes fixed P2
with `--execute`. No run checkbox, source upload or manual file editing is required.
GPU metadata is preset, without guaranteeing Colab allocation or GPU type.
Existing environment/cached/secret credentials are tried first. Only actual
401/403 access denial triggers the necessary blocking hidden-token login prompt;
an existing usable login does not invoke `notebook_login` or request another login.
`runner.log` receives direct child stdout/stderr and tracebacks; `setup.log` keeps
installation output and setup failures. Both append across invocations.

Old one-shot and LF/HF paths/results are retained. The only shared-runtime code
change is an optional post-step callback for capturing step18 inside the common
suffix. Old callers leave it unset. Authorization/scope documents are prepended
with the current authorization while preserving the historical entries.

## Validation

Final publication-preparation suite: **74 passed in 22.54s**. `compileall`,
notebook regeneration/source consistency checks, credential-literal scanning
and `git diff --check` also passed. No real-model execution was performed.

The source-audit summary correction passed **66 tests in 21.98s** (the original
65 plus one unmarked-holdout failure regression). The publication preparation
adds credential/logging checks and replaces plan-only notebook checks with the
actual Run all execution route using CPU subprocess substitutes. It includes the new solver,
full synthetic four-trial orchestration, independent holdout failure handling,
actual runtime with multilayer CPU fake models, source/SDXL ownership, complete
physical envelope accounting, save/restore, notebook checks, and retained
runtime/single-source regression tests.

```bash
python -m pytest -q -o addopts= \
  tests/unit/test_stage_protection_v2.py \
  tests/unit/test_stage_protection_runner_v2.py \
  tests/unit/test_stage_protection_notebook_v2.py \
  tests/unit/test_stage_protection_colab_v2.py \
  tests/integration/test_stage_protection_runtime_v2.py \
  tests/integration/test_channel_precompensation_runtime.py \
  tests/integration/test_channel_precompensation_single_source.py
```

Tests used the existing CPU environment at
`/home/richar/projects/CEG-WM/alive/CEG-WM/.venv/bin/python` (torch 2.5.1+cpu).
`compileall`, default plan invocation and `git diff --check` also completed.
The notebook was validated with nbformat; every actual code cell was executed
in order along its default P2 route. Drive mount, CUDA admission and Hub access
were stubbed; pip/model commands were replaced with small CPU subprocesses.
The actual logging helper ran, preserving stdout/stderr and traceback for both
success and failure routes. The test asserts the real command contains
`--execute`, without any run switch. Embedded source bytes were compared to
current files; the snapshot excludes credential files and cell outputs and is
scanned for literal HF tokens/private keys. This does **not** test real Google
Drive/Hub authentication, Colab UI/rendering, installation/downloads, CUDA, or
the actual model execution behind that command.
No notebook visual/real-Colab execution is claimed.

The maximum-schedule runtime test physically observed 40 fake SD3 forwards /
258 denoising steps, SDXL 44 ordinary + 12 proxy = 56 forwards, RGB and latent
VJP each 16, and reader encode 70 result ordinary + 16 auxiliary ordinary +
16 proxy = **102**. Its SDXL fake deliberately has three steps, so its observed
step count is **168**. Real SDXL's frozen six-step setting implies the **336-step
planned upper bound**, not a measured real-model result.

During verification, the same-basename pytest collection conflict was repaired
by renaming the integration test. A holdout-failure test was corrected to account
for identical-RGB reuse; failure reuse now also prevents reattempting the same
RGB under another output alias. Raw proposal and returned-result metrics are
computed independently so unavailable E holdout does not erase a known raw
proposal result. These were engineering fixes, with no method retuning.

The source audit independently reran the original 65 tests and identified a
summary inconsistency when unmarked holdout fails but the four trials complete.
The added regression reproduced the stale positive overall outcome before the
fix. `Canary.evaluate` now sets the overall outcome to `ENGINEERING_INCOMPLETE`,
clears overall and group operational signals, and defers quality review.
Original trial-only outcome/group signals remain explicitly named diagnostic
fields; trial rows, strict-improvement bits and denominators are unchanged.
The embedded-source notebook was regenerated and rechecked for exact source
consistency and default sequential execution. No method/parameter changes,
real-model execution, push or merge accompanied this correction.

## Source-session review and real-run prerequisites

1. Review the local diff, frozen config, protocol correspondence and notebook
   snapshot. The user has now authorized commit/push and a directly runnable
   link. The source audit chat owns final verification, publication and remote
   link validation; this implementation chat has not committed or pushed.
2. Confirm the explicit reporting interpretation: IC/IR are the two fixed
   clean/design task cells; holdout target bit is the third channel. Holdout
   location relative to the same frozen cR is separately reported, and no
   holdout-derived target cell or extra threshold is chosen. All underlying
   readouts are saved for auditing this interpretation.
3. Resource feasibility remains UNKNOWN: eleven-step backward CPU activation
   storage, host RAM, GPU peak, exact CUDA kernels, model access/downloads,
   Drive capacity/behavior, and wall-clock cost. Old last-step VJP success
   supplies no L4 guarantee. The notebook requires a single active VM per run.
4. Any operational P2 signal still needs quality-tradeoff review. There is no
   invented PSNR cutoff or automatic P3 permission. Both content groups must
   exhibit strict eligible C>B improvement with no other-bit regression.

No mathematical, seed, position, reader, cell-selection or budget revision was
needed for this implementation. P3 fixed-margin D and P4 zero-bit attribution /
independent CAL/TEST remain conditional later stages, not hidden extra runs.

Authoritative plan and audit remain at
`D:/CodexData/research-audits/ceg-full-plan-review-20261109/PLAN_V2.txt`
and `AUDIT_DECISION.txt`. The source audit chat retains subsequent authorization
and scientific interpretation responsibilities.
