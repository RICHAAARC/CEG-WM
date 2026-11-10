# Fixed V2 checkpoint continuation — 2026-11-10

This separate entry implements the user's authorization to continue the existing
attempt and retry its interrupted `source06_fur_knit/bit1/early1/latent_vjp` once.
It includes the [memory preparation patch](stage_protection_v2_memory.md). It is
engineering delivery only; no real models, GPU or Colab run were started here.
The source audit chat owns review, commit/push and remote-link verification.

Use [stage_protection_v2_resume_colab.ipynb](../notebooks/stage_protection_v2_resume_colab.ipynb)
with Run all. First cell is only Drive mount. No upload, switch, manual path or
parameter change is needed. Dependencies and existing model access are still
required. The unchanged output argument is:

`/content/drive/MyDrive/CEG-WM_Result/stage-protection-v2-20261109/attempt01`

## Saved evidence and the one new attempt

The audit verified that the original target has only STARTED status, no result
or partial payload, and the preceding RGB VJP is RETURNED. Original counts
include transformer 46 attempted / 45 returned, suffix forwards 7 / 6 and one
full source generation. These numbers are audit observations, not new execution
results from this implementation chat.

Before provider construction, `stage_protection_resume.preflight` reads the
saved method config and counters, then individually loads/validates/releases
every required RETURNED artifact (and any additional original completed
operations): source06 source/canonical/unmarked; bit0 E, common late VJPs and
all A/B/C solve/action/render/read checkpoints; bit1 early1 RGB VJP. It checks
readability, finite values and required runtime fields. Missing/corrupt data or
a changed original run stops recovery; it never regenerates the first source.
There are no SHA, filename-identity or byte-length admission gates.

Original operation files, counts and aggregate results are never modified.
All new operations and results live in the fixed child directory:

`repairs/source06_bit1_early1_latent_vjp_attempt01/`

`reservation.json` records the one-target authorization and original counters
and status. The new target's operation status is durably STARTED **before** its
function runs. A RETURNED retry is reused; a FAILED/STARTED retry blocks future
invocations. The base `OperationStore` keeps its existing no-retry policy.
The target is only invoked if the unchanged Canary actually reaches that key.
Retry failure stops immediately without freezing it as a completed selection
or proceeding to source07. Keyboard interruption also leaves STARTED spent.

After a successful target, the original fixed P2 runner continues its remaining
operations. It reuses saved late solver outputs instead of calling the solver
again. New source07 generation is allowed; source06 regeneration is not. All
selections still freeze before any holdout. No seeds, reader cells, basepoints,
write budgets, thresholds, candidate order, four-trial denominator or P3/P4
authorization changes are introduced.

The child directory contains `selection_frozen.pt/json`, `evaluated.pt`,
`summary.json`, and `operations/`. The notebook reads these child results.
When a child summary already exists, repeated Run all returns it after preflight
without constructing a provider. Other new STARTED/FAILED operations retain the
normal no-retry policy; this notebook is not a general repair interface.

## Accounting and diagnostics

The original root `physical_counts.json` is read-only. The child counter file is
initialized with all original attempted/returned counts and grows durably with
new work. `continuation_counts.json` separately reports original, new, target
retry, later continuation and cumulative counts. The original 40 SD3 forwards /
258 steps and other uninterrupted-plan maxima are **not** claimed as cumulative
recovery bounds. No spent interrupted work is erased or treated as free.
The denoising-step display now reads the actual module instrumentation keys
`sd35_transformer_attempted` / `sdxl_unet_forward_attempted` rather than the
pipeline-style `*_calls` keys; returned counts and physical execution are unchanged.

`runner.log`, `setup.log` and `runner_resources.jsonl` append in the original
root. Earlier exits remain. Physical counters and STARTED reservation remain
the durable evidence if SIGKILL prevents the derived count report from refreshing;
the latter is refreshed on normal checkpoints/exit. Resource telemetry is best
effort and cannot prove this process caused a shared-cgroup OOM event.

Use only one active VM for this directory; the Linux lock coordinates one VM,
not multiple Google Drive clients. These are trusted own-run torch/PIL
checkpoints, not an upload/import format for untrusted pickle files.

## Validation and limits

Complete targeted regression: **110 passed in 37.89s**, including the memory
patch, resume policy, real adapter with CPU fake modules, both notebooks and
retained runtime/single-source tests. `compileall` and `git diff --check` passed.

Targeted CPU checks cover cache reuse without compute; exactly one target
attempt; STARTED-before-function; retained original bytes/status/counts; failure
and interruption consuming the authorization; missing, corrupt and nonfinite
prerequisites blocking before provider construction; distinct retry/continuation
costs; unchanged general failure policy; no provider for an existing summary;
cached late solver reuse; and both notebook Run-all flows with subprocess
substitutes, append-only logs and exact embedded source comparison.
An adapter-level test also creates typed captures/RGB sensitivities with tiny
CPU modules, loads them through the recovery preflight and executes the saved
11-step VJP without regenerating the source. Its initial missing reader-loader
stub caused one Hub VAE-config request to fail with 401 (no weights or real model
execution); the stub was repaired and the test now explicitly rejects all HTTP
requests. These checks are engineering evidence only.

Real Drive durability, Hub access, CUDA, model downloads, numerical precision,
11-step memory feasibility and scientific outcomes remain untested here.
Rebuild both notebooks after source changes:

```bash
python -m scripts.build_stage_protection_v2_notebook
python -m scripts.build_stage_protection_v2_resume_notebook
```
