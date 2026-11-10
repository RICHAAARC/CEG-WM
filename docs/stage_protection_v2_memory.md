# V2 memory preparation — 2026-11-10

Subsequent explicit authorization adds a separate [one-target continuation](stage_protection_v2_resume.md).
The no-retry description below records this memory patch's original scope.

## Subsequent Colab compatibility repair

The source audit relayed a user-run failure of published commit `2262713`:
`source_suffix_vjp` could not import `is_non_overlapping_and_dense` from
`torch._prims_common` in the Python 3.13 Colab environment. The authorized repair
is FAILED. Reported new counts are source-pipeline load 1/1 and zero other model
operations; cumulative transformer counts remain 46/45 and full generation 1.
This is an import compatibility failure, not new evidence about memory feasibility.

The saved-weight eligibility check now uses public tensor layout, shape, stride
and numel metadata. Non-singleton dimensions sorted by stride must exactly
extend a contiguous span; singleton strides do not introduce overlap. Scalars
and dense permutations/transposes remain eligible. Empty, holey, overlapping and
nonzero-offset views keep the old copy fallback. Version/lifetime/storage guards,
activation copies, model calls and method semantics are unchanged.

CPU regression simulates the missing private symbol and compares exact forward
values, gradients, saved strides and retained storage against the old copy path.
The complete targeted suite passed **118 tests in 51.27s**, including both
notebook bundles/Run-all substitutes and unchanged one-retry failure handling.
`compileall` and `git diff --check` passed. This is local Python 3.12 / CPU torch
2.5.1 validation, not a real Python 3.13 Colab or CUDA run.
Both original notebooks are rebuilt with this fix and still block the consumed
FAILED retry. The subsequently clarified delivery adds a [separate fixed
compatibility continuation](stage_protection_v2_compat_resume.md) only for this
exact import-only failure, retaining all previous cost/status and allowing one
new attempt of the same target. No real model was run for this implementation.

## Original memory patch

Local engineering follow-up on clean base `fd0f048453147c09f9a6726e6949f8205ce3fba1`.
No real models were run and no commit/push was made by this implementation chat.
No candidate, reader, seed, position, budget, threshold, physical model-call
allowance, failed result or operation-retry rule was changed.

The source audit reports that source06/bit1/early1/latent_vjp exited with SIGKILL
(-9): transformer 36/36 before the attempt, 46 attempted/45 returned at exit;
source loader 5/5 to 6/6; suffix forward 6/6 to 7/6; decode unchanged at 6/6;
latent VJP unchanged at 2/2. This places the interruption near step18, the tenth
step of the eleven-step differentiable forward, before backward. Host OOM is
strongly suspected, not established: the original run has no resource telemetry
or kernel OOM evidence. Source06 bit0's late path had completed; source07 and
holdout had not been reached. Those historical records are not edited here.

## Changes and static savings

1. `stage_protection_v2.load_v2_source` uses the public diffusers 0.39.0
   `StableDiffusion3Pipeline.from_pretrained` component overrides. Suffix loads
   pass all three text encoders and tokenizers as `None`, and pass the existing
   frozen reader VAE. Full generation still loads all text components and also
   reuses the reader VAE when available. A completed full pipeline drops text
   modules after embeddings have been captured. A later new source explicitly
   replaces a suffix-only pipeline with a full one. SD3 and SDXL do not coexist.
   This removes unused text weights/tokenizers from suffix residency and avoids
   constructing a temporary replacement VAE. Legacy two-argument loader injection
   remains supported; custom factories can use the new component-aware protocol.

2. The shared `_cpu_saved_tensors` helper has an explicit source-only opt-in;
   all old callers and SDXL paths keep their default copy policy. V2 source
   transformer/VAE weights stay resident for the synchronous VJP, so eligible
   saved views retain that existing storage rather than allocate CPU copies and
   later device copies. Activation copies are unchanged. Eligible views must
   be non-overlapping/dense with zero storage offset. Transposes can preserve
   copy layout; holey, overlapping and nonzero-offset views conservatively fall
   back to the previous copy path. Object/version/storage/device/dtype/shape/
   stride and context lifetime checks detect ordinary writes, replacement and
   migration. There is **no TorchDispatchMode/global operator interception**.
   Raw-storage and `.data` mutations that bypass version tracking are unsupported
   operations in this fixed frozen runtime; this is not a guard for arbitrary
   third-party writers. The dtype-mutating SDXL VAE is never opted in.

3. Parent `run_logged` observes cgroup v2 before launch, at child start and at
   exit, including a returned -9. `runner_resources.jsonl` holds observed
   `memory.events`, current/peak/limit and event deltas. Missing files, permissions,
   counter resets and unsupported hierarchy discovery remain null/unknown with
   reasons. A group counter increase is not single-process OOM attribution.
   Telemetry writes are best-effort; if their file fails, a reason is attempted
   in the existing runner log. They cannot block a child or mask its exit code.
   Normal result/store IO failure policy is unchanged. If the parent/VM also
   dies, a final snapshot cannot be guaranteed.

4. Source VJP stage/step boundaries emit `SOURCE_MEMORY` rows to existing stdout
   logging, with process RSS and saved-tensor CPU-copy byte counters. Copy bytes
   are cumulative allocations for this VJP, **not live RSS or a measured peak**.
   The record immediately before a step survives if that step is killed, subject
   to the existing filesystem/VM limitations. Observation does not control any
   candidate, threshold or retry.

The installed 0.39.0 sources were inspected: SD3 constructor accepts missing
text/tokenizer components, and pipeline `from_pretrained` removes explicitly
`None` components and reuses supplied component objects. Model revision, dtype,
scheduler/RNG/conditions and the existing forward/decode expressions are unchanged.
No activation checkpoint recomputation, lossy representation or graph truncation
was added. These are static sources of savings, not a measured GPU/host peak.

## Validation boundary and remaining risk

Final targeted suite: **94 passed in 35.15s**, including the necessary retained
runtime/single-source regressions and notebook checks. The regenerated notebook
contains 69 source/config/requirements files, verified against current source.
`compileall` and whitespace checks also completed. These were CPU/static/fake
checks only, with no real models, commit/push or STARTED-state edits.

Targeted CPU checks cover public loader kwargs, full → suffix → reconstruction
→ suffix → new-source lifecycle, old loader injection, weak-reference text
release, reader identity, forward/gradient equality against the copy path,
transpose/slice/overlap fallback, ordinary weight mutation and replacement,
scope violations, unchanged physical counts, missing/denied cgroup records,
SIGKILL child exit records and failed telemetry writes.

The full-envelope fake runtime still checks SD3 40 forwards/258 steps, SDXL 56
forwards, RGB/latent VJP 16 each and reader encode 102. Fake SDXL has three steps;
its 168 observed steps do not replace the real six-step 336-step planned limit.
CPU equivalence is not proof of CUDA numerical equivalence. The real 11-step
activation footprint may remain too large even after these savings. Real
memory peak, CUDA kernels, Colab resources and OOM cause remain unverified.

The notebook embeds the revised source and describes resource logs. Regenerating
it **does not authorize a rerun of STARTED work**. The same output directory and
no-automatic-retry behavior are retained; any further real-run decision stays
with the user/source audit. No P3/P4 or new experiment was started.
