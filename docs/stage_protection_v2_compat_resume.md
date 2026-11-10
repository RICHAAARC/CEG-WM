# Fixed continuation after the import-only failure — 2026-11-10

The later early2 interruption uses the [isolated-worker/disk continuation](stage_protection_v2_isolated_resume.md),
which retains this entry's completed early1 results and spent costs.

Use [stage_protection_v2_compat_resume_colab.ipynb](../notebooks/stage_protection_v2_compat_resume_colab.ipynb)
with Run all for the observed private-PyTorch-symbol failure. First cell only
mounts Drive; source, output path and mode are fixed. No upload, manual path or
run switch is required. This completes the original checkpoint-continuation
delivery for the same target, without changing research scope. The audit chat
owns review/commit/push; no real-model/GPU execution occurs in implementation.

## Exact supported predecessor

Before constructing a provider, the entry validates the original method config,
source06 completed checkpoints and original STARTED target, as in the first
continuation. It additionally requires all of the following in the first repair:

- The only operation is `source06_fur_knit/bit1/early1/latent_vjp`, with FAILED
  status and the exact `RuntimeStageError: source_suffix_vjp: ImportError` for
  `is_non_overlapping_and_dense` from `torch._prims_common`. An optional Python
  module-path suffix in that ImportError is allowed; no specific path is required.
- Original reservation and before-counts agree with the root. Final counters
  agree with FAILED `counts_after`. The only increments are
  `source_pipeline_load_attempted=1` and `source_pipeline_load_returned=1`;
  every other counter increment, including unknown keys, is zero.
- There is no other operation or later frozen/evaluated/summary result, including
  incomplete aggregate temporary files, and no `result.pt` or result temporary.
  An existing `partial.pt` is retained as failure evidence; it is not reused or
  treated as a completed result.

Any mismatch blocks before provider/model construction. This does not generally
permit FAILED/STARTED retries, choose another target, or use hashes/identity gates.
The old continuation entry still rejects its FAILED target.

## Fixed output and cost

The root output remains:

`/content/drive/MyDrive/CEG-WM_Result/stage-protection-v2-20261109/attempt01`

The new attempt and later fixed P2 outputs are isolated in:

`repairs/source06_bit1_early1_latent_vjp_compat_attempt01/`

All original root and first-repair statuses, payloads, partials and counters remain
read-only. New cumulative counters start from the first repair's final counts,
including its reported loader 7/7; they do not reset to root 6/6. The notebook
appends existing logs and displays new child results and durable current counts.
`continuation_counts.json` distinguishes original-root cost, starting cumulative
cost, previous failed repair increments, this entry's new cost, target cost,
later continuation and total cumulative cost. The original 40/258 plan cap is
not claimed as a cumulative recovery cap. Inventory retains the original
STARTED, first-repair FAILED and new attempt with separate origin labels.

The fixed new reservation permits one physical attempt of this same target;
STARTED is written before its function. RETURNED is reused on repeated Run all;
new STARTED/FAILED blocks future invocations. The attempt is lazy: merely opening
the entry does not compute the target. Success resumes the original P2 runner,
which reuses saved results and solver decisions, retains the same basepoint,
seeds, reader/cells and budget, and freezes all selections before holdout.
No P3/P4 or additional research stage is added. One active VM is required.

## Engineering validation

Final targeted regression: **150 passed in 50.94s**. `compileall` and
`git diff --check` passed. All three notebooks embed the same 72 current
source/config/requirements files and passed sequential Run-all substitute checks.

CPU/static/mock checks cover the precise error and zero-computation boundary,
counter consistency, missing/corrupt prerequisites, optional retained partials,
old bytes unchanged, correct 7/7 starting point and retained loader cost,
three-attempt inventory, lazy target invocation, RETURNED reuse, new failure and
interruption consuming the attempt, unchanged old entry, no provider for a cached
summary, freeze-before-holdout orchestration, typed real-adapter checkpoints with
small CPU fake modules and no network, and all three notebook Run-all routes.

The private-symbol fix and its copy/reference forward-gradient checks remain in
[the memory patch record](stage_protection_v2_memory.md). No real Python 3.13
Colab/CUDA execution, memory-peak measurement or scientific success is claimed.

```bash
python -m scripts.build_stage_protection_v2_notebook
python -m scripts.build_stage_protection_v2_resume_notebook
python -m scripts.build_stage_protection_v2_compat_resume_notebook
```
