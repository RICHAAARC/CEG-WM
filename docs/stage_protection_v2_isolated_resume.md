# Early2 continuation with isolated workers and local disk saves — 2026-11-11

Run-all entry: [stage_protection_v2_isolated_resume_colab.ipynb](../notebooks/stage_protection_v2_isolated_resume_colab.ipynb).
First cell only mounts Drive; code and paths are preset. No upload or run switch.
Output stays under the original `stage-protection-v2-20261109/attempt01`, with
new results in `repairs/source06_bit1_early2_rgb_vjp_isolated_attempt01/`.

## Observed starting point

The 2026-11-10 Drive inspection confirmed durable counts SD3 transformer 68/67,
suffix forwards 9/8, latent VJP 3/3, source loads 8/8, SDXL UNet 54/54, input
encode 9/9, output decode 9/8 and proxy forwards 3/2. Full source generation
remains 1. Compat early1 VJP returned in 82.83 seconds. The later inspection on
2026-11-11 confirmed VJP/action/render/clean results, but `design_image` has only
STARTED status (no result or temporary payload), and `design_read` is empty.
The earlier inference from six folders to six persisted results was incorrect.
The early2/rgb_vjp folder exists but its status was not cloud-visible.
The separate derived report still showed SDXL 48/48; it is not the cost authority.
The user's final log reports -9 during SDXL decode. OOM remains unconfirmed.

The new store reuses original, import-failed and compat RETURNED checkpoints,
including the completed early1 results. All required caches are scanned before
reservation/model loading, with all missing items reported together. The two
fixed early1 design dependencies first search the same operation's result,
temporary result and complete partial payloads; readable results are reused even
with stale status. Only if absent/unreadable may each dependency be recomputed
once, through the same ProcessProvider with the original image and design seed.
Existing VJP/action/render/clean are never recomputed. Dependency recomputation
costs are separate in `continuation_counts.json`; all prior spent counts remain.
`source06_fur_knit/bit1/early2/rgb_vjp` gets one independent new attempt.
STARTED, FAILED, or an existing directory with
unsynced status and durable post-early1 cost are accepted. A usable saved target
result is reused even if its final status has not synced. Original files are
read-only. New dependency or target STARTED/FAILED cannot loop; failure stops
before freezing decisions. Any other missing prerequisite blocks before models;
no source06 regeneration. Latest durable counts initialize the
new cumulative ledger; prior attempts and derived-report lag remain explicit.

## Memory mechanism and boundary

Old single-source precompensation already released SD3/SDXL in turn and retained
the frozen reader VAE; its source suffix was one step. Its saved-tensor hooks
kept CPU copies until backward ended. Releasing models or CUDA caches could not
release graph tensors still required by the complete differentiable SDXL path.

Each expensive provider operation now runs in a separate Python process. The
coordinator holds CPU checkpoints only, waits/reaps the child before loading its
result, and never constructs models. Worker results, partials, exit code and
latest physical counts are durable; SIGKILL costs survive. Process exit releases
models without an unnecessary final VAE-to-CPU move. Parent death kills the
worker; owned local job directories are cleaned only after workers exit.

Saved tensors retain the old `to('cpu', copy=True)` layout/dtype/value semantics,
then serialize individually to local `/content/ceg_stage_protection_scratch`.
Graph/cache entries retain paths, not those CPU tensors. Files survive repeated
unpack and two-row `retain_graph` VJPs until context exit. Eligible stable source
weights still retain their existing storage. No compression, truncation, lower
precision/resolution, probe or graph recomputation is introduced by these hooks.

The retained graph-wide CPU-copy total is removed; pack has one transient copy
at a time. Unpacked tensors live with active backward consumers. This does NOT
bound model weights, current operator workspace, allocator retention, simultaneous
backward operands or kernel file cache. Free disk space is checked at startup
and every tensor; Drive/FUSE/network and tmpfs/ramfs are rejected. fsync plus
advisory cache hints reduce dirty-page accumulation; hint failure is diagnostic.
Disk capacity and I/O may still limit completion and are not predicted by CPU tests.

Worker receipts report copy maximum, scratch peak, read/write bytes/time and exit
state separately from RSS and GPU allocated/reserved/peaks. SDXL encode/decode
before/after, each UNet step and backward boundaries are observed. Cgroup paths
escaping the visible namespace remain unavailable; no alternate group is guessed.
More independent model loads and disk I/O are real overhead. The original
computation schedule, seeds, reader, dtype, budgets, candidates and holdout order
are unchanged. Recovery of the two missing dependencies adds one ordinary
reconstruction and one reader call if neither result can be recovered; these
costs are additional to the early2 retry and retained prior work.

## Bounded CPU evidence

One 192-MiB saved graph (12 × 16 MiB) was measured in independent CPU processes.
Forward-end RSS increment was 374,128,640 bytes for CPU-copy and 38,965,248 bytes
for disk saves. The disk scratch peak was 201,340,752 bytes and maximum pack copy
16,777,216 bytes; observed writes/reads took 0.294/0.115 seconds. Value 67 and
gradient 479001600 matched exactly. RSS includes allocator effects, so it is not
equated to saved bytes. This is a non-gating CPU mechanism observation, not a
model/GPU memory guarantee. The measurement's disk snapshot is before cleanup;
normal cleanup and kill cleanup are separately checked. Report:
`artifacts/stage-protection-isolated-memory-cpu.json`; reproduction script:
`python -m scripts.measure_stage_saved_tensor_memory --output <report.json>`.

Targeted tests cover exact dtype/layout, weak-reference release, repeated unpack,
two-row gradients, actual SD3/SDXL adapters with CPU fakes, no extra model calls,
worker isolation/reap, cache without workers, kill cost/cleanup, layered recovery,
unsynced status, no-repeat failures and the notebook's default Run-all path.
An execute-level CPU fixture has a genuinely missing design image/read and uses
the real observation path and process transport to check same-input/seed repair,
cached VJP/render/clean, separate costs, unchanged old bytes and immediate stops.
Real Colab/CUDA, full local scratch peak, elapsed time and scientific outcome
remain unverified. No real model was run, committed or pushed by implementation;
the source audit owns publication. P3/P4 are not added.
