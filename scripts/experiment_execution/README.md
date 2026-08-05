# Experiment execution servers

## Development exploration

`development_exploration_server.py` is the Colab-neutral and server-direct entrypoint for
the frozen 13-module development exploration. Given one clean exact repository revision,
an absolute persistent root, an ephemeral cache root, a fixed run ID, and a unique session
ID, it checks basic GPU and disk availability, installs the version-frozen dependency list
without hash mode, downloads the configured model ID and revision without model-file hash
validation, then calls `development_exploration_entrypoint.py`.

The entrypoint owns the formal runner, records, create-only intent/bundle/`COMMITTED`
protocol, cross-session recovery, and result or diagnostic ZIP. It observes the frozen
21-hour soft stop and 24-hour hard cap without changing the unit or attempt budget.
`HF_TOKEN` and `CEG_WM_ROOT_KEY` are passed only through the worker environment and never
persisted in receipts or artifacts. Colab-side copying of ZIP, receipt, and `SHA256SUMS`
is a delivery convenience; only verified persistent `COMMITTED` bundles establish unit
completion.

The checked-in thin Notebook invokes the server from detached execution revision
`67bf7ea0cc9cfaf5083e1487ab593d605eda68eb` with run ID
`ceg_wm_development_exploration_module_outcome_replay_execution`. The prior
`ceg_wm_development_exploration_detector_crossfit_execution` run and all of its
scientific records, operational records, and diagnostic artifacts remain unchanged and
are not read, migrated, rewritten, or deleted. The prior
`ceg_wm_development_exploration_science_first_v42` run namespace, records, dangling
attempts, and full artifacts remain unchanged and are not read or migrated. The prior
`ceg_wm_development_exploration_scientific_execution` run remains unchanged with two
operational commits, zero scientific commits, dangling unit 0002 attempt 0, and diagnostic
`builtins.AssertionError`. Any existing
`ceg_wm_development_exploration_joint_record_execution` directory also remains unchanged;
neither prior run is read, migrated, or deleted.
That execution authority is intentionally
separate from the later Notebook delivery revision and must not be replaced by a mutable
branch.

Rebuilding the deterministic tracked-tree execution package from that exact revision
produces 4,530,056 bytes with SHA-256
`eeea6a1bf6d235be834d693b4a7ac02dcf9d3d07244b1b769b4ed240912c0c94`.

## HF-only threshold-fit GPU execution

The separate schema-v2 HF-only threshold-fit GPU execution delivery
package. It executes one preregistered fit shard at an exact committed
revision. It cannot approve tau, unlock untouched-confirmation data, or support
a scientific claim by itself.

## Frozen dependency closure

`requirements_hf_only_threshold_fit_gpu_execution.txt` is the hf_only_reference_validation-specific complete transitive
dependency lock for Linux x86_64, CPython 3.12, and the frozen SD3.5 Colab GPU
candidate. It contains 62 exact distributions, including
`torch==2.11.0+cu128`, and has SHA-256
`07a4c1bbe6fc5e7e6b38334c5a9919a8565b810a9aae7820b61c24cee91270de`.

The lock was generated from the eight frozen top-level requirements with pip's
resolver in dry-run report mode. Resolution used PyPI, the official PyTorch
cu128 index, and the official NVIDIA index, with CPython 3.12 ABI and explicit
Linux wheel tags `manylinux_2_28_x86_64`, `manylinux_2_27_x86_64`,
`manylinux_2_18_x86_64`, and `manylinux2014_x86_64`. The lock entries are the
normalized, sorted name/version pairs from all 62 `install` records in that
resolver report, not only the eight requested distributions.

Before package import, the package-external bootstrap verifies the lock digest
and exact syntax. It reuses the global environment only when every lock entry
has the exact version. Otherwise it installs every locked distribution into
ephemeral storage with `--no-deps`, then requires exact equality of the target
distribution set and all versions. The entrypoint repeats the frozen-lock and
installed-version checks and records all 62 versions plus the exact imported
torch local version in the execution facts.

## Build and execute

`build_experiment_execution_package.py` reads only tracked blobs from one clean
exact commit and writes a deterministic external ZIP plus adjacent delivery
sidecar. Its exact allowlist includes the hf_only_reference_validation lock and excludes the shared
runtime-qualification lock, untouched-confirmation manifest, baselines,
comparison protocol, synthetic runtime, governance, Notebook, checked-in
outputs, builder, and package-external bootstrap.

After independent review, build to a new path outside the repository:

```bash
python scripts/experiment_execution/build_experiment_execution_package.py \
  --root . \
  --output-zip '<outside-repository>/ceg_wm_hf_only_threshold_fit.zip' \
  --committed-revision '<exact 40-hex HEAD>'
```

Independently record the SHA-256 of the external bootstrap, archive, sidecar,
and embedded manifest before upload. The thin output-free Colab Notebook passes
those trust values, the exact revision, run ID, shard index, and Secrets to the
separately distributed schema-v2 bootstrap. It does not install dependencies,
unpack the archive, import the package, or validate its own result.

Each invocation runs one frozen shard. Persistent attempt records bind exact
revision, run, and shard identities for resume; every result or diagnostic ZIP
uses a unique name. A bootstrap, resource, execution, exclusion, incomplete, or
scientific diagnostic is not a successful shard and cannot be promoted into
method evidence.

Historical non-hf_only_reference_validation runtime-qualification materials remain in their explicitly
named repository files. They are not part of this hf_only_reference_validation schema-v2 entrypoint or
its authority; this README intentionally provides no historical commands.

## Unified server entrypoint

`hf_only_threshold_fit_server.py` is the complete Colab-neutral and
server-direct orchestration entrypoint for one frozen threshold-fit shard. It
requires a clean checkout at an explicit 40-hex revision plus disjoint absolute
scratch, cache, and output roots. It checks the registered GPU/VRAM floor and
available storage, builds the dedicated schema-v2 package from that exact Git
tree, delegates the frozen dependency installation and package trust boundary
to `experiment_execution_bootstrap.py`, downloads the runtime configuration's
exact `model_id` and `model_revision` into the supplied cache, and then invokes
the existing package entrypoint and formal runner. The runner remains the only
records writer.

Both `HF_TOKEN` and `CEG_WM_ROOT_KEY` are read from the process environment and
are never included in the machine-readable receipt. A server invocation is:

```bash
python scripts/experiment_execution/hf_only_threshold_fit_server.py \
  --repository-root /absolute/clean/CEG-WM \
  --expected-revision 0123456789abcdef0123456789abcdef01234567 \
  --scratch-root /absolute/scratch \
  --cache-root /absolute/cache \
  --output-root /absolute/output \
  --run-id hf-only-content-threshold-fit \
  --shard-index 0
```

The stdout JSON receipt identifies the result or diagnostic ZIP, its SHA-256,
the revision/run/shard, the package trust digests, and the frozen model
identity. This entrypoint does not fit or approve tau, unlock confirmation,
access held-out evaluation, or support scientific claims.
