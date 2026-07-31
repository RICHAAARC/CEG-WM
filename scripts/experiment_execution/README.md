# C1 HF threshold-fit experiment execution

This directory's active delivery path is the schema-v2 C1 HF threshold-fit
package. It executes one preregistered fit shard at an exact committed
revision. It cannot approve tau, unlock untouched-confirmation data, or support
a scientific claim by itself.

## Frozen dependency closure

`requirements_c1_threshold_fit.txt` is the C1-specific complete transitive
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
sidecar. Its exact allowlist includes the C1 lock and excludes the shared
runtime-qualification lock, untouched-confirmation manifest, baselines,
comparison protocol, synthetic runtime, governance, Notebook, checked-in
outputs, builder, and package-external bootstrap.

After independent review, build to a new path outside the repository:

```bash
python scripts/experiment_execution/build_experiment_execution_package.py \
  --root . \
  --output-zip '<outside-repository>/ceg_wm_c1_threshold_fit.zip' \
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

Historical non-C1 runtime-qualification materials remain in their explicitly
named repository files. They are not part of this C1 schema-v2 entrypoint or
its authority; this README intentionally provides no historical commands.
