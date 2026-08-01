# CEG-WM HF-only threshold-fit GPU execution execution package

This deterministic schema-v2 archive contains only the exact-revision hf_only_reference_validation
threshold-fit entrypoint, its required runner/import closure, the implemented
method/runtime code, and frozen fit-split assets.

It deliberately excludes the untouched-confirmation manifest, baseline and
comparison implementations, synthetic runtime and package tests, governance,
Notebook, checked-in outputs, builder, and package-external bootstrap. It has
no tau-approval or confirmation-unlock capability.

Run this package only through the separately distributed schema-v2
`ceg_wm_experiment_execution_bootstrap`. Independently verify the bootstrap,
archive, adjacent delivery sidecar, embedded manifest, and exact revision. The
trusted sidecar binds authority digests derived by the clean-revision builder;
the Notebook does not supply or override them.

The hf_only_reference_validation-specific `requirements_hf_only_threshold_fit_gpu_execution.txt` is a complete 62-item
Linux x86_64 / CPython 3.12 transitive lock for the frozen SD3.5 candidate.
The bootstrap verifies the frozen lock identity before package import, reuses
global packages only when every locked version is exact, otherwise installs
the lock with dependency resolution disabled, and requires exact target
distribution-set and version equality before importing torch or this package.

One invocation runs exactly one explicitly selected frozen shard. Unit attempt
records persist under exact revision/run/shard identity for resume, while each
result or diagnostic ZIP is uniquely named. A resource or execution diagnostic
is not method evidence, and a completed shard result still does not approve
tau or support a scientific claim.
