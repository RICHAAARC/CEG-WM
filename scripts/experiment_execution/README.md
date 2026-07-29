# Runtime qualification execution package

This package is a revision-bound execution surface for the frozen SD3.5
runtime candidate. It is not runtime/GPU evidence until the included runner
finishes on a real supported GPU and the returned result zip is independently
replayed.

Run only through the included qualification runner. The first run profile is
`smoke`; `qualification` follows only after smoke succeeds. `replay` is
optional. Model and package caches, temporary tensors, and uncompressed
results must stay in the caller-supplied ephemeral root. Never persist an HF
token, method key, model cache, or raw tensor in the result directory.

The Colab entrypoint does not unpack or trust this archive directly. A
separately reviewed, package-external
`runtime_qualification_bootstrap.py`, bound to package schema version 1,
must receive the full independently audited archive SHA-256 at run time.
The bootstrap streams the Drive archive once into a unique ephemeral `xb`
snapshot while computing that digest. A mismatch removes the snapshot without
unpacking; a match causes all ZIP/manifest/file checks to use only the local
snapshot before installing requirements or starting this package's runner. The
bootstrap is deliberately excluded from this execution package, avoiding a
self-verification loop.

The runner catches ordinary Python/backend failures after it can resolve the
requested result path and writes a minimal failure zip. A Python interpreter
crash, an OS kill (including hard OOM), or an unwritable result filesystem is
outside any in-process guarantee; report that case as incomplete/resource
failure and do not manufacture a success archive in the Notebook.

## Build from the final runtime candidate

Run the builder only after the Batch-4 revision has been committed and the
repository is clean. The builder rejects a non-HEAD revision, a dirty tree,
untracked delivery files, unsafe paths, and missing allowlisted files. It reads
tracked blobs from the exact commit rather than copying the working tree.

```bash
RUNTIME_CANDIDATE_REVISION="$(git rev-parse HEAD)"
test -z "$(git status --porcelain)"
PYTHONDONTWRITEBYTECODE=1 python \
  scripts/experiment_execution/build_runtime_qualification_package.py \
  --root . \
  --runtime-candidate-revision "${RUNTIME_CANDIDATE_REVISION}" \
  --output-zip \
  "<outside-repository>/ceg_wm_runtime_execution_${RUNTIME_CANDIDATE_REVISION}.zip"
```

The final archive path is outside the repository. Upload that one archive to
the revision-specific Google Drive `execution_packages/<revision>/` directory;
do not rebuild it in Colab.

Use the following fixed archive and Notebook-ingress rules:

1. The sole authoritative immutable archive is
   `execution_packages/<runtime_candidate_revision>/ceg_wm_runtime_execution_<runtime_candidate_revision>.zip`.
   Never overwrite or modify a frozen revision-specific archive.
2. `execution_packages/current/ceg_wm_runtime_execution.zip` is only the fixed
   Notebook ingress alias. It is not revision or evidence authority.
3. Create the alias by copying the authoritative archive byte for byte. Do not
   rebuild, repack, recompress, or otherwise modify the zip for the alias.
4. After the copy, compute SHA-256 for both files and require the digests to be
   identical.
5. The `current` path supplies no runtime identity. After extraction and
   verification, only `runtime_execution_manifest.json` field
   `runtime_candidate_revision` identifies the run.
6. The Notebook writes results under
   `runs/<runtime_candidate_revision>/<run_id>/`, using the verified manifest
   revision. To change candidates, overwrite only the alias with a byte-for-byte
   copy of another frozen authoritative archive and recheck equal SHA-256
   digests. Never overwrite an existing authoritative archive or historical
   results.

## Run through the package-external bootstrap

The repository copy of the schema-v1 bootstrap is:

```text
scripts/experiment_execution/runtime_qualification_bootstrap.py
```

Freeze and distribute that file separately, record its full SHA-256, and make
the Notebook read it once, verify those bytes, write them with `xb` to a new
`/content` snapshot, verify the snapshot again, and invoke only that snapshot.
The expected package SHA-256 is pasted at run time from the independent
delivery audit; do not read it from a replaceable sidecar stored beside the
package.

The bootstrap CLI accepts `--profile`, `--package-zip`,
`--expected-package-sha256`, `--ephemeral-root`, `--persistent-root`, and an
optional `--replay-source`. It reads `HF_TOKEN` and `CEG_WM_ROOT_KEY` only from
the process environment. It writes neither value to an archive, log, Drive
file, nor unpacked package.

Before any pip invocation, package import, or runner launch, the bootstrap
checks the complete archive SHA-256, ZIP path/member/size/symlink safety,
manifest schema/profile/readiness/revision, allowlist, complete file set,
per-file size/hash, and exact frozen requirements. Only then does it install
the dependencies into ephemeral cache space and start the runner.

Runner exit `0`, `1`, or `2` retains a validated formal result under
`runs/<runtime_candidate_revision>/<run_id>/`. An ingress, unpacking,
manifest, pip, or pre-runner failure instead produces exactly one independent
`bootstrap_failure.json` inside
`bootstrap_failures/<run_id>/ceg_wm_runtime_bootstrap_failure_<run_id>.zip`.
That diagnostic schema is not a qualification result and cannot support a
runtime-stage claim.

## Direct runner contract inside an independently unpacked package

Unpack into a new ephemeral directory. Set `PYTHONDONTWRITEBYTECODE=1` before
starting Python so package verification does not see interpreter cache files.
Model and pip caches must also remain on ephemeral storage, never Google Drive.

```bash
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/content/hf_cache
export PIP_CACHE_DIR=/content/pip_cache
python -m pip install --cache-dir "${PIP_CACHE_DIR}" \
  --requirement requirements_runtime_qualification.txt
export HF_TOKEN='<read from Colab Secret>'
export CEG_WM_ROOT_KEY='<read from Colab Secret>'
python -m scripts.experiment_execution.runtime_qualification_runner \
  --profile smoke \
  --run-id '<UTC run id>' \
  --package-root . \
  --runtime-candidate-revision '<40 hex revision from manifest>' \
  --result-zip /content/ceg_wm_runtime/result.zip \
  --ephemeral-root /content/ceg_wm_runtime \
  --persistent-root /content/drive/MyDrive/CEG-WM/runtime_qualification
```

`--result-zip`, `--ephemeral-root`, and `--persistent-root` are explicit
required arguments and have no runner defaults. The result target must be
strictly inside the ephemeral root; the ephemeral and persistent roots must
be disjoint in both ancestor directions. Only `replay` accepts
`--replay-source`, and that source must be strictly inside the persistent
root. Smoke and qualification must omit it. The backend receives the same
persistent root and independently rejects a model-cache root that equals,
contains, or is contained by it. The Notebook therefore produces the runner
zip under ephemeral storage first and only then copies it into the
manifest-revision/run-ID Drive directory.

The runner independently re-verifies the complete manifest file set, file hashes/sizes,
revision, requirements lock, and every installed dependency version before
importing `main` or `runtime`. Exit `0` means the requested profile passed;
exit `1` means a completed runtime/resource/integrity/budget/Q/K/determinism
failure; exit `2` means incomplete/preflight failure. Both nonzero exits should
still produce a failure result zip when the Python process and result storage
remain usable. Never reinterpret a missing archive, a nonzero exit, or a
`failed` summary as success.

The result zip contains only `run_summary.json`,
`environment_summary.json`, `runtime_checks.jsonl`, and `failures.jsonl`.
`run_id`, profile, runtime candidate revision, seed, prompt digest, and key
control role are recorded in the result. The optional `replay` profile also
requires an existing passed qualification zip for the same revision,
seed/prompt identity, and record digests; it independently reruns the complete
qualification path.
