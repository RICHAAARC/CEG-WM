# CEG-WM

CEG-WM is a method-first research repository for keyed watermark attribution in generated images.

The repository has restarted from a minimal framework. The previous implementation and its full governance history are preserved separately at `../CEG-WM-Archive`; they are read-only provenance, not current method or evidence authority.

## Current scope

The authorized work surface is Stage A:

- establish a reproducible HF attribution anchor;
- test whether an LF carrier provides independent keyed attribution and attack complementarity;
- retain clean, registered-key, wrong-key, and primary-null observations on fixed denominators;
- build only the runtime and experiment code required to answer that question.

Content-adaptive allocation, crop fault decomposition, Q/K geometry, joint recovery, formal calibration, and paper artifacts are later stages. They must not be used to mask a failed Stage-A attribution channel.

## Layout

```text
src/cegwm/              importable research package
experiments/stage_a/    Stage-A experiment entrypoints
configs/stage_a/        Stage-A run configurations
tests/                  lightweight research tests
notebooks/              thin, output-free entrypoints
governance/             detachable dependency/notebook/extraction checks
```

## Local checks

```bash
python -m pytest -q
python -m pytest -q governance/tests -c governance/pytest.ini
python -m governance.harness.run_all
```

The extraction tool exposes two structural profiles. A profile may be structurally valid while `release_candidate_ready` remains false until substantive method or execution code and its tests exist.

```bash
python -m governance.tools.extract_package --profile method_core --output release_packages/method_core --dry-run
python -m governance.tools.extract_package --profile stage_a_execution --output release_packages/stage_a_execution --dry-run
```

The source and reduction decisions for the migrated harness are recorded in [docs/framework_migration.md](docs/framework_migration.md).
