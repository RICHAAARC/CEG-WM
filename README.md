# CEG-WM

CEG-WM is a method-first research repository for keyed watermark attribution in generated images.

## Current scope

The validated method completes the content chain:

- simultaneous content-adaptive LF+HF embedding;
- blind final-image scoring from a detection key and frozen public assets;
- calibrated weighted-joint attribution;
- separate fixed-denominator 8/8/32/32 stability strata.

Geometry, fixed-FPR calibration, attack robustness, and paper promotion remain separate work.

## Layout

```text
src/cegwm/              importable research package
experiments/run_content_chain.py  completed content-chain runner
configs/content_chain/  content-chain protocols, rosters, and public assets
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

The extraction tool exposes one complete content-chain profile.

```bash
python -m governance.tools.extract_package --profile content_chain_execution --output release_packages/content_chain_execution --dry-run
```

The source and reduction decisions for the migrated harness are recorded in [docs/framework_migration.md](docs/framework_migration.md).
