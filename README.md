# CEG-WM

CEG-WM is a method-first research repository for keyed watermark attribution in generated images.

## Current scope

The current research candidate on `dev/reconstruction-channel-v1` is
[one-shot two-channel precompensation](docs/channel_precompensation_v1.md),
following the retained reconstruction-direction mismatch measurement. Its
[self-contained Colab notebook](notebooks/channel_precompensation_v1_colab.ipynb)
has been run by the user; that run stopped at ordinary/proxy forward checks,
before any marked write. The [continuation notebook](notebooks/channel_precompensation_v1_continue_colab.ipynb)
reuses its saved zero observations after a writer preprocessing-layout repair.
The repair has CPU/fake checks; its effect on the real L4 execution remains
unverified. No bit-recovery result was collected. Quality is report-only.

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
