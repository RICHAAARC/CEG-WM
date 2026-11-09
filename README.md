# CEG-WM

## Single-source method validation (2026-11-09)

[Run the fixed source05 method trial in Colab](https://colab.research.google.com/github/RICHAAARC/CEG-WM/blob/dev/reconstruction-channel-v1/notebooks/channel_precompensation_single_source_v1_colab.ipynb).
It reuses the existing Drive capture, builds a coherent current suffix baseline,
and attempts joint/gradient writes for both bits with ordinary RGB evaluation.
Historical image mismatch is diagnostic only; gradients are recomputed at the new
basepoint. See [method and fixed budget](docs/channel_precompensation_single_source_v1.md).
Engineering delivery only; actual model execution remains user-run.

CEG-WM is a method-first research repository for keyed watermark attribution in generated images.

## Current scope

The current research candidate on `dev/reconstruction-channel-v1` is
[one-shot two-channel precompensation](docs/channel_precompensation_v1.md),
following the retained reconstruction-direction mismatch measurement. Its
[self-contained Colab notebook](notebooks/channel_precompensation_v1_colab.ipynb)
has been run by the user. Following preprocessing-layout and checkpoint-loading
repairs, the latest run reused all zero observations and saved eight RGB
sensitivities, then stopped at the source-suffix pixel check. The
[continuation notebook](notebooks/channel_precompensation_v1_continue_colab.ipynb)
reuses those observations and sensitivities while repairing omitted scheduler
`sigmas` device restoration. The user run confirmed both schedule tensors on
CUDA, but all four sources still failed the pixel check. The adopted next step
is the [single-source NG1 / NG2 / G diagnostic](docs/channel_suffix_diagnostic_v1.md),
with its [independent Colab notebook](notebooks/channel_suffix_diagnostic_v1_colab.ipynb).
No bit-recovery result was collected. Quality is report-only.

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
