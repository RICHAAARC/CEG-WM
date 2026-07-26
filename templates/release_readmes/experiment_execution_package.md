# Experiment Execution Package Candidate

This directory contains the repository modules and configurations needed for Colab or GPU-server execution. Notebook and governance control-plane implementations are intentionally excluded.

Before consuming compute resources:

- load a completed experiment configuration;
- validate the `ComparisonProtocol` and retain its `PreflightApproval`;
- confirm required external baseline provenance and vendored sources;
- run the package's integration or smoke tests explicitly;
- confirm `extraction_manifest.json` has no safety violations or missing paths.

Example test selection when such tests are present:

```bash
python -m pytest -q -m "integration or smoke"
```

An extraction candidate is not evidence that an experiment is ready or valid.
