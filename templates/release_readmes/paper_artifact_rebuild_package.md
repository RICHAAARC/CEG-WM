# Paper Artifact Rebuild Package Candidate

This directory contains protocol structures, artifact rebuild code, supporting documentation, and lightweight functional tests. It does not contain the watermark method or experiment runner.

Before distribution, confirm that frozen records and manifests are supplied through the intended external evidence bundle, rebuild commands work, and `extraction_manifest.json` has no safety violations or missing paths.

Run the included lightweight tests with:

```bash
python -m pytest -q
```

An extraction candidate is not evidence that paper claims are supported.
