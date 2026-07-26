# Minimal Method Package Candidate

This directory is an extracted candidate for the independently installable core method. It contains `main/`, project packaging metadata, and method-scoped configuration or tests when the source project provides them.

Before publishing, confirm that:

- `pyproject.toml` contains the concrete project's build and distribution metadata;
- `main/` contains the designed method rather than an empty or weakened implementation;
- method behavior tests from the source readiness manifest are present and pass;
- `extraction_manifest.json` has no safety violations or missing paths.

Basic import check:

```bash
python -c "import main"
```

An extraction candidate is not release approval.
