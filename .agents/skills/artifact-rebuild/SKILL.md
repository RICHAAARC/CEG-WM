---
name: artifact-rebuild
description: Build or revise governed paper tables, figures, reports, manifests, and their rebuild paths from frozen records. Use for changes under paper_artifacts, artifact schemas, rebuild scripts, provenance, or publication evidence generation.
---

# Artifact Rebuild

## Workflow

1. Identify the governed records and frozen manifest that are the artifact's facts.
2. Keep record schemas in `experiments/protocol/` and builders in `paper_artifacts/`.
3. Record input paths, output paths, configuration digest, code version, and rebuild command.
4. Make tables, figures, and reports deterministic from their declared inputs.
5. Add lightweight rebuild tests using `tmp_path`.

## Blocking Rules

- Do not hand-edit formal result tables or figure data.
- Do not rerun methods or attacks from `paper_artifacts/`.
- Do not create a formal artifact without provenance.
- Do not write formal outputs into checked-in `outputs/`.

## Required Validation

- Rebuild the smallest representative artifact in a temporary directory.
- Run artifact tests, default tests, and all audits.
