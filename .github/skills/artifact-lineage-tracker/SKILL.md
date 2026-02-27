---
name: artifact-lineage-tracker
description: Track artifact lineage from inputs and configs to generated outputs for auditability and debugging. Use when users ask for run provenance mapping, output traceability, or artifact dependency review.
---
# artifact-lineage-tracker

Build a clear lineage map from input decisions to output files.

## Execute

1. Identify run roots, manifests, and artifact naming conventions.
2. Map each output to configuration digest, policy path, and runtime metadata.
3. Verify lineage links are complete and machine-readable.
4. Detect orphan artifacts or outputs without provenance anchors.
5. Report lineage gaps and required metadata additions.

## Output Contract

- `issue`
- `lineage_gap`
- `evidence`
- `severity`
- `recommended_fix`
