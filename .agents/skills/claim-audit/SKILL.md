---
name: claim-audit
description: Audit research claims against governed records, tables, figures, reports, and manifests. Use when writing, revising, reviewing, or promoting quantitative or qualitative claims for papers, reports, READMEs, or release materials.
---

# Claim Audit

## Workflow

1. Enumerate each supported claim and its exact scope.
2. Map the claim to governed artifacts and their manifests.
3. Verify sample scope, split, metric, threshold, exclusions, uncertainty, and code/config version.
4. Label unsupported statements as hypotheses, plans, or observations.
5. Update the claim-to-evidence mapping when artifacts change.

## Blocking Rules

- Do not support claims with placeholders, notebook output, logs, dry runs, or unaudited files.
- Do not generalize beyond the governed sample and protocol.
- Do not treat code volume or harness success as experimental evidence.

## Required Validation

- Run artifact rebuild checks and the relevant formal evidence gate before declaring support.
