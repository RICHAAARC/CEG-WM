---
name: negative-result-recorder
description: Standardize recording of negative results, failed runs, and counterexamples for transparent research reporting. Use when users ask for failure logging standards, ablation failure documentation, or rebuttal ready evidence organization.
---
# negative-result-recorder

Capture failed or negative outcomes as first-class research artifacts.

## Execute

1. Define minimal fields for negative result records: setup, failure mode, and evidence.
2. Verify failed runs are retained with reasons and reproducible context.
3. Ensure records link to configs, seeds, and artifact paths.
4. Prevent selective omission by enforcing append-only recording policy.
5. Summarize negative-result patterns that affect research claims.

## Output Contract

- `issue`
- `negative_result_visibility_risk`
- `evidence`
- `severity`
- `recommended_fix`
