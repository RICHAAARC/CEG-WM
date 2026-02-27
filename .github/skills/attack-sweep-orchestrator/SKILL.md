---
name: attack-sweep-orchestrator
description: Orchestrate and audit structured attack sweeps with stable grids, reproducible execution, and comparable summaries. Use when users ask for attack sweep setup, robustness curve generation, or batch attack evaluation governance.
---
# attack-sweep-orchestrator

Run and audit attack sweeps with consistent protocol and summary integrity.

## Execute

1. Define sweep grids from protocol config and experiment matrix inputs.
2. Verify attack parameters and seeds are explicit and reproducible.
3. Ensure batch execution writes outputs under controlled run roots.
4. Ensure summaries preserve protocol labels and group keys.
5. Report incomplete coverage, schema mismatches, or comparison hazards.

## Output Contract

- `issue`
- `sweep_impact`
- `evidence`
- `severity`
- `recommended_fix`
