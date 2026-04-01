# paper_workflow Stage 01

## Scope

This directory is an orchestration layer for paper evaluation.
It reuses the existing method implementation and the stable stage-01 runner path.
It does not re-implement method internals from main.

Implemented in stage 01:

- PW00 family manifest build.
- PW01 isolated positive_source shard execution.
- Event identity with prompt_index + seed + sample_role.
- Drive-first output layout under paper_workflow/families/<family_id>/.

Explicitly not implemented in stage 01:

- Global merge.
- Global thresholds or calibration merge.
- clean_negative generation.
- attacked_positive generation.
- Attack orchestration.
- release/signoff flow.

## Layout

paper_workflow/families/<family_id>/

- manifests/
- snapshots/
- source_shards/positive/
- runs/
- logs/
- runtime_state/
- exports/

## Entrypoints

- paper_workflow/scripts/PW00_Paper_Eval_Family_Manifest.py
- paper_workflow/scripts/PW01_Source_Event_Shards.py

The notebooks in paper_workflow/notebook are thin wrappers that call these scripts.
