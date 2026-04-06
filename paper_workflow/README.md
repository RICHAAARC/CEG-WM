# paper_workflow Stage 01-04

## Scope

This directory is the orchestration layer for paper evaluation.
It reuses the existing method implementation under main and only closes the
paper_workflow stage boundaries around frozen manifests, shard execution,
merge/materialize, and export publication.

Implemented in stage 01-04:

- PW00 family manifest build, source split freeze, attack event grid, and shard plans.
- PW01 isolated source shard execution for positive_source, clean_negative, and optional diagnostic control-negative roles.
- PW02 source merge, global thresholds, finalize manifest, and clean-side formal / derived exports.
- PW03 attacked_positive shard execution, attacked-image materialization, and staged detect records.
- PW04 attack shard merge, formal overlay materialization from PW02 thresholds, attack metrics exports, and clean / attack overview tables.

Explicitly not implemented in the current paper_workflow stage boundary:

- PW05.

## Layout

paper_workflow/families/<family_id>/

- manifests/
- snapshots/
- source_shards/positive/
- source_shards/negative/
- source_shards/control_negative/
- attack_shards/
- runs/
- logs/
- runtime_state/
- exports/
- exports/pw02/
- exports/pw04/

PW04 top-level exports live under:

- exports/pw04/manifests/attack_merge_manifest.json
- exports/pw04/attack_positive_pool_manifest.json
- exports/pw04/formal_attack_final_decision_metrics.json
- exports/pw04/formal_attack_attestation_metrics.json
- exports/pw04/derived_attack_union_metrics.json
- exports/pw04/per_attack_family_metrics.json
- exports/pw04/per_attack_condition_metrics.json
- exports/pw04/tables/attack_event_table.jsonl
- exports/pw04/tables/attack_family_summary.csv
- exports/pw04/tables/attack_condition_summary.csv
- exports/pw04/clean_attack_overview.json

## Entrypoints

- paper_workflow/scripts/PW00_Paper_Eval_Family_Manifest.py
- paper_workflow/scripts/PW01_Source_Event_Shards.py
- paper_workflow/scripts/PW02_Source_Merge_And_Global_Thresholds.py
- paper_workflow/scripts/pw03_run_attack_event_shard.py
- paper_workflow/scripts/pw04_merge_attack_event_shards.py

The notebooks in paper_workflow/notebook remain thin wrappers that call these scripts.
