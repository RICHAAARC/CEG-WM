# paper_workflow Stage 01-02

## Scope

This directory is an orchestration layer for paper evaluation.
It reuses the existing method implementation and the stable stage-01 runner path.
It does not re-implement method internals from main.

PW02 currently closes the stage-02 source merge layer on top of PW00/PW01.
It aggregates completed positive_source and clean_negative PW01 shards,
reuses the PW00-frozen split plan, runs global calibrate/evaluate for
content_chain_score and event_attestation_score, and writes top-level export
artifacts under paper_workflow/families/<family_id>/exports/pw02/.

Implemented in stage 01-02:

- PW00 family manifest build.
- PW01 isolated positive_source and clean_negative shard execution.
- Event identity with prompt_index + seed + sample_role.
- PW02 source merge from completed PW01 shard manifests.
- PW02 global thresholds for content and attestation score families.
- PW02 top-level source pool manifests, finalize manifest, threshold exports, and clean-evaluate exports.
- PW02 honest system_final derived metrics export (not an independent formal evaluate record).
- Drive-first output layout under paper_workflow/families/<family_id>/.

Explicitly not implemented in the current paper_workflow stage boundary:

- PW03.
- PW04.
- PW05.
- attacked_positive generation.
- Attack orchestration.
- Independent system_final formal evaluate workflow.

## Layout

paper_workflow/families/<family_id>/

- manifests/
- snapshots/
- source_shards/positive/
- source_shards/negative/
- runs/
- logs/
- runtime_state/
- exports/

PW02 top-level exports live under:

- exports/pw02/positive_source_pool_manifest.json
- exports/pw02/clean_negative_pool_manifest.json
- exports/pw02/paper_source_finalize_manifest.json
- exports/pw02/thresholds/content/thresholds.json
- exports/pw02/thresholds/attestation/thresholds.json
- exports/pw02/evaluate/clean/content/evaluate_record.json
- exports/pw02/evaluate/clean/attestation/evaluate_record.json
- exports/pw02/system_final_metrics.json

## Entrypoints

- paper_workflow/scripts/PW00_Paper_Eval_Family_Manifest.py
- paper_workflow/scripts/PW01_Source_Event_Shards.py
- paper_workflow/scripts/PW02_Source_Merge_And_Global_Thresholds.py

The notebooks in paper_workflow/notebook are thin wrappers that call these scripts.
