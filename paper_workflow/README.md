# paper_workflow Stage 01-05

## Scope

This directory is the orchestration layer for paper evaluation.
It reuses the existing method implementation under main and only closes the
paper_workflow stage boundaries around frozen manifests, shard execution,
merge/materialize, and export publication.

Implemented in stage 01-05:

- PW00 family manifest build, source split freeze, attack event grid, and shard plans.
- PW01 isolated source shard execution for positive_source, clean_negative, and optional diagnostic control-negative roles.
- PW02 source merge, global thresholds, finalize manifest, and clean-side formal / derived exports.
- PW03 attacked_positive shard execution, attacked-image materialization, and staged detect records.
- PW04 attack shard merge, formal overlay materialization from PW02 thresholds, legacy attack metrics exports, canonical paper-facing exports, and clean / attack overview tables.
- PW05 release packaging, signoff emission, and frozen paper-family bundle export.

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
- exports/pw05/

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

Append-only paper-facing exports also live under:

- exports/pw04/metrics/paper_metric_registry.json
- exports/pw04/metrics/content_chain_metrics.json
- exports/pw04/metrics/event_attestation_metrics.json
- exports/pw04/metrics/system_final_metrics.json
- exports/pw04/metrics/bootstrap_confidence_intervals.json
- exports/pw04/tables/main_metrics_summary.csv
- exports/pw04/tables/attack_family_summary_paper.csv
- exports/pw04/tables/attack_condition_summary_paper.csv
- exports/pw04/tables/rescue_metrics_summary.csv
- exports/pw04/tables/bootstrap_confidence_intervals.csv
- exports/pw04/figures/attack_tpr_by_family.png
- exports/pw04/figures/clean_vs_attack_scope_overview.png
- exports/pw04/figures/rescue_breakdown.png
- exports/pw04/tail/estimated_tail_fpr_1e4.json
- exports/pw04/tail/estimated_tail_fpr_1e5.json
- exports/pw04/tail/tail_fit_diagnostics.json
- exports/pw04/tail/tail_fit_stability_summary.json

PW05 top-level release outputs live under:

- exports/pw05/signoff_report.json
- exports/pw05/release_manifest.json
- exports/pw05/workflow_summary.json
- exports/pw05/run_closure.json
- exports/pw05/stage_manifest.json
- exports/pw05/package_manifest.json

The legacy PW04 exports remain authoritative compatibility outputs. The new paper-facing tables and figures only use the canonical scope names content_chain, event_attestation, and system_final.

## Entrypoints

- paper_workflow/scripts/pw00_paper_eval_family_manifest.py
- paper_workflow/scripts/pw01_source_event_shards.py
- paper_workflow/scripts/pw02_source_merge_and_global_thresholds.py
- paper_workflow/scripts/pw03_run_attack_event_shard.py
- paper_workflow/scripts/pw04_merge_attack_event_shards.py
- paper_workflow/scripts/pw05_release_signoff.py

The notebooks in paper_workflow/notebook remain thin wrappers that call these scripts.
