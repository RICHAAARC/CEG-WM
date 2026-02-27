---
name: paper-figure-table-exporter
description: Standardize paper-ready figure and table exports from evaluation outputs with stable columns, group keys, and protocol labels. Use when users ask for publication tables, figure generation, metric export normalization, or final report formatting for submission.
---
# paper-figure-table-exporter

Export publication-ready figures and tables with stable, reproducible structure.

## Execute

1. Identify source metrics, grouping keys, and protocol labels from evaluation outputs.
2. Verify stable column ordering and key canonicalization rules.
3. Export tables and figure-ready CSV/JSON artifacts without recomputing thresholds.
4. Verify required anchors for ablation and attack comparisons are present.
5. Report missing data or schema mismatches with exact upstream fixes.

## Repository Anchors

- `main/evaluation/table_export.py`
- `main/evaluation/report_builder.py`
- `main/evaluation/metrics.py`
- `tests/test_table_export_column_order_is_stable.py`
- `tests/test_metrics_group_key_canonicalized.py`
- `tests/test_evaluation_report_contains_required_anchors.py`

## Output Contract

For each export issue include:
- `issue`
- `affected_table_or_figure`
- `evidence`
- `impact_on_claim`
- `recommended_fix`
