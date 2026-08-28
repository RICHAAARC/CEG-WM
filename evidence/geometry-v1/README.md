# Geometry-V1 Evidence Package

This branch contains the bounded public evidence package for `geometry_v1_natural_qk_geometry_observation`, final state `OPERATIONAL_UNRESOLVED`, and `science_denominator=0`. It remains separate from `Geometry-V1`.

Files:

- [human status card](../../docs/evidence/Geometry-V1.md)
- [machine index](index.json)
- [Drive provenance](data/provenance.json)
- [comparison records](data/comparison_records.jsonl)
- [24-layer and focus summary](data/comparison_summary.json)

Each JSONL row has the fixed fields `stage/protocol/run/layer/kind/control/transform/reference_or_pair/metric/value/support/count/status/source_sidecar_id`. Rows are ordered by stage, layer, Q/K kind, control, transform, pair, and metric. The summary explains which fields are directly comparable and which public sidecars do not expose the requested grain.

Drive access for this revision was read-only. Only compact public sidecars and folder metadata were used. ZIP payloads were not read or committed. Unknown source hashes remain `null`; they are not inferred from connector transport.

D0.1 and D0.2 remain selection observations; D1 and D2 remain confirmation observations. All material is operational evidence only, and Geometry cannot create a positive watermark decision.
