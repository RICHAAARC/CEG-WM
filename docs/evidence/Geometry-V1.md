# Geometry-V1 Operational Evidence

## Identity and final state

- Branch family: `Geometry-V1`
- Method identity: `geometry_v1_natural_qk_geometry_observation`
- Final semantic identity: passive natural-Q/K geometry observation
- Final state: `OPERATIONAL_UNRESOLVED`
- Scientific denominator: `0`
- Semantics-freeze exact: `95dd416d541450145436b685897d5ee210eb04e5`

Geometry-V1 records bounded operational coordinate-recovery observations. Geometry cannot create a positive watermark decision. The evidence branch remains separate from Geometry-V1.

## Stage results

| Stage | Units | Frozen public state | Drive folder |
| --- | ---: | --- | --- |
| D0 | 768/768 calculated, 0 failed | `D0_UNRESOLVED`; 24 ordered layer shards | [artifact](https://drive.google.com/drive/folders/1swYs6hQUQYsDWA6vBNPLJT7aPSJJIp1E) |
| D0.1 | 768 audited | `D01_CANDIDATES_FROZEN`; layers 6, 13, 18 | [artifact](https://drive.google.com/drive/folders/1DXelqcoC6LLJ-TmFVPuog-aXQVk3fkWk) |
| D0.2 | 768 audited | `DIRECTION_TWO_CANDIDATES_FROZEN`; layers 23, 14 | [artifact](https://drive.google.com/drive/folders/1EfpzIO-ENWDHGlSTlo5NCHTkvgQmvkkm) |
| D1 | 96/96 calculated, 0 failed | `D1_UNRESOLVED` | [artifact](https://drive.google.com/drive/folders/1ye3mFWkF6JJqJQg1IhSldIMqUEOCSvc6) |
| D2 | 64/64 calculated, 0 failed | `D2_UNRESOLVED` | [artifact](https://drive.google.com/drive/folders/1QbtPkS2l5phSNDkWLdOKyrQDyiNLZz2y) |

D0 plan digest is `96e1e5ae6fb8ae66a545b1b10d6c896176989272c81ef1fd737184dcdfaea7b8`; roster digest is `88850de32ae0783427f86d0a5c82c6272a30811931ca0f883f6888cf8b83ac9e`.

## Detailed comparison data

The bounded package adds:

- `data/provenance.json`: Drive folder, sidecar, and shard metadata plus read-only validation facts.
- `data/comparison_records.jsonl`: one deterministic public-derived statistic per line.
- `data/comparison_summary.json`: the 24-layer D0/D0.1/D0.2 table and focused 6/13/18/14/23 cross-stage view.

D0.2 contributes all 24 layers, Q/K aggregate direction values, eight pair medians per layer-kind, per-transform two-reference statistics, common-finite support, eligibility, selected state, and route audit. D1 and D2 contribute their public layer-kind, pair, and available per-transform statistics. D0 contributes only the aggregate and shard facts present in its public receipt. D0.1 contributes finite/null support, record counts, eligibility, frozen selected state, and its opaque public selection tuple.

The D0.2 public receipt does not expose coverage, recovery error, fit residual, ambiguity gap, or layer-level null counts. These fields are explicitly marked unavailable rather than reconstructed. Selection observations and confirmation observations remain distinct.

## Focus comparison

| Layer | D0.1 selected | D0.2 Q/K | D0.2 selected | D1 Q/K | D2 Q/K |
| ---: | --- | --- | --- | --- | --- |
| 6 | yes | -2.75 / -2.25 | no | 0 / 0 | — |
| 13 | yes | -5.5 / -4 | no | -1 / -0.75 | — |
| 18 | yes | -8 / -8 | no | -0.75 / -0.25 | — |
| 14 | no | -9.25 / -8.75 | yes | — | -3.5 / -3.5 |
| 23 | no | -15 / -10 | yes | — | -3.5 / 0 |

D2's frozen rule requires all four layer-kind statistics to be strictly negative. Layer 23 K is `0`; similarity and crop-rescale are `0` in all four D2 layer-kind cells, and the public route audit records `route_level_transform_instability=true`.

## Evidence ceiling

This package is operational evidence only. The route remains `OPERATIONAL_UNRESOLVED` with `science_denominator=0`. It does not establish geometric reliability, content detection, watermark attribution, or scientific success.
