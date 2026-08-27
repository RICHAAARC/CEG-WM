# Geometry-V1 Operational Evidence

## Identity and final state

- Branch family: `Geometry-V1`
- Method identity: `geometry_v1_natural_qk_geometry_observation`
- Final semantic identity: passive natural-Q/K geometry observation
- Final state: `OPERATIONAL_UNRESOLVED`
- Scientific denominator: `0`
- Semantics-freeze exact: `95dd416d541450145436b685897d5ee210eb04e5`

Geometry-V1 records whether natural Q/K observations retain useful geometric direction under the frozen operational protocols. Geometry only provides coordinate-recovery evidence and cannot create a positive watermark decision. All entries below are operational evidence; none is a scientific conclusion or a claim of method or detector success.

## Evidence summary

| Stage | Protocol | Runner exact | Run | Public result |
| --- | --- | --- | --- | --- |
| D0 | `geometry-v1-qk-d0-all-layer-discovery-v1` | `4732211beefbeface95cb842c117b9719e362f1a` | `geometry-v1-qk-d0-4732211beefb` | 24 shards x 32 = 768 calculated, 0 failed; `D0_UNRESOLVED` |
| D0.1 | `geometry-v1-qk-d01-artifact-selection-v1` | `ccfb7bcefbb18f9812a4e800bbea18b91b031ebb` | `geometry-v1-qk-d01-ccfb7bcefbb1` | 768 source units; ordered layers 6, 13, 18; `D01_CANDIDATES_FROZEN` |
| D0.2 | `geometry-v1-qk-direction-all-layer-selection-v1` | `41742d462d62525189855c8ebb2ee1995fb9230a` | `geometry-v1-qk-direction-all-layer-41742d462d62` | 768 source units; ordered layers 23, 14; `DIRECTION_TWO_CANDIDATES_FROZEN` |
| D1 | `geometry-v1-qk-d1-independent-confirmation-v1` | `69171346d8fe8889dc2202d4f34c1cd4a834be34` | `geometry-v1-qk-d1-69171346d8fe` | 96/96 calculated, 0 failed; `D1_UNRESOLVED` |
| D2 | `geometry-v1-qk-d2-independent-confirmation-v1` | `d54492044e7789da3883fc75e2075a253ac22c75` | `geometry-v1-qk-d2-d54492044e77` | 2 shards x 32 = 64 calculated, 0 failed; `D2_UNRESOLVED` |

## Bound identities

D0 used plan digest `96e1e5ae6fb8ae66a545b1b10d6c896176989272c81ef1fd737184dcdfaea7b8` and roster digest `88850de32ae0783427f86d0a5c82c6272a30811931ca0f883f6888cf8b83ac9e`.

D0.1 froze the ordered observations `transformer_blocks.6.attn`, `transformer_blocks.13.attn`, and `transformer_blocks.18.attn` from the immutable 768-unit D0 source.

D0.2 performed artifact-only all-layer direction selection over the immutable 768-unit D0 source. It froze the ordered observations `transformer_blocks.23.attn` and `transformer_blocks.14.attn`; this stage is selection evidence, not confirmation evidence.

D1 reported the following frozen layer-kind statistics:

| Layer | Q | K |
| --- | ---: | ---: |
| 6 | 0 | 0 |
| 13 | -1 | -0.75 |
| 18 | -0.75 | -0.25 |

D2 used plan digest `81394d4bd2ed9e437a8914c707b3dca60cb0842c67f79c716b39b5b8610db310` and the ordered observations 23 and 14. Its independently audited frozen statistics were:

| Layer | Q | K |
| --- | ---: | ---: |
| 23 | -3.5 | 0.0 |
| 14 | -3.5 | -3.5 |

The frozen D2 rule requires all four statistics to be strictly below zero. The sole unmet term was layer 23 K at `0.0`. Similarity and crop-rescale each reported zero across all four layer-kind cells, and `route_level_transform_instability` was `true`.

## Public artifact locations

- D1 folder: https://drive.google.com/drive/folders/1ye3mFWkF6JJqJQg1IhSldIMqUEOCSvc6
- D2 folder: https://drive.google.com/drive/folders/1QbtPkS2l5phSNDkWLdOKyrQDyiNLZz2y
- D2 layer 23 shard, 58,142 bytes: https://drive.google.com/file/d/1zwkGICM-WxusY9cJ-xCLhJ47hyZZ73Cb/view
- D2 layer 14 shard, 58,579 bytes: https://drive.google.com/file/d/12psW3azzr2Ie3SU1sh8GIWBSScj6rr68/view

No trusted folder URL was supplied for D0, D0.1, or D0.2. No trusted artifact hash was supplied for the listed Drive artifacts, so the machine-readable index records those fields as `null` rather than inferring them.

## Evidence ceiling

This package records bounded public operational observations only. The route remains `OPERATIONAL_UNRESOLVED` with `science_denominator=0`. It does not establish geometric reliability, content detection, watermark attribution, or scientific success.
