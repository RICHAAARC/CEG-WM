# Geometry-V1 Status

## Identity

- Branch: `Geometry-V1`
- Method: `geometry_v1_natural_qk_geometry_observation`
- Route type: passive natural-Q/K geometry observation
- Final state: `OPERATIONAL_UNRESOLVED`
- Science denominator: `0`
- Evidence-code baseline: `43ef528eabb6fa2096376a85bba898acd904e089`

Geometry-V1 is complete as an operationally audited observation route. It is
not a positive watermark authority. Geometry may recover coordinates only;
content statistics remain the sole source of a positive watermark decision.

## Evidence summary

| Node | Protocol result | Unit evidence | Frozen layer record |
| --- | --- | --- | --- |
| D0 | `D0_UNRESOLVED` | 768/768 calculated, 0 failed | all 24 layers audited |
| D0.1 | `D01_CANDIDATES_FROZEN` | immutable D0 artifact analysis | 6, 13, 18 |
| D0.2 | `DIRECTION_TWO_CANDIDATES_FROZEN` | immutable D0 artifact analysis | ordered 23, 14 |
| D1 | `D1_UNRESOLVED` | 96/96 calculated, 0 failed | fixed 6, 13, 18 |
| D2 | `D2_UNRESOLVED` | 64/64 calculated, 0 failed | fixed ordered 23, 14 |

The D2 aggregate values were layer 23 Q = -3.5, layer 23 K = 0.0,
layer 14 Q = -3.5, and layer 14 K = -3.5. The fixed rule required all four
values to be strictly below zero. The layer 23 K value therefore left the
route unresolved.

Similarity and crop-rescale directional observations were 0.0 across all four
D2 layer-kind cells. The audited record sets
`route_level_transform_instability=true`.

## Evidence identities

- D0 execution exact: `4732211beefbeface95cb842c117b9719e362f1a`
- D0 run: `geometry-v1-qk-d0-4732211beefb`
- D0 protocol: `geometry-v1-qk-d0-all-layer-discovery-v1`
- D0 plan digest:
  `96e1e5ae6fb8ae66a545b1b10d6c896176989272c81ef1fd737184dcdfaea7b8`
- D0.1 runner exact: `ccfb7bcefbb18f9812a4e800bbea18b91b031ebb`
- D0.1 run: `geometry-v1-qk-d01-ccfb7bcefbb1`
- D0.2 runner exact: `41742d462d62525189855c8ebb2ee1995fb9230a`
- D0.2 run: `geometry-v1-qk-direction-all-layer-41742d462d62`
- D1 runner exact: `69171346d8fe8889dc2202d4f34c1cd4a834be34`
- D2 runner exact: `d54492044e7789da3883fc75e2075a253ac22c75`
- D2 plan digest:
  `81394d4bd2ed9e437a8914c707b3dca60cb0842c67f79c716b39b5b8610db310`

## Claim boundary

This status card records complete bounded operational artifacts and their
predeclared outcomes. It does not claim method success, detector success,
watermark presence, formal false-positive-rate performance, or scientific
adjudication. `science_denominator=0` is unchanged.

The normative route definition and fuller evidence interpretation are in
[geometry_v1_method_route.md](geometry_v1_method_route.md).
