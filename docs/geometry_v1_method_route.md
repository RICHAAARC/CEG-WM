# Geometry-V1 Natural Q/K Geometry Observation

## Method identity and final state

The normative method identity of this branch is
`geometry_v1_natural_qk_geometry_observation`.

Geometry-V1 is a passive natural-Q/K geometry observation method. It observes
sample-side Q/K features recomputed from the current RGB image and measures
whether their natural correspondences retain geometric direction under a fixed
attack roster. Its final route state is `OPERATIONAL_UNRESOLVED`, with
`science_denominator=0`.

Geometry is limited to coordinate recovery. It cannot emit a positive
watermark decision. Content statistics remain the only positive watermark
authority.

## Observation and decision boundary

The observation path is:

`current RGB -> VAE posterior mode -> public noise/time -> SD3 transformer -> sample-side to_q/to_k`

The Geometry observation consumes the current image and frozen public
observation assets. It does not consume an original image, prompt, embed
record, private latent, embed-side route, cached embedding Q/K, or true attack
parameters. Known transforms and reference images exist only as experiment
truth and do not enter the observation API. Any content decision remains bound
to the current image, detection key, and frozen public content-method assets.

Observed Q/K tensors are reduced in process to bounded public derived metrics.
Raw Q/K, tokens, prompts, latents, secrets, model weights, image bytes, and
private paths are not persisted in public artifacts.

The permissible Geometry output contract is limited to coordinate quantities:
corners, `H_canonical_to_observed`, `H_observed_to_canonical`, valid support,
and raw reliability observations. Any reliable inverse rectification must
preserve the same content detector, key semantics, preprocessing, and
calibrated threshold. Geometry does not recover image content deleted by a
crop.

## Fixed evaluation rule

Matched and shuffled controls are compared only at the same reference-token
indices where both true-match ranks are finite. Each fixed reference/attack
pair produces the median of `matched_rank - shuffled_rank`. A layer-kind
statistic is the equal-weight median over the eight fixed pair medians. A
directional confirmation requires every predeclared Q/K layer-kind statistic
to be strictly below zero.

Missing common-finite support makes the corresponding fixed statistic
ineligible. Units, transforms, and denominators remain fixed. Recovery error,
fit residual, ambiguity gap, coverage, null counts, and per-transform values
are retained audit observations rather than post-run gates.

## Completed operational evidence chain

| Node | Fixed scope | Final operational fact |
| --- | --- | --- |
| D0 | 24 layers x 32 records = 768 retained units | `D0_UNRESOLVED`; 768 calculated and 0 failed in the audited artifact |
| D0.1 | Artifact-only missingness-aware selection | `D01_CANDIDATES_FROZEN`; layers 6, 13, and 18 |
| D0.2 | Artifact-only all-layer directional selection | `DIRECTION_TWO_CANDIDATES_FROZEN`; ordered layers 23 and 14 |
| D1 | Three fixed layers, 96 retained units | 96 calculated, 0 failed, `D1_UNRESOLVED` |
| D2 | Two fixed layers, 64 retained units | 64 calculated, 0 failed, `D2_UNRESOLVED` |

### D0

Protocol `geometry-v1-qk-d0-all-layer-discovery-v1` evaluated the contiguous
sample-side paths `transformer_blocks.0.attn` through
`transformer_blocks.23.attn`. Two deterministic asymmetric procedural RGB
references were crossed with identity, D4, similarity, and crop-rescale,
Q/K, and matched/shuffled controls. The audited artifact contains all 768
predeclared public derived records.

### D0.1

Protocol `geometry-v1-qk-d01-artifact-selection-v1` consumed the immutable D0
artifact without model execution. Its fixed strata selection produced
`transformer_blocks.6.attn`, `transformer_blocks.13.attn`, and
`transformer_blocks.18.attn` as an operational candidate record.

### D0.2

Protocol `geometry-v1-qk-direction-all-layer-selection-v1` recomputed the
fixed directional statistic across all 24 D0 layers from the immutable public
artifact. Its predeclared ordering froze
`transformer_blocks.23.attn` followed by `transformer_blocks.14.attn` for an
independent confirmation run.

### D1

Protocol `geometry-v1-qk-d1-independent-confirmation-v1` used new procedural
references and attack instances with the fixed 6/13/18 layers. The real
artifact contains 96 of 96 calculated units and no failed unit. The fixed
six-statistic strict-negative conjunction resolved to `D1_UNRESOLVED`.

### D2

Protocol `geometry-v1-qk-d2-independent-confirmation-v1` used fresh procedural
references, attack instances, and observation seed 73 with the fixed ordered
23/14 layers. The real artifact contains 64 of 64 calculated units and no
failed unit. Independently recomputed layer-kind aggregates were:

| Fixed layer | Q aggregate | K aggregate | Strict-negative result |
| --- | ---: | ---: | --- |
| `transformer_blocks.23.attn` | -3.5 | 0.0 | not satisfied |
| `transformer_blocks.14.attn` | -3.5 | -3.5 | satisfied |

The layer-23 K aggregate of 0.0 did not satisfy the predeclared strict
`< 0` rule, so the complete run resolved to `D2_UNRESOLVED`. Identity and D4
directional observations were negative across all four layer-kind cells;
similarity and crop-rescale observations were 0.0 across all four cells. The
artifact therefore records `route_level_transform_instability=true`.

## Evidence ceiling

The completed chain is operational evidence that the fixed passive natural-Q/K
observation protocol ran with bounded public artifacts and resolved its
predeclared rules. `OPERATIONAL_UNRESOLVED` is the final Geometry-V1 route
state. `science_denominator=0` remains fixed.

These records do not establish method success, detector success, watermark
presence, fixed-FPR behavior, or a scientific conclusion. Artifact integrity,
complete unit rosters, and clean execution do not raise that evidence ceiling.

## Stable invariants

- Geometry only supplies coordinate-recovery evidence; content statistics are
  the sole positive watermark authority.
- The observation API remains blind to experiment truth and private
  embedding-side state.
- The same content detector, key semantics, preprocessing, and threshold must
  be preserved around any future coordinate rectification.
- Predeclared units and failed units remain visible in fixed denominators.
- Public artifacts remain bounded derived records and exclude sensitive or raw
  model/image material.
- The final branch state is descriptive operational evidence with zero science
  denominator.
