# Geometry-V2 Operational Evidence Card

## Final route identity

- Method: `geometry_v2_keyed_neural_corner_sync`.
- Source branch final: `Geometry-V2@bc147f8985d5e54477d0bbd47f7d44a73f70a6e6`.
- Executed protocol: `geometry-v2-keyed-neural-corner-sync-n0-v1`.
- Execution exact: `d82efc292db8a16f60d272635e577a4186ed866a`.
- Run: `geometry-v2-neural-corner-sync-n0-d82efc292db8`.
- Final status: `OPERATIONAL_UNRESOLVED`; run status: `N0_UNRESOLVED`.
- Evidence ceiling: operational evidence only, with `science_denominator=0`.

The real N0 run trained and evaluated the frozen small PyTorch keyed residual embedder and attacked-RGB-only corner extractor on fixed procedural RGB. It completed all predeclared confirmation units and produced a complete bounded artifact. It did not pass all predeclared candidate gates.

## Frozen execution design

| Split | Seeds | Images | Evaluations |
|---|---:|---:|---:|
| Training | 1000–1127 | 128 | four attack classes sampled with equal fixed weight per epoch |
| Validation | 2000–2031 | 32 | 128 observations |
| Independent confirmation | 3000–3031 | 32 | 128 retained units: 32 each for identity, rotate90, similarity, and crop-rescale |

Training used seed 73, batch size 8, eight epochs, and Adam at `1e-3`. The reliability rule was fixed before confirmation: complete support, score `clamp(1 - mean_corner_error / 0.25, 0, 1)`, threshold `0.5`.

## Candidate-gate result

| Frozen gate | Observed | Result |
|---|---:|---|
| all 128 units calculated | 128 calculated, 0 failed | pass |
| median corner error `<0.05` | 0.29130835680109846 | fail |
| p95 corner error `<0.10` | 0.719844602879589 | fail |
| reliable fraction `>=0.75` | 0.0 | fail |
| actual residual maximum `<=4/255` | 0.013113617897033691 | pass |

The public comparison data retain all 128 per-unit records. Per-attack median corner errors were 0.22914715144119968 for identity, 0.7181837848646667 for rotate90, 0.2474215952053131 for similarity, and 0.33472790520471607 for crop-rescale.

## Route decision and claim boundary

The route ends at N0: no N1, further training, parameter tuning, or checkpoint production is planned. This is a decision to stop further engineering investment after the frozen N0 result. It is not evidence that neural geometric synchronization is theoretically impossible and does not predict the outcome of a differently scoped SD3.5 implementation.

Geometry has coordinate authority only. This evidence cannot create positive watermark evidence, establish method or detector success, establish attack robustness, or support a scientific conclusion. A later reliable rectification request would still have to reuse the same content detector, content key semantics, preprocessing, and calibrated threshold unchanged.

## Evidence package

The bounded package lives on the independent `Geometry-V2-Evidence` branch and must not be merged into `Geometry-V2`. It contains the complete 128-row public metrics file plus independently recomputed summaries and exact Drive provenance. It excludes source images, geometry secret material, access tokens, trained parameters, checkpoints, prompts, latents, and any non-public local path.
