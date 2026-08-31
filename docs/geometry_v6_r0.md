# Geometry-V6 public R0 pilot

`geometry_v6_public_fixed_unkeyed_roundtrip_pilot_v1` is a public deterministic, unkeyed geometry pilot. It is coordinate/synchronization observability only: never authentication, ownership, or positive watermark evidence. `science_denominator=0`.

The unchanged content callback writes at step 18. After the final scheduler update, step 19 applies one fixed-amplitude frozen-VAE `E(D(z))` adjoint update to the full public pilot in strict `0.24 < normalized radius < 0.58` support; final decode follows. The engine owns the fixed amplitude sequence with no feedback, retry, selection, or threshold tuning.

The public support is deterministically split into non-overlapping `search`, `fit`, and `validate` subsets. Future search may propose candidates, fit freezes them, and validate is holdout-only. R0 records only aggregate/subset raw observations from ordinary RGB plus frozen public VAE; it does not validate H or reliability.

R0 records immutable content-only/unwatermarked and per-amplitude content+geometry/geometry-only arms at the same prompt/seed. Content remains the unchanged frozen whitening-LF/HF/V9 detector with content key and 16 content wrong keys. A positive-to-negative content flip is fail-closed; geometry never votes or rescues it. Carrier window and conditional-flow FPR are `NOT_ADJUDICATED`.

Future conditional flow is documentation only: retain an initial content positive; after a negative, public geometry may estimate H; only RELIABLE geometry permits one rectification and one same-detector/same-tau retry. Formal FPR remains unadjudicated.

The Colab notebook mounts Drive first, clones and detached-checks out an approved exact, runs one fresh-subprocess full-sequence diagnostic, and create-only writes its small JSON under `Geometry-V6/R0`. It accepts only content and HF secrets.
