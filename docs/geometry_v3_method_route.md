# Geometry-V3 method route

## Identity and scope

- Branch: `Geometry-V3`.
- Method identity: `geometry_v3_keyed_qk_canonical_anchor`.
- Route role: active keyed geometric synchronization through a canonical
  relation anchor written into predeclared SD3.5 Q/K/attention feature
  placements.
- Current delivery: local pure-CPU contract scaffold only.
- Evidence ceiling: engineering structure and deterministic contract tests.
  No SD3.5 writer, model execution, recoverability result, detector result,
  watermark result, or scientific conclusion exists at this stage.

Geometry can recover coordinates only. It has no authority to create a
positive watermark decision. Any image rectified by this route must be judged
again by the same content detector under the same content-key semantics,
preprocessing identity, and frozen threshold identity.

## Canonical relation anchor

The generation side derives an ordered two-dimensional canonical relation
anchor from a domain-separated `geometry_key`. The frozen domain is distinct
from content-key carrier domains. Derivation is deterministic and returns only
bounded normalized public coordinates plus a public digest; raw key material
is never part of the returned contract.

The anchor is a geometric synchronization target, not content evidence and not
a payload. A future writer must bind its public anchor identity to its exact
placement and budget declarations before model execution.

## Writer declaration

There are no default writer layers. Before a writer phase may start, a later
experiment must independently predeclare all of the following:

1. the exact SD3.5 module path and feature role (`q`, `k`, or `attention`) for
   every placement;
2. a placement-study protocol identity independent of passive observation;
3. a finite positive total relative-L2 budget and a per-placement budget no
   greater than the total;
4. a content-interference test protocol identity.

Placements must be non-empty and unique. No layer pair from another route is
inherited or treated as a writer conclusion. Writer placement, perturbation
budget, anchor recoverability, and interference with the content chain require
their own predeclared validation. This stage performs no all-layer selection.

## Recoverability production path

The frozen phase order is:

1. admit a complete writer declaration;
2. produce final RGB;
3. produce the attacked RGB instance;
4. recompute fresh Q/K observations from that current attacked RGB only;
5. align those observations to the keyed canonical relation anchor;
6. estimate four corners, a bounded invertible homography, support, and an
   independent reliability score;
7. fail closed if either support or reliability is below its predeclared
   minimum;
8. when reliable, authorize inverse rectification and submit the rectified RGB
   to the unchanged content detector binding.

Detection must not consume original RGB, embedding records, embedding latent,
embed-side routes, or cached embedding Q/K. The scaffold records only public
identities and geometric estimates; it does not carry raw Q/K tensors.

## Reliability and decision boundary

Reliability is computed from explicit support and reliability values against
their independently declared minima. It is not a caller-supplied pass flag.
Corners must be four distinct normalized 2-D points. The homography must be a
finite, normalized, bounded, invertible 3-by-3 matrix. Any malformed estimate,
phase violation, provenance mismatch, insufficient support, insufficient
reliability, or content-detector identity change stops rectification.

Passing this geometry contract means only that coordinate recovery may be used
for rectification. It never means that a watermark was detected. The content
detector remains the sole positive authority.

## Next separately authorized work

Model-backed work must be separately exact-bound and must freeze placements,
budgets, interference measures, attack roster, fixed denominators, retained
failures, artifact limits, and stopping rules before execution. The first
method experiment should validate the complete active-writer to final-RGB to
attacked-RGB to fresh-Q/K recovery path at predeclared placements. It must not
begin with adaptive all-layer selection, automatic fallback, or route mixing.

Real SD3.5, GPU, Colab, Drive, Hugging Face access, push, formal evaluation,
and scientific adjudication remain outside this scaffold delivery.
