# Geometry-V3 method route

## Identity and scope

- Branch: `Geometry-V3`.
- Method identity: `geometry_v3_keyed_qk_canonical_anchor`.
- Route role: active keyed geometric synchronization through a canonical
  relation anchor written into predeclared SD3.5 Q/K/attention feature
  placements.
- Current delivery: the P0 active-writer discovery implementation has one real
  SD3.5 operational artifact, followed by a bounded P0D single-configuration
  diagnostic implementation.
- Evidence ceiling: the real P0 artifact is controlled operational evidence
  with `science_denominator=0`; P0D local fake/CPU/static results are
  engineering evidence until a separately authorized real run. Neither class
  is a recoverability, detector, watermark, or scientific conclusion.

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

The complete intended method route is therefore fixed as follows. A
domain-separated geometry root/key derives V3 key material and a two-dimensional
keyed canonical relation anchor. Generation writes that anchor only at an
independently predeclared SD3.5 Q/K/attention placement under a hard budget and
an independent content-interference check. The final RGB is attacked by the
predeclared attack roster. Detection reconstructs fresh Q/K from that attacked
RGB alone and aligns it to the keyed canonical anchor to estimate corners, H,
support, and reliability. Reliability fails closed. Only a reliable estimate
may authorize rectification, after which the unchanged content detector must
run with the same content key, preprocessing identity, and frozen threshold.
Geometry has coordinate authority only and cannot produce positive content
watermark evidence.

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

## P0 active-writer discovery freeze

The first model-backed protocol identity is
`geometry-v3-keyed-qk-active-writer-p0-v1`. It freezes SD3.5-medium at 512 by
512, 20 inference steps, seed 73 and one public prompt identity. The active
writer has six independently declared configurations only: blocks 4, 12 and
20, each writing sample-side `attn.to_q` and `attn.to_k` at hard relative-RMS
budget 0.0025 or 0.005. A paired no-writer generation uses the same public
prompt, seed and model load. There is no default or inherited placement and no
all-layer scan.

The writer is armed after denoising step 17, writes Q and K exactly once in
the zero-based transformer call for step 18, and is verified after that call.
A topology that is not exactly one transformer call per denoising step stops
the configuration; it cannot select another step, layer or budget. The keyed
anchor becomes a transient zero-mean unit-RMS low-rank token-grid by channel
pattern. Actual relative RMS and call count are public scalar records; keys,
anchor tensors and Q/K tensors are not artifacts.

Every final writer RGB and the paired no-writer RGB is attacked by identity,
90-degree rotation, the frozen similarity transform and the frozen
crop-rescale transform. Fresh observation begins only from that current RGB:
the public VAE posterior mode, public noise seed 9073 and public timestep 500
feed a direct transformer call with frozen zero conditioning. It cannot use an
embedding cache, prompt, route, latent or homography input. Known homography is
truth metadata used only to transport the public anchor for derived metrics.

The fixed denominator is 144 retained derived units: six configurations by
four attacks by Q/K by correct-key-anchor, wrong-key-anchor and no-writer
controls. The per-attack margin is correct-anchor normalized correlation minus
the larger of wrong-anchor and no-writer correlation. A configuration is
eligible only when all 24 units are calculated and both Q and K equal-weight
four-attack median margins are strictly positive. Eligible configurations are
ordered by descending worst-side margin, descending two-side median, ascending
budget and ascending block index; exactly one is frozen. The only statuses are
`P0_STOPPED`, `P0_UNRESOLVED` and `P0_WRITER_CANDIDATE_FROZEN`, always with
`science_denominator=0`. A frozen P0 candidate is discovery output and is not
confirmation.

RGB MSE/PSNR and an unchanged-content-detector identity hook are record-only;
they cannot alter P0 eligibility. Artifacts are create-only receipt, manifest,
terminal and metrics JSONL under a two-MiB aggregate bound. They exclude
images, prompts, keys, raw anchors, Q/K, latents, weights and private paths.

## Next separately authorized work

### Real P0 operational state

The completed real P0 run retained its fixed 144-unit roster as 0 calculated
and 144 failed units. It stopped at the public `writer_generation` boundary;
all six writer configurations retained `runtime_error`, while writer
measurements, interference records, fresh attacked-RGB Q/K observations, and
candidate selection were not reached. Its status is `P0_STOPPED` with
`science_denominator=0`. This is a complete controlled-failure artifact, not a
failure adjudication of the keyed-Q/K-anchor method, and the same full P0 exact
must not be rerun unchanged.

### P0D single-configuration diagnostic

The next operational protocol is
`geometry-v3-keyed-qk-active-writer-p0d-v1`. It freezes exactly one run of
`block4-qk-rms0p0025`; it has no baseline substitute, alternate placement,
alternate budget, retry, fallback, or candidate selection. It reuses the P0
writer semantics and exposes only these bounded public facts:

1. pipeline-load and writer-session setup counts;
2. denoising callback count and whether transformer call 18 was reached;
3. transformer-root call count;
4. block-4 `to_q`/`to_k` hook-hit and injection counts;
5. session-completion and final-RGB-validation counts;
6. one finite failure-point enum and generic error class.

The only statuses are `P0D_STOPPED` and `P0D_DIAGNOSTIC_COMPLETE`, always with
`science_denominator=0`. Create-only receipt, manifest, and terminal artifacts
exclude exception text, tensors, raw Q/K, anchor material, latent, prompt text,
geometry key, Hugging Face token, model weights, and private paths. P0D does not
attack RGB, observe fresh Q/K, select a writer, or make a scientific decision.

After writer generation is mechanically repaired, P0 discovery may be run
under a separately reviewed exact. Only after it freezes one placement and
budget may an independent confirmation use new generation and attack instances
with no reselection. It must preserve retained failures, bounded artifacts, and
the same coordinate-only evidence ceiling.

Real SD3.5, GPU, Colab, Drive, Hugging Face access, push, formal evaluation,
and scientific adjudication remain separately authorized actions.
