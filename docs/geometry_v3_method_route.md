# Geometry-V3 method route

## Identity and scope

- Branch: `Geometry-V3`.
- Method identity: `geometry_v3_keyed_qk_canonical_anchor`.
- Route role: active keyed geometric synchronization through a canonical
  relation anchor written into predeclared SD3.5 Q/K/attention feature
  placements.
- Current delivery: the P0 active-writer discovery implementation has one real
  SD3.5 controlled-stop artifact, and the bounded P0D sequence has culminated
  in a real single-configuration writer-completion canary.
- Evidence ceiling: the real P0 and P0D artifacts are controlled operational
  evidence with `science_denominator=0`; local fake/CPU/static results remain
  engineering evidence. Neither class is a recoverability, detector,
  watermark, or scientific conclusion.

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

The completed real P0 run `geometry-v3-qk-p0-9b5085c805b6` at runner exact
`9b5085c805b6e3580fadc153598aac93fcc41eab` retained all 144 derived units as
calculated with zero failed units. It completed the writer-final-RGB, attack,
and fresh attacked-RGB Q/K path and froze the single discovery candidate
`block12-qk-rms0p0025`. Its status is `P0_WRITER_CANDIDATE_FROZEN` with
`science_denominator=0`. This is discovery evidence only. It does not establish
independent confirmation, blind H/corners recovery, content detection,
watermark success, or a scientific conclusion.

### P1 independent active-anchor confirmation

The separately authorized protocol is
`geometry-v3-keyed-qk-active-writer-p1-confirmation-v1`. Before model work it
fail-closed validates the immutable bounded P0 receipt, manifest, terminal and
144 public metrics against the exact source run, protocol, execution commit,
plan digest, roster digest, complete status, 144/0 counts, unique selected
configuration and zero science denominator. It reads no raw Q/K, image, prompt
text, key, token, latent, model weights or private path from that artifact.

P1 fixes `block12-qk-rms0p0025`: block 12 sample-side Q/K, writer step 18 and
hard relative-RMS budget 0.0025. There is no reselection, placement or budget
switch, tuning, retry or fallback. Its independent generation uses public
prompt identity `geometry-v3-p1-public-prompt-01`, generation seed 173,
observation noise seed 19073 and timestep 500. These differ from P0. The four
new attack instances are identity; Pillow `ROTATE_270`; similarity at -11
degrees, scale 0.89 and translation (-17,+9) about image centre; and BICUBIC
crop-rescale with box (46,28,470,482). Their coordinate mappings are frozen
before execution.

The fixed 24-unit roster is one configuration by four attacks by Q/K by
correct-key anchor, wrong-key anchor and no-writer controls. Every failure is
retained. Each attack/kind margin is correct-key correlation minus the larger
of wrong-key and no-writer correlation. Confirmation requires only the Q
equal-weight four-attack median and K equal-weight four-attack median to be
strictly positive; per-transform margins remain audit records and are not
separate gates. The only statuses are `P1_STOPPED`, `P1_UNRESOLVED`, and
`P1_ACTIVE_ANCHOR_CONFIRMED`, always with `science_denominator=0`.

Even `P1_ACTIVE_ANCHOR_CONFIRMED` means only that the frozen active anchor was
observed in fresh attacked-RGB Q/K under independent instances. It is not a
blind H/corners, reliability, rectification, content-watermark, detector, or
scientific conclusion.

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

The real P0D artifact reached denoising step 18 and entered the block-4 Q
production hook, but stopped after hook entry and before the public Q
measurement was recorded. Its K hook for that transformer call, session
completion, and final-RGB validation were not reached. The complete artifact
and zero science denominator establish only this bounded operational location;
they do not establish writer, placement, budget, anchor, or method success or
failure, and the stopped exact is not rerun unchanged.

### P0D.1 inner-Q diagnostic

The next protocol is `geometry-v3-keyed-qk-active-writer-p0d1-v1`. It keeps the
single fixed `block4-qk-rms0p0025` configuration, one generation, step-18 hook
timing, and hard 0.0025 relative-RMS budget unchanged. A default-disabled
observer inside the production Q hook may count these ordered, value-free
checkpoints only: `q_output_contract_pass`, `q_pattern_materialized`,
`q_base_rms_validated`, `q_delta_materialized`, `q_ratio_validated`,
`q_budget_validated`, and `q_measurement_recorded`. It exposes no shape, dtype,
device, tensor, scalar measurement, pattern, anchor, exception text, prompt,
latent, key, token, weight, or private path. There is no baseline, attack,
fresh-Q/K observation, retry, fallback, alternate placement or budget, dynamic
selection, or threshold adjustment. The only statuses are `P0D1_STOPPED` and
`P0D1_DIAGNOSTIC_COMPLETE`, with `science_denominator=0`.

The real P0D.1 artifact completed callbacks 0 through 17, armed the writer,
entered transformer call 18 and the block-4 Q production hook, and retained
the checkpoints through `q_ratio_validated`. It stopped before
`q_budget_validated`, Q measurement recording, the corresponding K hook,
session completion, or final-RGB validation. This establishes only that the
first actual Q ratio was finite and positive and that the stop occurred after
that validation but before hard-budget validation. The public
`runtime_error` does not distinguish correction-path tensor operations from
the explicit hard-budget rejection and does not authorize changing the
writer or its budget.

### P0D.2 final inner-Q narrowing

The final Stage-1 narrowing protocol is
`geometry-v3-keyed-qk-active-writer-p0d2-v1`. It preserves the same single
`block4-qk-rms0p0025` configuration, step-18 timing, one generation, writer
math, hard budget, and production hook. Between `q_ratio_validated` and the
existing `q_budget_validated`, the default-disabled observer may additionally
count only these ordered, value-free events:

1. `q_initial_budget_comparison_completed`, emitted after the fixed soft-limit
   comparison whether or not correction is needed;
2. `q_correction_branch_entered`, only when that branch is entered;
3. `q_corrected_output_materialized`;
4. `q_corrected_delta_materialized`;
5. `q_post_correction_ratio_computed`;
6. `q_hard_budget_rejected`, immediately before the existing explicit
   hard-budget exception; or
7. `q_hard_budget_accepted`, after the hard check passes and before the
   existing `q_budget_validated` checkpoint.

P0D.2 publishes integer counts and finite public stage/error enums only. It
publishes no ratio, tensor, scalar measurement, dtype, device, shape, Q/K,
pattern, anchor, latent, prompt, key, token, model weight, exception text, or
private path. It has no baseline, attack, fresh-Q/K observation, retry,
fallback, alternate placement/budget, candidate selection, or threshold
adjustment. Its only statuses are `P0D2_STOPPED` and
`P0D2_DIAGNOSTIC_COMPLETE`, always with `science_denominator=0`. P0D.2 is a
diagnostic location result only: it cannot be described as a budget failure or
as authorization for mechanical repair before its real artifact is audited.

### Stage-3 single-configuration writer closure

The real P0D.2 completion canary used runner
`a27a940ae1ef4d1141925c6304caab55b89ec999` and run
`geometry-v3-qk-p0d2-a27a940ae1ef`. Its create-only artifact is complete with
status `P0D2_DIAGNOSTIC_COMPLETE`: the fixed block-4 Q and K hooks each
injected exactly once, all 20 pipeline callbacks and 20 transformer-root calls
completed, the Q hard-budget path recorded one acceptance and zero rejections,
and Q measurement, writer-session completion, and final-RGB validation each
completed once. The artifact retains `science_denominator=0`.

This is engineering and operational closure of the one frozen
`block4-qk-rms0p0025` writer path only. It unlocks the separately controlled
full P0 discovery stage, but does not establish attacked-RGB fresh-Q/K
recoverability, writer placement or budget selection, blind coordinate
recovery, content-watermark detection, or scientific success.

## Frozen progression after P0D.2

The route advances only in this order:

1. **Mechanical writer repair.** The last completed checkpoint can authorize
   only its adjacent narrow correction: make the pattern follow the Q device,
   apply a controlled dtype conversion at the frozen position, correct the
   predeclared pattern expansion, or correct the RMS/ratio calculation domain.
   A genuine budget stop first requires checking unit-RMS delta normalization;
   it does not authorize increasing 0.0025. Placement, budget, hook timing, and
   method semantics remain frozen.
2. **Single-configuration writer completion canary.** Still using
   `block4-qk-rms0p0025`, Q and K must each write exactly once, both measured
   relative RMS values must remain within the hard budget, the session must
   complete, and final RGB must be finite and valid. This closes only writer
   engineering, not recoverability.
3. **Full P0 discovery.** Only after that canary may the frozen six
   placement/budget configurations and 144 retained units run. Final writer RGB
   receives all four attacks; detection recomputes fresh attacked-RGB Q/K and
   compares correct-key anchor, wrong-key anchor, and no-writer controls under
   the frozen selection rule. Status remains `P0_STOPPED`, `P0_UNRESOLVED`, or
   `P0_WRITER_CANDIDATE_FROZEN`, with `science_denominator=0`. Discovery data
   cannot also serve as confirmation.
4. **Independent confirmation.** One frozen placement/budget is tested with
   new prompts, seeds, generations, and attack instances. There is no
   reselection, budget change, margin tuning, or transform deletion. A passing
   result shows only that the active anchor is observable in fresh attacked-RGB
   Q/K.
5. **Blind H/corners recovery.** Attacked RGB produces fresh Q/K; keyed-anchor
   alignment estimates H within a predeclared search domain and returns
   corners, support, and fail-closed reliability. The H domain, optimization,
   stopping rule, reliability rule, retained failures, and independent
   validation are frozen before execution. Ground-truth attack H is evaluation
   metadata only and cannot enter estimation or guide search. Correlation after
   truth-H transport is signal evidence, not blind coordinate recovery.
6. **Content-watermark end to end.** The content watermark and V3 anchor writer
   produce final RGB, attacks are applied, V3 estimates H blindly, reliability
   gates rectification, and the unchanged content detector reruns with the same
   content key, preprocessing, and threshold. Writer interference with the
   content chain is recorded. Only that content detector can issue positive
   watermark evidence.

No stage is unlocked merely because its artifact is complete. Automatic
fallback, post-result budget or threshold tuning, unchanged repetition of a
stopped exact, and scientific adjudication remain outside this route.

Real SD3.5, GPU, Colab, Drive, Hugging Face access, push, formal evaluation,
and scientific adjudication remain separately authorized actions.

## P1M0 posterior mechanism audit

The diagnostic protocol
`geometry-v3-keyed-qk-canonical-anchor-p1m0-mechanism-audit-v1` follows the
independent P1 operational result without reopening discovery or advancing to
blind coordinate recovery. Before model loading it validates the immutable P0
and P1 receipt, manifest, terminal, metrics, hashes, byte counts, roster and
zero-denominator identities. It retains the selected P0 configuration's 24
public score components, all 24 P1 public score components, and the six
identity-control differences labelled only as `two_instance_displacement`.
Two instances do not establish a population variance.

P1M0 fixes `block12-qk-rms0p0025`, writer step 18, identity geometry, one new
public prompt identity and seed. It performs exactly one paired no-writer
generation and one writer generation. It cannot retry, switch placement or
budget, select a candidate, tune a threshold, or use another attack. A
default-disabled production writer observer may retain only bounded public
scalars and contract facts: pre/post correct- and wrong-anchor normalized
correlations, actual relative RMS, Q/K module identity, row-major y/x token
axis, square token-grid and channel counts, zero-mean unit-RMS normalization,
positive injection sign, and finite booleans.

Writer-step-18 and final predecode latents are transient in-process snapshots.
They and the final RGB re-encode are each observed under the same frozen fresh
noise, timestep and zero-text contract. The paired no-writer path is observed
at the same three stages. Public output contains only derived scores,
writer-versus-control separation, and stage-to-stage score changes. It cannot
contain images, Q/K, latents, anchor or pattern material, prompt text, keys,
tokens, model weights, exception text, or private paths.

The predeclared diagnostic statuses are limited to
`P1M0_IMPLEMENTATION_MISMATCH_INDICATED`,
`P1M0_OBSERVABILITY_INSUFFICIENCY_INDICATED`, `P1M0_INCONCLUSIVE`, and
`P1M0_STOPPED`. A mismatch indication requires a public layout,
normalization, module, or positive-injection-sign contract failure.
Observability insufficiency requires all such contracts to pass, positive Q
and K writer-hook lifts, and nonpositive Q and K separation after final-RGB
re-encoding. Every other complete pattern is inconclusive. These are
mechanism-diagnostic indications, not method, detector, watermark, or
scientific success/failure. `science_denominator=0` remains fixed.
