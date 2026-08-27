# Geometry-V1 Method Route and Evidence Gates

## 1. Authority, identity, and scope

This document is the normative method-route authority for the `Geometry-V1`
branch.  The current user authorization and the active project contract and
research scope take precedence if any text here conflicts with them.  Historic
repositories, historical thresholds, readiness statements, artifacts, and
agent verdicts are not current authority.

The method identity is **Geometry-V1** / `geometry_v1_qk_relation_similarity`.
Geometry is a coordinate-recovery component only: it can never create a
positive watermark decision.  Content statistics remain the sole positive
watermark authority.

## 2. Stable end-state route

The intended route is ordered and bounded as follows:

1. Establish oracle-first recoverability for the proposed attack family.
2. Use a domain-separated `geometry_key` as the active Q/K-relation
   synchronization anchor.
3. Independently observe Q/K from the current attacked RGB image.
4. Estimate an eight-hypothesis D4 transform with bounded, proper residual
   rotation, strictly positive isotropic scale, translation, and crop support.
5. Produce corners, `H_canonical_to_observed`,
   `H_observed_to_canonical`, valid support, and raw reliability metrics.
6. Apply an independently fitted, fail-closed reliability decision.
7. Inverse-rectify when reliable, then re-run the same content detector with
   the same key, preprocessing, and calibrated `tau`.

No inpainting is permitted, and Geometry makes no claim to recover content
deleted by a crop.

## 3. Detection and observation boundary

Formal detection accepts only the current image, detection key, and frozen
public method assets.  It must not consume a prompt, original or reference
image, embed record, private latent, embed-side route, cached embedding Q/K,
or true attack parameters.  An evaluation-only reference image or known `H`
may be retained by an experiment solely as truth; it must never enter a
detector API.

The observation contract is: RGB -> VAE posterior mode -> public noise and
time -> direct SD3 transformer -> selected sample-side `to_q`/`to_k`.
Hooks run exactly once per selected observation point.  Their observations are
detached CPU `float32` values; raw tensors are not persisted.  This is an
image-only observation path, not an embedding-side cache or a substitute for
the content detector.

## 4. Transform and search boundary

Search enumerates eight D4 hypotheses.  The residual rotation is proper
(`det=+1`), residual isotropic scale is strictly positive, and translation and
crop support are explicit.  The frozen homography convention is canonical to
observed for `H_canonical_to_observed`; its inverse is
`H_observed_to_canonical`.

Perspective, free homography, local non-rigid deformation, and generative
redraw are excluded from V1.  Existing numeric boxes, budgets, and thresholds
are implementation constraints only and are not inherited as scientific or
final parameters.

## 5. Present semantic gap and the Geometry-V1-QK-E0 gate

The present `keyed_qk_relation` is an observed-by-observed Gram/projection.
The geometry key changes its keyed projection, but does not create a
canonical-axis identity.  A generic relation-to-correspondence bridge built
only from synthetic canonical-by-observed matrices is therefore prohibited:
it would be a self-confirming proxy rather than evidence of blind recovery.

`Geometry-V1-QK-E0` is frozen as a representation/equivariance feasibility
gate only, with `science_denominator=0`.  Its fixed plan is two asymmetric RGB
references crossed with identity, D4, similarity, and crop-rescale transforms:
eight pairs.  For each pair it independently recomputes reference and attacked
observations at two layers, for Q and K, with matched-H and shuffled-H
comparisons: 64 retained units.  Known `H` is truth-only.  E0 neither fits a
threshold or reliability rule.  E0 does not conclude method success, detector
success, or a scientific result.

Its possible descriptive outcomes are:

- `QK_ROUTE_STOPPED`
- `QK_ROUTE_UNRESOLVED`
- `ELIGIBLE_FOR_KEYED_ANCHOR_DESIGN`

The last outcome is not method success.  If the Q/K route stops, FFT latent
sync can be evaluated only after separate user authorization; it is never an
automatic fallback or mixed route.  SyncSeal remains a separate external
baseline, not a Geometry-V1 fallback or per-sample chooser.

## 6. Status snapshot (2026-08-27)

This section is dated status, not a stable method invariant.

- Main content is `content_chain_method_complete`; Geometry, fixed-FPR, and
  the complete system remain incomplete.
- A real operational/artifact success is bound to
  `aa9d4c8211cce5dab7ccddbf37ace6cefb7e5507`, with science denominator zero.
- The E0 CPU/fake harness reached and was pushed at
  `dee90d7f0931e5b0112054d35ecd8343eef665dc`.
- Before this documentation commit, local Geometry HEAD is
  `244f28a6c671c0d9fc134b350ecd7c6d7f7d2018`, clean and two commits ahead of
  `origin/Geometry-V1`.
- `244f28a` changed only the E0 operational runner and integration test.
  Historical receipts record 20/20 integration and 98/98 targeted CPU/fake
  tests.  Those are engineering evidence only.
- A2 requested changes because `MAX_PLAN_BYTES` is 98,304 while plan-digest
  serialization is 65,536, allowing a late wrong-stage failure.
- A3 requested changes because bounded dynamic production-contract coverage
  remains missing, as identified in session 11.
- A4 has not started.  No real paired E0, sync writer, canonical anchor,
  reliability fit, detector recheck, or scientific result exists.

## 7. Authorized progression nodes

0. Under separate exact-bound authorization, close the `244f28a` B1 blockers
   in the existing two runner/test files.
1. Obtain fresh A2 and A3 reviews of that final exact; start A4 only if both
   approve, and authorize a controlled push separately.
2. Prepare a thin B2 Colab handoff and run one real paired E0 only with
   separate model, GPU, and Drive authorization.
3. Interpret E0 without retry, layer switching, or fallback.
4. If eligible, perform a crop-oracle recoverability gate before investing in
   an active keyed anchor or end-to-end development.
5. Design an active keyed canonical-anchor writer, including no-sync and
   wrong-key controls and a separate geometry budget.
6. Build a true `S[canonical_anchor, observed_token]` correspondence and
   bounded transform raw metrics; reliability remains false.
7. Fit reliability on independent data and fail closed.
8. Inverse-rectify and recheck the unchanged detector, key, preprocessing, and
   `tau`.
9. Conduct fixed-denominator end-to-end evaluation and separately authorized
   cross-route integration.

## 8. Stop and drift rules

Geometry must not emit a positive decision.  Do not alter a threshold, `tau`,
or `tau_rescue` without explicit authorization.  Never delete failure units,
retry opportunistically, use a fallback, choose a route per sample, promote a
proxy, inherit old artifacts or thresholds, or hide detector inputs.

CPU/fake results are engineering evidence.  An operational real run retains a
zero science denominator.  Artifact integrity is not scientific evidence.
Every mutating or external step must bind the branch, exact, allowlist, action,
identities, stop rules, and evidence ceiling before execution.
