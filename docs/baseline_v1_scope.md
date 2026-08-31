# Baseline-V1: four-method generative-watermark main table

The only Baseline-V1 main-table methods are Tree-Ring, Gaussian Shading,
Shallow Diffuse, and T2SMark.  Tree-Ring, Gaussian Shading, and Shallow Diffuse
are reserved for method-faithful SD3.5 Medium adaptations.  T2SMark is reserved
for its official `run_sd35.py` native path. Detached official sources are kept
under ignored `external_sources/` as source-audit input, not project method code
or executable adapters.

Each observation records the method/source/adapter identity, prompt/seed/base
latent commitment, attack family and unresolved-or-frozen condition, continuous
score/direction, method-specific threshold identity and decision, quality/runtime,
status/failure, and artifact digests. Calibration observations use only
unwatermarked negatives and contain a score but no decision; the threshold is
then independently method-calibrated at a common target FPR. The target FPR,
attack conditions, seed, sample count, and denominator are all
`pending_user_freeze` in this engineering contract. A threshold is never shared
between methods.

The main-table builder admits only watermarked evaluation and unwatermarked
negative evaluation records from these four baselines. It counts TP/FN/FP/TN and
failures for one baseline/threshold/attack identity. Wrong-key is an optional
method-native diagnostic role: it is excluded from calibration, rejected by the
main-table builder, and never mixed into the unwatermarked FPR. Proposed CEG or
Geometry-V4 rows are not baseline IDs and are rejected.

The source audit registers method-native score directions, but adapter exacts
and threshold provenance remain unresolved. Therefore the current registry
rejects `observed` records; failed and `not_available` records
cannot contain placeholder scores or decisions.  Future observed records must bind a
registered method direction, a `baseline_id:calibration:` provenance identity,
a validated source/adapter registry state, lowercase 40-character source and
adapter Git exacts, and named SHA-256 source/adapter/threshold artifact digests.
Those six identities are frozen in the method registry and must match each
observed record exactly; current adapter and calibration identities are unresolved.

No record here supplies geometry (`H_hat` or corners), alters Geometry-V4, or
constitutes a model execution, threshold calibration, attack roster, denominator,
or scientific result.  Other baseline families are future extensions and are
outside this main table.
