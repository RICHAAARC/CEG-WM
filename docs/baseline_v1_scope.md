# Baseline-V1: four-method generative-watermark main table

The only Baseline-V1 main-table methods are Tree-Ring, Gaussian Shading,
Shallow Diffuse, and T2SMark.  Tree-Ring, Gaussian Shading, and Shallow Diffuse
are reserved for method-faithful SD3.5 Medium adaptations.  T2SMark is reserved
for its official `run_sd35.py` native path. Detached official sources are kept
under ignored `external_sources/` as source-audit input, not project method code
or executable adapters.

Each observation records method identity, prompt/seed/base latent commitment,
frozen attack family and condition, continuous score/direction, method-specific
threshold identity and decision, quality/runtime, and status/failure. Source,
adapter, exact, digest, license, and other provenance metadata are optional
context rather than score, adapter, or main-table hard gates. Calibration uses
2,000 independent clean unwatermarked negatives per method and contains a score
but no decision; each method then independently calibrates its own threshold for
the frozen target FPR of 0.1%.
Clean testing uses 3,000 independent clean unwatermarked negatives.  It reports
`FP/3000`, observed FPR, and an exact two-sided 95% Clopper-Pearson interval.
An exact one-sided 95% upper bound is optional diagnostic context only; it is
not an admission condition and cannot suppress a result package.
Evaluation uses 1,000 physical units over the six frozen conditions: clean,
JPEG Q50, 50% bicubic restore, 80% center-crop restore, Gaussian blur sigma 1,
and the frozen +10-degree rotation. The prompt roster, sampling-seed roster,
and dataset identity remain unresolved; this document does not invent them. A
threshold is never shared between methods.

The main-table builder admits only watermarked evaluation and unwatermarked
negative evaluation records from these four baselines. It counts TP/FN/FP/TN and
failures for one baseline/threshold/attack identity. Wrong-key is an optional
method-native diagnostic role: it is excluded from calibration, rejected by the
main-table builder, and never mixed into the unwatermarked FPR. Proposed CEG or
Geometry-V4 rows are not baseline IDs and are rejected.

The source audit registers method-native score directions. Shallow Diffuse uses
exactly `negative_mask_l1diff_mean`; its p-value alternative is outside this
contract. Tree-Ring, Gaussian
Shading, and Shallow Diffuse have an `implemented_unexecuted` SD3.5 adapter
status: the mechanism and CPU fixtures were migrated from the non-git SLM-WM
adapter archive, which supplies neither a source exact nor a real GPU result.
Their `result_status` remains `not_available` and `paper_claim_support` remains
false. `observed` and
`confirmation_observed` records require a real score/decision, that method's
direction, and a `baseline_id:calibration:` threshold provenance prefix; optional
source/adapter exacts or artifact digests never decide their admissibility.
Failed and `not_available` records still cannot carry placeholder scores or
decisions. The registry and adapter plan describe implementation work, not a
claim that an adapter, calibration, or scientific result is complete.

No record here supplies geometry (`H_hat` or corners), alters Geometry-V4, or
constitutes a model execution, threshold calibration, attack roster, denominator,
or scientific result.  Other baseline families are future extensions and are
outside this main table.
