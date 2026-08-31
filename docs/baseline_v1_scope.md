# Baseline-V1: four-method generative-watermark main table

The only Baseline-V1 main-table methods are Tree-Ring, Gaussian Shading,
Shallow Diffuse, and T2SMark.  Tree-Ring, Gaussian Shading, and Shallow Diffuse
are reserved for method-faithful SD3.5 Medium adaptations.  T2SMark is reserved
for its official `run_sd35.py` native path.  This repository currently contains
neither external source code nor an executable adapter for any of them.

Each observation records the method/source/adapter identity, prompt, seed and
base-latent commitment, split/sample role/attack, continuous score and its
direction, that method's threshold provenance and decision, quality/runtime,
status/failure, and artifact digests.  A threshold is method-specific and may
not be borrowed across methods.  Failed units remain records; there is no retry,
replacement, or success-subset selection in this interface.

All four detector score directions are intentionally unresolved until a
method-faithful source/adapter audit is separately authorized.  Therefore the
current registry rejects `observed` records; failed and `not_available` records
cannot contain placeholder scores or decisions.  Future observed records must bind a
registered method direction, a `baseline_id:calibration:` provenance identity,
and SHA-256 artifact digests.

No record here supplies geometry (`H_hat` or corners), alters Geometry-V4, or
constitutes a model execution, threshold calibration, attack roster, denominator,
or scientific result.  Other baseline families are future extensions and are
outside this main table.
