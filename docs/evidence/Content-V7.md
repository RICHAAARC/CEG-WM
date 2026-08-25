# Content V7 evidence

## Method identity

Content V7 is the V6 two-pass detector-domain ISS structure applied to the
Content V3 unweighted LF write and ordinary blind LF scorer. It does not load
or consume the whitening operator or the V6 whitened-domain gain-target asset.
The callback-free first pass is the sole primary null, the independently reset
same-seed second pass is the sole embedded candidate, ISS multiplies only the
LF preprojection delta, HF preprojection is unchanged, the common actual-dtype
budget remains at most `0.012`, and within each cohort joint remains
`min(LF, HF)`.

The runtime-fitted ordinary-domain asset is preserved with SHA-256
`34a5dda304ae431309dd0e54d0bf2eebc1be85d0fa678709f998f16b4ae124a1`.

## Frozen execution identity

- Source exact: `884a4f6524440cf111def89e46d54258ee7c9e1a`
- Formal protocol ID: `cegwm-stage-a-content-v7-ordinary-iss-formal-initial-v1`
- Formal protocol digest: `2c2ce18c3d0fd3b1573ad335d0c9fe62b9583a7162f97d75194c8322efa0db0e`
- Public-key digest: `805bc21e173a83898f3b7034d75e6ed02f65894a6885377d9659ee3091b4dd77`
- Old-roster SHA-256: `dd30c719ae5a48b2a9a652420a3237adb74ffd26af8bac90e25c1d03fe845b88`
- Current-V6-roster SHA-256: `20058788bfe7d75878e7263efda2b8de94c6fdcd3a963f64368f2ba4d594868f`
- Formal archive SHA-256: `eb588864d9d1af3182b96748adfcd8de1b10aaaa7e6dcfcec9654337b77848db`

The single RC0 invocation completed the fit and both evaluations. Each cohort
retained an independent 8-unit/16-record denominator; no pooled or combined
result was produced, and neither cohort controlled the other.

### Old-roster evaluation

- LF Gate A/B: `5/8`, `8/8`
- HF Gate A/B: `8/8`, `8/8`
- Joint Gate A/B: `7/8`, `8/8`
- Mechanical requirements: `8/8`
- Failed units and ties: none
- `all_predeclared_gates_pass=false`

### Current-V6-roster evaluation

- LF Gate A/B: `6/8`, `8/8`
- HF Gate A/B: `8/8`, `8/8`
- Joint Gate A/B: `8/8`, `8/8`
- Mechanical requirements: `8/8`
- Failed units and ties: none
- `all_predeclared_gates_pass=false`

## Personal adjudication

Under the user's explicit V6/V7/V8 adjudication authorization, this exact
Content V7 result is recorded as `SCIENTIFIC_NEGATIVE` for both independently
fixed cohorts. The preregistered conjunction failed because LF Gate A reached
only `5/8` and `6/8`, respectively, below `7/8`. Passing joint and HF Gates do
not replace the frozen LF requirement.

The immutable evaluation results retain `scientific_status=not_adjudicated`;
the later adjudication exists only on this Evidence branch. This is not a
general invalidity claim and does not establish calibrated threshold/FPR,
attack, robustness, geometry, Stage/main, paper, retry, tuning, replacement,
or promotion authority.

## Branch role

`Content-V7` is the canonical method branch. `Content-V7-Evidence` is a
results-only evidence leaf, not a development base and not mergeable to
`main`. Its current tree retains the final formal Notebook, the exact public
runtime asset, and the portable scalar evidence.
