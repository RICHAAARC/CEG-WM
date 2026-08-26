# Content V5 Evidence

## Method

Content V5 retains the Content V4 unweighted LF write, clean-null-whitened LF
scorer, adaptive HF path, shared actual-dtype budget, blind final-image
detection boundary, key/PRG identities, wrong keys, and primary null. Its sole
decision change is a branchwise logical OR applied after strict within-branch
comparisons. It never computes `max(raw LF score, raw HF score)`.

For each unit, Gate A passes when either the LF registered score strictly
exceeds all 16 LF wrong-key scores or the HF registered score strictly exceeds
all 16 HF wrong-key scores. Gate B applies the same logical OR to the LF and HF
registered-versus-same-unit-primary-null comparisons. Ties fail within each
branch, and each Gate requires at least 7 of 8 units.

## Frozen execution identity

- Source exact: `ed5ed5a303c4eb3a8e7cafb8410109b68f5e41fa`
- Run ID: `content-v5-c5a0c4bf7d6d-805bc21e173a`
- Protocol ID: `cegwm-stage-a-content-v5-whitened-lf-adaptive-hf-branchwise-or-clean-v1`
- Protocol digest: `c5a0c4bf7d6d3521ae233756ea07753dd002d842662b50f82a86de6a0f96c204`
- Public-key digest: `805bc21e173a83898f3b7034d75e6ed02f65894a6885377d9659ee3091b4dd77`
- Reference manifest SHA-256: `dd30c719ae5a48b2a9a652420a3237adb74ffd26af8bac90e25c1d03fe845b88`
- Primary manifest SHA-256: `5303a0284e36d2e6e159526c7ba61a7106fb3db72de35f0ada98fcfd5da2ec2c`
- Artifact ZIP SHA-256: `c58c904dd8a030abd4ad0c19faed7c952b79aafe55083d462d1c661175cb2c39`

The single umbrella invocation completed RC0. It unconditionally executed the
reference cohort followed by the primary cohort. Each cohort retained its own
fixed 8-unit/16-record denominator, aggregates, failures, and Gate decision.
The cohorts were not pooled and neither cohort controlled whether the other was
executed or reported.

## Result

### `control_1` reference cohort

- LF diagnostic Gate A/B: `4/8`, `8/8`
- HF diagnostic Gate A/B: `8/8`, `8/8`
- Branchwise-OR Gate A/B: `8/8`, `8/8`
- All mechanical requirements: pass on `8/8`
- No ties and no failed unit
- `all_predeclared_gates_pass=true`

### `primary_1` evaluation cohort

- LF diagnostic Gate A/B: `5/8`, `8/8`
- HF diagnostic Gate A/B: `8/8`, `8/8`
- Branchwise-OR Gate A/B: `8/8`, `8/8`
- All mechanical requirements: pass on `8/8`
- No ties and no failed unit
- `all_predeclared_gates_pass=true`

`formal_fpr_claim=false` for both cohorts.

## Personal adjudication

The user personally adjudicated this exact bound Content V5 result as
`CONTENT_CHAIN_MECHANISM_COMPLETE`. The adjudication applies only to the
branchwise logical-OR clean-only mechanism, source exact, run, protocol,
public-key identity, two independent fixed cohorts, and artifact listed above.

The immutable artifact continues to record each cohort's original
`scientific_status=not_adjudicated`; it is not rewritten. The later personal
adjudication is recorded only on this Evidence branch.

This result does not establish independent LF completion: LF Gate A was `4/8`
on the reference cohort and `5/8` on the primary cohort, while HF Gate A/B was
`8/8` on both. It therefore does not by itself establish LF/HF
complementarity. It also establishes no calibrated threshold, fixed FPR,
attack performance, robustness, geometry, generalization, Stage/main/paper
promotion, or permission for retry, tuning, replacement, or new execution.

## Branch role

`Content-V5-Evidence` is a complete executable evidence snapshot under
`Content-V5`. It retains the canonical execution tree together with the final
formal Notebook and portable scalar result, and the Notebook clones this
Evidence branch directly. It is not a method-development base and must not be
merged into `main`.

## Portable scalar evidence

The exact terminal artifact members are committed at
`evidence/content-v5/content-v5-c5a0c4bf7d6d-805bc21e173a/`.

- Exact `receipt.json` SHA-256:
  `e4b118eb696e36bf982264aa1dea2b3ecdd02ff45d70b63cf4b487866d63f8c4`
- Exact `result.json` SHA-256:
  `f043fcca0a5a483d2419751730155d221e3e69a1b8ac735224b8e7c05cc42c68`

The package preserves every scalar score and aggregate required for read-only
recomputation. The terminal `audit-state.json` is intentionally omitted because
it duplicates committed records and is unnecessary for comparison or
recalculation; its source-archive hash remains recorded in the manifest. The
package contains no image, raw prompt text, raw key, token, latent, delta,
route, mask, tensor, or private embed state. Any future statistic recomputed
from these records is a new analysis and cannot retroactively change this
frozen adjudication.
