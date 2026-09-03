# Paper FPR Policy V1

## Frozen status and scope

This policy is frozen on branch `PaperFPR-V1` from
`main@e12c7eae91cc36edc5d1a1d96249780a3925eccb`.

The complete execution semantics are frozen in
`docs/formal_experiment_contract_v1.md`. If this policy's earlier no-retry
wording is read more narrowly, the complete contract controls: no replacement
is allowed, while one predeclared retry of the identical unit is allowed only
for typed operational failures and retains every attempt.

It freezes only threshold calibration and FPR reporting semantics. It does not
freeze calibration/test denominators, prompt or image rosters, attack-table
membership, generative-attack models, baseline readiness, or execution
authorization. The existing BlindDetection-V1 `N_dev=256` threshold remains an
engineering asset and is not the paper threshold.

## Score orientation and target

Every method must expose one continuous detection score whose normalized
orientation is "higher means more watermark evidence". A method whose native
score has the opposite direction must declare and apply one fixed monotone
orientation conversion before calibration and evaluation.

The paper calibration target is

`alpha = 0.001` (target FPR `0.1%`).

This is a calibration target, not a requirement that the observed test FPR
equal `0.1%`, and not an admission gate for producing the result package.

## Per-method threshold calibration

Each method receives its own threshold from its own independent calibration
scores. Thresholds are never shared across methods.

Calibration uses clean, unwatermarked negatives only. Attacked negatives do
not tune the threshold. They are evaluated later with the same frozen
threshold and are reported separately by attack condition.

For `N_cal` complete normalized calibration scores sorted in nondecreasing
order as `c_(1), ..., c_(N_cal)`, define

`k = ceil((1 - alpha) * N_cal)`

and

`tau = c_(k)`.

This is the nearest-rank empirical `99.9%` quantile. The decision is strictly

`positive iff score > tau`.

Equality is negative. This strict rule also resolves ties at the threshold.
The calibration denominator and roster will be frozen separately before any
formal run.

Calibration data must be disjoint from confirmation and formal test data.
Formal test scores, labels, attacks, or aggregate outcomes may not be used to
select or revise `tau`.

If the fixed calibration roster contains a terminal failure or missing score,
every row and attempt remains in the fixed denominator. No threshold may be
fabricated from a filtered subset. The result package is still produced with
an explicit incomplete threshold status; there is no replacement or post-hoc
roster change. The complete contract permits at most one predeclared retry of
the identical unit for typed operational failures only.

## Formal evaluation

The same frozen per-method threshold is used for clean positives, attacked
positives, clean unwatermarked negatives, and every attacked-negative
condition. There is no attack-specific retuning.

Formal evaluation always retains and reports:

- raw normalized detection scores;
- threshold identity and strict decision;
- TP, FN, FP, and TN counts;
- planned, observed, failed, and missing denominators;
- TPR and observed FPR when the required rows are complete;
- exact two-sided 95% binomial confidence intervals;
- per-condition attacked-negative FPR;
- every operational failure and method-complete fail-closed row.

An optional one-sided 95% FPR upper confidence bound may be reported as a
diagnostic. Neither that upper bound nor `observed FPR > alpha` blocks result
package production. Instead, the package records a report-only
`operating_point_deviation` field. It never retunes the threshold or suppresses
the result.

The accurate primary metric name is:

`TPR at a threshold calibrated for target FPR = 0.1%`.

It may be abbreviated as `TPR@0.1%-target-calibrated-threshold`. The paper must
not label it `TPR@FPR=0.1%` unless the accompanying observed test FPR supports
that statement.

## Supplementary threshold-free reporting

Formal artifacts retain enough raw positive and negative scores to compute a
ROC curve, ROC AUC, and low-FPR partial AUC without rerunning the detector.
These supplement the frozen-threshold result and do not authorize choosing a
new production threshold from the test set.

## Claim ceiling

Freezing this policy establishes an evaluation rule only. It does not establish
a paper threshold, a fixed denominator, an observed FPR, attack robustness,
baseline completion, or paper readiness. Results remain publishable as
complete measurements even when the observed operating point misses the
target; only the strength of the conclusion changes.
