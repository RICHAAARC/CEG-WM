---
name: np-threshold-calibrator
description: Calibrate and audit Neyman-Pearson thresholds under explicit FPR constraints with reproducible reports. Use when users ask for threshold calibration, FPR control, FNR tradeoff review, calibration-evaluation boundary checks, or NP policy compliance.
---
# np-threshold-calibrator

Calibrate thresholds with strict FPR-first semantics and auditability.

## Execute

1. Identify threshold sources in config, calibration CLI, and evaluation CLI.
2. Verify threshold values are produced only by calibration outputs, then consumed read-only by evaluation.
3. Verify reports log target FPR, achieved FPR, corresponding FNR, and threshold provenance.
4. Run targeted tests to confirm no hidden threshold recomputation.
5. Report violations with file-level evidence and repair actions.

## Repository Anchors

- `main/watermarking/fusion/neyman_pearson.py`
- `main/cli/run_calibrate.py`
- `main/cli/run_evaluate.py`
- `tests/test_thresholds_readonly_guard.py`
- `tests/test_no_threshold_recompute_in_aggregate.py`
- `tests/test_no_threshold_recompute_under_attack.py`

## Output Contract

For each finding include:
- `issue`
- `risk_level`
- `evidence`
- `decision_basis`
- `recommended_fix`
