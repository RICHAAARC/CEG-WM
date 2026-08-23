# Content Runtime V3 scientific negative adjudication

## Immutable identity

- Method: Content V2 method + Runtime Asset Contract V3
- Formal exact revision: `4d8b0df5bf7840d242115669f1d3115cdf6810cc`
- Run: `content-adaptive-v2-e3fe3fd32ca2-805bc21e173a`
- Protocol ID: `cegwm-stage-a-content-adaptive-dual-branch-v2-semantic-gate-runtime-asset-contract-v3`
- Protocol digest: `e3fe3fd32ca2df7a1b1d2afe0318ff6c81cd67765b1f1be79a3ed89db7e87345`
- Public key digest: `805bc21e173a83898f3b7034d75e6ed02f65894a6885377d9659ee3091b4dd77`
- Roster SHA-256: `dd30c719ae5a48b2a9a652420a3237adb74ffd26af8bac90e25c1d03fe845b88`
- Artifact SHA-256: `d29873a61baebe47deb41d989e19ab5a78008e279807e78f0302479acea65b77`

## Fixed outcome

RC0 status: complete/evaluable; 8 units; 16 records; 0 failures.

| Preregistered gate | Outcome | Decision |
| --- | ---: | --- |
| LF A | 5/8 | FAIL |
| LF B | 8/8 | PASS |
| HF A | 8/8 | PASS |
| HF B | 8/8 | PASS |
| Joint A | 7/8 | PASS |
| Joint B | 8/8 | PASS |

Strict ties fail; no ties occurred. `all_predeclared_gates_pass=false` and `formal_fpr_claim=false`.

LF A failures: `0001`, `0003`, `0004`. Joint A failure: `0001`.

## Adjudication

`SCIENTIFIC_NEGATIVE` applies only to this exact clean-only simultaneous LF/HF candidate on this exact fixed 8-unit roster. The preregistered all-gates conjunction failed at LF Gate A because 5/8 is below 7/8.

This does not establish general LF invalidity, general HF invalidity, full Content invalidity, or CEG-WM invalidity. It makes no attack, complementarity, robustness, calibrated threshold, FPR, geometry, crop, Stage, main, paper, or publication claim.

There was no retry, resample, replacement, tuning, or change to any threshold, roster, prompt, seed, model, backend, dtype, budget, or Gate. Old RC2 provenance remains immutable, is excluded from this adjudication, and was not retried, resumed, or counted in this run.
