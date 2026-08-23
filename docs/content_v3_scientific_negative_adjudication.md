# Content V3 scientific negative adjudication

## Immutable identity

- Method: `content_v3_unweighted_lf_adaptive_hf_v1`
- Candidate: `content_v3_unweighted_lf_adaptive_hf_semantic_gate_v1`
- Formal exact revision: `aa2ff4476901033bff2564d93298889a9967303c`
- Run: `content-v3-6b812bbef380-805bc21e173a`
- Protocol ID: `cegwm-stage-a-content-v3-unweighted-lf-adaptive-hf-clean-v2`
- Protocol digest: `6b812bbef380085b67c33ea380444c379278faad1822762d4028465ecfd6058c`
- Public key digest: `805bc21e173a83898f3b7034d75e6ed02f65894a6885377d9659ee3091b4dd77`
- Roster SHA-256: `dd30c719ae5a48b2a9a652420a3237adb74ffd26af8bac90e25c1d03fe845b88`
- Artifact SHA-256: `6fe33b16fdefba29de14a9e3e03fa9c1772bfd12b25b5ceaaa99557eb1a63355`

## Fixed outcome

RC0 status: complete/evaluable; 8 units; 16 records; zero failures.

| Preregistered gate | Outcome | Decision |
| --- | ---: | --- |
| LF A | 5/8 | FAIL |
| LF B | 8/8 | PASS |
| HF A | 8/8 | PASS |
| HF B | 8/8 | PASS |
| Joint A | 7/8 | PASS |
| Joint B | 8/8 | PASS |

Strict ties fail; no ties occurred. `all_predeclared_gates_pass=false` and
`formal_fpr_claim=false`.

LF A failures: `0001`, `0003`, `0004`. Joint A failure: `0001`.

## Adjudication

`SCIENTIFIC_NEGATIVE` applies only to this exact clean-only candidate on this
exact fixed roster. The preregistered conjunction failed because LF Gate A was
5/8, below 7/8.

This does not establish general LF invalidity, general HF invalidity, general
Content invalidity, or CEG-WM invalidity. It makes no attack, complementarity,
FPR, robustness, geometry, Stage, main, paper, publication, or other promotion
claim.

There was no retry, tuning, or change to any threshold, roster, prompt, seed,
model, or denominator. Content V2 provenance and artifacts remain immutable,
are excluded from this adjudication, and are not Content V3 evidence.
