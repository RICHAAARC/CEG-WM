# Content V3 Evidence

## Method

Content V3 keeps the Content V2 real DINO, RGB texture, 64-probe content analysis, scalar LF/HF allocation, adaptive HF spatial transform, simultaneous embedding, shared actual-dtype budget, blind final-RGB detector boundary, and `joint=min(LF,HF)`. Its defining change is an unweighted standard LF write direction:

`delta_LF = A_LF(content) * normalize(c_LF)`

The LF production delta does not consume LF tile weights. Content statistics still determine the scalar LF/HF shares and amplitudes; HF continues to consume its adaptive spatial weights.

## Frozen execution identity

- Source exact: `aa2ff4476901033bff2564d93298889a9967303c`
- Run ID: `content-v3-6b812bbef380-805bc21e173a`
- Protocol ID: `cegwm-stage-a-content-v3-unweighted-lf-adaptive-hf-clean-v2`
- Protocol digest: `6b812bbef380085b67c33ea380444c379278faad1822762d4028465ecfd6058c`
- Public-key digest: `805bc21e173a83898f3b7034d75e6ed02f65894a6885377d9659ee3091b4dd77`
- Roster SHA-256: `dd30c719ae5a48b2a9a652420a3237adb74ffd26af8bac90e25c1d03fe845b88`
- Artifact ZIP SHA-256: `6fe33b16fdefba29de14a9e3e03fa9c1772bfd12b25b5ceaaa99557eb1a63355`

The run completed RC0 with all 8 fixed units, 16 ordered records, and no failed unit.

## Result

- LF Gate A: `5/8`, fail against required `7/8`
- LF Gate B: `8/8`, pass
- HF Gate A: `8/8`, pass
- HF Gate B: `8/8`, pass
- Joint Gate A: `7/8`, pass
- Joint Gate B: `8/8`, pass
- Mechanical budget, nonzero-branch, six-counterfactual, probe-count, share, and PSNR requirements: pass on `8/8`
- `all_predeclared_gates_pass=false`
- `formal_fpr_claim=false`

LF Gate-A failures were units `0001`, `0003`, and `0004`. The exact clean-only candidate therefore failed its preregistered all-gates conjunction.

## Adjudication boundary

The user personally adjudicated this exact bound result as a narrow
`SCIENTIFIC_NEGATIVE`. The adjudication applies only to the source exact, run,
protocol, roster, public-key identity, artifact, and preregistered conjunction
listed above: LF Gate A passed only `5/8`, below the required `7/8`, so the
all-gates conjunction failed.

The immutable artifact itself continues to record its original
`scientific_status=not_adjudicated`; it is not rewritten or reinterpreted.
This evidence-branch adjudication establishes no general LF/HF invalidity and
supports no retry, resampling, tuning, threshold change, attack,
complementarity, fixed-FPR, robustness, geometry, Stage, main, or paper claim.

The original canary/formal revisions are merge parents of this branch, and their runners and notebooks are retained here. The canonical `Content-V3` branch is a clean method-first reconstruction; the historical artifact remains bound only to its original source exact.

## Portable scalar evidence

The exact artifact members are committed at
`evidence/content-v3/content-v3-6b812bbef380-805bc21e173a/`. The package
preserves every scalar score and aggregate needed for later read-only
recomputation without retaining images or private runtime state.

- Exact `receipt.json` SHA-256:
  `f2b370959c67bc34bc896aea6d548de92384c6bd0500c5baeabb5b61a0c03f9c`
- Exact `result.json` SHA-256:
  `b62e81f0530fe51bbc79a106390892bd1c0db1b413902dccf97029ed54cdbf13`

The package does not record the later 50/50 posthoc analysis. Any future
statistic recomputed from these records remains a new analysis and cannot
retroactively change the frozen Content V3 adjudication.
