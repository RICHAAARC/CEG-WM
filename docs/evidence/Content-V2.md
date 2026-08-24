# Content V2 Evidence

## Method

Content V2 is the simultaneous content-adaptive LF/HF clean candidate implemented by the `Content-V2` branch. It uses real DINO semantic analysis, RGB texture, 64 independent baseline-differenced probes, adaptive LF/HF allocation, the balanced-block LF carrier, the HF tail carrier, one shared actual-dtype combined relative-L2 budget not exceeding `0.012`, ordinary final RGB, blind LF/HF scoring, and `joint=min(LF,HF)`.

Runtime Asset Contract V3 is the DINO public-size asset contract used by this method run; it is not a Content method version.

## Frozen execution identity

- Source exact: `4d8b0df5bf7840d242115669f1d3115cdf6810cc`
- Run ID: `content-adaptive-v2-e3fe3fd32ca2-805bc21e173a`
- Protocol ID: `cegwm-stage-a-content-adaptive-dual-branch-v2-semantic-gate-runtime-asset-contract-v3`
- Protocol digest: `e3fe3fd32ca2df7a1b1d2afe0318ff6c81cd67765b1f1be79a3ed89db7e87345`
- Public-key digest: `805bc21e173a83898f3b7034d75e6ed02f65894a6885377d9659ee3091b4dd77`
- Roster SHA-256: `dd30c719ae5a48b2a9a652420a3237adb74ffd26af8bac90e25c1d03fe845b88`
- Artifact ZIP SHA-256: `d29873a61baebe47deb41d989e19ab5a78008e279807e78f0302479acea65b77`

The run completed RC0 with all 8 fixed units and 16 ordered records evaluable and no retained operational failure.

## Result

- LF Gate A: `5/8`, fail against required `7/8`
- LF Gate B: `8/8`, pass
- HF Gate A: `8/8`, pass
- HF Gate B: `8/8`, pass
- Joint Gate A: `7/8`, pass
- Joint Gate B: `8/8`, pass
- All mechanical budget, nonzero-branch, counterfactual, probe-count, share, and PSNR requirements: pass on `8/8`
- `all_predeclared_gates_pass=false`
- `formal_fpr_claim=false`

LF Gate-A failures were units `0001`, `0003`, and `0004`. The preregistered conjunction therefore failed.

## Adjudication boundary

The preserved narrow adjudication is `SCIENTIFIC_NEGATIVE` only for the exact method, run, roster, key identity, and artifact listed above. It does not establish general LF/HF invalidity, attacks, complementarity, fixed-FPR detection, robustness, geometry, Stage completion, main promotion, or paper evidence. It authorizes no retry, resampling, tuning, threshold change, or roster replacement.

The original execution and handoff revisions are merge parents of this evidence branch. The canonical `Content-V2` branch is a clean method-first reconstruction; the historical artifact remains bound only to its original source exact.

## Portable scalar evidence

The exact artifact members are committed at
`evidence/content-v2/content-adaptive-v2-e3fe3fd32ca2-805bc21e173a/`.
The package preserves every scalar score and aggregate needed for later
read-only recomputation without retaining images or private runtime state.

- Exact `receipt.json` SHA-256:
  `1ed043d527bad4e92b33ceeea4409c140a384f83538b351426c90e480f4ef011`
- Exact `result.json` SHA-256:
  `2556f71a39a9b6fce13d943f2314eb6bf16b113679a46da5a73008b348eeb471`

The package does not record the later 50/50 posthoc analysis. Any future
statistic recomputed from these records remains a new analysis and cannot
retroactively change the frozen Content V2 adjudication.
