# Content V4 evidence

## Method identity

Content V4 retains the Content V3 unweighted LF write and content-adaptive
scalar allocation, while replacing the formal LF detector with the audited
clean-null-whitened matched score. Its blind LF path is ordinary final RGB,
the frozen public image processor and SD3.5 VAE posterior-mode observation,
per-channel affine-plane detrending, orthonormal DCT-II, the frozen public
whitening operator, and keyed matched cosine. The HF path is unchanged and the
joint statistic remains `min(LF, HF)`.

The public whitening asset is bound to producer exact
`79f67646595bd99cc8b066cad0e4b12e96a22cbb`, asset SHA-256
`a7021dd8b98bc4282b98ed5d1fe276236d99a3c9e80b9bdce015d28cf715633f`,
and sidecar-file SHA-256
`c900cce0980348eeadcf07d782b6169c4d46ac55d7154db0fc0a0a878cce0ced`.
The 32-sample fit is construction evidence with scientific denominator zero;
it is not reused as formal Content V4 evidence.

## Formal result identity

- Source exact: `0387eefc1e2b943cfe2a7f16ba9cc2073693dd9a`
- Run ID: `content-v4-a9fdf3e5d384-805bc21e173a`
- Protocol digest: `a9fdf3e5d384976c11bbd542c3248483806473ed1dca91dc5d753ab10ec5beb0`
- Public-key digest: `805bc21e173a83898f3b7034d75e6ed02f65894a6885377d9659ee3091b4dd77`
- Roster SHA-256: `dd30c719ae5a48b2a9a652420a3237adb74ffd26af8bac90e25c1d03fe845b88`
- Formal archive SHA-256: `27f4ff5a148174bf81e74106c8e03a9f1ebac39b68b158baadb0b42f6094282f`

The authenticated RC0 artifact contains all eight ordered units and sixteen
ordered records with no missing denominator entries. All mechanical
requirements passed 8/8. The independently recomputed preregistered gates were:

- LF Gate A: 4/8, fail
- LF Gate B: 8/8, pass
- HF Gate A: 8/8, pass
- HF Gate B: 8/8, pass
- Joint Gate A: 5/8, fail
- Joint Gate B: 8/8, pass

Therefore `all_predeclared_gates_pass=false`; `formal_fpr_claim=false`.

## Adjudication boundary

The user personally adjudicated this exact bound result as a narrow
`SCIENTIFIC_NEGATIVE`. The adjudication applies only to the source exact, run,
protocol, roster, public-key identity, formal archive, and preregistered
conjunction listed above: LF Gate A passed only `4/8` and Joint Gate A only
`5/8`, both below the required `7/8`, so the all-gates conjunction failed.

The immutable artifact itself continues to record its original
`scientific_status=not_adjudicated`; it is not rewritten or reinterpreted.
This evidence-branch adjudication establishes no general LF, HF, Content, or
CEG-WM invalidity; authorizes no retry, resume, tuning, replacement, or new
formal execution; and supports no attack, complementarity, fixed-FPR,
robustness, geometry, Stage, main, or publication claim.

## Branch role

`Content-V4-Evidence` is an evidence leaf under `Content-V4`. It is not a
method-development base and must not be merged into `main`. The corresponding
W-fit, non-roster canary, and formal-run notebooks are retained here so that
the execution provenance stays attached to the method version that produced
it.
