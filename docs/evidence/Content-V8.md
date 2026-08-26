# Content V8 evidence

## Method identity

Content V8 preserves the Content V2 spatially weighted LF write
`normalize(lf_tile_weights * carrier)` and ordinary blind LF scorer while
adding the V6 two-pass detector-domain ISS control flow. The callback-free
first pass is the sole primary null; the independently reset same-seed second
pass is the sole candidate. The fitted multiplier applies to the complete LF
preprojection delta without changing its spatial direction. HF preprojection,
the common actual-dtype budget at most `0.012`, wrong keys, blind boundary, and
within-cohort `joint=min(LF,HF)` remain frozen. No whitening operator or V6/V7
gain-target payload enters the formal Content V8 path.

## Frozen execution identity

- Source exact: `bd6269861412b8628009238ace84d3278ff1e17a`
- Run ID: `content-v8-bd6269861412-0ba01f405106`
- Protocol ID: `cegwm-stage-a-content-v8-v2-spatial-lf-detector-domain-iss-formal-initial-v1`
- Protocol digest: `7670d54434906ae246ef76c097bf54f997c3a0ca6b036c0f55e0fe5b31489a1c`
- Public-key digest: `805bc21e173a83898f3b7034d75e6ed02f65894a6885377d9659ee3091b4dd77`
- Old-roster SHA-256: `dd30c719ae5a48b2a9a652420a3237adb74ffd26af8bac90e25c1d03fe845b88`
- Current-V6-roster SHA-256: `20058788bfe7d75878e7263efda2b8de94c6fdcd3a963f64368f2ba4d594868f`
- Runtime asset SHA-256: `a122a8dea1bc2615cc80e17ed1fb55235ede23b8a7498c1618ab47ecf151a438`
- Formal archive SHA-256: `c0626ee67fb7c96506fb6cc6a45d7971c0ff911eb8c12c5cfced6108928e86ca`

The RC0 invocation completed the runtime fit and both evaluations. Each cohort
retained its own 8-unit/16-record denominator and neither pooling nor
cross-roster outcome control was applied.

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
Content V8 result is recorded as `SCIENTIFIC_NEGATIVE` for both independent
cohorts. LF Gate A reached only `5/8` and `6/8`, respectively, below the
required `7/8`; passing joint and HF Gates do not replace that frozen member of
the conjunction.

The immutable artifact retains `scientific_status=not_adjudicated`; the later
adjudication is recorded only on this Evidence branch. The result establishes
no general Content V2 or ISS invalidity and no calibrated threshold/FPR,
attack, robustness, geometry, Stage/main, paper, retry, tuning, replacement,
or promotion claim.

## Branch role

`Content-V8` is the canonical method branch. `Content-V8-Evidence` is its
complete executable evidence snapshot, not a development base and not
mergeable to `main`. Its current tree retains the canonical execution path,
one formal Notebook, the exact public runtime asset, and the portable scalar
result. The Notebook clones this Evidence branch directly.

## Drive-first Notebook rerun

The formal runner completed RC0 at exact
`c9be80952ef6b23627f5ff45411addc955316950` on `20260826T040739Z`, with
independent `[8V1]` and `[8V3]` evaluations following `[32V1]` development.
Its ZIP SHA-256 is
`1f73303a12a8e15e1160b796b284f4462b53ad212ec7cc5438ec4bf1432caf7c`.

The append-only record is preserved under
`evidence/content-v8/content-v8-c9be80952ef6-94673c645cf3/reruns/20260826T040739Z-c9be80952ef6/`.
The Notebook's final `artifact_pair_validation` output was an operational false
negative caused by a stale fixed run ID: the dynamic runner directory already
contained the valid terminal ZIP/SHA pair. The lookup was repaired later at
Evidence exact `062c6b0687d7d81c5ec1b39d2be223384959de40`; the original handoff
output is not rewritten.

After excluding execution provenance, both scalar results reproduce the prior
evidence exactly, and the public asset differs only in `producer_exact`. The
original `scientific_status`, Gate outcomes, and narrow personal adjudication
remain unchanged.
