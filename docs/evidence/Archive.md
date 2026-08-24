# CEG-WM Archive negative evidence

## Scope and authority

This branch preserves reviewed negative evidence from the repository now kept
at `CEG-WM-Archive`. It is an evidence leaf, not a current method branch. It
must not be merged into `main`, and Archive code, assets, thresholds, results,
or readiness do not become current CEG-WM authority through this branch.

The branch has provenance-only merge parents for Archive `main` exact
`11eb3eb81215defb99f63ae8d9f464c3473a086f` and the otherwise unmerged
diagnostic delivery exact
`6b5e302f4cefb051b34f0da3dc71b29b191b6ba2`. The current tree deliberately
does not copy the Archive implementation. The original implementation can be
inspected at the producer exacts and paths listed below with `git show`.

The Archive registry distinguishes `authenticated_development_negative` from
formal scientific adjudication. The three records below retain their original
evidence levels; they are not relabeled as the formal, user-adjudicated
`SCIENTIFIC_NEGATIVE` results used by current Content V2-V4.

## 1. Semantic-texture soft-route candidate selection

### Evidence identity

- Status: `AUTHENTICATED_DEVELOPMENT_NEGATIVE`
- Producer revision: `c46ef185c04ea76e48b8630cf5c0a70011cb9df5`
- Run: `semantic-texture-soft-route-candidate-selection-ab0dbf93033e4eada8b4a47920913f78`
- Detector asset bundle identity:
  `f9dd6df410cb4f7895376c65c5f6d3e764f6cfddabd0d64d525fdaaefd93de3d`
- Fixed denominator: `160/160` generations and `384/384` detector records

### Implemented method

The candidate family combined these exact identities:

- `routing_semantic_texture_soft`
- `content_embedding_semantic_texture_soft_lf_hf`
- `lf_semantic_texture_soft_whitened_matched_score`
- `hf_semantic_texture_soft_direct_score`
- `content_combination_semantic_texture_max_standardized`

The semantic map `M` was obtained from the InSPyReNet finest raw `d0`, one
sigmoid, and bilinear resizing to `64 x 64`. The texture map `T` was derived
from the same public RGB8 image using grayscale Sobel magnitude, area
downsampling, and a strictly positive exact-nearest-rank per-image P95 map.
The positive sum-one soft routes were:

```text
m_hf = (1 + M*T) / (2 + M)
m_lf = (1 + M*(1-T)) / (2 + M)
```

The embed direction was:

```text
normalize(normalize(m_hf * T_hf) + normalize(m_lf * T_lf))
```

under the shared binary32 content relative-L2 budget `3/250`. There was no
scalar LF/HF allocation, hard mask, erosion, connected-component rule, or
attack-conditioned switch. A route-disabled control used
`m_hf=m_lf=0.5`. LF used its separately fitted clean-null whitening operator
and matched score; HF used its direct score. Each branch was standardized from
its primary-null distribution, and the diagnostic combination was fixed as
`max(z_hf_soft, z_lf_soft)`.

Inspect the original implementation at:

- `c46ef185...:main/content_chain/routing.py`
- `c46ef185...:main/content_chain/embedder.py`
- `c46ef185...:main/content_chain/lf_detector.py`
- `c46ef185...:main/content_chain/detector.py`
- `c46ef185...:experiments/protocol/semantic_texture_soft_route_mechanism_validation.py`
- `c46ef185...:experiments/runners/semantic_texture_soft_route_mechanism_validation.py`

### Failure result and closed route

The first failed frozen gate was identity wrong-key attribution: soft-routed
wrong-key maximum positives were `5`, above the allowed maximum `3`. Multiple
later raw-crop gates also failed. Therefore
`candidate_selection_passed=false`; untouched confirmation was not authorized,
and no formal threshold, formal FPR, promotion, or paper-effect conclusion was
created.

This result closes only the exact five-candidate formula, assets, W/CDF,
provisional threshold, selection roster, split, and max-statistic identity. It
must not be revived by relaxing the wrong-key or crop gates, tuning on the
failed selection, reusing its W/CDF/threshold, or claiming that a future
geometry chain would excuse the raw identity/crop failure. It does not reject
content-adaptive LF/HF generally or a new carrier, detector, or constrained
allocation identity.

## 2. `routing_stqr` fixed-half directional diagnosis

### Evidence identity

- Status: `AUTHENTICATED_DIRECTIONAL_DIAGNOSTIC_NEGATIVE`
- Producer revision: `925c2cbc727e3b18e91c0b3981eeed1b470a955a`
- Run: `ceg_wm_content_routing_positive_reference_support_correction_diagnosis`
- Fixed denominator: `42/42` terminal

### Implemented method

This development diagnosis compared `routing_stqr` with the
`routing_uniform_control` at a fixed LF/HF mixing coefficient `0.50` and total
content relative-L2 budget `3/250`. The routed maps used four public
observations: semantic `S`, texture `T`, latent response `R`, and local
sensitivity `Q_sens`:

```text
A       = ((1-S) * (1-R) * (1-Q_sens)) ** (1/3)
mask_lf = A * (1-T)
mask_hf = A * T
```

The protocol contained two operational preflights, 32 cross-fit reference
clusters, and eight paired routed-versus-uniform directional probes, with four
wrong keys per arm. The public score remained HF-only; LF detector use was
explicitly prohibited in this diagnosis. A probe's incremental indicator
tested whether routed attribution improved over its paired uniform control.
Passing required the mean of eight indicators to be strictly greater than
`0.5`, positive route coverage, complete execution, and no identity,
integrity, nonfinite, or `3/250` budget violation.

Inspect the original implementation at:

- `925c2cbc...:main/content_chain/routing.py`
- `925c2cbc...:experiments/protocol/content_routing_directional_diagnosis.py`
- `925c2cbc...:experiments/metrics/content_routing_directional_diagnosis.py`
- `925c2cbc...:experiments/runners/content_routing_directional_diagnosis.py`

### Failure result and closed route

The eight ordered incremental indicators were `1,1,1,0,0,0,0,0`, so the mean
was `3/8 = 0.375`, below the strict `>0.5` requirement. Clusters `1`, `5`, and
`6` also exceeded the RGB relative-L2 budget and could not count as successful
clusters.

This diagnosis must not be reused to choose S/T/R/Q references, masks,
thresholds, coverage, or mixture parameters. It must not be converted into a
winner by deleting clusters, adding attempts or samples, relaxing `3/250`, or
selecting only the margin-positive subset. It does not establish that all
routing mechanisms fail.

## 3. `content_uniform_combination` directional diagnosis

### Evidence identity

- Status: `AUTHENTICATED_DIRECTIONAL_DIAGNOSTIC_NEGATIVE`
- Producer revision: `7c0d86d6eac5ffcfc4a30f2f5fb22884aaa848da`
- Delivery revision: `6b5e302f4cefb051b34f0da3dc71b29b191b6ba2`
- Run: `ceg_wm_content_uniform_combination_budget_observation_correction_diagnosis`
- Artifact ZIP SHA-256:
  `698dd70f0d6798a86310987f21d54056dd2cc27d4e67743bb1671a9927d31435`
- Fixed denominator: one operational unit, 32 reference clusters, and eight
  probes, all 41 attempt-zero units committed without retry or duplicate

### Implemented method

This disabled-routing diagnosis used `routing_uniform_control` and evaluated
fixed embed coefficients `0.25`, `0.50`, and `0.75` under the binary32 content
relative-L2 limit `3/250`. LF used `lf_null_whitened_matched_score`. Thirty-two
clean reference clusters produced four-fold cross-fit branch-null
standardization for eight six-image probes with four wrong keys.

The finite candidate menu was:

```text
hf_only_standardized_score
weighted_hf_lf_standardized_score, weight in {0.25, 0.50, 0.75}
maximum_hf_lf_standardized_score
```

Each candidate was checked against its registered-primary-null margin,
registered-maximum-wrong-key margin, improvement over the HF-only baseline,
and maximum allowed identity loss. Passing also required all eight probes and
zero implementation, resource, identity, integrity, nonfinite, or budget
violations.

Inspect the original implementation at:

- `7c0d86d6...:experiments/protocol/content_uniform_combination_directional_diagnosis.py`
- `7c0d86d6...:experiments/metrics/content_uniform_combination_directional_diagnosis.py`
- `7c0d86d6...:experiments/runners/content_uniform_combination_directional_diagnosis.py`

### Failure result and closed route

Clusters `1` and `6` violated
`clean_to_watermarked_rgb_relative_l2`. The aggregate was
`mechanism_signal_not_observed`, the recommendation was
`candidate_not_recommended_for_selection`, and
`allow_request_for_content_combination_candidate_selection=false`.

This closes the old uniform-combination identity and its fixed `a/w/function`
menu. It must not be revived by selecting `0.70/0.30`, `0.50/0.50`, or another
old fixed combination after observing these results; relaxing the quality or
budget conjunction; deleting failed clusters; adding samples or retries; or
letting a margin-only subset override the actual budget violations. It does
not reject independent HF or LF carriers or every possible constrained joint
allocation.

## Records deliberately excluded from the negative list

The following remain visible in Archive history but are not authenticated
method negatives:

- the hard salient-object local-LF family was
  `superseded_without_scientific_adjudication`;
- Q/K geometry and the complete joint detector were `not_yet_tested` or
  `implemented_not_scientifically_validated`;
- InSPyReNet `EXDEV` and `No space left on device` events were operational or
  resource failures;
- the historical DirectHF center-crop `0.90 = 0/34` result was a
  non-authoritative historical observation, not evidence that the current Q/K
  recovery route failed.

These categories must not be rewritten as `SCIENTIFIC_NEGATIVE` or used to
claim that complete CEG-WM succeeded or failed.
