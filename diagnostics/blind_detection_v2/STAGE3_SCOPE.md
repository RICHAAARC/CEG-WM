# Stage 3 candidate and real A/B package

Local implementation only until controller approval for this exact package.
No stage-4 samples, formal thresholds, main merge or automatic GPU execution.

## Implementation

`cegwm.runtime.blind_detection_v2.detect_watermark_v2(image,key,assets)` binds
the real V1 scorer internally by default. It preserves original + legal raw H,
121 fixed coarse rotations including identity, stable top3, and joint
{-0.25,0,+0.25}° × {-1,0,+1}px² refinement. Maximum 200 scorer calls. It
returns every complete per-candidate m/statistic, errors and an uncalibrated
maximum, never an official positive. V1 source and interfaces are unchanged.

The optional `reuse_observation=True` path is in a separate implementation
commit. It shares only the current candidate's image-derived VAE mode, and
only if LF/HF reference the same VAE and processor objects. Different
contexts use the unchanged scorer. Original LF CPU float32→float64 DCT,
HF device float32 FFT→float64 reduction, 17-key order, weighting and maxima
are unchanged. No latent is cached across calls or accepted from a caller.
Actual scoring modes are reported, including fallback. The optimization can
be disabled independently of the search candidate.

## Fixed data and attacks

`stage3.py` uses the first two already existing stage-2 mechanism units,
`v2-mechanism-00` and `v2-mechanism-01`, each with its saved content-only and
primary-null image. It applies SyncSeal multiplier 0.75 once to content-only
RGB. There is no new diffusion generation and no source selection by outcome.

Each pair supplies positive and negative for three conditions:

| Condition | Rotation | Observed translation |
|---|---:|---|
| identity | 0° | (0,0) px |
| offgrid_rotation | -11.7° | (0,0) px |
| offgrid_joint | +18.3° | (+0.75,-0.75) px |

Total 2×3×2=12 observed images. The nonzero angles are not on the 0.5°
coarse grid. The joint transform uses one bicubic image resampling with
reflect padding. These reused two base pairs provide stage-3 mechanism
evidence only; they cannot count as stage-4 independent validation.

## A/B comparisons and failures

Both searches independently rank their own coarse scores and select their
own top3/refinements. For a candidate visited by either path, evaluate both
original and optimized scorers on that same RGB. Save all 17 LF, HF and
weighted values, each complete m, all differences, actual optimization mode,
encoder counts, timings, and changes relative to the old descriptive tau.
Pair execution order alternates original/optimized and optimized/original.
No numeric tolerance is used to suppress or re-label differences.

A per-image dictionary reuses only scalar and branch records for the same
candidate parameters; it contains no images or latents and is discarded
before the next image. A single observed-image geometry estimate supplies
the same raw-H candidate to both paths. Candidate-set divergence is retained,
with full ordered rankings, top3 and maximum differences. Search elapsed
time is affected by this reuse; paired candidate timing/encode totals are
the optimization comparison, not the wall time of the second search alone.

V1 is also actually called through its existing core, using the reference
tau only for descriptive development comparison. V1/coarse/full values are
reported. Oracle scoring and nearest coarse-angle membership enter only
after both searches have finished. They diagnose whether the grid or ranking
missed useful alignment; neither can change candidate generation or selection.

If optimization fails, retain its errors while the original path continues.
Failures remain in the 12-image roster. Summary distinguishes image-level,
candidate A/B, incomplete-search and V1 errors. No automatic retries or
rewrites of prior results. `error=None` for a pilot is not an equivalence or
performance pass: search completion and A/B errors are shown separately.

## Work estimate and user operation

The two searches visit at most 121+1+78+78=278 unique search candidates.
Adding the diagnostic oracle gives at most 279 paired evaluations, plus
at most two old-scorer V1 evaluations: at most 560 scorer invocations/image,
or 6720 for all 12 images. This is a conservative bound; coincident rankings
usually reduce the union. These are scorer calls, not independent samples.

The notebook first runs only image 0 (unit00 identity positive), records its
actual complete A/B image time, and then displays an estimate for the 11
remaining images. It also reports paired cost and failures. Only the next
separate cell continues the fixed remaining roster. Attack and ranking
divergence can cost more than the pilot; no real V2 time or speedup is claimed
before that measurement. Initialization and sample preparation remain visible
in wall time and are not confused with per-candidate paired timing.

Input directory: `MyDrive/CEG-WM/BlindDetection-V2/stage2-mechanism-v1`.
New output: `MyDrive/CEG-WM/BlindDetection-V2/stage3-mechanism-v1`.
Required secrets remain HF_TOKEN and CEG_WM_ROOT_KEY. The existing runtime
requires CUDA but no GPU model or memory size is mandated. Do not rerun over
an existing output directory. The notebook is prepared for user operation,
not executed by this local task.

Local tests validate injected arithmetic, encoder call counts, cache
isolation, candidate selection and geometry. They do not establish real GPU
numerical equivalence, speedup, recovery or calibrated FPR. The entire V2
adaptive maximum still requires independent stage-5 calibration, and stage 4
requires separate fresh samples after the candidate is chosen.
