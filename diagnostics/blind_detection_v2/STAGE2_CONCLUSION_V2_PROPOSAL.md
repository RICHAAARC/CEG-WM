# Stage 2 conclusion and proposed V2 candidate

## Observed mechanism

The user-operated L4 run has 216 unique planned rows: 36 strength, 12
negative and 168 tolerance measurements, from only **four base pairs**.
There are no missing or unexpected keys. All 48 V1 detections completed.
One additional public-warp diagnostic failed because unit00 negative +7°
had an unsupported nonconvex prediction and no H. V1 correctly returned a
complete negative; native scoring completed at -1.8760153042. There are 47
public/native comparisons, not 48. `analyze_results.py` labels the missing H
in its derived report while retaining the original TypeError. Raw files are
unchanged; no rerun is needed.

| SyncSeal multiplier | -13° median corner error, px | +7° median corner error, px | Median PSNR vs content-only, dB | Rotated positive V1 / native / oracle |
|---|---:|---:|---:|---|
| 0.5 | 80.872 | 44.108 | 47.780 | 0/8 / 0/8 / 8/8 |
| 0.75 | 78.492 | 43.000 | 44.639 | 0/8 / 0/8 / 8/8 |
| 1.0 | 74.192 | 41.944 | 42.227 | 0/8 / 0/8 / 8/8 |

All three multipliers detected 4/4 unattacked positives. The additional
distortion at higher strength buys only a small geometric improvement,
without recovering detection. Native warp also fails to recover any rotated
positive. Neither increasing strength nor swapping warp alone is a justified
V2 choice from these results.

At +7° and multiplier 0.75, zero and ±0.25° alignment perturbations retained
4/4 positive scores above the old descriptive threshold. At -0.5° only 2/4
did so; at +0.5° all four did. ±1° and ±2° retained none. Single-axis ±1 px
retained four; ±2 px retained 1–4 depending on direction; ±4 px retained
none. The 84 negative tolerance measurements stayed below threshold, but
they are repeated observations of four images. This is **not** a joint
±0.25°/±1 px safe region, a population tolerance guarantee, or formal FPR.

Best rigid fits of the existing 0.75 corner predictions still estimate only
about -2.54° to +1.30°, leaving 5.70°–13° angular error. Removing homography
degrees of freedom or searching only ±1–2° around those estimates has no
evidence of reaching the observed narrow content peak. Combined with the
earlier native-input/raw-output equality, the supported cause is severe
geometric estimation error relative to content alignment tolerance. The
training/domain reason for SyncSeal's biased predictions remains unknown.

## Concrete candidate for controller decision

Propose a **bounded rotation search with joint local refinement**, keeping
the current embedding multiplier 0.75 and current per-image content method.
`search_prototype.py` is a local research prototype, not production V2.

1. Score the ordinary input and retain the legal raw-H recovery candidate.
2. Independently of SyncSeal's predicted angle, examine a public fixed
   [-30°,30°] interval at 0.5° steps around the 512-canvas center. There are
   121 coarse candidates including the original. This is a declared narrow
   rotation operating domain, not a fit of +7° or -13° and not an all-angle
   robustness claim. The runtime takes no attack label or truth angle.
3. Rank the coarse candidates by their **complete** registered-minus-exact16-
   wrong-key-maximum m, with fixed stable tie order and no threshold cutoff.
   At the top three, evaluate angle offsets {-0.25°,0,+0.25°} jointly with
   x/y translations {-1,0,+1}px in canonical coordinates. Skip already
   evaluated parameter tuples and angles outside the declared domain.
4. Each image is sampled directly from the original observed RGB once.
   Refined transforms are composed as matrices, never sequential image warps.
   Keep every score and operational failure. A missing legal raw H does not
   disable the independent rotation search; inconsistent/error geometry
   records prevent a claim of complete execution.

Maximum default budget: 121 + 1 raw-H + 3×26 = **200 content scorer calls**.
The coarse-only subtotal can be inspected from the same run, without a
second scoring pass. The candidate set contains no original clean image,
embed record, attack label, true transform or embedding latent.

Coarse translation is zero, local translations are only ±1 px, and the
top three may cluster around one false peak. Thus neither joint tolerance
nor recovery of large translation, scale, cropping or compounded attacks is
guaranteed. The raw-H candidate retains the previous route as a comparison,
not as proof these attacks remain good. A 1° coarse grid could reduce cost
but has a greater narrow-peak miss risk; it is not the first proposal.

The new statistic is z(X,k)=max of the complete m for all candidates visited
by this exact search. Ranking and refinement depend on content. Never take
max registered score from one candidate minus max wrong-key score from a
different candidate. There is deliberately no `positive` field or calibrated
threshold in the prototype. The entire adaptive search requires independent
stage-5 calibration before formal decisions; old tau is descriptive only.

## Cost and smallest implementation plan

The L4 tolerance rows measured a median 3.696579 seconds per current scorer
call. A 200-call image would be about 739 seconds (~12.3 minutes) at that
rate, excluding initialization/geometry/warping; this is an estimate, not
measured V2 runtime. Row time in the completed canary totals 1410.719 seconds,
excluding generation and initialization.

The existing scorer repeats VAE encoding 34 times per candidate (17 keys ×
LF/HF). The real builder shares the VAE and image processor. A minimal
production implementation should encode each candidate RGB once internally,
then preserve LF's CPU float32→float64 DCT/weighting and HF's original-device
float32 FFT/float64 cosine, and the existing key/weight/max arithmetic.
This is an implementation optimization, **not** a license to transform one
cached latent instead of re-encoding each warped RGB. No external latent,
cross-image cache, cache by PIL object identity, or global monkeypatch.
Distinct VAE/processor contexts must remain distinct. Speedup is unmeasured.

After controller approval, stage 3 should bind the real image/key/assets
scorer internally, verify all 17 LF/HF/weighted values and m against the old
path on real candidate images, and check A/B/A images and changed keys/assets
for cache leakage. Count encoder calls and measure wall time; preserve any
numeric differences and threshold-adjacent cases. Local injected tests do
not establish real-model equality or watermark recovery.

The next separately authorized model test should first use development-only
material to compare V1, the coarse subtotal and the full search with shared
candidate evaluations, including off-grid angles and paired negatives. Do
not consume stage-4 samples while adjusting this candidate. Stage 4 requires
its own predeclared fresh roster and rotations/other attacks/quality/cost.
No new model test, push, merge or formal stage is authorized by this proposal.

## Evidence locations and status

Original [Drive result folder](https://drive.google.com/drive/folders/1gqZyQFjG2zD4Zd7ZY8JOhht06wSwAAVF)
contains [rows](https://drive.google.com/file/d/1q819GkoR9ELx-dL12NkVft3qK87T5Mtv/view),
[summary](https://drive.google.com/file/d/1eO_ki8SCiNE1aBDvC1PPA02X-P-oXmWK/view),
and [plan](https://drive.google.com/file/d/1f4QN4h5QeaLC0yrdRc77pgEzBopezUqq/view).
Local copies are in `/home/richar/projects/CEG-WM/diagnostics/BlindDetection-V2/stage2-mechanism-v1/`;
the derived report is `/home/richar/projects/CEG-WM/diagnostics/BlindDetection-V2/stage2-analysis.json`.

Stage 2's three requested comparisons now have measured evidence within the
four-pair canary. V2 method improvement is unproven. The local prototype has
six tests for search mechanics, one-warp behavior, off-grid synthetic
landmarks, score failures and typed geometry consistency. Stage 3 production
implementation and stage 4 independent validation remain pending controller
decision and subsequent real results. All current evidence has science_denominator=0.
