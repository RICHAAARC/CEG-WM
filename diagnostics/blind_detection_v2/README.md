# BlindDetection-V2 mechanism development

Stage 2 is in progress. No V2 detector has been selected or implemented, and
stage 4 has not run. All development has `science_denominator=0`.

## Observed native comparison

On 2026-09-07 the official paper TorchScript model ran on local CPU,
PyTorch 2.5.1+cpu, using all 100 historical diagnostic pairs, two arms and
two conditions. The 400 unique rows completed without operational errors.
Independently constructed native RGB tensors and adapter tensors matched
exactly, as did every native/adapter raw corner. Rotation-positive corner
RMSE had median 56.1149141164 px and minimum 25.6869168063 px; 13 identity
matrices were reproduced. Native detect, detector and head have no
confidence-conditioned identity fallback.

This excludes adapter changes to input/raw predictions in this comparison.
It does not establish that all upstream preprocessing semantics are correct.
The main geometric error is already present in native predictions. There is
no evidence for fixing it by simply inverting H or changing half pixels.

Native unwarp versus production Pillow warp has median MAE 2.1093322039/255
and median RMSE 7.1860525872/255. The native path truncates public endpoints
to integers after scaling; the additional coordinate displacement is under
1 pixel per axis. Sampling/quantization/boundary differences remain. Their
effect on content scores has **not** been tested locally and must not be
declared harmless.

Original rows and actual JIT method code are retained outside Git at
`/home/richar/projects/CEG-WM/diagnostics/BlindDetection-V2/native_compare_01/`.
The source model is the official
[paper checkpoint](https://dl.fbaipublicfiles.com/wmar/syncseal/paper/syncmodel.jit.pt);
the [upstream scripted implementation](https://github.com/facebookresearch/wmar/blob/main/syncseal/syncseal/models/scripted.py)
was read alongside actual JIT code. After this completed run,
`native_compare.py` gained per-stage failure preservation and endpoint fields;
the original rows are unchanged and do not contain these additional fields.
No rerun is needed to fill those optional fields.

The historical 0056 negative replay score anomaly remains unresolved. The
current CPU comparison does not execute the content scorer and cannot
attribute that anomaly to hardware or explain it away.

## Next mechanism canary: 216 rows

`development.py` predeclares four new prompts and seeds 2026090700–2026090703.
It generates fresh content-only/primary-null pairs using the existing ISS
content production path. It reads no historical diagnostic images and uses
neither the old formal seed ranges nor the V1 calibration seeds. These four
units are mechanism development only and are excluded from stage 4 and
future formal calibration/testing.

Per unit:

* 9 strength rows: positive C+SyncSeal at multipliers 0.5, 0.75 and 1.0,
  each at angles 0, -13 and +7 degrees. One official `embed` residual is
  shared across multipliers. Each row records V1 behavior, geometric error,
  native/public warp scores from the same raw prediction, and RGB MSE/PSNR
  against both content-only and clean images.
* 3 negative rows: the same primary-null image at those three angles.
  Unwatermarked negatives do not depend on the SyncSeal multiplier; each
  row is reused descriptively across the three strength comparisons and is
  never counted three times as independent evidence.
* 42 tolerance rows: 21 perturbations for each positive/negative arm, using
  the +7-degree attacked image and 0.75 positive multiplier. They include
  zero, angular errors ±0.25/0.5/1/2 degrees, and separate horizontal and
  vertical offsets ±1/2/4 px. There is no angle×translation product grid.
  Truth is used only to diagnose alignment tolerance, never in detection.

Thus 4×(9+3+42)=216. Unique keys are `(unit_id,kind,strength,angle)` for
strength rows, `(unit_id,kind,angle)` for negatives, and
`(unit_id,kind,arm,angle_error,dx,dy)` for tolerance rows. Rows are incremental;
failed generation/scoring retains every dependent planned row. A failed
runtime initialization leaves the written plan but no completed experiment.

The historical threshold 1.2657276026437319 is a descriptive reference only.
There is no fitting, threshold change, sample selection, or formal FPR claim.
Native/public warp scoring is diagnostic and does not change production.
Poor margins and failures are results, not reasons to suppress the package.

## Execution and next decision

The accompanying notebook is for user-operated Colab after controller review.
It mounts Drive first, clones the reviewed code, reads Colab secrets
`HF_TOKEN` and `CEG_WM_ROOT_KEY`, and writes a new output directory. It does
not force A100 or a GPU memory size. Existing production loading requires
CUDA; model/device/version are recorded. No global local environment repair
or external model execution has been performed in this task.

Local CPU landmark tests validate the diagnostic sampler at -13, +7 and +21
degrees. They do not validate content robustness. Full canary execution is
pending: this local machine has CPU Torch, no cached content model, and no
HF_TOKEN or CEG_WM_ROOT_KEY in the task environment. The notebook has not
been executed top-to-bottom in Colab.

After the 216 rows, compare geometry error, score margins and quality jointly.
Do not assume 0.75 is too weak. Use the measured tolerance to decide whether
a carrier change, geometry estimator refinement or bounded candidate search
is justified. Submit that concrete V2 choice to the controller. A content
maximum over candidates changes the full detector statistic and needs new
independent stage-5 calibration. Stage 4 needs a separately fixed new roster
and representative rotations/other attacks/paired negatives; none of these
four mechanism units or historical diagnostic units may enter that roster.
