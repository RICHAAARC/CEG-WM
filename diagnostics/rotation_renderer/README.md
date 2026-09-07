# RotationRenderer-Diagnostic-V1

This diagnostic reuses the original eight Geometry-V7 R0 evaluation CG PNGs
at residual multiplier 0.75. It performs no embedding, image generation,
content scoring, threshold change, or search. Main, V2 and old results stay intact.

The only matrix is eight images × {-15°, +15°} ×
{black/bilinear, reflect/bilinear, black/bicubic, reflect/bicubic} = 64 real
SyncSeal CPU detections. The first black/bilinear 16 are run and reviewed before
the remaining 48. They are part of the 64, never an additional warmup or rerun.
Input mapping comes from the original R1A `r0_input.ordered_evaluation_cg_inputs`,
with comparison against its original `raw_records`, not its PASS label.
The located source is `Geometry-V7/4f0bf1560805672f786dc86dd50d793aec18aae7/r0-f1`,
`images/evaluation/multiplier-0.75/content-v6-iss-eval-0001..0008/CG.png`.
The local `inputs/source_mapping.json` maps each unit to its downloaded file;
the runner retains both original and local paths and reads JSON with optional BOM.

The historical renderer, adapter and geometry contracts have no changes between
`ac590330e91aacf4b3283df1e94572a0e4f983a0` and baseline
`e12c7eae91cc36edc5d1a1d96249780a3925eccb`.

## Rendering and measurements

All four conditions use the historical inverse rotation mapping, angle sign and
512 output, with one Pillow PERSPECTIVE sample. Black uses the original image
and `fillcolor=(0,0,0)`. Reflect uses `numpy.pad(mode="reflect")` with 128 pixels
on each edge and adds 128 to the two source translation coefficients. Padding
copies pixels without sampling. It does not change the rotation center or add
a resize/crop. The padding covers the +/-15 degree footprint and cubic kernel.

The historical matrix is expressed about 255.5 and passed unchanged to Pillow's
edge-coordinate sampler. Its effective pixel-center rotation center is 255.0.
This shared historical convention has an approximately 0.184592 pixel offset
from strict pixel-center truth at these angles. We retain the old convention
for both historical comparison and all new conditions. Do not interpret
subpixel differences as a center correction. Integer padding translations can
also change isolated floating-point quantization ties by one uint8 level;
the interior synthetic check bounds this difference and is not a claim of
pixelwise identity for the reflection implementation.

The primary metric is sqrt(mean over four corners of squared Euclidean pixel
error). Historical `prediction_rmse` averages over eight normalized coordinates,
so the conversion is `old * 255.5 * sqrt(2)`. Identity baseline is approximately
94.326467 pixels at either angle. Raw predicted corners, public coordinates,
H, legality/status, identity, error and time are preserved for every attempt.
Identity means normalized H is within absolute 1e-6 of I (zero relative
tolerance); absent H is unknown, not a negative identity determination.
Coordinate errors on illegal H remain diagnostic and do not imply recovery.
Group medians require all eight coordinate measurements, and errors are counted
separately. No failure is replaced or implicitly successful.

## Execution

Use the existing CPU venv and official local JIT. Run from this worktree:

```bash
PYTHONPATH=src /home/richar/projects/CEG-WM/alive/CEG-WM/.venv/bin/python \
  -m diagnostics.rotation_renderer.run \
  --history /path/to/original-r1a-result.json \
  --r0-root /path/to/original-r0-artifact-root \
  --model /home/richar/projects/CEG-WM/diagnostics/BlindDetection-V2/runtime/syncmodel.jit.pt \
  --output /home/richar/projects/CEG-WM/diagnostics/RotationRenderer-Diagnostic-V1/renderer-v1 \
  --phase baseline
```

After reporting the first 16 historical comparisons, the same command with
`--phase remaining` runs only the other 48. There is no automatic retry or
overwrite. A poor baseline reproduction requires investigating inputs, model,
call and renderer before claiming a fill effect. If an explicit new attempt
is needed, retain and report the first attempt.

Each phase writes incremental raw JSONL, environment and input records, a
summary and a same-image paired CSV. The second summary/table include the
preserved first 16 rows. The CPU thread count (default four) is recorded.
The runner does not enforce a GPU, library version, model hash or PASS gate.

The fixed sample supports paired renderer attribution only. Black/bilinear
good and reflect/bilinear bad supports a fill effect; black/bilinear good and
black/bicubic bad supports an interpolation effect; only reflect/bicubic bad
suggests interaction. If all are good, investigate old/new image or embedding
differences next. No content-detection or formal robustness claim follows.
`science_denominator=0` remains explicit.

Lightweight checks (no real model):
`python -m pytest diagnostics/rotation_renderer/test_renderer.py -q -o addopts=`.
