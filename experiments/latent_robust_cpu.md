# Fixed four-arm latent rotation CPU diagnosis

This is an isolated development experiment. `latent_sync.estimate_rotation` and
the content detector remain unchanged. It does not load a model or score content.

Run from the repository with the existing NumPy/SciPy CPU environment:

```sh
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python experiments/run_latent_robust_cpu.py \
  --cache-dir /path/to/objective-cache \
  --output-dir /path/to/diagnosis
```

The input is `report.json` plus exactly eight `observation` NPZ arrays: the
`plain`, `content_only`, `latent_anchor`, and `v2_rgb_sync` sources, each suffixed
`__clean.npz` and `__rotation.npz`. Arrays must be finite float32 (16,64,64).
The report supplies geometric truth only after all blind fits and curves finish.

The fixed arms are original, covariance whitening, transfer processing, and their
combination. Whitening uses the current processed observation, centered spatially
only for covariance estimation: K=.75S+.25trace(S)/C I. Its symmetric inverse
square root is fixed across angles and applied to both fields without recentering.
Transfer processing uses continuous zero-extension bilinear rendering, followed
by sigma=.5 smoothing then the original sigma=3 highpass on both sides. This is a
robust comparison approximation, not an estimated VAE transfer or exact shift
probability integral. All arms keep the public template, native pixel center,
[-15,15] bounds, and xatol=.01. The 121-point curve never selects the result.

The 2026-09-10 eight-cache development result completed all 32 fits. Anchor clean
angles were -0.550761, +0.001114, -0.697677, +0.000713 degrees respectively;
rotation angles were -11.122496, -11.093654, -11.216768, -11.451599 for a -10
degree reference. The combination failed to repair rotation despite improving
clean localization. Continuous rendering removed the old local boundary jump.
No further parameter search or GPU candidate follows from these observations.

`robust_cpu_result.json` retains all rows, curves, spectra, numerical failure
status, and same-W/common-denominator privileged differences. Differences require
reference observations and are diagnostic-only. The eight arrays originate from
one already-seen image pair, not eight independent test images. No-anchor sources
can return angles; geometric correlation is not watermark evidence. Correlations
from different objectives must not be compared as performance improvements.

Focused numerical checks:

```sh
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python -m pytest tests/test_latent_robust_sync.py -q
```
