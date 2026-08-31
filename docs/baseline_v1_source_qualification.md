# Baseline-V1 source qualification

The four external sources were shallow-fetched from their official HTTPS remotes
at the exacts and source-archive SHA-256 values in `cegwm.baselines.registry`.
Each checkout is detached and clean under ignored `external_sources/`; it is not
project method code and is not an execution result.

| Method | Source status | License | Native score and direction | Official path |
| --- | --- | --- | --- | --- |
| Tree-Ring | qualified | MIT | inverted Fourier-key distance; lower is positive | `run_tree_ring_watermark.py`, `_detect.py:detect` |
| Gaussian Shading | qualified | MIT | watermark bit accuracy; higher is positive | `run_gaussian_shading.py`, `watermark.py:eval_watermark` |
| Shallow Diffuse | blocked_license_missing | no license file at frozen exact | negative L1 distance / negative p-value; higher is positive | `run_shallow_diffuse_t2i.py`, `get_metrics` |
| T2SMark | qualified | Apache-2.0 | key-decode L1 norm; higher is positive | `run_sd35.py` |

The source facts do not create a calibrated threshold, performance result, or
executability. Tree-Ring, Gaussian Shading, and Shallow Diffuse require an
explicitly authorized SD3.5 adaptation review before any implementation choices
are frozen. T2SMark's source path is official SD3.5, but model/runtime execution
remains prohibited. Shallow Diffuse cannot advance beyond source audit until an
official license is identified or a user authorizes a different legal basis.
