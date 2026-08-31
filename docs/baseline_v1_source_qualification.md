# Baseline-V1 source qualification

The source registry records official paths and optional source metadata for the
four methods. Such metadata is not an observation, adapter, score, or main-table
hard gate, and this document does not claim an execution result.

| Method | Source status | License | Native score and direction | Official path |
| --- | --- | --- | --- | --- |
| Tree-Ring | qualified | MIT | direct Fourier-key L1 distance; lower is positive | `run_tree_ring_watermark.py`, `_detect.py:detect` |
| Gaussian Shading | qualified | MIT | watermark bit accuracy; higher is positive | `run_gaussian_shading.py`, `watermark.py:eval_watermark` |
| Shallow Diffuse | qualified for method implementation | license not evaluated in this method-implementation stage | negative L1 distance / negative p-value; higher is positive | `run_shallow_diffuse_t2i.py`, `get_metrics` |
| T2SMark | qualified | Apache-2.0 | `norm1_w` under master key; higher is positive | `run_sd35.py` |

The source facts do not create a calibrated threshold, performance result, or
completed adapter. Tree-Ring, Gaussian Shading, and Shallow Diffuse have
method-faithful SD3.5 adaptation work remaining; T2SMark has an official SD3.5
native path but no completed runtime is claimed. T2SMark's `norm1_no_w` is a
fake-master-key comparator on the same watermarked sample, not an unwatermarked
control. Shallow Diffuse licensing is not evaluated in this
method-implementation stage and is not a technical blocker here.
