# Saved-image phase / VAE mechanism diagnostic

[Run in Colab](https://colab.research.google.com/github/RICHAAARC/CEG-WM/blob/diagnostics/crop-bottleneck-v1/diagnostics/CropMechanism-V1/CEG_WM_crop_mechanism_v1.ipynb)

Use the successful Python 3.12 GPU runtime and the existing HF_TOKEN and
CEG_WM_ROOT_KEY Colab secrets. Run all cells. No image generation is invoked.

Read-only source: `/content/drive/MyDrive/CEG-WM/crop-bottleneck-v1/attempt01`.
Output: `/content/drive/MyDrive/CEG-WM/crop-mechanism-v1/attempt01`.
Only the pinned SD3.5 VAE is loaded; the prior VAE-only support reader is embedded
unchanged and SHA256-bound. Historical source revision is
`d9ef4adb4f8424f1102f577bb76347fbba112bd9`.

## Fixed design

Eight positive/negative pairs, seven shifts: (0,0), (±2,0), (0,±2), (8,0), (0,8)
RGB pixels. Exactly 112 record slots. All 16 saved Oracle scores must replay before
probing. RGB translation uses exact integer array copies: no new RGB interpolation.
This is a diagnostic about saved Oracle RGB, not an exact reproduction of the
previous crop-to-canonical one-pass residual renderer.

For z=E(x), b=E(black), r=Q(z-b), Q removes even-grid Nyquist row/column. Compare:

- b + periodic unitary Fourier shift of r (norm/round-trip checked).
- b + zero-extended Fourier shift, with fixed 32/64 latent-cell padding on odd grids.
- b + periodic/zero-extended bilinear shift of r.
- b + Q(E(integer-shifted RGB)-b), with zero and circular RGB boundary controls.
- Known inverse latent translation of the actual shifted encoding.
- Periodic Fourier shift of Q(z), without the encoded-black baseline convention.

All counterfactual HF scores use the same non-Nyquist mask and frozen 17 keys.
Actual RGB paths retain the original full LF/HF/joint score and historical reference
tau. The counterfactual scores have **no detection threshold**. Full-mask cached HF
must match the original scorer at replay. Record original and projected HF scores,
removed Nyquist energy, full/inner latent discrepancies, padding convergence,
correct-key evidence, wrong-key maximum and keyed gap.

For the registered HF and its wrong-key gap separately, signed changes telescope:

`base - actual = (base - phase_periodic) + (phase_periodic - phase_pad32) + (phase_pad32 - actual)`.

The bilinear comparison is also retained. These are model-conditional score changes,
not causal percentages. No fractions with near-zero denominators are interpreted.
An extra residual supports encoding-chain non-equivariance relative to the chosen
translation model, not irreversible watermark information loss. Strong boundary,
interpolation, Nyquist or baseline sensitivity leaves the mechanism unresolved.
The eight sources are a development probe, not a population FPR estimate; correlated
shifts and wrong keys do not increase the number of independent source images.

## Stop and resume

Stop on missing/changed source, key/version mismatch, failed 16-image replay,
undefined scores, failed zero-shift or Fourier inverse checks. Failures retain their
record slots and attempt logs. Completed records are skipped on resume, source data
are never overwritten, and no generation fallback exists. Concurrent writers to the
same output are unsupported. There is no automatic expansion, training or search.

## Engineering validation

`test_mechanism.py` uses synthetic CPU observations and validates Fourier sign,
norm/inverse, Nyquist rejection, boundaries/interpolation, original HF equivalence,
checkpoint resume, failure gates, source immutability and notebook embedded scripts.
These checks do not substitute for the user's GPU run. The exported notebook is
unexecuted: GPU scores and the result figure must be inspected after Colab Run all.

Rebuild with `python diagnostics/CropMechanism-V1/build_notebook.py`; the builder
reuses the prior published Colab setup notebook and unchanged support script under
`diagnostics/CropResidual-V1/`. The built notebook itself is self-contained.
