# Fixed cross-channel spectrum reader: zero-shift gate first

[Run in Colab](https://colab.research.google.com/github/RICHAAARC/CEG-WM/blob/diagnostics/crop-bottleneck-v1/diagnostics/CropCrossSpectrum-V1/CEG_WM_crop_cross_spectrum_v1.ipynb)

Use the successful Python 3.12 GPU runtime and existing `HF_TOKEN` /
`CEG_WM_ROOT_KEY` secrets. Run all cells. No new image generation, model training,
frequency/channel selection, parameter search or best-per-image decision exists.

Source (read-only): `MyDrive/CEG-WM/crop-bottleneck-v1/attempt01`.
New output: `MyDrive/CEG-WM/crop-cross-spectrum-v1/attempt01`.
Historical source code: `d9ef4adb4f8424f1102f577bb76347fbba112bd9`.
The self-contained notebook embeds unchanged, SHA256-bound prior support scripts;
only the VAE is loaded. Source images and public model/key identities are checked.

## Frozen candidate and comparison

B is the historical centered real-spectrum HF cosine on the common mask excluding
the even-grid Nyquist row and column. C uses the same mask, all 120 unordered pairs
of 16 channels, and complex off-diagonal features
`R_cd(f) = Z_c(f) conj(Z_d(f)) / sum_a |Z_a(f)|^2`.
Its score is the real part of the complex cosine against identically transformed
original keyed carriers. No diagonal terms, black-image subtraction, extra
frequency weighting, absolute correlation, LF mixture, learned weights or search.
Zero-energy frequencies contribute zero; a zero/nonfinite overall norm is invalid.

Common per-frequency nonzero complex gain cancels in this feature. Real VAE
re-encoding need not have that form; channel mixing and weak second-order keyed
signal can invalidate the candidate. This run tests those limitations, not a
guaranteed method improvement. Templates preserve relative rather than absolute
channel signs; the 17 fixed relation templates must not collide.

## Sequential gates

1. Replay all 16 original Oracle images against full LF/HF/joint historical scores.
2. Evaluate both readers at zero shift, with the original key and 16 fixed wrong keys.
3. Require C positive top-1 >=7/8, paired positive registered score > negative >=7/8,
   median standardized keyed gap >0 and >=80% of B, and negative top-1 <3/8.
   A nonpositive B median stops as an uninformative reference.
4. Only on PASS, perform 96 pure phase controls of Qz. All C key scores must remain
   within absolute 1e-6 of their zero-shift values. No encoded-black baseline here.
5. Only after mathematical checks, score exact RGB copies at x/y +/-2 and x/y +8.
   Zero and circular boundary results must be byte-identical; otherwise stop.
6. C must reduce median per-source worst standardized gap change over the four
   +/-2 conditions by >=30% vs B, with >=6/8 sources improving. Each such condition
   requires top-1 >=6/8, paired score wins >=6/8, and median gap >=50% of C zero.
   Negative top-1 must be <3/8 at every condition. +8 points are references only.

Gap = registered score minus the largest of the fixed 16 wrong-key scores. For
each reader the report scale is frozen from zero-shift negative images: subtract
each image's 16-wrong-key mean, pool the 8x16 squared deviations, take their mean
and square root (population RMS). This is report normalization, not a calibrated
decision threshold or reader parameter. Ties do not count as wins. Zero scale is
invalid. All raw scores are retained. No pooled independence or formal FPR claim.

The first stage uses 16 observations. A scientific zero-gate stop records 96 actual
shift observations and 96 phase controls as not run, keeping the planned counts.
Passing it allows a total of 112 actual observations and 96 phase controls. Both
readers share each VAE observation; no embedding or generation path is invoked.

## Stop, resume, and validation

`STOPPED_ZERO_GATE` and `STOPPED_UNINFORMATIVE_REFERENCE` are persistent terminal
results. Run all again does not reopen them. Full runs finish as
`COMPLETE_CANDIDATE_PASSED` or `COMPLETE_CANDIDATE_REJECTED`. Operational errors are
separate and retain attempts/fixed record slots. Interrupted runs can resume;
successful shift records are skipped, and baseline replay is repeated before
continuing. Input/output may not overlap. Concurrent writers are unsupported.

This notebook is delivered unexecuted on real saved images. Local tests are
synthetic CPU engineering validation only. They cover the actual keyed math,
phase cancellation, gate boundaries, early stop, interrupted resume, immutable
source, notebook embedding/logging and plot paths. GPU scientific results and
plots must be inspected after Colab Run all.

Rebuild: `python diagnostics/CropCrossSpectrum-V1/build_notebook.py`.
The builder reuses the previously published residual setup notebook and support
scripts from the residual/mechanism directories; the built notebook is standalone.
