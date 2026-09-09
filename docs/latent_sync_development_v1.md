# Intermediate latent synchronization development

This branch implements the first A mechanism and a fit-only two-objective
candidate selector. No real candidate comparison has yet been executed.
V2 ISS content allocation, carriers, beta, and registered-minus-max16wrong score
are reused. At callback step_index 18, the original content callback runs first,
then a public asymmetric five-anchor, three-width channel-coded template is
added. No final RGB SyncSeal residual is added or model loaded by the new entry.

Detection encodes only the current RGB with the frozen SD3.5 VAE, matches the
public analytic template with coarse hypotheses plus continuous Powell fitting,
and returns one reference-to-observed similarity H. Pixel-center conjugation
transfers latent H to RGB. One RGB resampling produces the post image. Pre and
post both use the original V2 content scorer; geometry correlation is never
positive evidence. Valid regions do not change scoring or normalization.
No per-image template, original, prompt, embedding latent or Q/K reaches detection.

The fixed search covers +-25 degrees, scale .6 to 1.45 and translation +-15%
of the canvas. The template's survival through remaining generation and VAE
re-encoding, and the VAE's geometric response, are hypotheses to measure.
The scipy search is CPU and can be slow. It selects H using template correlation
only, never queries content scores during H selection.

Run `python -m experiments.run_latent_sync_development` for a no-model plan.
Explicit real execution adds `--execute --output <new-directory>` and optionally
`--tolerance`. Dependencies include the existing real SD3.5 environment plus
`scipy>=1.10`; HF_TOKEN and CEG_WM_ROOT_KEY are read from the environment.
Default seed 2031010000 is new. One plain/marked pair creates two images and 18
blind development paths across the nine shared conditions. Nine positive oracle
paths are diagnostic only. With tolerance enabled there are 117 total diagnostic
paths (13 offsets including oracle per condition), not 117 plus nine.
Output retains exceptions and missing path counts. PSNR/SSIM/LPIPS compare
marked against plain and therefore include content/sync interaction. 2x2 local
MSE is also recorded. Anchor RMS alone is not an image-quality budget.

Generate each candidate on the same fit pairs with `--split fit --anchor-rms ...
--anchor-seed ...`. Select with `python -m experiments.select_latent_sync_candidates
--reports <report paths> --max-mse <RGB [0,1] squared budget> --output selection.json`.
This compares mean corner error versus mean positive-part oracle content score
minus predicted post score, with a shared worst-pair final-image MSE ceiling.
It records all candidates. This is an empirical content tolerance loss, not a
trained geometric residual metric. PSNR/SSIM/LPIPS still need review.
Validation passes `--anchor-config selection.json --objective geometry` or
`--objective content_tolerance --split validation --seed <independent seed>`.
Reused fit seeds are rejected; validation never reselects. The objectives may
legitimately select the same candidate. The current tolerance offsets are axis slices
and two diagonal translations, not a full interaction surface. No observed
quality-matched gain, real-model robustness, formal threshold or FPR
claim is made here. Calibration must include this complete max(pre,post) path;
the V2 threshold must not be reused for this changed synchronization method.
