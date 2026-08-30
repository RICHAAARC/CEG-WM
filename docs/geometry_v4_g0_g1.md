# Geometry-V4 G0/G1 generated SD3.5 path

This is a local engineering route only; it makes no watermark-positive,
robustness, P1D/P1C, or scientific claim.  Geometry continues to be coordinate
only and never votes for presence.

The sole writer placement is the zero-based callback step 19 of a 20-step
`stabilityai/stable-diffusion-3.5-medium` run, immediately before the final VAE
decode.  `FinalLatentAnchorCallback` rejects a second invocation.  A clean
same-seed pass and the marked pass each materialize ordinary final RGB; the
writer's internal latent update is never itself evidence.

The writer and RGB-only detector now share one deterministic keyed RGB-luma
basis: twelve signed directional multiscale global components and sixteen
keyed local tiles, orthogonalized with the frozen .40/.60 energy split. At
step 19 the writer takes exactly one differentiable SD3 VAE adjoint gradient
of that same signed objective, removes DC, normalizes it once, and applies the
unchanged fixed latent amplitude. It fails closed on a missing, nonfinite, or
zero VAE gradient; it does not search, iterate, observe final comparison
results, or use truth.

G0 is exactly seeds 5101--5104 with its one predeclared prompt and identity.
Its unit passes only if correct-key final-RGB anchor score exceeds its
wrong-key score, PSNR is above 40, SSIM is above .98, Rec.709 luma RMS and peak
are at most 2/255 and 8/255, and the supplied unchanged content detector's
two independent RGB calls differ by less than .05.  Over-budget and runtime
failures are retained, never tuned or retried.

The content detector is not a Geometry proxy: it is constructed automatically
from the same `content_iss_engine._load_pipeline_and_assets` result as the
generation pipeline.  It accepts only current RGB and normalized key bytes,
then applies the existing whitening LF scorer, HF scorer, and frozen V9
weighted-joint asset.  Its scorer and asset identities are retained in every
run record.

G1 is the one-shot holdout roster: seeds 6101--6104 with their frozen prompts
crossed with identity, rotation 5 degrees, scale .9, translation .08/0, and
crop .9.  It may only start after a separately reviewed G0 4/4 freeze, and it
must not feed tuning.

The Colab notebook only accepts an uploaded or Drive-stored source archive and
a SHA-256 supplied by the controller.  It verifies that archive before
extracting it; it never clones GitHub or uses an unpushed branch.  All runtime
artifacts are create-only paths under the dedicated Drive directory and must
be separately SHA-256 checked before any result is reported.
