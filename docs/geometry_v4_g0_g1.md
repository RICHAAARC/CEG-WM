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

The G1 detector is a separate generated-anchor recovery path rather than the
P1 proxy detector. It receives exactly one attacked ordinary RGB array and one
normalized detection key. It builds the unchanged generated RGB anchor basis,
uses the frozen two-stage rotation/scale grid, and estimates translation by
valid-mask Hann-windowed normalized cross-power phase correlation. Candidate
ranking uses signed combined-anchor evidence, translation PSR, and a fixed
lexicographic tie-break. Interpolation is NumPy bilinear at pixel centers;
out-of-frame samples use current-RGB channel medians; rotation is around
normalized center (.5,.5); composition is centered rotation/scale followed by
translation. The public matrix direction is always attacked-to-canonical.

Reliability is fail-closed and frozen before G1: signed anchor score at least
3.0, translation PSR at least 4.0, at least six independently positive local
tiles at score .05, coverage of at least three 2x2 macro regions, positive
global evidence, and valid convex corners. A failed invariant emits no H or
corners. Correct and wrong keys each execute this complete path independently;
neither may reuse the other's candidates or H.

A G1 unit passes only when the unchanged final-RGB observability gate passes,
correct-key geometry is RELIABLE with H/corners/support, wrong-key geometry is
UNRELIABLE, and the current attacked RGB rectified by the correct H has higher
correct-key than wrong-key anchor evidence. This coordinate output never votes
for watermark presence and never changes the content threshold. All 4x5=20
units remain in the denominator, including exceptions; there is no retry,
replacement, fallback, or attack-conditioned detector path. The command-line
summary counts `attacked_rgb.passed`, so twenty final-RGB-only passes cannot be
reported as a G1 pass.

The preceding real G0 artifact at source exact
`5b29a275151d436dbe1d51789cffe8e6908966b7` passed 4/4 with no failure; its
records SHA-256 is
`ddff80537a023e3f4e0ad368e625ec413c49a2ade540e67980ed617ec3640773`.
This establishes only final-RGB observability and does not predict G1.

The Colab notebook checks out one controller-pinned pushed source exact in a
fresh subprocess. All runtime artifacts are create-only paths under the
dedicated Drive directory and must be separately SHA-256 checked before any
result is reported.
