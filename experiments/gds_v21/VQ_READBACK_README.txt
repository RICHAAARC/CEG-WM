Frozen native VQ RGB readback, 2026-11-28

Authorized scope: existing 48 G1 PNGs only, one encoder forward per image,
zero training, generation, attacks, decoder forwards, checkpoint search or
automatic expansion. The user executes the notebook in Colab. This delivery
does not modify the G1/G2-A protocols or their existing results.

Only the fixed original vqvae component is downloaded/loaded, not the diffusion
transformer, text encoder or pipeline. The full VQModel includes an unused decoder;
a forward hook prohibits invoking it. eval, no_grad/inference_mode, float32.
from_pretrained output_loading_info must have no missing, unexpected, mismatched
or error keys. Record all checkpoint and code hashes. Do not invent a fallback
encoder or initialize missing weights. The fixed model/revision matches G1.

RGB is the exact saved 256x256 RGB PNG, transformed by x/127.5-1, no resize/crop/
attack/scaling_factor. Native encode() already includes quant_conv; call official
quantize() on its [1,128,32,32] output. Flatten indices in official row-major
BHWC order. Native public g() maps the indices to 16 groups. All raw labels and
seeds/prompts/source IDs are excluded from the encoder/quantizer input.
Original final Tokens are used only after prediction for offline metrics.

The diagnostics reuse the same encoded features for distances to the 4096 public
codewords. No second encoder call. torch.cdist argmin must equal official indices.
Original Token rank breaks equal-distance ties by codeword index. top1/4/16 are
diagnostics only, never selecting a prediction using the correct label.
Group accuracy must be >= exact Token accuracy at the same sites. Report ordinary
and balanced group accuracy, confusion, Token-error same-group/cross-group rates,
and seen/unseen-codeword subsets with original 4/2/2 source split.

The existing G2-A helpers are reused for read-only source checks, priors, metrics
and shuffle controls. No Reader is instantiated; no training routine is called.
G2-A priors use only its original 8 fitting images, all variants remain in their
source split. Compare against saved G2-A metrics without running its U-Net again.
Do not compare hard one-hot clipped CE with U-Net soft CE: only accuracy/BA and
confusion are used for that comparison. Two heldout sources are not a fresh
confirmation set; 1024 sites and six arms do not create independent sources.

Clean geometry: each 32x32 group output is copied into 2x2 cells to form the same
64x64 field used by original clean labels. Compare identity with the same eight
fixed wrong hypotheses in geometry.py, using original scoring, no normalization
change. Score saved U-Net probabilities and ideal original Token fields alongside
native output. This is a finite clean-template diagnostic, not attacked RGB
inference, a full blind-search evaluation or a watermark-positive decision.
N has no reference and is diagnostic, not a required identity-pass unit.

Limits: at most 48 native encoder forwards, 30-minute job limit including model
loading (package setup outside). Failure/OOM/missing data stops, preserving every
completed/failed/not-run unit and partial output. No automatic retry or resume.
Output directory is fixed and refuses overwrite. Model/input integrity failures
invalidate the run; source hashes before/after are retained. Complete results
require manual review, no automatic geometric attack or new training stage.

Inputs from persistent Drive:
gds-v21/runs/g1-v21-20261128-8sources-48attempts (48 images + original trajectories)
gds-v21/runs/g2a-clean-existing-g1-v1 (completed report, input hashes, metrics,
and saved 64x64 probabilities). review ZIP alone is not a substitute for G1 images.
Outputs: gds-v21/runs/vq-readback-existing-g1-v1 and its _review.zip.
All original input files remain read-only. Derived indices, rank diagnostics,
tables, figures and per-unit ledger persist on Drive. No input images/checkpoint
weights are embedded in the public notebook or review ZIP.

Interpretation: good native readback with weak U-Net suggests a learnability/
initialization bottleneck, not a causal separation of data versus initialization.
Poor native readback cannot prove information-theoretic irrecoverability; native
encoder is not guaranteed to invert every generated decoder output. Widespread
cross-group near-codeword substitutions motivate reviewing the random grouping,
not silently regrouping old outputs and claiming a validated new writing method.
