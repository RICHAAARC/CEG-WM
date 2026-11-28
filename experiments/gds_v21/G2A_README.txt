GDS v2.1 G2-A: existing-image clean readability pilot, 2026-11-28

Authorization amendment: the user explicitly moved image quality from an entry
gate to a continuously recorded evaluation item, and requested continued method
validation delivered for user execution in Colab. Previous G1 results and paper
protocol remain unchanged; no missing quality rating is marked passed.

This is a NEW, smaller preliminary branch before the full paper G2. It is not
the previously proposed 256-source training / 64-source development experiment.
It creates zero new images, loads no diffusion/VQ/TrustMark models, and uses only
the existing 48 G1 RGB files and their actual final Token labels. Source split is
SHA256(domain + source_id), first 4 train, next 2 development, last 2 held out.
All six arms of a source stay together. The G1 images were previously examined,
so 'held out' means held out of reader fitting, NOT a fresh scientific confirmation.

Only N/B0 from the four training sources enter fitting (8 images). Development
uses N/B0 from two other sources (4 images), for monitoring only. One fixed U-Net
is trained for at most 2000 AdamW steps, batch 4, lr .001, wd .0001, seed 281128201,
float32, no augmentation. Architecture follows the paper proposal: 32/64/128/256
two-convolution blocks, three average-pooling levels, bilinear decoder with skip
connections, 16 logits averaged into 64x64 cells before softmax. There is no
coordinate/template/arm/prompt/Token input. Only the reader is trained.

Targets are g(actual final Token), expanded 2x2 to the 64x64 clean grid, not
the ideal reference address. Loss: ordinary mean cross entropy. No P-specific
fitting, tuning after heldout inspection, checkpoint selection, automatic retry,
resume, or extension. Final step only. A 90-minute whole-job wall limit stops
with retained checkpoint and INCOMPLETE status; CUDA OOM/nonfinite data/failed
input checks stop, never silently reduce model or batch size. Checkpoints every
100 steps and logs/report are saved to Drive. Native Colab torch is used and
versions/GPU are recorded; no inherited G0 GPU identity requirement for training.

After a complete run, all 48 units are scored and labelled train/dev/heldout.
Only the two heldout sources provide the fit-independent diagnostic (12 images,
including all six arms). Metrics are computed at the 32x32 Token grid by averaging
the four posterior cells for each Token. Report accuracy, per-image balanced
accuracy (present classes only, with support), cross entropy, confusion counts,
and source-level comparisons. Neither 1024 Tokens nor six arms create additional
independent sources. No low-FPR, synchronization, or attack-robustness claim.

Shortcut controls: uniform; training-only group-frequency prior; training-only
position prior (Laplace 1); ideal-address prediction (diagnostic only); reader
posterior for another source's same-arm RGB within the same split; spatially
shuffled reader posterior with unchanged per-image frequencies. Report N
separately, because it carries no imposed address template. Predictions for the
fixed-address control use no true Token labels. Unknown-class supports are not
silently assigned zero recall. No unseen-address-layout evaluation exists here;
that limitation is explicit rather than pretending the original G1 had such data.

Frozen preliminary signal flag: for each of the four heldout N/B0 units, reader
CE must beat uniform/frequency/position and both shuffle controls, and balanced
accuracy must beat uniform/frequency/position/address. Strict numerical margin
1e-6 is a tie rule, not a scientific effect-size claim. All four passing is only
'clean signal in this small pilot; review', not automatic G2-B admission. A
failure is 'signal not established under this small-data budget', not proof that
all RGB readers are impossible. Persistent train/heldout gaps motivate diagnosing
sample insufficiency or shortcuts, not hiding failures or restarting training.

Clean symbol readout does not establish geometric recovery. We intentionally do
not warp predicted symbols as a substitute for attacking RGB, and do not run the
70 ideal-Token comparisons as purported blind RGB results. Geometric search,
JPEG, main-watermark integration, fresh layouts/sources and the full paper G2
remain separate later decisions. Quality records remain unchanged and available.

Input: Drive gds-v21/runs/g1-v21-20261128-8sources-48attempts.
Output: Drive gds-v21/runs/g2a-clean-existing-g1-v1 (one attempt; refuses reuse).
Report ZIP contains configuration, input hashes, split, loss/metrics, predictions,
confusions and diagnostic plots; latest.pt remains separately on Drive. Original
G1 inputs are hashed before/after and never modified. No private images/tokens are
embedded in the published notebook. Local tests are software fixtures only;
real Colab training has not been run by the assistant.
