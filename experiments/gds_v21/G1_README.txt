GDS v2.1 G1 — frozen limited validation, 2026-11-28

Prerequisite observed: user's g0-20261128T073448Z-bde6b4 completed both native
generations on NVIDIA L4, torch 2.14.1+cu130; all 100 tokens/RNG states, decoded
RGB and Gumbel replays matched. All 50 frozen t=60..11 steps had native exchange
opportunities. The original G0 Drive report and nine payload file hashes are
verified before any G1 generation. This G1 delivery has not run a real model yet.

Exactly 8 new authored source/seed pairs, each N/B0/B1/B2/P/P-E once: at most
48 attempts. No G0 regeneration, training, decoder fitting, primary watermark,
attack sweep, confirmation-set access or automatic expansion. Stop at first
failed attempt or wall-time budget; failed/unrun units retain their denominator.
No resume/retry implementation: repeated notebook execution refuses an existing
experiment output directory, including a partially completed one. Return logs
for review before any recovery; do not delete records to restart this roster.

The original pipeline pre-processing and q_posterior are retained. N uses the
native step at all times. All others use the original step outside t=60..11.
Within that window: independent MASK events -> bounded deterministic swaps ->
full conditional content draw on selected sites; unselected MASK sites are Dirac
MASK; previously visible sites sample full original K, including natural remask.
Each arm/domain seed = first 8 SHA256 bytes of UTF8
'GDS-v2.1-G1:{source_seed}:{arm}:{event|content|visible}', modulo 2^63-1.
The pipeline native RNG starts from the same source seed for all arms; event,
content, visible RNG streams are mutually separate and arm-specific. N's shadow
events never alter its native draw. Counterbalanced arm order is frozen by source
index. N is a descriptive native reference; finite sample pairing does not remove
the effects of different sampling streams.

At each actual window history, B1/B2/P/P-E use exactly the same S0 and hash pools;
only the actual arm is executed. All six arms pay this diagnostic overhead.
Same-history differences, initial/executed/shadow sets, ledger, bottleneck indices,
window geometry assets, original event log mass changes, KL-content-only values,
RNG states, actual Token reference and costs are retained. Selection sufficient
statistics are reused for swap scores; CPU tests compare to naive enumeration.
Transformer forwards are counted: exactly 100 per completed image, no extra NFE
for shadows. 'original_event_log_mass_change' is NOT a KL estimate. Summed content
KL is NOT total transition/path KL. Latent/token records are private diagnostics.

Per-image metrics verify final sum(F) against direct ideal-symbol rendering.
Reported reference retention at t10/t0 means address-group or exact-token retention
at previously successful selected positions, not an attribution/survival proof.
Symbol maps are ideal Token fields, NOT RGB readouts. All-histories P/P-E equality
stops progression; inequality only shows mechanism activation. Positive average
Token changes do not automatically authorize G2 or establish quality equivalence.

Output: report.json (48-row ledger), summary.csv, each unit's image.png,
final_symbol_map.svg, private_tokens.npz, steps.jsonl, metrics.json, failure logs.
blind_quality contains anonymous complete images and an EMPTY two-reviewer form;
quality_key_PRIVATE.csv is excluded from that folder. No human quality review has
been performed by the runner. Missing/failed images remain in the main ledger.

The notebook uses the current G0-matching Colab runtime and existing model cache;
it pins diffusers 0.39.0 / transformers 4.57.6 and verifies all recorded package
versions and GPU type match accepted G0. It does not install an old CEG environment
or replace torch/CUDA. A mismatch stops before generation for explicit review.
Loading time and CPU/GPU diagnostics count toward reported wall time. Nominal
limits: 900 seconds per attempt, 4 hours including loading for the run; not a
promise of completion. Keep the same Colab runtime open for the whole 48-run roster.
