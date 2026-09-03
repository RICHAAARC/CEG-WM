# CEG-WM Formal Experiment Contract V1

## Status and authority

After the producer exacts are bound in all seven notebooks and the local static
and CPU suite passes, this contract state is
`FORMAL_EXPERIMENT_CONTRACT_FROZEN / EXECUTION_READY / EXECUTION_NOT_AUTHORIZED`.
`EXECUTION_READY` is not permission to run a model, GPU, Colab, Drive job, or a
formal denominator.  A separate user authorization is required.

The proposed method is defined only by the frozen PaperFPR producer descended
from `main@e12c7eae91cc36edc5d1a1d96249780a3925eccb`. Baselines remain on
Baseline-V1 and enter only through their result files. The existing
BlindDetection-V1 N_dev=256 max-score threshold and N=4 smoke remain engineering
evidence with science_denominator=0; neither is a paper threshold or FPR result.

The notebook-executed producer exacts are
`PaperFPR-V1@e0deb60d3796a59891cd669fe6f071589897885d` and
`Baseline-V1@23862e0c47411d67e66a617cf35dbd54bbdc0435`.

## Shared population contract

All five methods use the same ordered logical roster defined by
`configs/paper_experiment/formal_experiment_v1.json`. The 64-prompt corpus is
expanded with nonoverlapping seed ranges into:

- 2,000 `threshold_calibration` clean unwatermarked negatives per method;
- 3,000 independent `clean_negative_test` clean unwatermarked negatives per method;
- 1,000 `formal_evaluation_pairs` per method, each with one clean negative arm
  and one watermarked positive arm sharing prompt and base-latent seed.

Roster index, unit ID, prompt-source ID, prompt, seed, role, and condition are
fixed before scores. Automatic checks require 6,000 unique `(prompt_id, seed)`
identities across the three partitions. No hash, receipt, signature, URL,
filename, or byte-length chain is an execution gate.

## Threshold and decisions

Each method exposes one continuous score normalized so higher means stronger
watermark evidence. Each method calibrates independently; thresholds are never
shared. Calibration sorts the 2,000 complete scores and freezes
`k=ceil(0.999*2000)=1998`, `tau=c_(1998)`. The decision is strictly
`score > tau`; equality is negative. Attacked negatives and all test results are
excluded from calibration and cannot revise tau.

The proposed method's calibration score is `max(m_pre,m_post)` when one legal
rectification completes, otherwise `m_pre` for a method-complete fail-closed
route. Operationally incomplete rows have no score. Tree-Ring uses negative
Fourier-key L1 distance, Gaussian Shading uses bit accuracy, Shallow Diffuse
uses exactly `negative_mask_l1diff_mean`, and T2SMark uses `norm1_w_master_key`.

If any calibration unit remains operationally failed after its frozen attempts,
the calibration result is still published as `INCOMPLETE_OPERATIONAL`, no
threshold is fabricated from a successful subset, and downstream evaluation
stops before model loading. The method terminal package is still created with
the unchanged planned denominators, zero scored downstream rows, all remaining
rows marked missing, and the corresponding widest planned-denominator bounds.

## Main evaluation and quality

The six ordered conditions are clean, JPEG Q50, 50% bicubic resize/restore,
80%-area center crop/restore, Gaussian blur sigma 1.0 px, and +10 degree
bicubic reflect-pad center-crop rotation. The same deterministic transform is
applied to both arms. Every attacked-negative condition has fixed N=1,000 and
is reported separately from the N=3,000 clean-negative test. A rotation+scale
combination is absent.

The one frozen per-method threshold is reused for clean/attacked positive and
negative arms. There is no attack-specific threshold. Quality is PSNR, SSIM,
and LPIPS only, computed on the already generated clean evaluation pairs. No
quality-only image generation or quality gate exists.

## Partial results and nonblocking statistics

For every role and condition the result includes `N_planned`, `N_scored`,
`N_failed`, `N_missing`, coverage, scored-only conditional TPR/FPR, its exact
two-sided 95% Clopper-Pearson interval, and planned-denominator bounds. For
negatives the bounds are
`FP/N_planned <= FPR <= (FP+failed+missing)/N_planned`; positives use the
analogous TP bounds. A method-complete fail-closed negative is TN or FN by
truth role. Operational failure is never TN or FN.

Observed FPR, exact intervals, and an optional one-sided UCB are report-only.
They never suppress a result package, retune a threshold, replace a sample, or
delete a row. The primary name is "TPR at a threshold calibrated for target
FPR=0.1%", not `TPR@FPR=0.1%` without observed support.

## Minimal ablations

The fixed subset is the first 100 formal evaluation pairs, selected before any
outcome. Each variant uses clean plus exactly one representative attack:

- `no_content_adaptive`: uniform tile weights and 0.5/0.5 branch shares under
  the same total budget; representative Gaussian blur sigma 1.0 px;
- `lf_only`: the LF direction receives the complete shared budget;
  representative JPEG Q50;
- `hf_only`: the content-weighted HF direction receives the complete shared
  budget; representative JPEG Q50;
- `no_geometry`: unchanged full embedding and content statistic, but no
  rectification; representative +10 degree rotation.

The first three retain the same detector statistic and route, so they reuse the
formal threshold. `no_geometry` is explicitly a controlled same-threshold
ablation and is not described as independently calibrated at the 0.1% target.
An independently target-calibrated no-geometry comparison would require a new
2,000-negative calibration and is outside this minimal contract.

## Reconstruction supplement

The supplement covers the proposed method only. It uses the first
`max(100,ceil(0.01*1000))=100` formal pairs, SDXL base 1.0 image-to-image at
repository revision `462165984030d82259a11f4367a4eed129e94a7b`, empty prompt,
strength 0.3, guidance 1.0, 20 steps, and a fixed per-unit seed. It reuses the
paper threshold and reports the same partial-result fields and intervals.

With 100 negative samples the empirical FPR resolution is 1%. This supplement
cannot validate a 0.1% attacked FPR, does not enter the main table, and cannot
support a claim of comprehensive generative-attack resistance.

## Recovery and Drive state

JOB_ID/RUN_ID and Drive directories are stable across Colab restarts. Each unit
has a create-only terminal record. A score already committed is skipped forever.
Only typed `CUDA_OOM_TRANSIENT` and `MODEL_RUNTIME_TRANSIENT` failures may retry
the identical unit once, for two total attempts. Prompt, seed, base latent,
method, role, attack, and configuration cannot change, and every attempt is
retained. No result, score, decision, or aggregate may trigger a retry.

Every 25-unit shard end and every two hours publishes a new append-only numbered
checkpoint. Final results are create-only and published only after all planned
units in that stage are terminal. `progress.json` may be overwritten for human
monitoring but is not statistical evidence. `/content` is disposable; Drive is
the only cross-runtime state. There is no force-rerun-all, alternate JOB_ID to
escape failures, lock, lease, heartbeat, or concurrent scheduler. One runtime
per JOB_ID is an operator obligation.

## Seven notebook entries and stages

The only logical entries are:

1. `paper_main_worker_colab.ipynb`;
2. `paper_baseline_worker_colab-t2smark.ipynb`;
3. `paper_baseline_worker_colab-treering.ipynb`;
4. `paper_baseline_worker_colab-gaussian-shading.ipynb`;
5. `paper_baseline_worker_colab-shallow-diffuse.ipynb`;
6. `paper_reconstruction_worker_colab.ipynb`;
7. `paper_results_finalize_colab.ipynb`.

Every first code cell is exactly the two-line Drive mount without
`force_remount`; outputs and execution counts are empty at commit. Notebooks
select only fixed JOB_ID and expected exact. They do not contain scientific
threshold, roster, attack, checkpoint, retry, or aggregation logic and do not
print per-unit scores.

Calibration completes before one threshold is created. Evaluation refuses to
load a model if its threshold is absent or mismatched. Main evaluation, clean
negative testing, and same-threshold ablations follow threshold freeze.
Reconstruction requires the main threshold and complete formal image stage.
Finalize reads Drive only, checks exacts, thresholds, method duplicates,
roster/matrix coverage, missing rows, and failure states, then writes the table
before publishing the unified JSON package last. It cannot run a model.

## Result package and violations

The Drive result tree contains the contract/config, per-unit raw rows, all
attempts and failures, quality values, and append-only checkpoints. The unified
package contains per-method thresholds, clean-negative and six-condition
positive/negative summaries, minimal-ablation summaries, the reconstruction
supplement, a long CSV table, and a terminal unified JSON result. Missing method
or reconstruction files are represented explicitly as all-missing terminal
entries; they do not prevent publication of that package.

Contract violations include cross-method or attack-specific thresholds, use of
the N_dev=256 engineering threshold, test feedback, roster overlap or
replacement, outcome-driven retries, rerunning a scored unit, hiding attempts,
turning failures into TN/FN, blocking the package on performance, adding a
rotation+scale main-table condition, changing the Shallow statistic, calling
no-geometry target-calibrated without a separate N=2,000 calibration, adding
quality generation, promoting reconstruction to the main table, using canary
rows as paper results, changing a stable JOB_ID, concurrent use of one JOB_ID,
or executing without separate authorization.

Poor TPR, observed FPR above 0.1%, a wide interval, operational incompleteness,
and deviation from the 70-95 or 350-580 GPU-hour planning ranges are reportable
outcomes, not contract violations.
