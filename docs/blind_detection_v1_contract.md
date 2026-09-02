# BlindDetection-V1 single-image closure contract

## Identity and evidence ceiling

BlindDetection-V1 starts from exact `3d0ebe699c4319fb2435e9eacca7e943096e9abd` on branch `BlindDetection-V1`. Its fixed ceiling is: **BlindDetection-V1 implementation authorized; engineering single-image closure only; N_dev=256; science_denominator=0**. This implementation is not fixed-FPR evidence, a reliability result, a robustness result, or a paper conclusion. It does not authorize a push, merge, GPU/model/Colab/Drive run, N=256 execution, or N=4 callback.

## Production method

The single-image content statistic is exactly

`m(image,key) = registered weighted_joint - max(exact 16 wrong-key weighted_joint)`.

The implementation reuses the accepted content-V9 `blind_weighted_scores`, unchanged wrong-key derivation domain and key normalization, and the ordinary-RGB preprocessing route. Paired Gate B, a paired/null image, and an inherited `tau=0` are absent. Correct-key attribution against exact 16 wrong keys is part of `m`; unwatermarked false-positive estimation is a different experiment with its own numerator and fixed denominator.

`detect_watermark(image,key,assets)` accepts only the current ordinary RGB image, the detection key, and frozen public assets. The API and its implementation have no original image, U/G pair, paired null, prompt, seed, embed record, private latent, attack, truth label, outcome, or stored H input. Production detection refuses to run if the generated N_dev=256 threshold asset is absent.

For finite `m_pre > tau_blind`, detection is positive and Geometry is not called. Every finite `m_pre <= tau_blind`, including equality, enters Geometry-Direct; there is no `b_low_blind`. Geometry is called once on the current image. A legal, finite, invertible raw `H_observed_to_canonical` rectifies that same image once. The recovered image is rescored by the same internal scorer, normalized key, preprocessing, public content assets, weighted-joint asset, exact-16 wrong-key derivation, and threshold. Only finite `m_post > tau_blind` is positive; equality is negative. No H, illegal/nonfinite/singular H, Geometry error, rectification error, or post-score error is retained as a fail-closed negative. There is no retry, D4 search, fallback, proxy RGB, or alternate scorer. SyncSeal logit, H legality, and Reliable never vote positive; Reliable remains an optional nonblocking ablation outside the primary method.

`embed_watermark` is a small injected adapter: it invokes the existing content embed once, validates its final ordinary RGB, then invokes frozen SyncSeal final-RGB embedding once. It does not alter the existing content method or the final-RGB order. Real model objects are injected by an execution caller; this implementation and its tests do not load or execute them.

## Development threshold protocol (prepared, not executed)

Before any score is computed, freeze an ordered, source-stratified roster of exactly 256 independent unwatermarked ordinary RGB base images. It must declare disjointness from Geometry-V7 development data, future paper calibration data, and future paper test data. Multiple transforms of one base image cannot count as independent samples. Unit IDs, base-image IDs, source strata, image references, order, and the fixed denominator are hashed into the roster identity.

Each row retains `m_pre` and exactly one complete Geometry-Direct attempt. If a legal raw H is available, the row retains `m_post`; define `z_i=max(m_pre_i,m_post_i)`. A complete fail-closed no-H, invalid-H, Geometry-error, or rectification-error row keeps finite `m_pre` and uses `z_i=m_pre_i`, covering later direct-positive risk. Image I/O, pre/post content scorer interruption, model/GPU interruption, a missing row, identity/order drift, a nonfinite required value, or an incomplete contract blocks threshold creation without deleting the row or changing 256.

Only after all 256 rows validate may the runner set `tau_blind=max_i(z_i)` and serialize it as exact IEEE-754 binary64 big-endian hexadecimal. The runner then replays the full 256 rows under strict `z_i > tau_blind`; anything other than empirical false positives `0/256` blocks the asset. The create-only output path is `configs/blind_detection/assets/blind_detection_v1_thresholds.json`. Because N_dev=256 was not executed in this implementation task, that file is intentionally absent; there is no placeholder and no default threshold.

Wrong-key attribution must be run later as an explicitly separate fixed-denominator experiment. It cannot be pooled with or substituted for primary unwatermarked FPR.

## N=4 image-only callback (prepared, not executed)

The callback notebook consumes a previously frozen production `tau_blind`, fixes N=4, and predeclares coverage of: a direct content positive, a positive only after one raw-H recovery, and an unwatermarked image that remains negative after the full Geometry route. Every detection call supplies only that current image, key, and public assets. The notebook uses a detached exact checkout, GPU guard, a runner-exactly-once marker, and a create-only output. Its first executable cell is exactly the two required Drive mount statements. No callback, Drive write, model, checkpoint, or GPU work was executed here.

## Read-only historical diagnostic boundary

The historical interfaces identify R0/F1 at `4f0bf1560805672f786dc86dd50d793aec18aae7`, R1B repair at `3b9819d80b07704a4caab8b7aaa581cf9eb8a3c5`, R4 engineering replay at `da2daeb9f2ab1e9c2b534b0075fe4daddda33b75`, and the prior R4-Direct callback at `02c40a51b46a35c1163609db2fe3f44a08d476f7`. Their raw result files are Drive paths and were not present in this local worktree or mounted local drives during implementation. Therefore no raw-score range or count is claimed. The included diagnostic command can read supplied `result.json` files without mutation and reports only status plus finite weighted-joint count/range; such output remains an engineering diagnostic, not an independent calibration or paper evidence.

## Stop conditions

Paired/null dependency, a default or inherited zero threshold, a Geometry/logit/Reliable positive vote, a pre/post scorer/key/preprocess/assets/tau change, post-hoc roster membership, deleted failures, retry/fallback, proxy RGB, or truth/stored-H input is `REQUEST_CHANGES / METHOD_DEVIATION`. A future complete real run whose frozen threshold misses the predeclared N=4 recovery gate is `METHOD_FAILED`; the threshold must not be lowered. GPU, checkpoint, dependency, Colab, Drive, or I/O interruption is `OPERATIONAL_BLOCKED`, not method failure.
