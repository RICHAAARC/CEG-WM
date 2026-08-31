# Geometry-V6 R0.1 fixed carrier candidate

R0.1 is an independent four-unit, fixed-denominator diagnostic. It reuses the first four ordered, public novel-seed rows from `content_chain_novel_seed_stability.jsonl` (SHA-256 `33613cb24de87c86a573ac0dda80523912e001c922494051f5d89a9e2851831b`), whose fixed content-chain runtime binds the same SD3.5-medium model. These rows do not include the R0 prompt/seed.

For each unit and each fixed R0 amplitude, both matched pairs must have all four public raw pilot deltas (`search`, `fit`, `validate`, `aggregate`) strictly greater than zero. Both matched RGB pairs must also have PSNR strictly greater than 40 dB and SSIM strictly greater than 0.98, using Geometry-V4's fixed RGB scalar semantics. Missing, failed, non-finite, tied, or non-positive values fail closed; no unit, arm, split, or amplitude may be replaced or removed.

All four units must pass all fixed requirements at a given amplitude, including the unchanged content correct-key/16-wrong-key/same-unit null rule, before that amplitude passes. All three amplitudes must pass to emit `CARRIER_WINDOW_CANDIDATE`. This remains a carrier-window candidate only: `science_denominator=0`, no formal FPR, robustness, or scientific-success claim.
