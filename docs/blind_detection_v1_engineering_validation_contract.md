# BlindDetection-V1 Engineering Validation Contract

## Claim ceiling

This route is prepared but not executed. Any later result is an engineering observation only, not paper-ready evidence, an FPR generalization, a reliability claim, or a robustness conclusion. `science_denominator=0` remains fixed. There is no performance success criterion.

## Frozen inputs and rosters

`configs/blind_detection/blind_detection_v1_engineering_validation.json` freezes two ordered 32-prompt Git sources. The positive denominator is exactly 64 distinct prompt-seed base units: clean, `core_rotation_pos15`, `core_fixed_canvas_zoom_0_8`, and the exact chain rotation then scale each contain 16 units, balanced 8+8 across the two sources. Positive seeds are `2026111000` through `2026111003` in that stratum order. The negative denominator is exactly 256 distinct plain SD3.5 units: all 64 prompts under each seed `2026112000` through `2026112003`, seed-major. The separate canary uses the first development-source prompt with seed `2026110999`.

The loader expands every unit before formal work and checks order, counts, attack/source balance, logical uniqueness, positive/negative/canary separation, and disjointness from BlindDetection N_dev=256, callback N4, and the declared Geometry-V7 development/evaluation pairs. These Stage3 pairs are reserved away from future paper calibration and test. No failed unit may be replaced, retried, resumed, dropped, or moved between denominators.

## Runtime and blind boundary

The canary runs once before formal roster allocation. It may check only imports, real public runtime construction, device, one plain 512x512 ordinary-RGB generation, and one typed real Geometry invocation. It never calls watermark scoring or detection, has no method conclusion, and contributes to no numerator or denominator.

After a passing canary, every positive uses the existing real content embed followed by final-RGB SyncSeal exactly once. Every negative uses plain SD3.5. Frozen R1A attacks are applied only to positive current RGB before detection. Preparation saves only create-once current candidate RGBs; generation requests, primary nulls, originals, and attack context do not cross the blind helper. Detection reloads one current RGB at a time and calls `detect_watermark(current_rgb, detection_key, public_assets)` through a three-argument helper. Public assets use the repository threshold `configs/blind_detection/assets/blind_detection_v1_thresholds.json`, with frozen tau `1.1328391433063743` (`3ff2201bf0021293`); Drive is not a threshold input. Content statistics remain the sole positive authority, Geometry supplies coordinates once, and the same key, scorer, preprocessing, assets, and strict `m > tau` rule apply before and after rectification. Equality is negative. The separate wrong-key experiment is excluded; the exact-16 wrong-key terms intrinsic to `m` remain unchanged.

## Result contract

The result fixes positive denominator 64 and negative denominator 256 and retains every allocated row and operational failure. Any incomplete formal row yields `OPERATIONAL_BLOCKED` without shrinking either denominator. If all rows are complete, status is the neutral `ENGINEERING_VALIDATION_COMPLETE` regardless of measured performance. Metrics report positive numerators overall and by the four attack strata, full-route negative false-positive numerator over 256, direct/recovered and other actual route counts, operational-incomplete counts, retries `0`, wrong-key experiment `excluded`, the engineering-only ceiling, and `science_denominator=0`.

The dedicated Colab notebook will bind a detached producer exact, use only `CEG_WM_ROOT_KEY` and `HF_TOKEN`, invoke the formal runner once, and publish one unique create-only terminal ZIP to Drive. Git and public model locations are inputs; Drive is output-only. The ZIP retains public config, result, positive rows, negative rows, canary information, logs, and current candidate RGBs when available. It excludes keys, private latents, originals, primary nulls, stored H, manifests, receipts, signatures, sidecars, and checksum or byte-size gates. Its final cell is read-only and never reruns the method.
