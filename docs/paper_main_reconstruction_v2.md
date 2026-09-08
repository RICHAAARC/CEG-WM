# paper-main-reconstruction-v2

This is the umbrella experiment ID for all paper results. It is not a rename
of only the old reconstruction worker. Code is delivered from main; each worker
keeps a separate output directory below `CEG-WM/paper-main-reconstruction-v2`.

## Coverage inherited from V1

| Work | V2 coverage | Output |
|---|---|---|
| Main + T2SMark, Tree-Ring, Gaussian Shading, Shallow Diffuse | Each: 2000 clean calibration, 3000 independent clean negatives, 1000 pairs × six conditions × two roles | Five method_final.json files |
| Main ablations | First 100 pairs, four variants, two conditions and two roles each: 1600 records; 300 new marked images | Main ablations |
| Quality | Each method's 1000 existing evaluation pairs: PSNR/SSIM/LPIPS | Five quality summaries |
| Reconstruction | Main first 100 pairs × two roles, SDXL strength .3 / 20 steps / empty prompt | reconstruction_final.json |
| Finalization | All five methods and reconstruction, including failures | Unified JSON, 60-row main CSV, full binary and quality CSVs, descriptive PNG/PDF figures |

The plan contains 86800 detection records, excluding preflight, retries and the
multiple content scores inside an individual detection. The original V1 exporter
provided JSON and the 60-row CSV; additional tables/figures format those same
results without adding observations.

The six conditions remain clean, JPEG50, resize .5 with restoration, .8-area center
crop with restoration, blur sigma 1 and rotation +10. Only rotation changes:
`rotation_10_bilinear_black_fixed_canvas_v2` uses the validated R1A inverse matrix,
Pillow perspective sampling, bilinear interpolation and black fill on 512×512.
All five methods use this same renderer. V1 reflect/bicubic remains available
under its original condition and in the old branch/results; no new reflect run
is included in this matrix.

## Method and independence

Main uses current registered-minus-max16wrong content evidence, observation reuse,
pre early return, otherwise one continuous predicted H and same-scorer post.
Geometry never supplies positive evidence. Calibration remains max(pre,post),
2000 clean negatives, rank 1998, strict greater-than; each method gets its own
new threshold. No global search is included. Old tau is not used for V2 formal
decisions. FPR/CI are report-only, including adverse results.

V2 uses the same prompt corpus and seed starts shifted by 1000000:
2028010000 / 2028020000 / 2028030000, with `v2-` unit IDs. These are new generated
samples, not a new prompt distribution. No selection is made from observed scores.

## Operation

`python -m experiments.run_paper_v2 --worker main --mode preflight
--drive-root /content/drive/MyDrive/CEG-WM --runtime-root /content/paper-v2-runtime`

Select one worker per runtime: main, t2smark, tree_ring, gaussian_shading,
shallow_diffuse, reconstruction or finalize. Default mode is preflight. Main and
baseline preflight generate one pair and exercise all six conditions. Main
explicitly evaluates continuous geometry/post even if production would early-return,
and compares original/reused branches once. Preflight has science_denominator=0;
there is no detection-rate pass threshold. Reconstruction reuses its small existing
canary. GPU/version are recorded, with no A100 requirement.

Formal main and four baseline workers may run independently. Reconstruction needs
main's new threshold and first 100 evaluation pairs. Finalize waits for all six
result sources. The notebook selects only one worker and one mode; Run all must
never silently start the entire matrix. Code revision is recorded, not a clean
checkout/producer-equality gate. Existing output identities and failure rows are
retained to prevent mixing jobs. Main's untracked `.codex/skills` is untouched.
