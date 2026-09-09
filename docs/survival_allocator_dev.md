# HF survival allocation development

The allocator comparison fixes an actual per-image LF reference and changes HF
spatial weights and its budget scale. Whitening, RGB SyncSeal and complete V2
registered-minus-max16wrong pre/post scoring remain unchanged. Original V2 is a
historical system reference; it retains its existing joint ISS projection.

For uniform, probes and survival, an original-allocation/ISS reference scale is
computed once. Freeze L=cast(base+a_ref*LF) and its actual component delta_L=L-base.
The new development-only solver forms Y=cast(L+b*HF); only b changes, and its
actual HF component is Y-L. The total 0.012 budget is checked on Y-base, including
the LF/HF cross term. The same L is reused for every allocator variant. Over-budget
LF, mismatched replay base or failure to find nonzero feasible HF remains an error
row. Quantized search does not claim a globally maximal amplitude or impossibility.
This fixes the injected LF reference component, not the final latent's LF spectral
projection: spatial weighting and quantization can add low-frequency energy.
The historical production ISS path is not changed to use this solver.

The three comparison arms are original HF, uniform HF and one learned HF rule.
Original uses the unchanged nearest interpolation. Uniform, four macroblock
probes and the fitted candidate use bilinear interpolation at the actual latent
resolution. Uniform is identical under either interpolation and is the shared
label reference. Uniform versus survival is the only allocator-effect comparison
with the same fixed LF reference and smoother. Comparisons with original include
both projection and smoothing differences and are labeled historical-system
comparisons. The new weights are positive, bounded, unit-mean and spatially smooth.
This does not establish that their final images have matched quality.

A low-capacity ridge fit maps 2x2 DINO saliency, texture and latent-energy features
to bounded allocation logits. Energy is a simplified candidate feature, not a
perturbation response or generation Jacobian. Features and labels are embedding
or development inputs only; none enter blind detection. No feature expansion,
tail coding, PRC, LF/HF reweighting or injection-time optimization is included.

For each fit pair, four separate HF probes replay the same generation seed to
step 18 and execute remaining step 19. They then undergo final RGB synthesis,
the unchanged RGB synchronization and clean, AWGN sigma .02 in RGB [0,1] with
clipping, and JPEG50. Original allocation is computed once per image and reused.
Its old decode/encode probes are features, never substitutes for continuation.

Each label stores complete pre/post registered and wrong-key scores for the probe,
uniform reference and unwatermarked image. Utility is mean complete-path margin
increment minus a fixed coefficient (default 1) times final LPIPS increment.
The coefficient has score/LPIPS units. Final PSNR/SSIM/LPIPS and four local MSEs
include content/synchronization interaction; no quality is inferred from latent L2.

Validation applies a frozen fit asset to independent IDs/seeds, never refits using
the validation attacks, and reports paired positive-negative separation, negative
q95/max, per-key components and paired quality differences. A quality comparison
flag is descriptive only. Poor scores are retained; uniform may win. No same-quality
gain, .1% FPR, robustness or innovation result exists until real measurements.

From this worktree, using its installed package or `PYTHONPATH=src`:

```bash
PYTHONPATH=src python -m experiments.run_survival_allocator_dev --mode plan --roster configs/parallel_method_dev/fit.json --output unused
PYTHONPATH=src python -m experiments.run_survival_allocator_dev --mode fit --roster configs/parallel_method_dev/fit.json --output fit-output --runtime-root runtime
PYTHONPATH=src python -m experiments.run_survival_allocator_dev --mode validation --roster configs/parallel_method_dev/validation.json --output validation-output --runtime-root runtime --allocator fit-output/allocator.json
```

Fit8: 56 images, 32 HF macroblock continuations and 168 score paths. Validation24:
96 images and 288 paths. Total: 152 images and 456 paths; no optional second
amplitude is scheduled. All marked variants replay the 20-step prefix for simple
scheduler isolation, so 32 labels do not mean only 32 single denoising steps.
Quality, original allocator probes and ISS host scores are additional internal work.
The old 264-image/3528-path plan is withdrawn. Model execution is separate from
code preparation; HF_TOKEN and CEG_WM_ROOT_KEY are needed only for fit/validation.

History is a constraint: Content V2-V8 LF wrong-key competition was uneven, while
V9 joint clean attribution passed its four strata. Content-Curve already measured
AWGN and HF tail carriers. This candidate therefore asks only whether HF spatial
allocation adds independent survival at comparable final quality. If it does not,
retain the outcome and stop adding features. Crop and reconstruction remain later
research questions. No real model execution has occurred in this implementation.
