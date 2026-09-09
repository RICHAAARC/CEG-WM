# Neighbor and mechanism review, 2026-09-09

Four actual agent instances cover five roles: the interface agent also performs
this sequential neighbor review after the tool declined a fifth instance. This
is not five independent approvals. Sources below were opened in this review.

## What the neighbors establish

| Source and original section | Useful mechanism | Boundary for this implementation |
|---|---|---|
| [AnchorMark](https://arxiv.org/html/2607.27551v1), Rotation Synchrony and AnchorMark sections | Initial-latent central multi-frequency phase anchor; latent matching; payload-confidence local refinement after image correction | Latent anchors and task-guided refinement already exist. Its residual decomposition does not establish small errors for SD3.5 step-18 injection followed by final VAE-only observation. |
| [SyncSeal](https://arxiv.org/html/2509.15208v1), sections 3.1–3.3 | Learned RGB synchronization embedder/extractor; corner supervision and inverse homography before original decoding | Modular synchronization is prior art. A must replace the RGB carrier rather than silently add another carrier. Geometric accuracy alone need not optimize this content detector. |
| [DIFT](https://arxiv.org/html/2306.03881v2), method and correspondence experiments | Diffusion feature maps and nearest-neighbor cosine matching between images | Pairwise correspondence supplies no blind absolute reference frame by itself. Any later use needs a frozen public reference/anchor; the source image cannot enter detection. |
| [Shallow Diffuse](https://arxiv.org/html/2410.21088), sections 2.2–4 | Intermediate-time injection motivated by local linearity and low-rank posterior-mean-prediction Jacobian; continue sampling and invert for detection | The assumptions/theorems are not demonstrated for SD3.5 callback index 18. VAE-only probes and latent energy are not the remaining-generation Jacobian. |
| [T2SMark](https://arxiv.org/html/2510.22366), sections 3.2–3.6 and Appendix A | Tail-truncated initial Gaussian sampling creates projection margins; session keys restore sampling diversity | Tail margin is a future controlled hypothesis here. Initial-noise distribution and projection-AWGN analysis do not automatically describe intermediate latent perturbations or RGB AWGN. |
| [PRC watermark](https://arxiv.org/html/2410.07369), section 3.2 | Pseudorandom error-correcting code signs with Gaussian magnitudes; recover initial randomness; soft detection | Computational indistinguishability depends on its construction and assumptions. This allocator/public anchor inherits neither cryptographic undetectability nor PRC false-positive guarantees. |

Candidate differences are the concrete intermediate-injection/final-observation
combination, fitting synchronization toward residual content tolerance, and fitting
allocation from post-continuation attacked keyed evidence versus perceptual cost.
These are research hypotheses, not established novelty or gains. Moving a known
template into a latent tensor or combining neighboring ideas is insufficient.

## Implementation inspection

Current narrowed scope supersedes the earlier implementation snapshot below:
A uses one writer-reader candidate plus a no-anchor reference on clean and the
exact V2 +10 renderer, with no multi-candidate selection or broad RST search.
B changes HF spatial weights only, fixing original per-image LF allocation,
branch shares, ISS and V2 synchronization/scoring; active conditions are clean,
AWGN .02 and JPEG50. The 264-image / 3528-path envelope is withdrawn. The snapshot
below records what was inspected earlier, not proof of the latest code's effect.

Inspected the A/B method, runtime and development CLI files. A wraps the existing
content injection, adds a public asymmetric anchor at step 18, and omits the old
RGB synchronization embed call. Its image scoring estimates H from current VAE
observation, rectifies once and applies the unchanged content scorer. Template
correlation is descriptive, never positive attribution. The complete max(pre,post)
path needs fresh calibration. Truth H appears only in the separate development
tolerance diagnostic. Any fitted content-target selector must be frozen before
held-out validation and must not use truth or positive labels at detection.

B replays the same seeded sampler independently for each variant, replaces only
the step-18 allocation, and checks that step 19 actually runs. It preserves the
old RGB synchronization carrier and max(pre,post) content score. Four smoothed,
bounded positive macroblock perturbations generate labels relative to uniform
allocation after all nine attacks. Final LPIPS increments enter utility, while
PSNR/SSIM and local distortion are retained. Its features are semantic attention,
texture and latent energy; the third feature is not a measured local response.
The full-support detector does not receive allocation masks or prompt.

The two H conventions are explicitly different: A helpers use reference-to-
observed integer pixel centres; historical V2 adapters use observed-to-canonical
normalized coordinates. Pixel-centre conjugation and one-operation combined
attacks are covered by the shared protocol tests.

The protocol's earlier 264-image total is withdrawn. Actual CLI plans
must derive image, score and oracle counts from selected roster and options.
Review feedback requested an explicit remaining-step check for A and disjoint
oracle/additional-tolerance counts; these are engineering refinements, not new
external execution gates. No real model was run by this review.

The subsequently inspected candidate selector minimized positive-part
oracle-minus-post loss. This can prefer oracle=.2/post=.2 over oracle=3/post=2.8,
even though absolute recovered evidence is much weaker. The latest scope retires
that selection objective; its historical implementation is not evidence of
optimizing recovered detection benefit. Fit-seed isolation and the separate
oracle diagnostic had no identified blind-input leakage.

## Evidence still needed

Real final-image anchor observability, residual transform accuracy inside content
tolerance, and allocation utility generalization remain unmeasured. The selector
may choose identical candidates for both objectives; preserve that outcome. A/B
same-quality comparisons need total final RGB distortion including LF/HF/sync
interactions. Fit/validation development runs cannot support a 0.1% FPR claim.
JPEG, blur, crop-rescale, generative reconstruction, optional feature-assisted
matching, and later tail/ECC variants remain staged work.
