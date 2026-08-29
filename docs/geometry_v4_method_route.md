# Geometry-V4 method route (P0)

Geometry-V4 is coordinate recovery only. Its fixed public output is
`(H_hat,corners_hat,support,reliability,status)` and it never votes for
watermark presence. Content and Geometry are sibling domain-separated
HKDF-SHA256 derivations from the same root key, never Geometry from a content
subkey; raw root/key/pattern artifacts are forbidden. Existing content key
bytes, preprocessing, assets, and weighted-joint score remain unchanged.

The detector accepts current attacked ordinary RGB only. Clean/original or
pre-attack RGB, tensors, residuals, true H/corners, and attack parameters are
forbidden from all detector roles; true transforms are evaluation-only after
output freeze. H is finite row-major attacked-to-canonical, h[8]=1; corners
are attacked TL,TR,BR,BL. P1 fits similarity and H0 model class is predeclared.

Global anchors are keyed asymmetric 8/16/32 cycles/image and 0/45/90/135
degrees. Local anchors are a fixed 4x4 row-major grid IDs 0..15 at centers
{.125,.375,.625,.875}², side .25, fixed 2x2 macro regions. Adaptation only
multiplies amplitude in [.75,1.25], then uses one normalization. RGB luma caps
are RMS 2/255 and peak 8/255 with .40/.60 global/local energy; excess STOPPED.

P1D and disjoint P1C each use 8 sources x16 fixed attacks (128 positives) and
matching unwatermarked negatives. Wrong key is one same-unit control, never a
denominator. No replacement, retry, fallback, P1C tuning, or reselection.
Reliability is fail-closed on finite PSR, support, inlier/coverage, reprojection
RMS, condition, cross-scale spreads, and valid corners; RELIABLE needs all
gates and aggregate >.5.

Only F0 jointly freezes tau/delta for the whole negative roster. s0 is attacked
RGB with the unchanged content path; s1 is eligible only for RELIABLE geometry
and s0 in [tau-delta,tau), using the exact same content path/tau. Positive is
`s0>=tau OR (eligible AND s1>=tau)`; Geometry never votes. G0 is only step-19
final-latent-before-decode, four fixed canaries, one budget, and final-RGB
observability (anchor, PSNR>40, SSIM>.98, caps, drift<.05) or STOPPED. P0 is
local/static engineering evidence with science denominator 0.
