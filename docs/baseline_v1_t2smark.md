# Baseline-V1 T2SMark SD3.5 interface

This module follows the official T2SMark SD3.5 path in
`external_sources/t2smark/run_sd35.py`, `src/t2s.py`, and
`src/inversion/inverse_diffusion3.py`. Its defaults are key length 16, message
length 256, `tau=0.674`, latent shape `(1,16,64,64)`, key channels `(0,1,2,3)`,
and the other twelve channels for the message. For each partition,
`r=int(2*NormalCDF(-tau)*n/m)`. The seeded support and sign vector, repeated
bit codeword, largest-magnitude tail placement, smallest-magnitude central
placement, and decode `p.reshape(r,m).sum(0)` follow the official codec;
`norm1_w = L1(p)` is the native continuous score and is higher for watermarked
images.

`embed_t2smark_sd35` accepts a caller-supplied SD3.5 base latent and clones it
before writing the key and message partitions. Supplying the partition slices as
`base_noise` retains a clean/watermarked pair from the same base latent without
altering tail-truncated sampling. The clean pair is the original, unmodified
base latent; this interface does not generate an image.

`score_t2smark_rgb` is blind at the image boundary: it accepts only H×W×3
uint8 RGB, an official-compatible SD3.5 inversion pipeline, master detection
key bits, and inversion steps (default 10, matching `run_sd35.py`). It applies
RGB `/255`, maps to `[-1,1]` BCHW, calls `get_image_latents(..., sample=False)`,
then `naive_forward_diffusion(..., num_inference_steps=10)`, and decodes the
reversed key channels. It returns only the finite native continuous `norm1_w`;
there is no threshold decision or synthetic fallback.

SLM-WM is only an engineering reference for passing an explicit base-noise
partition through the same official mechanism. This code has not run a real GPU
model. CPU fixtures check formula and interface wiring only, with
`science_denominator=0`; they are not robustness or scientific results. The
next authorized execution node is a Colab one-physical-unit clean/watermarked
pair and clean plus five-attack real-score canary.
