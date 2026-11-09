# Single-source one-shot precompensation validation

The previous diagnostic found the same suffix tensors in NG1, NG2 and G, while
all differed slightly from the historical full-generation RGB. The user adopted
method-first validation on 2026-11-09. This entry executes source05 (the first
predeclared source), keeping the other three roster entries explicitly unmeasured.

## Coherent basepoint and actual writes

Reuse the saved latent, conditioning, scheduler and keyed reader. Produce a new
zero image with `replay_suffix_write`; read its ordinary RGB and recompute both
clean and fixed-design-reconstruction RGB sensitivities there. Historical zero
images and all eight old sensitivities remain retained in the original directory,
but are not substituted for observations at this new basepoint.

Two latent VJPs share one differentiable suffix forward. Its reference is the
new zero image, so cotangents and baseline scores correspond to the same image.
The historical PNG mismatch is recorded only. This current-basepoint check does
not establish an exact derivative of quantized RGB: these are proxy gradients.

The existing solver and matched-information gradient-ray control each solve once
for bit0 and bit1, with unchanged nearest cells, reader, actual-dtype budget and
quality reporting only. Each solved write runs the identical restored suffix
from captured latent + delta. No complete prefix replay, tuning, cell search,
extra probe or response-based selection is performed. Original full-generation
results are not overwritten or described as equivalent to this suffix mapping.

Ordinary design-zero and evaluation reconstructions share the resident frozen
execution policy and use no_grad. Ordinary blind reads of clean/design/holdout
RGB are the method outcome. For each
write report predicted and observed clean/design score changes, signed residuals,
residuals in fixed cell units, sign opposition and response ratio. Use the actual
post-cast delta in predictions. These observations assess finite-budget behavior;
they neither prove exact gradients nor constitute an independent validation set.

## Fixed maximum new work

| Work | Maximum |
|---|---:|
| Full generation | 0 |
| Suffix forward / final denoising step / decode | 6 each |
| New unmarked / marked source RGB | 1 / 4 |
| RGB / latent VJP | 2 / 2 |
| Ordinary SDXL reconstruction | 10 |
| Additional differentiable design reconstruction | 1 |
| Public RGB feature reads | 15 |

Six suffixes are zero1 + VJP1 + four writes. The 10 ordinary reconstructions are
two zero references and two seeds per write. Component counters overlap: a design
RGB VJP includes the SDXL backward and must not be counted twice as autograd.
Failures and unsolved arms can reduce work but never remove the two bit trials per
method. An intentional no-op for one bit cannot establish both-bit control.

## Run and output

Notebook: `notebooks/channel_precompensation_single_source_v1_colab.ipynb`.
First code cell only mounts Drive; all source is embedded, with no manual upload.
Input is the existing `precompensation-20261108/continuation_suffix_device_v1`.
Output is the sibling `single_source_method_v1`; an existing directory is read-only.
Model execution is user-run on Colab with GPU and existing HF_TOKEN access.

The result ZIP includes new zero/marked/reconstructed RGB, ordinary features,
new RGB and latent sensitivities, requested/actual writes, blind reads and
response assessments, counts, source snapshot and startup logs. The existing
three-arm diagnostic ledger is added to prior counts when its completed sibling
summary matches the capture checkpoint ancestry; otherwise count scope explicitly
remains the capture checkpoint. A missing accounting file does not block science.

Failures identify preparation/acquisition, unsolved fixed constraints,
actual-dtype collapse or budget excess, finite-budget prediction miss, or holdout
reconstruction target miss. There is no automatic strength adjustment or retry.
A current-basepoint mismatch cannot be hidden as a valid gradient.

CPU fake-model checks verify lifecycle, caching, actual perturbations, solving,
blind reporting and packaging. They are not GPU execution or scientific success.
Even successful two-bit control on this source does not establish four-source
completion, watermark presence, attribution, FPR or general reconstruction robustness.
