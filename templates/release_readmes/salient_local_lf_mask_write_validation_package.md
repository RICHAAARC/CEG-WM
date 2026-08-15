# CEG-WM salient-local-LF mask/write validation package

This exact-revision package runs two operational preflights followed by eight
preregistered scientific mask/write observations, each with one attempt. It
requires `HF_TOKEN`, `CEG_WM_ROOT_KEY`, and an explicit
`CEG_WM_INSPYRENET_CHECKPOINT_PATH` bound to the frozen checkpoint identity.

It does not fit whitening, run a detector, execute Q/K, calibrate a threshold
or FPR, or promote a candidate. Passing only permits requesting an independent
32-clean masked-LF null fit.
