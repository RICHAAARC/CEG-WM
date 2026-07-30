# CEG-WM experiment-execution package

This exact-revision package contains the implemented method, runtime adapters,
governed experiment runner, frozen configurations, infrastructure definitions,
and explicit integration/smoke checks needed by the package-contained
entrypoint.

The default entrypoint performs one deterministic CPU/synthetic development
wiring check. It proves only that the packaged public method callables connect
to the governed A3a runner and record/replay surfaces. It does not execute a
model or GPU, access held-out evaluation data, calibrate thresholds, compare a
baseline, or provide scientific-effect evidence.

Run the package only through the separately distributed schema-v1
`ceg_wm_experiment_execution_bootstrap`. The bootstrap must be verified against
an independently supplied identity, version, and full SHA-256 before it checks
the independently supplied archive SHA-256 and expected revision/config/input
digests. The bootstrap is deliberately absent from this archive.
