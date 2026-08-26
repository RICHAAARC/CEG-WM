# Content-V8 Drive-first formal rerun

This directory preserves the public runtime asset, scalar result, receipt, and
immutable archive metadata from the real Drive-first runner invocation at
execution exact `c9be80952ef6b23627f5ff45411addc955316950`.

- Run UTC: `20260826T040739Z`
- Unit sets: `[32V1]` development, independent `[8V1]` and `[8V3]` evaluations
- Run ID: `content-v8-c9be80952ef6-94673c645cf3`
- Drive run root: `/content/drive/MyDrive/CEG-WM/Content/Content-V8-c9be80952ef6-20260826T040739Z`
- ZIP SHA-256: `1f73303a12a8e15e1160b796b284f4462b53ad212ec7cc5438ec4bf1432caf7c`
- ZIP CRC, outer sidecar, and runtime-asset sidecar bindings: pass
- Evaluation results: two independent 8-unit/16-record results, 0 failed units, runner RC0

The Notebook emitted `artifact_pair_validation` only because its final cell
looked up a stale fixed run ID. The runner-created ZIP/SHA pair existed under
the dynamic run ID above and independently passed validation. This operational
handoff false negative was fixed later at Evidence exact
`062c6b0687d7d81c5ec1b39d2be223384959de40`; it does not rewrite the original
Notebook output into a successful handoff.

After excluding execution provenance, both scalar evaluation results are
identical to the prior Content-V8 evidence. The public runtime asset differs
only in `producer_exact`. The prior narrow personal adjudication remains
unchanged.
