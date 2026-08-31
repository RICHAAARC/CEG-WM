# Geometry-V4 freeze evidence card

## Decision

- Evidence branch: `Geometry-V4-Evidence`
- Source branch: `Geometry-V4`
- Frozen source exact: `12488ad69bd6d2bf8ccc8d0c8d590cfa44bf372b`
- Method: `geometry_v4_keyed_multiscale_sync_anchor_v1`
- Final status: `DECODER_OUTPUT_BASELINE_METHOD_PARTIAL`
- Scientific denominator: `0`
- Merge policy: `do_not_merge_back`

V4 is frozen because the preserved evidence does not show a route that jointly
meets final-RGB observability and safe blind attacked-to-canonical recovery.
This is a method-partial result, not a runtime failure and not a theoretical
proof that keyed geometry is impossible.

## Evidence boundaries

G0 passed 4/4 and proves only that the selected writer placement can leave a
measurable final-RGB trace. The old G1 completed 20/20, but its old gate passed
2/20. Reclassification under the frozen `.02` corner/center, `2 degree`, and
`.03` log-scale tolerances finds 19 correct-key `RELIABLE` outputs, all unsafe,
and 18 wrong-key `RELIABLE` outputs, all unsafe.

G1R corrected the semantics: `RELIABLE` means only that H is safe to apply;
wrong-key reliability is not automatically a classification failure; geometry
never forms a positive. Across six preserved real development artifacts, every
run completed 20/20 with zero unsafe arm but also zero correct safe `RELIABLE`
recovery. Final-RGB observability was respectively 2/4, 2/4, 3/4, 2/4, 1/4,
and 0/4.

The final frozen implementation's deterministic CPU snapshot again completed
20/20 with no unsafe or failed arm, but correct safe recovery remained 0/20.
R/S truth appeared in the fixed top five for 5/20, and identity selected-fit
translation PSR reached 8 for 2/4 carriers. Its status is `CPU_METHOD_PARTIAL`,
and its formal denominator remains zero.

## Failed method families retained

- Legacy joint search/fit evidence: frequent fail-open-like confidence in an
  inaccurate H; closed by G1R semantics, not converted into success.
- Decoder-output placement alone: anchor reached some final RGBs but did not
  produce sufficient spatial fit support.
- Keyed normalized phase search: R/S candidate recall improved to 10/20 while
  fit support remained at most two.
- Diffuse-luma spatial spreading: truth-H fit and holdout remained invalid.
- Opponent-color carrier: source observability fell to 1/4 and truth-H recovery
  remained invalid.
- Sparse Gaussian fiducials: local support rose to 3--7, but source
  observability fell to 0/4 and R/S top-five recall stayed 0/20.
- Balanced bipolar PRN microcode: the final CPU carrier improved only to 5/20
  top-five recall and still yielded 0/20 safe recovery.

All per-unit records and their producer sidecars are under
`evidence/geometry-v4/raw/`. The recomputation source of truth is
`evidence/geometry-v4/derived/route_ledger.json`; narrative text must not
override it.

## Reuse rule

Future research should start by replaying the local verifier and naming a new
mechanism that addresses a retained failure. It must not rerun 6101--6104 or
the G1R development seeds as confirmation, select a favorable subset, relax
the frozen safety/budget gates, or treat geometry evidence as watermark
presence. A materially different writer/detector mechanism should use a new
method identity rather than silently reopening V4.
