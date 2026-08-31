# Geometry-V4 Evidence

This is the immutable, local-recomputable record of why
`geometry_v4_keyed_multiscale_sync_anchor_v1` was frozen at source exact
`12488ad69bd6d2bf8ccc8d0c8d590cfa44bf372b` with status
`DECODER_OUTPUT_BASELINE_METHOD_PARTIAL`.

It is an evidence branch, not a method branch. Do not merge it back into
`Geometry-V4`, do not delete failed units, and do not use any geometry score as
positive watermark evidence. Geometry remains coordinate-only. The evidence
ceiling is engineering and method-development evidence; the synthetic CPU
snapshot has `formal_denominator=0`.

## What is preserved

- Exact Drive bytes and original SHA-256 sidecars for the passing 4-unit G0,
  the failed 20-unit G1, and all six 20-unit G1R development artifacts.
- All per-unit correct-key, wrong-key, and unwatermarked-negative arms,
  including failures, blind outputs, private engineering diagnostics, and
  post-freeze truth probes when the producer recorded them.
- A fresh deterministic 4x5 CPU snapshot from the final frozen exact, with all
  twenty per-unit records and `formal_denominator=0`.
- A machine-readable route ledger, freeze decision, Drive provenance, and file
  index under `derived/` and `index.json`.
- A stdlib-only verifier/recalculator. Only the optional `--capture-cpu` mode
  imports NumPy/Torch/Pillow from the frozen repository.

## Recomputed route outcome

| Route | Exact | Final-RGB observable sources | Correct safe `RELIABLE` | Key diagnostic |
|---|---|---:|---:|---|
| G0 observability | `5b29a27` | 4/4 | n/a | minimal writer-to-RGB chain only |
| legacy G1 | `22db26d` | 2/4 | 0/20 | old gate 2/20; 19/19 correct reliable H were unsafe |
| initial G1R | `331e565` | 2/4 | 0/20 | no retained fit/search diagnostics |
| decoder-output writer | `7b8649a` | 2/4 | 0/20 | selected fit support 0/20 |
| keyed phase search | `c5ed4f1` | 3/4 | 0/20 | R/S top-5 10/20, fit support at most 2 |
| diffuse-luma spread | `a3877d6` | 2/4 | 0/20 | truth-H fit support at most 2; holdout 0/20 |
| opponent-color carrier | `40261d2` | 1/4 | 0/20 | truth-H fit/holdout 0/20 |
| sparse Gaussian atlas | `c0a7f9b` | 0/4 | 0/20 | support improved, but R/S top-5 0/20 and truth PSR max 2.679 |
| balanced bipolar PRN CPU | `12488ad` | not a GPU test | 0/20 | R/S top-5 5/20; identity translation PSR >=8 only 2/4 |

The important negative result is joint, not a single threshold miss. G0 showed
that an anchor can reach final RGB. Later variants separately improved search
or local support, but none simultaneously achieved the frozen writer
observability requirement and safe blind attacked-to-canonical recovery.
Changing placement, normalized phase search, spatial spread, color axis,
sparsity, and balanced PRN microcode did not close that joint gap under the
unchanged budget and gates.

## Local verification and recomputation

From the repository root, verify every imported sidecar and every committed
derived file without model, network, GPU, Colab, or Drive access:

```bash
python3 evidence/geometry-v4/scripts/recompute_geometry_v4_evidence.py
```

Rebuild the derived ledger and index from the preserved raw bytes:

```bash
python3 evidence/geometry-v4/scripts/recompute_geometry_v4_evidence.py --write-derived
```

The optional final-exact CPU recapture requires the repository's existing
environment and must still be described as synthetic-only:

```bash
PYTHONPATH=src:. /path/to/project-python \
  evidence/geometry-v4/scripts/recompute_geometry_v4_evidence.py --capture-cpu
```

## How a future route should use this package

1. Recompute the ledger locally before proposing a new carrier or detector.
2. State which preserved failure mechanism the new idea changes: final-RGB
   observability, R/S proposal, spatial fit, or independent holdout safety.
3. Compare against every retained route, not a selected successful subset.
4. Use new seeds and a new method identity if the mechanism changes materially.
5. Keep the content detector, content key, preprocessing, and positive threshold
   outside geometry; a safe H may permit correction but can never add a vote.
