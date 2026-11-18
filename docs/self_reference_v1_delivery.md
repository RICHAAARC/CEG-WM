# Self-reference v1 engineering delivery — 2026-11-18

Scope: independent local branch `research/self-reference-relation-v1`, based on
`f9740b03d68064e92ba8549913bee7483948d3d0`. Only the new research module, frozen
configuration, CPU tests and two documents are included. No existing research
implementation, test, LF/HF path, runtime observation or project contract is edited.
The precise delivery commit is reported to the audit session after committing.

Entry point: `PYTHONPATH=src python -m cegwm.self_reference`. The implementation
supports plan, cache validation, train/resume, checkpoint selection, dev/test
evaluation and fixed-host four-write diagnostics. All actions default to plan-only
unless `--execute` is supplied. See `docs/self_reference_v1.md` for the complete
cache contract, commands, fixed configuration, gates, limits and audit checklist.

Frozen configuration hash:
`6c74bd18231e18f56b4dd6e717804624e5f78929511645e108a4d8971c866495`.
DCT order and attacks are included explicitly in the JSON. Capacity excluding VAE:
B0=187824 parameters, B1=C=188465 (approximately 0.34% difference).

## Validation actually performed

Environment: WSL `Ubuntu-24.04`, Python executable
`/home/richar/projects/CEG-WM/alive/CEG-WM/.venv/bin/python`, torch `2.5.1+cpu`.
Commands below run in `/home/richar/projects/CEG-WM/alive/CEG-WM`.

| Command | Observed result |
|---|---|
| `PYTHONPATH=src .venv/bin/python -m cegwm.self_reference plan --config configs/self_reference_v1.json` | Passed; no VAE/model load |
| `.venv/bin/python -m pytest tests/unit/test_self_reference_v1.py tests/unit/test_final_image_observation.py tests/unit/test_lf_method.py tests/unit/test_content_chain_protocol.py tests/unit/test_shared_keys_prg.py -q` | **66 passed**, 2.30s; includes all **24 new tests** |
| `.venv/bin/python -m pytest tests/unit -q` | **174 passed, 2 failed, 16 deselected**, 4.92s; failures reproduced at base commit |
| `.venv/bin/python -m pytest -q` | Existing collection failure: unit/integration modules share four basenames |
| `.venv/bin/python -m pytest --import-mode=importlib -q` | Existing collection failure: `test_paper_v2.py` imports bare `test_continuous_corners` |

The two unit failures are existing bit-exact golden-hash mismatches, not repaired
or rebaselined in this candidate:

* `test_content_whitening_detector_matches_independent_oracle_golden_and_real_boundary`:
  expected `ca91ddfa70e2f675f8d5aa994c1c05bb13528e2c5ffa9281131b8a597a0e8453`,
  observed `103792327c6673257720f98319df67d81285d354ee509d7272d52b7fcffe5de7`.
* `test_production_carrier_is_repeatable_and_key_dependent`:
  expected `5d35e543e3890fc5dedae6e31bd8ac4efd049430abbe3ed065a7534b5d34759b`,
  observed `9d1fb0e26120c2db7941eb814421f67af4ab29a03f4a42e436629849880568dc`.

Both observed hashes were reproduced using the same interpreter against a
`git archive f9740b0` extraction of `src`, `tests`, `configs`, and `pyproject.toml`
in `/tmp/cegwm-baseline-6i9xrxnk`. This temporary baseline is not a Git worktree.
The first whitening reproduction lacked configs; after extracting the base-commit
configs, its original golden-hash failure reproduced exactly. No cause beyond
pre-existence in this environment is claimed.

CPU evidence covers actual fake-VAE CLI training for all arms, interruption and
resume, checkpoint selection, dev/test evaluation, diagnostics, model/optimizer/
RNG/loss equality with uninterrupted training, and matching shared input schedules.
Spies establish that selection reads only model_selection and a failed A gate
does not call any nonzero-shift model measurement. A deliberately injected test
gate exercises B wiring without promoting the fake fixture to evidence. NaN and
bad clean-cache outputs retain all failed units. Missing/failed LPIPS never gives
research PASS; per-image metric failures preserve A measurements while quality
is not_evaluable. Tests also cover blind-reader input isolation, locality, reference
sampling, correlation degeneracy/ties, exact pixel shifts, latent supervision
coordinates, split leakage, file identity and unequal training-budget rejection.
CUDA training requires `CUBLAS_WORKSPACE_CONFIG=:4096:8` before process startup;
CPU spies verify missing or incorrect values fail before model loading. This is
a preflight check, not a successful CUDA execution claim.

## User-owned changes preserved

Neither file below is staged or included in the delivery commit. Before/after
hashes are identical:

* `README.md` — `ff2fe25f0348c510b16799008d79698ad04ffff65d8bd67176a147867130d114`
* `.codex/skills/ceg-wm-five-agent-gated-delivery/SKILL.md` —
  `09c96c7dfd8f3917a27bb6d44938887b13f394d027740551199193b1dd416adc`

## What was not run or established

No true SD3.5 generation, model download, GPU job, real training, real cache
evaluation, LPIPS weight download, push or merge was performed. A trusted real
cache and complete local LPIPS weights are not supplied by this delivery. GPU
memory, compute cost, real processor/VAE behavior, writeability, key distinction,
translation benefit and quality matching remain unmeasured. Only CPU engineering
behavior is validated. Approximate capacity matching does not establish equal
optimization difficulty or equal real cost. Self-reference novelty, implicit
shortcut exclusion, low FPR and scientific success are not claimed.
