# Extraction Profiles

## `method_core`

Contains `src/`, lightweight unit tests, package metadata, and a package-local README. It excludes experiments, notebooks, models, outputs, `.codex`, and governance.

## `stage_a_execution`

Contains `method_core` plus `experiments/stage_a`, `configs/stage_a`, and explicitly selected integration tests. It excludes notebooks, models, outputs, `.codex`, and governance.

For both profiles:

- `structurally_valid` means required paths exist and safety scans pass;
- `release_candidate_ready` additionally requires substantive profile-owned implementation and tests;
- extraction never promotes scientific readiness.
