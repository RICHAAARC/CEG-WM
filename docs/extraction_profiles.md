# Extraction Profiles

## `content_chain_execution`

Contains the research package, content runners, `configs/content_chain`, unit tests, and content integration tests. It excludes notebooks, models, outputs, `.codex`, and governance.

For this profile:

- `structurally_valid` means required paths exist and safety scans pass;
- `release_candidate_ready` additionally requires substantive profile-owned implementation and tests;
- extraction never promotes scientific readiness.
