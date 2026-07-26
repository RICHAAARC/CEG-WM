---
name: minimal-release
description: Prepare and validate minimal method, GPU experiment-execution, or paper-artifact rebuild packages. Use when extracting, packaging, publishing, or reviewing release contents and reproducibility attachments.
---

# Minimal Release

## Workflow

1. Select the declared extraction profile.
   - `minimal_method_package` for a core-method release candidate.
   - `experiment_execution_package` for Colab or GPU-server execution without governance, notebooks, or paper-building layers.
   - `paper_artifact_rebuild_package` for rebuilding governed paper artifacts from frozen records.
2. Run outer-only `governance/tools/extract_release_package.py` in dry-run mode and inspect copied files, missing paths, safety violations, validation violations, exclusions, and `release_candidate_ready`.
3. Confirm that the package contains only the dependencies required by its purpose.
4. Build into an untracked temporary or release directory.
5. Validate imports, package-owned documentation, lightweight tests, and rebuild commands inside the extracted package.
6. Confirm the package contains no `.agents/`, `.codex/`, `governance/`, or pure governance documentation and no executable delivery code imports `governance`.
7. Include research scripts only from the profile-specific `scripts/experiment_execution/` or `scripts/artifact_rebuild/` path; never copy the extraction tool or the entire scripts root by default.

## Blocking Rules

- Exclude `.codex/`, `.agents/`, `governance/`, `notebooks/`, audit reports, outputs, caches, secrets, private data, and local absolute paths unless a declared profile explicitly requires otherwise.
- Include `third_party/` only through the experiment profile's explicit `--include-third-party` flag and only when every vendored baseline has registered provenance.
- Do not distribute a package whose manifest reports `release_candidate_ready: false`.
- Do not reuse the framework root README as package documentation or accept broken repository-internal links in extracted documentation.
- Do not release placeholder capability as implemented behavior.
- Do not release while required tests or audits fail.

## Required Validation

- Run the extraction-contract tests, default tests, and all audits.
