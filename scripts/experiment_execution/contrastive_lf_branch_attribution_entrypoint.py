"""One production Stage-A null-fit plus candidate-selection entrypoint."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Callable, Sequence

from experiments.protocol.contrastive_lf_branch_attribution import (
    MANIFEST_PATHS,
    NULL_FIT_ROLE,
    SELECTION_ROLE,
    load_manifest,
)
from experiments.runners.contrastive_lf_branch_attribution import (
    StageAOperations,
    create_adapter_backed_stage_a_operations,
    execute_stage_a_null_fit_and_selection,
)
from scripts.experiment_execution.contrastive_lf_branch_attribution_server import (
    finalize_contrastive_lf_delivery,
    finalize_contrastive_lf_preexecution_failure,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def execute_contrastive_lf_entrypoint(
    *,
    observed_repository_revision: str,
    run_id: str,
    output_root: str | Path,
    operations_factory: Callable[[], StageAOperations],
) -> tuple[int, dict[str, object]]:
    operations: StageAOperations | None = None
    result = None
    try:
        null_manifest = load_manifest(
            PACKAGE_ROOT / MANIFEST_PATHS[NULL_FIT_ROLE], expected_role=NULL_FIT_ROLE
        )
        selection_manifest = load_manifest(
            PACKAGE_ROOT / MANIFEST_PATHS[SELECTION_ROLE], expected_role=SELECTION_ROLE
        )
        operations = operations_factory()
        result = execute_stage_a_null_fit_and_selection(
            null_manifest, selection_manifest, operations
        )
        try:
            operations.close()
        except Exception as exc:
            result = replace(
                result,
                selection_result=None,
                result_classification="operational_failure",
                failure_reason=type(exc).__name__[:120],
            )
        operations = None
        return finalize_contrastive_lf_delivery(
            result,
            observed_repository_revision=observed_repository_revision,
            run_id=run_id,
            output_root=output_root,
        )
    except Exception as exc:
        if operations is not None:
            try:
                operations.close()
            except Exception:
                pass
        return finalize_contrastive_lf_preexecution_failure(
            observed_repository_revision=observed_repository_revision,
            run_id=run_id,
            output_root=output_root,
            failure_reason=type(exc).__name__[:120],
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--observed-repository-revision", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True)
    arguments = parser.parse_args(argv)
    if not arguments.execute:
        parser.error("--execute is required")
    code, receipt = execute_contrastive_lf_entrypoint(
        observed_repository_revision=arguments.observed_repository_revision,
        run_id=arguments.run_id,
        output_root=arguments.output_root,
        operations_factory=lambda: create_adapter_backed_stage_a_operations(
            implementation_revision=arguments.observed_repository_revision
        ),
    )
    print(json.dumps(receipt, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
