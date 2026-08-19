"""Soft-route selection entrypoint for one literal fixed attempt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Sequence

from experiments.protocol.semantic_texture_soft_route_mechanism_validation import (
    SELECTION_ROLE,
    load_soft_route_mechanism_configuration,
    load_manifest,
)
from experiments.runners.semantic_texture_soft_route_mechanism_validation import (
    SoftRouteMechanismOperations,
    SoftRouteMechanismSplitResult,
    create_adapter_backed_soft_route_mechanism_operations,
    execute_soft_route_mechanism_split,
)
from scripts.experiment_execution.semantic_texture_soft_route_candidate_selection_server import (
    finalize_soft_route_mechanism_candidate_selection_delivery,
    finalize_soft_route_mechanism_failure_delivery,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PACKAGE_ROOT / "configs/experiments/semantic_texture_soft_route_mechanism_validation.json"


class SoftRouteMechanismEntrypointError(RuntimeError):
    """The declared soft-route mechanism validation selection invocation is incomplete or invalid."""


def execute_soft_route_mechanism_candidate_selection_entrypoint(
    *,
    observed_repository_revision: str,
    run_id: str,
    output_root: str | Path,
    operations_factory: Callable[[], SoftRouteMechanismOperations],
) -> tuple[int, dict[str, object]]:
    """Run exactly the literal selection matrix through injected public operations."""

    operations: SoftRouteMechanismOperations | None = None
    try:
        configuration = load_soft_route_mechanism_configuration(CONFIG_PATH)
        manifest = load_manifest(
            PACKAGE_ROOT / configuration["candidate_selection_manifest_path"],
            expected_role=SELECTION_ROLE,
        )
        operations = operations_factory()
        result: SoftRouteMechanismSplitResult = execute_soft_route_mechanism_split(manifest, operations)
        operations.close()
        operations = None
        return finalize_soft_route_mechanism_candidate_selection_delivery(
            result,
            observed_repository_revision=observed_repository_revision,
            run_id=run_id,
            output_root=output_root,
        )
    except Exception as exc:
        failure_reason = type(exc).__name__
        if operations is not None:
            try:
                operations.close()
            except Exception as close_exc:
                failure_reason = (
                    f"{failure_reason}+close:{type(close_exc).__name__}"
                )[:80]
        return finalize_soft_route_mechanism_failure_delivery(
            observed_repository_revision=observed_repository_revision,
            run_id=run_id,
            output_root=output_root,
            stage="candidate_selection_entrypoint",
            failure_reason=failure_reason,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--observed-repository-revision", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--detector-asset-bundle", required=True)
    arguments = parser.parse_args(argv)
    if not arguments.execute:
        parser.error("--execute is required")
    code, receipt = execute_soft_route_mechanism_candidate_selection_entrypoint(
        observed_repository_revision=arguments.observed_repository_revision,
        run_id=arguments.run_id,
        output_root=arguments.output_root,
        operations_factory=lambda: create_adapter_backed_soft_route_mechanism_operations(
            configuration_path=CONFIG_PATH,
            detector_asset_bundle=arguments.detector_asset_bundle,
        ),
    )
    print(json.dumps(receipt, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
