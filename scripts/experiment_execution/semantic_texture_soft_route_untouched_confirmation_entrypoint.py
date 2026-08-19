"""Soft-route confirmation entrypoint with an exact artifact prerequisite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Sequence

from experiments.protocol.semantic_texture_soft_route_mechanism_validation import (
    CONFIRMATION_ROLE,
    load_soft_route_mechanism_configuration,
    load_manifest,
    load_selection_artifact,
)
from experiments.runners.semantic_texture_soft_route_mechanism_validation import (
    SoftRouteMechanismOperations,
    SoftRouteMechanismNullScoreRecord,
    SoftRouteMechanismProvisionalCalibration,
    SoftRouteMechanismSplitResult,
    create_adapter_backed_soft_route_mechanism_operations,
    execute_soft_route_mechanism_split,
)
from scripts.experiment_execution.semantic_texture_soft_route_untouched_confirmation_server import (
    finalize_soft_route_mechanism_untouched_confirmation_delivery,
    finalize_soft_route_mechanism_untouched_confirmation_failure_delivery,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PACKAGE_ROOT / "configs/experiments/semantic_texture_soft_route_mechanism_validation.json"


def _calibration_from_artifact(value: object) -> SoftRouteMechanismProvisionalCalibration:
    try:
        raw = value["provisional_calibration"]
        return SoftRouteMechanismProvisionalCalibration(
            selection_manifest_digest=raw["selection_manifest_digest"],
            hf_detector_identity=raw["hf_detector_identity"],
            lf_detector_identity=raw["lf_detector_identity"],
            hf_null_identity=raw["hf_null_identity"],
            lf_null_identity=raw["lf_null_identity"],
            tau_hf_provisional=raw["tau_hf_provisional"],
            tau_lf_provisional=raw["tau_lf_provisional"],
            tau_max_provisional=raw["tau_max_provisional"],
            hf_records=tuple(SoftRouteMechanismNullScoreRecord(**record) for record in raw["hf_records"]),
            lf_records=tuple(SoftRouteMechanismNullScoreRecord(**record) for record in raw["lf_records"]),
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("selection artifact calibration is invalid") from exc


def execute_soft_route_mechanism_untouched_confirmation_entrypoint(
    *, observed_repository_revision: str, run_id: str, output_root: str | Path,
    selection_artifact_root: str | Path, selection_artifact_sha256: str,
    operations_factory: Callable[[], SoftRouteMechanismOperations],
) -> tuple[int, dict[str, object]]:
    """Consume one authenticated selection artifact and never refit it."""

    operations: SoftRouteMechanismOperations | None = None
    calibration: SoftRouteMechanismProvisionalCalibration | None = None
    authenticated_selection_artifact_sha256: str | None = None
    authenticated_selection_manifest_digest: str | None = None
    try:
        configuration = load_soft_route_mechanism_configuration(CONFIG_PATH)
        selection_artifact_path = (
            Path(selection_artifact_root).resolve()
            / configuration["selection_artifact_relative_path"]
        )
        manifest = load_manifest(PACKAGE_ROOT / configuration["untouched_confirmation_manifest_path"], expected_role=CONFIRMATION_ROLE)
        artifact = load_selection_artifact(
            selection_artifact_path,
            expected_sha256=selection_artifact_sha256,
        )
        authenticated_selection_artifact_sha256 = selection_artifact_sha256
        authenticated_selection_manifest_digest = artifact[
            "selection_manifest_digest"
        ]
        calibration = _calibration_from_artifact(artifact)
        operations = operations_factory()
        result: SoftRouteMechanismSplitResult = execute_soft_route_mechanism_split(
            manifest, operations, provisional_calibration=calibration
        )
        operations.close()
        operations = None
        return finalize_soft_route_mechanism_untouched_confirmation_delivery(
            result,
            observed_repository_revision=observed_repository_revision,
            run_id=run_id,
            output_root=output_root,
            source_selection_artifact_sha256=(
                authenticated_selection_artifact_sha256
            ),
            source_selection_manifest_digest=(
                authenticated_selection_manifest_digest
            ),
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
        return finalize_soft_route_mechanism_untouched_confirmation_failure_delivery(
            observed_repository_revision=observed_repository_revision,
            run_id=run_id,
            output_root=output_root,
            stage="untouched_confirmation_entrypoint",
            failure_reason=failure_reason,
            source_selection_artifact_sha256=(
                authenticated_selection_artifact_sha256
            ),
            source_selection_manifest_digest=(
                authenticated_selection_manifest_digest
            ),
            provisional_calibration_digest=(
                None if calibration is None else calibration.digest()
            ),
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--observed-repository-revision", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--selection-artifact-root", required=True)
    parser.add_argument("--selection-artifact-sha256", required=True)
    parser.add_argument("--detector-asset-bundle", required=True)
    arguments = parser.parse_args(argv)
    if not arguments.execute:
        parser.error("--execute is required")
    code, receipt = execute_soft_route_mechanism_untouched_confirmation_entrypoint(
        observed_repository_revision=arguments.observed_repository_revision,
        run_id=arguments.run_id,
        output_root=arguments.output_root,
        selection_artifact_root=arguments.selection_artifact_root,
        selection_artifact_sha256=arguments.selection_artifact_sha256,
        operations_factory=lambda: create_adapter_backed_soft_route_mechanism_operations(
            configuration_path=CONFIG_PATH,
            detector_asset_bundle=arguments.detector_asset_bundle,
        ),
    )
    print(json.dumps(receipt, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
