"""Package-contained identity check using no repository or Git state."""

from __future__ import annotations

import json
from math import exp, log, sqrt
from pathlib import Path

import pytest
import torch

from experiments.attacks import (
    GeometricAttackSpec,
    apply_geometric_attack,
)
from main import (
    geometric_transform_estimator,
    geometry_reliability,
    qk_geometry_sync,
)
from main.geometry_chain import GeometricTransformEstimatorError
from scripts.experiment_execution.experiment_execution_entrypoint import (
    SYNTHETIC_ROOT_KEY,
    prepare_synthetic_wiring,
)


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.integration
def test_packaged_preparation_matches_bound_manifest(
    tmp_path: Path,
) -> None:
    manifest = json.loads(
        (ROOT / "experiment_execution_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    preparation = prepare_synthetic_wiring(
        package_root=ROOT,
        records_root=tmp_path / "records",
        workspace_root=tmp_path / "workspace",
        committed_revision=manifest["committed_revision"],
        run_id="packaged-integration",
    )

    assert preparation.candidate_config_digest == (
        manifest["candidate_config_digest"]
    )
    assert preparation.execution_config_digest == (
        manifest["execution_config_digest"]
    )
    assert preparation.input_manifest_digest == (
        manifest["input_manifest_digest"]
    )
    assert [
        (case.role, case.payload.attack_specification.attack_id)
        for case in preparation.cases
    ] == [
        ("geometry_non_identity_success", "scale"),
        ("resource_retry_resume", "crop"),
        ("execution_failure", "identity"),
    ]
    assert preparation.cases[0].payload.attack_specification == (
        GeometricAttackSpec(
            "scale",
            scale_factor=exp(-log(sqrt(2.0)) / 2.0),
        )
    )
    declaration = (
        preparation.payload.geometry_estimation_operation
        .formal_runner_semantic_declaration()
    )
    assert declaration["runtime_state"] == "ready"
    assert declaration["qk_public_callable"] == (
        "runtime.Sd35RuntimeAdapter.observe_detection_qk"
        " -> main.qk_geometry_sync"
    )


@pytest.mark.integration
def test_synthetic_qk_is_input_dependent_and_controls_fail_closed(
    tmp_path: Path,
) -> None:
    preparation = prepare_synthetic_wiring(
        package_root=ROOT,
        records_root=tmp_path / "records",
        workspace_root=tmp_path / "workspace",
        committed_revision="a" * 40,
        run_id="packaged-anti-weak",
    )
    case = preparation.cases[0]
    operation = case.payload.geometry_estimation_operation
    source = case.payload.source_artifact
    attacked = apply_geometric_attack(
        source,
        case.payload.attack_specification,
        registry=preparation.context.attack_registry,
    )
    source_observation = operation.runtime_adapter.observe_detection_qk(
        source.image
    )
    attacked_observation = operation.runtime_adapter.observe_detection_qk(
        attacked.attacked_artifact.image
    )

    assert all(
        not torch.equal(source_layer.query, attacked_layer.query)
        and not torch.equal(
            source_layer.attention_key,
            attacked_layer.attention_key,
        )
        for source_layer, attacked_layer in zip(
            source_observation.qk_layer_observations,
            attacked_observation.qk_layer_observations,
            strict=True,
        )
    )

    inconsistent = qk_geometry_sync(
        source_observation.qk_layer_observations,
        "synthetic-inconsistent-control-key",
    )
    with pytest.raises(
        GeometricTransformEstimatorError,
        match="Q/K geometry observation validation failed",
    ):
        geometric_transform_estimator(
            inconsistent,
            SYNTHETIC_ROOT_KEY,
            epsilon_inlier=0.8,
        )

    rotation = apply_geometric_attack(
        source,
        GeometricAttackSpec("rotation", rotation_degrees=8.0),
        registry=preparation.context.attack_registry,
    )
    rotation_estimation = operation(
        rotation.attacked_artifact.image,
        SYNTHETIC_ROOT_KEY,
    )
    rotation_reliability = geometry_reliability(
        rotation_estimation,
        case.payload.geometry_reliability_thresholds,
    )
    assert not rotation_estimation.transform.is_exact_identity
    assert not rotation_reliability.reliable
    assert rotation_reliability.failure_reasons
