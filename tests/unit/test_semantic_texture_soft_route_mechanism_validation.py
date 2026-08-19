"""Synthetic fixed-denominator tests for soft-route mechanism validation."""

from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256
import json
from math import isfinite, nextafter
from pathlib import Path
from struct import pack, unpack
from types import SimpleNamespace

import pytest
import torch

from main import ContentMaterializationObservation, ContentMaterializationResult
from runtime import ContentMaterializationMeasurement

from experiments.protocol.semantic_texture_soft_route_mechanism_validation import (
    CONFIRMATION_ROLE,
    SELECTION_ROLE,
    load_manifest,
    load_soft_route_mechanism_configuration,
    load_selection_artifact,
    provisional_tau,
    validate_split_disjointness,
)
from experiments.runners.semantic_texture_soft_route_mechanism_validation import (
    AdapterBackedSoftRouteMechanismOperations,
    SoftRouteMechanismBranchScores,
    SoftRouteMechanismGeneration,
    SoftRouteMechanismRunnerError,
    SoftRouteMechanismStandardizedScores,
    execute_soft_route_mechanism_split,
)


pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def _manifests():
    selection = load_manifest(
        ROOT / "configs/experiments/semantic_texture_soft_route_candidate_selection_manifest.json",
        expected_role=SELECTION_ROLE,
    )
    confirmation = load_manifest(
        ROOT / "configs/experiments/semantic_texture_soft_route_untouched_confirmation_manifest.json",
        expected_role=CONFIRMATION_ROLE,
    )
    return selection, confirmation


class SyntheticOperations:
    def __init__(self, *, fail_write_call: int | None = None) -> None:
        self.clean_calls = 0
        self.write_calls = 0
        self.observe_calls = 0
        self.fail_write_call = fail_write_call
        self.installed = None

    def clean(self, entry):
        self.clean_calls += 1
        return SoftRouteMechanismGeneration(entry.source_cluster_id, "clean_unwatermarked", None, None, 0.0)

    def write(self, entry, arm_id):
        self.write_calls += 1
        if self.write_calls == self.fail_write_call:
            raise RuntimeError("bounded synthetic failure")
        return SoftRouteMechanismGeneration(
            f"{entry.source_cluster_id}:{arm_id}", arm_id,
            f"materialization:{entry.source_cluster_id}:{arm_id}",
            "combined_relative_l2_3_250", 0.0,
        )

    def attack(self, entry, generation, attack_id):
        return f"{generation.image}:{attack_id}"

    def observe(self, image, *, wrong_key_index):
        self.observe_calls += 1
        value = float(self.observe_calls)
        if wrong_key_index is not None:
            value = -value
        return SoftRouteMechanismBranchScores(value, value, value, "d" * 64, "h" * 64, "l" * 64, wrong_key_index is None)

    def build_calibration(self, primary_null, *, partition_identity):
        assert len(primary_null) == 32
        return "h" * 64, "l" * 64, "a" * 64, "b" * 64

    def install_calibration(self, calibration):
        self.installed = calibration

    def standardize(self, scores):
        return SoftRouteMechanismStandardizedScores(scores.hf_score, scores.lf_score, scores.max_score, scores.detector_identity)

    def close(self):
        return None


class _BudgetAuthorityAdapter:
    def __init__(self, content_write, written_rgb8: torch.Tensor) -> None:
        self.content_write = content_write
        self.written_rgb8 = written_rgb8

    def execute_semantic_texture_content_arm_write_and_vae(
        self,
        _latent,
        _root_key,
        _semantic_runtime,
        *,
        arm_id,
    ):
        assert arm_id == "semantic_texture_route_disabled"
        return SimpleNamespace(
            result=SimpleNamespace(content_write_result=self.content_write)
        )

    def materialize_semantic_texture_written_rgb8(self, _observation):
        return self.written_rgb8


def _budget_authority_write_operation(
    budget_authority: object | None,
    *,
    measurement_identity: str = "m" * 64,
) -> AdapterBackedSoftRouteMechanismOperations:
    latent = torch.zeros((1, 16, 1, 1), dtype=torch.float16)
    measurement = ContentMaterializationMeasurement(
        attempt_index=1,
        callback_index=18,
        embedder_config_digest="e" * 64,
        materialization_scale=1.0,
        scaled_nominal_delta_digest="d" * 64,
        baseline_latent_actual=latent,
        written_latent_actual=latent.clone(),
        delta_content_actual=torch.zeros_like(latent, dtype=torch.float32),
        baseline_latent_digest="a" * 64,
        written_latent_digest="b" * 64,
        delta_content_actual_digest="c" * 64,
        tensor_replay_identity="t" * 64,
        materialization_replay_identity=measurement_identity,
        realized_total_l2=1.0,
        realized_relative_l2=unpack(">f", pack(">f", 3.0 / 250.0))[0],
        integrity_status="passed",
    )
    content_write = SimpleNamespace(
        clean_image=torch.zeros((1, 3, 2, 2), dtype=torch.float32),
        content_materialization=measurement,
        content_materialization_result=budget_authority,
    )
    backend = SimpleNamespace(
        set_development_generation_prompts=lambda _prompt, _negative: None
    )
    adapter = _BudgetAuthorityAdapter(
        content_write,
        torch.ones((1, 3, 2, 2), dtype=torch.uint8),
    )
    return AdapterBackedSoftRouteMechanismOperations(
        backend=backend,
        runtime_adapter=SimpleNamespace(),
        session=SimpleNamespace(
            image_height=8,
            image_width=8,
            selected_device="cpu",
        ),
        semantic_runtime=object(),
        adapter=adapter,
        whitening_asset=object(),
        root_key="synthetic-budget-authority-root",
        attack_registry=object(),
    )


def _accepted_budget_authority(
    *,
    replay_identity: str = "m" * 64,
) -> ContentMaterializationResult:
    binary32_limit = unpack(">f", pack(">f", 3.0 / 250.0))[0]
    observation = ContentMaterializationObservation(
        materialization_scale=1.0,
        baseline_norm=1.0,
        scaled_nominal_delta_digest="d" * 64,
        delta_content_actual=(binary32_limit,),
        realized_total_l2=binary32_limit,
        integrity_status="passed",
        deterministic_binary16_replay_passed=True,
        materialization_replay_identity=replay_identity,
    )
    return ContentMaterializationResult(
        embedding_result=None,
        observation=observation,
        content_relative_l2_nominal=binary32_limit,
        content_relative_l2_limit=binary32_limit,
        realized_total_l2=binary32_limit,
        realized_relative_l2=binary32_limit,
        budget_utilization=1.0,
        materialization_scale=1.0,
        attempt_count=1,
        integrity_status="passed",
        budget_status="accepted",
    )


def test_production_write_honors_accepted_binary32_budget_authority() -> None:
    binary32_limit = unpack(">f", pack(">f", 3.0 / 250.0))[0]
    assert binary32_limit > 3.0 / 250.0
    operation = _budget_authority_write_operation(_accepted_budget_authority())

    generation = operation.write(
        SimpleNamespace(prompt_text="a walnut", generation_seed=202608190200),
        "semantic_texture_route_disabled",
    )

    assert generation.arm_id == "semantic_texture_route_disabled"
    assert generation.materialization_replay_identity == "m" * 64
    assert generation.budget_identity == "combined_relative_l2_3_250"
    assert isfinite(generation.paired_rgb8_mse)


@pytest.mark.parametrize(
    "authority_case",
    ["missing", "wrong_type", "not_accepted", "identity_drift"],
)
def test_production_write_rejects_invalid_budget_authority(
    authority_case: str,
) -> None:
    authority = _accepted_budget_authority()
    if authority_case == "missing":
        authority = None
    elif authority_case == "wrong_type":
        authority = SimpleNamespace(
            budget_status="accepted",
            integrity_status="passed",
            observation=authority.observation,
        )
    elif authority_case == "not_accepted":
        authority = replace(authority, budget_status="rejected")
    else:
        authority = _accepted_budget_authority(replay_identity="x" * 64)
    operation = _budget_authority_write_operation(authority)

    with pytest.raises(
        SoftRouteMechanismRunnerError,
        match="content budget authority is invalid",
    ):
        operation.write(
            SimpleNamespace(prompt_text="a walnut", generation_seed=202608190200),
            "semantic_texture_route_disabled",
        )


def test_literal_manifests_roundtrip_tamper_skip_and_disjointness() -> None:
    selection, confirmation = _manifests()
    configuration = load_soft_route_mechanism_configuration(
        ROOT / "configs/experiments/semantic_texture_soft_route_mechanism_validation.json"
    )
    assert configuration["alpha_selection_float64_hex"] == float(0.1).hex()
    validate_split_disjointness(selection, confirmation)
    assert [entry.source_row for entry in selection.entries] == list(range(65, 82)) + list(range(83, 98))
    assert [entry.source_row for entry in confirmation.entries] == list(range(98, 130))
    assert [entry.generation_seed for entry in selection.entries] == list(range(202608190200, 202608190232))
    assert [entry.generation_seed for entry in confirmation.entries] == list(range(202608190300, 202608190332))
    assert selection.skipped_raw_rows[0]["raw_row"] == 82
    changed = replace(selection.entries[0], generation_seed=selection.entries[0].generation_seed + 1)
    with pytest.raises(ValueError, match="entry identity drifted"):
        changed.validate(ordinal=0, role_id=SELECTION_ROLE)


def test_runner_produces_exact_generation_and_detector_denominators() -> None:
    selection, _ = _manifests()
    operations = SyntheticOperations()
    result = execute_soft_route_mechanism_split(selection, operations)
    assert len(result.generations) == 32 * 5
    assert len(result.records) == 32 * 12
    assert all(record.execution_status == "completed" for record in result.generations)
    assert all(record.execution_status == "completed" for record in result.records)
    assert operations.clean_calls == 32
    assert operations.write_calls == 32 * 4
    assert operations.observe_calls == 32 * 12
    assert result.science_started is False
    assert result.candidate_promoted is False
    assert result.formal_tau_created is False


def test_first_failure_retains_failed_slot_and_unstarted_fixed_tail() -> None:
    selection, _ = _manifests()
    operations = SyntheticOperations(fail_write_call=2)
    result = execute_soft_route_mechanism_split(selection, operations)
    assert len(result.generations) == 160
    assert len(result.records) == 384
    statuses = [record.execution_status for record in result.generations]
    assert statuses[:3] == ["completed", "completed", "failed"]
    assert set(statuses[3:]) == {"unstarted"}
    assert result.generations[2].failure_reason == "RuntimeError"
    assert set(record.execution_status for record in result.records) == {"unstarted"}
    assert operations.write_calls == 2
    assert operations.observe_calls == 0


def test_provisional_tau_is_tie_safe_fourth_largest_plus_one_ulp() -> None:
    values = [float(value) for value in range(29)] + [100.0, 100.0, 100.0]
    assert provisional_tau(values) == nextafter(28.0, float("inf"))
    with pytest.raises(ValueError):
        provisional_tau(values[:-1])


def test_selection_artifact_requires_exact_sha_and_confirmation_does_not_refit(tmp_path: Path) -> None:
    selection, confirmation = _manifests()
    selection_result = execute_soft_route_mechanism_split(selection, SyntheticOperations())
    artifact = {
        "protocol_id": selection_result.protocol_id,
        "selection_manifest_digest": selection_result.manifest_digest,
        "provisional_calibration": asdict(selection_result.provisional_calibration),
        "candidate_selection_passed": True,
        "diagnostic_only": True,
        "science_started": False,
        "scientific_unit_count": 0,
        "candidate_promoted": False,
        "formal_tau_created": False,
        "formal_fpr_created": False,
    }
    path = tmp_path / "selection.json"
    blob = (json.dumps(artifact, sort_keys=True) + "\n").encode()
    path.write_bytes(blob)
    loaded = load_selection_artifact(path, expected_sha256=sha256(blob).hexdigest())
    with pytest.raises(ValueError, match="authority drifted"):
        load_selection_artifact(path, expected_sha256="0" * 64)
    drifted = {**artifact, "selection_manifest_digest": "f" * 64}
    drifted_path = tmp_path / "selection-manifest-drift.json"
    drifted_blob = (json.dumps(drifted, sort_keys=True) + "\n").encode()
    drifted_path.write_bytes(drifted_blob)
    with pytest.raises(ValueError, match="authority drifted"):
        load_selection_artifact(
            drifted_path,
            expected_sha256=sha256(drifted_blob).hexdigest(),
        )
    from scripts.experiment_execution.semantic_texture_soft_route_untouched_confirmation_entrypoint import _calibration_from_artifact
    confirmation_operations = SyntheticOperations()
    result = execute_soft_route_mechanism_split(
        confirmation,
        confirmation_operations,
        provisional_calibration=_calibration_from_artifact(loaded),
    )
    assert confirmation_operations.installed == result.provisional_calibration
    assert result.provisional_calibration.digest() == selection_result.provisional_calibration.digest()
