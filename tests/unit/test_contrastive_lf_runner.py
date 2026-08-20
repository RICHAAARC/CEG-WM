from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from experiments.protocol.contrastive_lf_branch_attribution import (
    CONFIG_DIGEST,
    MANIFEST_PATHS,
    NULL_FIT_ROLE,
    SELECTION_ROLE,
    load_manifest,
)
from experiments.runners.contrastive_lf_branch_attribution import (
    StageADetection,
    StageAGeneration,
    ContrastiveLfRunnerError,
    execute_null_fit,
    execute_selection,
    execute_stage_a_null_fit_and_selection,
)
from main import (
    ContrastiveLfRawObservation,
    HfDetectionResult,
    contrastive_lf_detector,
    derive_wrong_key_material,
    identify_root_key,
)
from scripts.experiment_execution.contrastive_lf_branch_attribution_server import (
    SELECTION_ARTIFACT_FILENAME,
    finalize_contrastive_lf_delivery,
)


ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.unit


class FakeOperations:
    implementation_revision = "1" * 40
    method_config_digest = CONFIG_DIGEST
    model_identity = "synthetic_model_identity"
    runtime_identity = "synthetic_runtime_identity"
    codec_identity = "pillow_rgb8_jpeg_exact_capability"
    root_key = "synthetic-stage-a-root"

    def __init__(self, fail_after: int | None = None) -> None:
        self.calls = 0
        self.fail_after = fail_after
        self.root_digest = identify_root_key(self.root_key).root_key_public_digest

    def _tick(self) -> None:
        if self.fail_after is not None and self.calls == self.fail_after:
            raise RuntimeError("synthetic_bounded_failure")
        self.calls += 1

    def clean(self, entry) -> StageAGeneration:
        self._tick()
        image = torch.full((1, 3, 4, 4), 10, dtype=torch.uint8)
        return StageAGeneration("clean_unwatermarked", image, image.clone(), None, None, None, 0.0)

    def write(self, entry, arm_id: str) -> StageAGeneration:
        self._tick()
        value = {"hf_only": 30, "multiscale_low_frequency_only": 20, "single_scale_low_frequency_only": 21}[arm_id]
        clean = torch.full((1, 3, 4, 4), 10, dtype=torch.uint8)
        image = torch.full((1, 3, 4, 4), value, dtype=torch.uint8)
        return StageAGeneration(arm_id, image, clean, "a" * 64, "b" * 64, "accepted", 0.0)

    def attack(self, entry, generation, attack_id: str) -> torch.Tensor:
        self._tick()
        return generation.image_rgb8.clone()

    def observe_hf_raw(self, image_rgb8, key) -> HfDetectionResult:
        self._tick()
        value = float(self.calls)
        return HfDetectionResult("hf_sparse_tail", value, "c" * 64, "d" * 64, self.root_digest, "registered", None, "e" * 64, "f" * 64)

    def observe_lf_raw(self, image_rgb8, key, candidate_id: str) -> ContrastiveLfRawObservation:
        self._tick()
        index = float(self.calls)
        feature = (index, float((self.calls * 7) % 13)) if candidate_id.startswith("lf_multiscale") else (index,)
        decoys = tuple(tuple(value - 0.2 * (decoy + 1) for value in feature) for decoy in range(8))
        return ContrastiveLfRawObservation(candidate_id, feature, decoys, "1" * 64, "2" * 64, self.root_digest, "registered", None, f"{self.calls:064x}"[-64:])

    def score_lf_raw(self, raw, asset):
        return contrastive_lf_detector(raw, asset)

    def observe_hf(self, image_rgb8, key, asset) -> StageADetection:
        self._tick()
        wrong = not isinstance(key, str)
        value = int(image_rgb8[0, 0, 0, 0])
        z = -5.0 if wrong or value == 10 else 8.0
        return StageADetection(z, z, (), asset.asset_digest, "3" * 64, "4" * 64)

    def observe_lf(self, image_rgb8, key, asset) -> StageADetection:
        self._tick()
        wrong = not isinstance(key, str)
        value = int(image_rgb8[0, 0, 0, 0])
        expected = 20 if asset.candidate_id.startswith("lf_multiscale") else 21
        z = -5.0 if wrong or value == 10 else 12.0 if value == expected else -3.0
        return StageADetection(z, z, tuple(-6.0 - index for index in range(8)), asset.asset_digest, "5" * 64, "6" * 64)

    def wrong_key(self, index: int):
        return derive_wrong_key_material(self.root_digest, index)

    def close(self) -> None:
        pass


def _manifests():
    return (
        load_manifest(ROOT / MANIFEST_PATHS[NULL_FIT_ROLE], expected_role=NULL_FIT_ROLE),
        load_manifest(ROOT / MANIFEST_PATHS[SELECTION_ROLE], expected_role=SELECTION_ROLE),
    )


def test_synthetic_stage_a_preserves_exact_denominators_and_selects_primary(tmp_path: Path) -> None:
    result = execute_stage_a_null_fit_and_selection(*_manifests(), FakeOperations())
    assert len(result.null_fit_records) == 128
    assert len(result.selection_records) == 4960
    assert all(record.execution_status == "completed" for record in result.null_fit_records)
    assert all(record.execution_status == "completed" for record in result.selection_records)
    assert result.selection_result is not None
    assert result.selection_result.result_classification == "success"
    assert result.selection_result.selected_candidate_id == "lf_multiscale_lowpass_contrastive"
    assert result.null_fit_artifact is not None
    assert result.null_fit_artifact.formal_tau_created is False
    result.validate_for_delivery()
    code, receipt = finalize_contrastive_lf_delivery(
        result,
        observed_repository_revision="1" * 40,
        run_id="contrastive-lf-branch-attribution-" + "a" * 32,
        output_root=tmp_path / "success",
    )
    assert code == 0
    assert receipt["selection_artifact_filename"] == SELECTION_ARTIFACT_FILENAME
    assert (tmp_path / "success" / SELECTION_ARTIFACT_FILENAME).is_file()


def test_first_failure_retains_one_failed_and_complete_unstarted_tail(tmp_path: Path) -> None:
    result = execute_stage_a_null_fit_and_selection(*_manifests(), FakeOperations(fail_after=4))
    statuses = [record.execution_status for record in result.null_fit_records]
    assert statuses[:4] == ["completed"] * 4
    assert statuses[4] == "failed"
    assert set(statuses[5:]) == {"unstarted"}
    assert result.selection_records == ()
    assert result.null_fit_artifact is None
    result.validate_for_delivery()
    code, receipt = finalize_contrastive_lf_delivery(
        result,
        observed_repository_revision="1" * 40,
        run_id="contrastive-lf-branch-attribution-" + "b" * 32,
        output_root=tmp_path / "failure",
    )
    assert code == 2
    assert receipt["selection_artifact_filename"] is None
    assert not (tmp_path / "failure" / SELECTION_ARTIFACT_FILENAME).exists()


def test_selection_rejects_tampered_null_population_asset() -> None:
    null_manifest, selection_manifest = _manifests()
    operations = FakeOperations()
    _, artifact, failure = execute_null_fit(null_manifest, operations)
    assert failure is None and artifact is not None
    tampered_asset = replace(
        artifact.multiscale_null_asset,
        contrastive_population=(
            artifact.multiscale_null_asset.contrastive_population[0] + 1.0,
            *artifact.multiscale_null_asset.contrastive_population[1:],
        ),
    )
    tampered = replace(artifact, multiscale_null_asset=tampered_asset)
    with pytest.raises(ContrastiveLfRunnerError, match="digest drifted"):
        execute_selection(selection_manifest, operations, tampered)
