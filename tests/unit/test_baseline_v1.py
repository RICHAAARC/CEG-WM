from __future__ import annotations

from dataclasses import replace
import json

import pytest

from cegwm.baselines import (
    FINAL_BASELINE_LONG_TABLE_FIELDS,
    FINAL_BASELINE_PRIMARY_CONDITION_ORDER,
    FORMAL_ATTACK_CONDITIONS,
    PRIMARY_BASELINES,
    BaselineObservation,
    adapter_plan,
    baseline_by_id,
    build_baseline_table_row,
    validate_observation,
)
def _record(**changes: object) -> BaselineObservation:
    record = BaselineObservation(
        baseline_id="tree_ring",
        source_exact="source-commit",
        adapter_exact="adapter-commit",
        prompt_id="prompt-001",
        seed=7,
        base_latent_commitment="sha256:latent",
        split="test",
        protocol_partition="evaluation",
        sample_role="evaluation_watermarked",
        attack_family="clean",
        attack_condition="pending_user_freeze",
        continuous_score=None,
        score_direction=None,
        threshold_provenance=None,
        decision=None,
        quality={"clip": 0.8},
        runtime_seconds=1.0,
        status="not_available",
        failure_reason=None,
        artifact_digests={"status": "sha256:" + "0" * 64},
        attack_provenance=None,
    )
    return replace(record, **changes)


@pytest.mark.unit
def test_registry_contains_only_the_authorized_four_methods() -> None:
    assert [item.baseline_id for item in PRIMARY_BASELINES] == [
        "tree_ring", "gaussian_shading", "shallow_diffuse", "t2smark"
    ]
    assert all(item.result_status == "not_available" for item in PRIMARY_BASELINES)
    assert all(not item.paper_claim_support for item in PRIMARY_BASELINES)
    assert baseline_by_id("t2smark").sd35_path == "official_run_sd35_native_path"
    with pytest.raises(ValueError, match="out-of-scope"):
        baseline_by_id("image_domain_method")


@pytest.mark.unit
def test_final_baseline_long_table_contract_and_condition_order_are_frozen() -> None:
    assert FINAL_BASELINE_LONG_TABLE_FIELDS == (
        "baseline_id", "threshold_identity", "attack_family", "attack_condition",
        "planned_positive_units", "observed_positive_units", "failed_positive_units", "planned_negative_units",
        "observed_negative_units", "failed_negative_units", "true_positive", "false_negative",
        "false_positive", "true_negative", "tpr", "tpr_ci95_lower", "tpr_ci95_upper", "fpr",
        "fpr_ci95_lower", "fpr_ci95_upper", "clean_confirmation_false_positives",
        "clean_confirmation_negatives", "clean_confirmation_failure_count", "clean_confirmation_ucb95",
        "clean_confirmation_gate_passed", "status",
    )
    assert FINAL_BASELINE_PRIMARY_CONDITION_ORDER == tuple(
        condition.condition for condition in FORMAL_ATTACK_CONDITIONS
    )


@pytest.mark.unit
def test_method_plans_preserve_method_identity_without_license_blocking() -> None:
    assert baseline_by_id("tree_ring").score_direction == "lower_is_watermarked"
    assert baseline_by_id("tree_ring").native_score_name == "fourier_key_l1_distance"
    assert baseline_by_id("t2smark").official_entrypoint == "run_sd35.py"
    assert baseline_by_id("t2smark").native_score_name == "norm1_w_master_key"
    assert adapter_plan("tree_ring").execution_status == "implementation_required"
    assert adapter_plan("t2smark").execution_status == "adapter_interface_available_gpu_unverified"
    assert adapter_plan("t2smark").blocker == "local GPU is unavailable; a Colab canary is required"
    assert adapter_plan("shallow_diffuse").execution_status == "implementation_required"
    assert adapter_plan("shallow_diffuse").blocker is None
    assert baseline_by_id("shallow_diffuse").source_license is None


@pytest.mark.unit
def test_method_first_observations_allow_optional_identity_metadata() -> None:
    assert validate_observation(_record()).as_dict()["baseline_id"] == "tree_ring"
    calibration = _record(
        source_exact=None, adapter_exact=None, protocol_partition="threshold_freeze",
        sample_role="calibration_unwatermarked_negative", continuous_score=0.25,
        score_direction="lower_is_watermarked", threshold_provenance=None, decision=None,
        status="calibration_observed", artifact_digests={},
    )
    assert validate_observation(calibration).status == "calibration_observed"
    observed = _record(
        source_exact=None, adapter_exact=None, continuous_score=0.25,
        score_direction="lower_is_watermarked", threshold_provenance="tree_ring:calibration:future",
        decision=True, status="observed", artifact_digests={},
    )
    assert validate_observation(observed).status == "observed"
    assert validate_observation(replace(observed, source_exact="anything", adapter_exact="not-a-git-exact")).status == "observed"
    with pytest.raises(ValueError, match="score_direction"):
        validate_observation(replace(observed, score_direction="higher_is_watermarked"))
    with pytest.raises(ValueError, match="threshold provenance"):
        validate_observation(replace(observed, threshold_provenance="t2smark:calibration:future"))
    with pytest.raises(ValueError, match="evaluation partition"):
        validate_observation(replace(observed, protocol_partition="threshold_freeze"))
    with pytest.raises(ValueError, match="evaluation sample role"):
        validate_observation(replace(observed, sample_role="calibration_unwatermarked_negative"))
    with pytest.raises(ValueError, match="cannot carry detection evidence"):
        validate_observation(_record(continuous_score=0.25))


@pytest.mark.unit
@pytest.mark.parametrize("invalid_score", ["placeholder", float("nan"), True, float("inf"), float("-inf")])
def test_observation_scores_must_be_finite_real_numbers(invalid_score: object) -> None:
    calibration = _record(
        protocol_partition="threshold_freeze", sample_role="calibration_unwatermarked_negative",
        continuous_score=invalid_score, score_direction="lower_is_watermarked", decision=None,
        status="calibration_observed", artifact_digests={},
    )
    observed = _record(
        source_exact=None, adapter_exact=None, continuous_score=invalid_score,
        score_direction="lower_is_watermarked", threshold_provenance="tree_ring:calibration:future",
        decision=True, status="observed", artifact_digests={},
    )
    confirmation = replace(
        observed, protocol_partition="clean_confirmation", sample_role="confirmation_unwatermarked_negative",
        attack_condition="clean_no_attack", status="confirmation_observed",
    )
    for record in (calibration, observed, confirmation):
        with pytest.raises(ValueError, match="finite real score"):
            validate_observation(record)


@pytest.mark.unit
@pytest.mark.parametrize("invalid_threshold", [123, object(), ""])
def test_observed_threshold_provenance_requires_nonempty_string(invalid_threshold: object) -> None:
    observed = _record(
        source_exact=None, adapter_exact=None, continuous_score=0.25,
        score_direction="lower_is_watermarked", threshold_provenance=invalid_threshold,
        decision=True, status="observed", artifact_digests={},
    )
    with pytest.raises(ValueError, match="non-empty method-specific threshold"):
        validate_observation(observed)


@pytest.mark.unit
def test_observed_rotation_still_requires_scientific_attack_provenance() -> None:
    observed = _record(
        source_exact=None, adapter_exact=None, continuous_score=0.25, score_direction="lower_is_watermarked",
        threshold_provenance="tree_ring:calibration:approved", decision=True, status="observed", artifact_digests={},
    )
    assert observed.as_dict()["status"] == "observed"
    with pytest.raises(ValueError, match="rotation observations require attack provenance"):
        replace(
            observed,
            attack_family="geometric",
            attack_condition="rotation_10_bicubic_reflect_center_crop_v1",
            attack_provenance=None,
        ).as_dict()


@pytest.mark.unit
def test_rotation_provenance_jsonl_round_trip_accepts_crop_box_array() -> None:
    rotation_provenance = {
        "attack_id": "rotation_10_bicubic_reflect_center_crop_v1", "angle_degrees": 10.0,
        "angle_convention": "Pillow visual counter-clockwise positive angle",
        "center_formula_id": "pixel_center_w_minus_1_over_2_v1", "padding_x": 8, "padding_y": 9,
        "bicubic_margin_pixels": 2, "padding_mode_rgb": "numpy.reflect_edge_not_repeated",
        "padding_mode_mask": "numpy.constant_zero", "rgb_interpolation": "PIL.Image.Resampling.BICUBIC",
        "mask_interpolation": "PIL.Image.Resampling.NEAREST", "crop_box": (8, 9, 81, 50),
        "positive_negative_pipeline_identical": True,
    }
    observed = _record(
        source_exact=None, adapter_exact=None, continuous_score=0.25, score_direction="lower_is_watermarked",
        threshold_provenance="tree_ring:calibration:approved", decision=True, status="observed",
        attack_family="geometric", attack_condition="rotation_10_bicubic_reflect_center_crop_v1",
        artifact_digests={},
        attack_provenance=rotation_provenance,
    )
    round_tripped = json.loads(json.dumps(observed.as_dict()))
    assert isinstance(round_tripped["attack_provenance"]["crop_box"], list)
    assert validate_observation(BaselineObservation(**round_tripped)).status == "observed"


@pytest.mark.unit
def test_main_table_uses_unwatermarked_fpr_rejects_wrong_key_and_counts_failures() -> None:
    common = {
        "source_exact": None, "adapter_exact": None, "continuous_score": 0.25,
        "score_direction": "lower_is_watermarked", "threshold_provenance": "tree_ring:calibration:approved",
        "status": "observed", "artifact_digests": {},
    }
    watermarked = _record(sample_role="evaluation_watermarked", decision=True, **common)
    negative = _record(sample_role="evaluation_unwatermarked_negative", decision=False, **common)
    row = build_baseline_table_row((watermarked, negative))
    assert (row.true_positive, row.false_negative, row.false_positive, row.true_negative) == (1, 0, 0, 1)
    with pytest.raises(ValueError, match="wrong-key diagnostics"):
        build_baseline_table_row((replace(watermarked, sample_role="wrong_key_diagnostic"), negative))
    retained_failure = replace(
        watermarked, continuous_score=None, score_direction=None, threshold_provenance=None, decision=None,
        status="failed", failure_reason="adapter stopped", artifact_digests={},
    )
    assert build_baseline_table_row((watermarked, negative, retained_failure)).failure_count == 1
    unrelated_failure = replace(
        watermarked,
        attack_condition="different_pending_user_freeze",
        continuous_score=None,
        score_direction=None,
        threshold_provenance=None,
        decision=None,
        status="failed",
        failure_reason="adapter stopped",
    )
    with pytest.raises(ValueError, match="share attack identity"):
        build_baseline_table_row((watermarked, negative, unrelated_failure))


@pytest.mark.unit
def test_failed_unit_remains_a_record_without_artifact_identity() -> None:
    failed = _record(
        source_exact=None,
        adapter_exact=None,
        continuous_score=None,
        score_direction=None,
        threshold_provenance=None,
        decision=None,
        status="failed",
        failure_reason="external source not available",
    )
    assert validate_observation(failed).status == "failed"
    assert validate_observation(replace(failed, artifact_digests={})).status == "failed"
    assert validate_observation(replace(failed, artifact_digests={"log": "not-a-digest"})).status == "failed"
    with pytest.raises(ValueError, match="cannot carry detection evidence"):
        validate_observation(replace(failed, continuous_score=0.25))
