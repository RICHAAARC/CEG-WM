"""CPU unit checks for the seven executable C1 HF-reference metrics."""

from __future__ import annotations

import ast
from dataclasses import replace
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import shutil

import pytest

from experiments.metrics import (
    C1HfActualDtypeIntegrityCase,
    C1HfConfirmationInputBundle,
    C1HfMetricCaseIdentity,
    C1HfMetricError,
    C1HfQualityPair,
    C1HfScoreCase,
    C1Rgb8Image,
    clopper_pearson_lower,
    clopper_pearson_upper,
    evaluate_c1_hf_actual_dtype_integrity,
    evaluate_c1_hf_confirmation_metrics,
    evaluate_c1_hf_paired_rgb8_quality,
    evaluate_c1_hf_paired_key_attribution,
    evaluate_c1_hf_primary_null_fixed_fpr,
    evaluate_c1_hf_registered_tpr,
    evaluate_c1_hf_rgb8_quality_pair,
    evaluate_c1_hf_wrong_key_false_accept,
    fit_c1_hf_tau,
    load_c1_hf_metric_implementation_binding,
    student_t_quantile_975,
    validate_c1_hf_confirmation_input_bundle,
    validate_c1_hf_confirmation_metric_results,
    validate_c1_hf_threshold_result,
)


ROOT = Path(__file__).resolve().parents[2]
C1_METRIC_MODULE = ROOT / "experiments/metrics/c1_hf_reference.py"
SOURCE_CLUSTER_COUNT = 4096
REGISTERED_KEY_DIGEST = hashlib.sha256(b"registered-key").hexdigest()
DETECTOR_CONFIG_DIGEST = hashlib.sha256(b"detector-config").hexdigest()
QUALITY_CLEAN_IMAGE = C1Rgb8Image(
    height=1,
    width=1,
    channels=3,
    dtype="uint8",
    values_hwc=bytes((10, 20, 30)),
)
QUALITY_MARKED_IMAGE = C1Rgb8Image(
    height=1,
    width=1,
    channels=3,
    dtype="uint8",
    values_hwc=bytes((11, 20, 30)),
)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def _binding():
    return load_c1_hf_metric_implementation_binding()


@lru_cache(maxsize=2)
def _ordered_analysis_units(split: str):
    binding = _binding()
    units = (
        binding.fit_analysis_units
        if split == "content_threshold_fit"
        else binding.confirmation_analysis_units
    )
    return tuple(sorted(units, key=lambda identity: identity.unit_id))


def _case_identity(index: int, split: str) -> C1HfMetricCaseIdentity:
    binding = _binding()
    manifest_digest = (
        binding.fit_manifest_digest
        if split == "content_threshold_fit"
        else binding.confirmation_manifest_digest
    )
    return C1HfMetricCaseIdentity(
        analysis_unit_identity=_ordered_analysis_units(split)[index],
        split=split,
        detector_identity="ceg_wm_hf_only_detector",
        detector_config_digest=DETECTOR_CONFIG_DIGEST,
        protocol_id="ceg_wm_internal_scientific_validation_v2",
        protocol_version="2.0.0",
        protocol_digest=binding.protocol_digest,
        c1_specification_digest=binding.c1_specification_digest,
        manifest_digest=manifest_digest,
        metric_registry_digest=binding.metric_registry_digest,
        registered_key_family_digest=binding.registered_key_family_digest,
    )


def _fit_cases() -> tuple[C1HfScoreCase, ...]:
    return tuple(
        C1HfScoreCase(
            identity=_case_identity(index, "content_threshold_fit"),
            key_role="unwatermarked_primary_null",
            score=float(index) / 8192.0,
            registered_detection_key_public_digest=REGISTERED_KEY_DIGEST,
            detection_key_public_digest=REGISTERED_KEY_DIGEST,
            control_identity="unwatermarked_image_with_registered_detection_key",
            image_digest=_digest(f"fit-clean-{index}"),
        )
        for index in range(SOURCE_CLUSTER_COUNT)
    )


def _confirmation_cases(tau: float) -> tuple[C1HfScoreCase, ...]:
    cases = []
    for index in range(SOURCE_CLUSTER_COUNT):
        identity = _case_identity(index, "untouched_confirmation")
        marked_digest = QUALITY_MARKED_IMAGE.digest()
        cases.extend(
            (
                C1HfScoreCase(
                    identity=identity,
                    key_role="unwatermarked_primary_null",
                    score=-1.0,
                    registered_detection_key_public_digest=REGISTERED_KEY_DIGEST,
                    detection_key_public_digest=REGISTERED_KEY_DIGEST,
                    control_identity=(
                        "unwatermarked_image_with_registered_detection_key"
                    ),
                    image_digest=QUALITY_CLEAN_IMAGE.digest(),
                ),
                C1HfScoreCase(
                    identity=identity,
                    key_role="registered_positive",
                    score=1.0,
                    registered_detection_key_public_digest=REGISTERED_KEY_DIGEST,
                    detection_key_public_digest=REGISTERED_KEY_DIGEST,
                    control_identity=(
                        "registered_watermarked_image_with_registered_detection_key"
                    ),
                    image_digest=marked_digest,
                ),
                C1HfScoreCase(
                    identity=identity,
                    key_role="wrong_key",
                    score=1.0 if index == 0 else tau if index == 1 else -1.0,
                    registered_detection_key_public_digest=REGISTERED_KEY_DIGEST,
                    detection_key_public_digest=_digest(f"wrong-key-{index}"),
                    control_identity=(
                        "watermarked_image_with_preregistered_wrong_detection_key"
                    ),
                    image_digest=marked_digest,
                ),
            )
        )
    return tuple(cases)


def _quality_pairs() -> tuple[C1HfQualityPair, ...]:
    return tuple(
        C1HfQualityPair(
            identity=_case_identity(index, "untouched_confirmation"),
            clean_image=QUALITY_CLEAN_IMAGE,
            registered_watermarked_image=QUALITY_MARKED_IMAGE,
            clean_image_digest=QUALITY_CLEAN_IMAGE.digest(),
            registered_watermarked_image_digest=QUALITY_MARKED_IMAGE.digest(),
        )
        for index in range(SOURCE_CLUSTER_COUNT)
    )


@pytest.mark.unit
def test_c1_metric_binding_binds_formula_registry_source_and_safe_dependencies() -> None:
    binding = _binding()
    assert (
        binding.c1_specification_digest
        == "aa21c78233926049fe3cc1d0f8ec14469330fcaa33174558b45254153a10f3be"
    )
    assert binding.implementation_source_sha256 == hashlib.sha256(
        C1_METRIC_MODULE.read_bytes()
    ).hexdigest()
    assert len(binding.formula_identity_digest) == 64

    tree = ast.parse(C1_METRIC_MODULE.read_text(encoding="utf-8"))
    imports = {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert not imports & {
        "runtime",
        "main",
        "torch",
        "scipy",
        "experiments.methods",
        "experiments.attacks",
        "experiments.runners",
        "governance",
    }


@pytest.mark.unit
def test_c1_metric_binding_rejects_candidate_authority_byte_tamper(tmp_path) -> None:
    binding = _binding()
    specification = json.loads(
        (ROOT / "configs/experiments/c1_hf_reference_run.json").read_text()
    )
    candidate = specification["candidate_binding"]
    temporary_root = tmp_path / "repository"
    paths = {
        Path("configs/experiments/c1_hf_metric_implementation.json"),
        Path("configs/experiments/c1_hf_reference_run.json"),
        Path("configs/experiments/internal_execution_components.json"),
        Path("experiments/metrics/c1_hf_reference.py"),
        Path(candidate["candidate_specification_path"]),
        Path(candidate["formal_method_adapter_config_path"]),
        Path(candidate["runtime_config_path"]),
        Path(specification["dataset"]["roster_path"]),
        Path(specification["dataset"]["dataset_snapshot_path"]),
        *(
            Path(manifest["path"])
            for manifest in specification["split_manifests"].values()
        ),
        *(Path(entry["path"]) for entry in candidate["method_source_files"]),
    }
    for relative_path in paths:
        destination = temporary_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative_path, destination)
    authority = temporary_root / candidate["candidate_specification_path"]
    authority.write_bytes(authority.read_bytes() + b"\nauthority tamper\n")

    with pytest.raises(ValueError, match="candidate_specification_authority_mismatch"):
        load_c1_hf_metric_implementation_binding(
            temporary_root
            / "configs/experiments/c1_hf_metric_implementation.json"
        )
    assert len(binding.binding_digest) == 64


@pytest.mark.unit
def test_clopper_pearson_shared_primitive_boundaries_and_golden_values() -> None:
    expected_upper_zero_of_ten = 1.0 - 0.05 ** (1.0 / 10.0)
    expected_lower_ten_of_ten = 0.05 ** (1.0 / 10.0)
    assert clopper_pearson_upper(0, 10) == pytest.approx(
        expected_upper_zero_of_ten,
        abs=2e-15,
    )
    assert clopper_pearson_lower(10, 10) == pytest.approx(
        expected_lower_ten_of_ten,
        abs=2e-15,
    )
    assert clopper_pearson_lower(0, 10) == 0.0
    assert clopper_pearson_upper(10, 10) == 1.0


@pytest.mark.unit
def test_student_t_quantile_matches_independent_reference_golden_values() -> None:
    references = {
        1: 12.706204736174696,
        2: 4.302652729696142,
        10: 2.2281388519649385,
        30: 2.0422724563012373,
    }
    for degrees_of_freedom, reference in references.items():
        assert student_t_quantile_975(degrees_of_freedom) == pytest.approx(
            reference,
            rel=1e-9,
            abs=1e-10,
        )


@pytest.mark.unit
def test_c1_tau_fit_uses_all_clusters_nextafter_and_rejects_drift() -> None:
    binding = _binding()
    cases = _fit_cases()
    threshold = fit_c1_hf_tau(cases, binding=binding)
    assert threshold.source_cluster_count == SOURCE_CLUSTER_COUNT
    assert threshold.tau == math.nextafter(max(case.score for case in cases), math.inf)
    assert threshold.tau_float64_hex == threshold.tau.hex()
    assert threshold.fit_false_positive_count == 0
    validate_c1_hf_threshold_result(threshold, binding)

    duplicate_identity = cases[0].identity
    duplicate_cases = (*cases[:-1], replace(cases[-1], identity=duplicate_identity))
    with pytest.raises(C1HfMetricError, match="analysis_unit_manifest_set_mismatch"):
        fit_c1_hf_tau(duplicate_cases, binding=binding)
    with pytest.raises(C1HfMetricError, match="finite_float64"):
        replace(cases[0], score=float("nan"))


@pytest.mark.unit
def test_c1_confirmation_metrics_are_separate_paired_and_threshold_bound() -> None:
    binding = _binding()
    threshold = fit_c1_hf_tau(_fit_cases(), binding=binding)
    cases = _confirmation_cases(threshold.tau)

    primary = evaluate_c1_hf_primary_null_fixed_fpr(
        cases,
        threshold,
        binding=binding,
    )
    registered = evaluate_c1_hf_registered_tpr(
        cases,
        threshold,
        binding=binding,
    )
    wrong = evaluate_c1_hf_wrong_key_false_accept(
        cases,
        threshold,
        binding=binding,
    )
    attribution = evaluate_c1_hf_paired_key_attribution(
        cases,
        threshold,
        binding=binding,
    )
    assert primary.event_count == 0
    assert primary.confidence_direction == "one_sided_upper"
    assert registered.event_count == SOURCE_CLUSTER_COUNT
    assert registered.confidence_direction == "one_sided_lower"
    assert wrong.event_count == 2
    assert wrong.confidence_direction == "one_sided_upper"
    assert attribution.event_count == SOURCE_CLUSTER_COUNT - 1
    assert attribution.confidence_direction == "one_sided_lower"
    assert len(
        {
            primary.result_identity,
            registered.result_identity,
            wrong.result_identity,
            attribution.result_identity,
        }
    ) == 4

    drifted = replace(threshold, tau=threshold.tau + 0.1)
    with pytest.raises(C1HfMetricError, match="threshold"):
        evaluate_c1_hf_registered_tpr(cases, drifted, binding=binding)
    missing = tuple(cases[:-1])
    with pytest.raises(C1HfMetricError, match="exact_three_roles"):
        evaluate_c1_hf_wrong_key_false_accept(
            missing,
            threshold,
            binding=binding,
        )


@pytest.mark.unit
def test_c1_quality_consumes_rgb8_bytes_and_uses_unclipped_student_t_interval() -> None:
    binding = _binding()
    pairs = _quality_pairs()
    single = evaluate_c1_hf_rgb8_quality_pair(pairs[0], binding=binding)
    assert single.relative_l2 == pytest.approx(1.0 / math.sqrt(1400.0))
    assert single.normalized_rgb8_mse == pytest.approx(1.0 / (255.0**2 * 3.0))

    case_results = tuple(
        evaluate_c1_hf_rgb8_quality_pair(pair, binding=binding)
        for pair in pairs
    )
    result = evaluate_c1_hf_paired_rgb8_quality(
        case_results,
        binding=binding,
    )
    assert result.pair_count == SOURCE_CLUSTER_COUNT
    assert result.relative_l2.mean == pytest.approx(single.relative_l2)
    assert result.relative_l2.student_t_degrees_of_freedom == 4095
    assert result.relative_l2.student_t_critical_975 > 1.96
    assert result.relative_l2.confidence_interval_lower == pytest.approx(
        single.relative_l2
    )
    assert result.normalized_rgb8_mse.mean == pytest.approx(
        single.normalized_rgb8_mse
    )

    zero = C1Rgb8Image(1, 1, 3, "uint8", bytes((0, 0, 0)))
    zero_pair = replace(
        pairs[0],
        clean_image=zero,
        clean_image_digest=zero.digest(),
    )
    with pytest.raises(C1HfMetricError, match="clean_l2_zero"):
        evaluate_c1_hf_rgb8_quality_pair(zero_pair, binding=binding)


@pytest.mark.unit
def test_c1_actual_dtype_integrity_keeps_denominator_and_failure_counts() -> None:
    binding = _binding()
    cases = [
        C1HfActualDtypeIntegrityCase(
            identity=_case_identity(index, "untouched_confirmation"),
            registered_watermarked_image_digest=QUALITY_MARKED_IMAGE.digest(),
            materialization_identity_digest=_digest(f"materialization-{index}"),
            materialization_integrity_passed=True,
            runtime_dtype="float16",
            measurement_dtype="float32",
            actual_relative_l2=0.01,
        )
        for index in range(SOURCE_CLUSTER_COUNT)
    ]
    cases[0] = replace(cases[0], materialization_integrity_passed=False)
    cases[1] = replace(
        cases[1],
        runtime_dtype="float32",
        actual_relative_l2=float("nan"),
    )
    cases[2] = replace(
        cases[2],
        measurement_dtype="float16",
        actual_relative_l2=0.013,
    )
    result = evaluate_c1_hf_actual_dtype_integrity(cases, binding=binding)
    assert result.registered_positive_count == SOURCE_CLUSTER_COUNT
    assert result.materialization_integrity_failure_count == 1
    assert result.runtime_dtype_failure_count == 1
    assert result.measurement_dtype_failure_count == 1
    assert result.non_finite_relative_l2_count == 1
    assert result.budget_exceeded_count == 1
    assert result.failed_case_count == 3


@pytest.mark.unit
def test_c1_confirmation_cross_binds_score_quality_and_actual_dtype_tables() -> None:
    binding = _binding()
    threshold = fit_c1_hf_tau(_fit_cases(), binding=binding)
    scores = _confirmation_cases(threshold.tau)
    quality_results = tuple(
        evaluate_c1_hf_rgb8_quality_pair(pair, binding=binding)
        for pair in _quality_pairs()
    )
    actual_cases = tuple(
        C1HfActualDtypeIntegrityCase(
            identity=_case_identity(index, "untouched_confirmation"),
            registered_watermarked_image_digest=QUALITY_MARKED_IMAGE.digest(),
            materialization_identity_digest=_digest(f"materialization-{index}"),
            materialization_integrity_passed=True,
            runtime_dtype="float16",
            measurement_dtype="float32",
            actual_relative_l2=0.01,
        )
        for index in range(SOURCE_CLUSTER_COUNT)
    )
    inputs = C1HfConfirmationInputBundle(
        threshold=threshold,
        score_cases=scores,
        quality_case_results=quality_results,
        actual_dtype_cases=actual_cases,
    )
    validation = validate_c1_hf_confirmation_input_bundle(inputs, binding)
    assert validation.source_cluster_count == SOURCE_CLUSTER_COUNT
    assert len(validation.cross_input_digest) == 64
    formal_results = evaluate_c1_hf_confirmation_metrics(
        inputs,
        binding=binding,
    )
    assert formal_results.cross_input_digest == validation.cross_input_digest
    assert formal_results.primary_null_fixed_fpr.metric_id == (
        "c1_hf_primary_null_fixed_fpr"
    )
    assert formal_results.paired_rgb8_quality.pair_count == (
        SOURCE_CLUSTER_COUNT
    )
    validate_c1_hf_confirmation_metric_results(formal_results, inputs, binding)
    with pytest.raises(C1HfMetricError, match="results_identity_mismatch"):
        validate_c1_hf_confirmation_metric_results(
            replace(formal_results, result_identity=_digest("forged-result")),
            inputs,
            binding,
        )

    drifted_quality = (
        replace(
            quality_results[0],
            clean_image_digest=_digest("different-clean"),
        ),
        *quality_results[1:],
    )
    with pytest.raises(C1HfMetricError, match="quality_case_result_identity"):
        validate_c1_hf_confirmation_input_bundle(
            replace(inputs, quality_case_results=drifted_quality),
            binding,
        )

    different_marked = C1Rgb8Image(
        height=1,
        width=1,
        channels=3,
        dtype="uint8",
        values_hwc=bytes((12, 20, 30)),
    )
    different_pair = C1HfQualityPair(
        identity=quality_results[0].identity,
        clean_image=QUALITY_CLEAN_IMAGE,
        registered_watermarked_image=different_marked,
        clean_image_digest=QUALITY_CLEAN_IMAGE.digest(),
        registered_watermarked_image_digest=different_marked.digest(),
    )
    valid_but_cross_drifted_quality = (
        evaluate_c1_hf_rgb8_quality_pair(
            different_pair,
            binding=binding,
        ),
        *quality_results[1:],
    )
    with pytest.raises(C1HfMetricError, match="cross_binding_mismatch"):
        validate_c1_hf_confirmation_input_bundle(
            replace(
                inputs,
                quality_case_results=valid_but_cross_drifted_quality,
            ),
            binding,
        )
