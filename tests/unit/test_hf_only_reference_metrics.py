"""CPU unit checks for the seven executable hf_only_reference_validation-reference metrics."""

from __future__ import annotations

import ast
from dataclasses import asdict, replace
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import shutil

import pytest

from experiments.metrics import (
    HfOnlyReferenceActualDtypeIntegrityCase,
    HfOnlyReferenceConfirmationInputBundle,
    HfOnlyReferenceMetricCaseIdentity,
    HfOnlyReferenceMetricError,
    HfOnlyReferenceQualityPair,
    HfOnlyReferenceRawRgb8QualityArtifact,
    HfOnlyReferenceScoreCase,
    HfOnlyReferenceRgb8Image,
    clopper_pearson_lower,
    clopper_pearson_upper,
    evaluate_hf_only_reference_actual_dtype_integrity,
    evaluate_hf_only_reference_confirmation_metrics,
    evaluate_hf_only_reference_paired_rgb8_quality,
    evaluate_hf_only_reference_paired_key_attribution,
    evaluate_hf_only_reference_primary_null_fixed_fpr,
    evaluate_hf_only_reference_registered_tpr,
    evaluate_hf_only_reference_rgb8_quality_pair,
    evaluate_hf_only_reference_wrong_key_false_accept,
    fit_hf_only_reference_tau,
    load_hf_only_reference_metric_implementation_binding,
    student_t_quantile_975,
    validate_hf_only_reference_confirmation_input_bundle,
    validate_hf_only_reference_confirmation_metric_results,
    validate_hf_only_reference_quality_case_result,
    validate_hf_only_reference_threshold_result,
)


ROOT = Path(__file__).resolve().parents[2]
HF_ONLY_REFERENCE_METRIC_MODULE = ROOT / "experiments/metrics/hf_only_reference_metrics.py"
SOURCE_CLUSTER_COUNT = 4096
REGISTERED_KEY_DIGEST = hashlib.sha256(b"registered-key").hexdigest()
DETECTOR_CONFIG_DIGEST = hashlib.sha256(b"detector-config").hexdigest()
QUALITY_CLEAN_IMAGE = HfOnlyReferenceRgb8Image(
    height=1,
    width=1,
    channels=3,
    dtype="uint8",
    values_hwc=bytes((10, 20, 30)),
)
QUALITY_MARKED_IMAGE = HfOnlyReferenceRgb8Image(
    height=1,
    width=1,
    channels=3,
    dtype="uint8",
    values_hwc=bytes((11, 20, 30)),
)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@lru_cache(maxsize=1)
def _binding():
    return load_hf_only_reference_metric_implementation_binding()


@lru_cache(maxsize=2)
def _ordered_analysis_units(split: str):
    binding = _binding()
    units = (
        binding.fit_analysis_units
        if split == "content_threshold_fit"
        else binding.confirmation_analysis_units
    )
    return tuple(sorted(units, key=lambda identity: identity.unit_id))


def _case_identity(index: int, split: str) -> HfOnlyReferenceMetricCaseIdentity:
    binding = _binding()
    manifest_digest = (
        binding.fit_manifest_digest
        if split == "content_threshold_fit"
        else binding.confirmation_manifest_digest
    )
    return HfOnlyReferenceMetricCaseIdentity(
        analysis_unit_identity=_ordered_analysis_units(split)[index],
        split=split,
        detector_identity="ceg_wm_hf_only_detector",
        detector_config_digest=DETECTOR_CONFIG_DIGEST,
        protocol_id="ceg_wm_internal_scientific_validation_v2",
        protocol_version="2.0.0",
        protocol_digest=binding.protocol_digest,
        hf_only_reference_specification_digest=binding.hf_only_reference_specification_digest,
        manifest_digest=manifest_digest,
        metric_registry_digest=binding.metric_registry_digest,
        registered_key_family_digest=binding.registered_key_family_digest,
    )


def _fit_cases() -> tuple[HfOnlyReferenceScoreCase, ...]:
    return tuple(
        HfOnlyReferenceScoreCase(
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


def _confirmation_cases(tau: float) -> tuple[HfOnlyReferenceScoreCase, ...]:
    cases = []
    for index in range(SOURCE_CLUSTER_COUNT):
        identity = _case_identity(index, "untouched_confirmation")
        marked_digest = QUALITY_MARKED_IMAGE.digest()
        cases.extend(
            (
                HfOnlyReferenceScoreCase(
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
                HfOnlyReferenceScoreCase(
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
                HfOnlyReferenceScoreCase(
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


def _quality_pairs() -> tuple[HfOnlyReferenceQualityPair, ...]:
    return tuple(
        HfOnlyReferenceQualityPair(
            identity=_case_identity(index, "untouched_confirmation"),
            clean_image=QUALITY_CLEAN_IMAGE,
            registered_watermarked_image=QUALITY_MARKED_IMAGE,
            clean_image_digest=QUALITY_CLEAN_IMAGE.digest(),
            registered_watermarked_image_digest=QUALITY_MARKED_IMAGE.digest(),
        )
        for index in range(SOURCE_CLUSTER_COUNT)
    )


def _quality_artifacts(tmp_path: Path) -> tuple[HfOnlyReferenceRawRgb8QualityArtifact, ...]:
    clean_path = tmp_path / "clean.rgb8"
    marked_path = tmp_path / "marked.rgb8"
    clean_path.write_bytes(QUALITY_CLEAN_IMAGE.values_hwc)
    marked_path.write_bytes(QUALITY_MARKED_IMAGE.values_hwc)
    return tuple(
        HfOnlyReferenceRawRgb8QualityArtifact(
            identity=_case_identity(index, "untouched_confirmation"),
            height=QUALITY_CLEAN_IMAGE.height,
            width=QUALITY_CLEAN_IMAGE.width,
            channels=QUALITY_CLEAN_IMAGE.channels,
            dtype=QUALITY_CLEAN_IMAGE.dtype,
            clean_artifact_path=str(clean_path.resolve()),
            clean_artifact_sha256=hashlib.sha256(
                QUALITY_CLEAN_IMAGE.values_hwc
            ).hexdigest(),
            registered_watermarked_artifact_path=str(marked_path.resolve()),
            registered_watermarked_artifact_sha256=hashlib.sha256(
                QUALITY_MARKED_IMAGE.values_hwc
            ).hexdigest(),
        )
        for index in range(SOURCE_CLUSTER_COUNT)
    )


@pytest.mark.unit
def test_hf_only_reference_metric_binding_binds_formula_registry_source_and_safe_dependencies() -> None:
    binding = _binding()
    assert (
        binding.hf_only_reference_specification_digest
        == "dce7479c5969567a118db250f81de6790590ecf416c2cca9ce68ddd6d325b483"
    )
    assert binding.implementation_source_sha256 == hashlib.sha256(
        HF_ONLY_REFERENCE_METRIC_MODULE.read_bytes()
    ).hexdigest()
    assert len(binding.formula_identity_digest) == 64

    tree = ast.parse(HF_ONLY_REFERENCE_METRIC_MODULE.read_text(encoding="utf-8"))
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
def test_hf_only_reference_metric_binding_rejects_candidate_authority_byte_tamper(tmp_path) -> None:
    binding = _binding()
    specification = json.loads(
        (ROOT / "configs/experiments/hf_only_reference_validation.json").read_text()
    )
    candidate = specification["candidate_binding"]
    temporary_root = tmp_path / "repository"
    paths = {
        Path("configs/experiments/hf_only_reference_metrics.json"),
        Path("configs/experiments/hf_only_reference_validation.json"),
        Path("configs/experiments/internal_execution_components.json"),
        Path("experiments/metrics/hf_only_reference_metrics.py"),
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
        load_hf_only_reference_metric_implementation_binding(
            temporary_root
            / "configs/experiments/hf_only_reference_metrics.json"
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
def test_hf_only_reference_tau_fit_uses_all_clusters_nextafter_and_rejects_drift() -> None:
    binding = _binding()
    cases = _fit_cases()
    threshold = fit_hf_only_reference_tau(cases, binding=binding)
    assert threshold.source_cluster_count == SOURCE_CLUSTER_COUNT
    assert threshold.tau == math.nextafter(max(case.score for case in cases), math.inf)
    assert threshold.tau_float64_hex == threshold.tau.hex()
    assert threshold.fit_false_positive_count == 0
    validate_hf_only_reference_threshold_result(threshold, binding)

    duplicate_identity = cases[0].identity
    duplicate_cases = (*cases[:-1], replace(cases[-1], identity=duplicate_identity))
    with pytest.raises(HfOnlyReferenceMetricError, match="analysis_unit_manifest_set_mismatch"):
        fit_hf_only_reference_tau(duplicate_cases, binding=binding)
    with pytest.raises(HfOnlyReferenceMetricError, match="finite_float64"):
        replace(cases[0], score=float("nan"))


@pytest.mark.unit
def test_hf_only_reference_confirmation_metrics_are_separate_paired_and_threshold_bound() -> None:
    binding = _binding()
    threshold = fit_hf_only_reference_tau(_fit_cases(), binding=binding)
    cases = _confirmation_cases(threshold.tau)

    primary = evaluate_hf_only_reference_primary_null_fixed_fpr(
        cases,
        threshold,
        binding=binding,
    )
    registered = evaluate_hf_only_reference_registered_tpr(
        cases,
        threshold,
        binding=binding,
    )
    wrong = evaluate_hf_only_reference_wrong_key_false_accept(
        cases,
        threshold,
        binding=binding,
    )
    attribution = evaluate_hf_only_reference_paired_key_attribution(
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
    with pytest.raises(HfOnlyReferenceMetricError, match="threshold"):
        evaluate_hf_only_reference_registered_tpr(cases, drifted, binding=binding)
    missing = tuple(cases[:-1])
    with pytest.raises(HfOnlyReferenceMetricError, match="exact_three_roles"):
        evaluate_hf_only_reference_wrong_key_false_accept(
            missing,
            threshold,
            binding=binding,
        )


@pytest.mark.unit
def test_hf_only_reference_quality_consumes_rgb8_bytes_and_uses_unclipped_student_t_interval() -> None:
    binding = _binding()
    pairs = _quality_pairs()
    single = evaluate_hf_only_reference_rgb8_quality_pair(pairs[0], binding=binding)
    assert single.relative_l2 == pytest.approx(1.0 / math.sqrt(1400.0))
    assert single.normalized_rgb8_mse == pytest.approx(1.0 / (255.0**2 * 3.0))

    case_results = tuple(
        evaluate_hf_only_reference_rgb8_quality_pair(pair, binding=binding)
        for pair in pairs
    )
    result = evaluate_hf_only_reference_paired_rgb8_quality(
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

    zero = HfOnlyReferenceRgb8Image(1, 1, 3, "uint8", bytes((0, 0, 0)))
    zero_pair = replace(
        pairs[0],
        clean_image=zero,
        clean_image_digest=zero.digest(),
    )
    with pytest.raises(HfOnlyReferenceMetricError, match="clean_l2_zero"):
        evaluate_hf_only_reference_rgb8_quality_pair(zero_pair, binding=binding)


@pytest.mark.unit
def test_hf_only_reference_actual_dtype_integrity_keeps_denominator_and_failure_counts() -> None:
    binding = _binding()
    cases = [
        HfOnlyReferenceActualDtypeIntegrityCase(
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
    result = evaluate_hf_only_reference_actual_dtype_integrity(cases, binding=binding)
    assert result.registered_positive_count == SOURCE_CLUSTER_COUNT
    assert result.materialization_integrity_failure_count == 1
    assert result.runtime_dtype_failure_count == 1
    assert result.measurement_dtype_failure_count == 1
    assert result.non_finite_relative_l2_count == 1
    assert result.budget_exceeded_count == 1
    assert result.failed_case_count == 3


@pytest.mark.unit
def test_hf_only_reference_confirmation_replays_fit_and_raw_quality_authority(
    tmp_path: Path,
) -> None:
    binding = _binding()
    fit_cases = _fit_cases()
    threshold = fit_hf_only_reference_tau(fit_cases, binding=binding)
    scores = _confirmation_cases(threshold.tau)
    quality_artifacts = _quality_artifacts(tmp_path)
    quality_results = tuple(
        evaluate_hf_only_reference_rgb8_quality_pair(pair, binding=binding)
        for pair in _quality_pairs()
    )
    actual_cases = tuple(
        HfOnlyReferenceActualDtypeIntegrityCase(
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
    inputs = HfOnlyReferenceConfirmationInputBundle(
        fit_primary_null_cases=fit_cases,
        threshold=threshold,
        score_cases=scores,
        raw_rgb8_quality_artifacts=quality_artifacts,
        actual_dtype_cases=actual_cases,
    )
    validation = validate_hf_only_reference_confirmation_input_bundle(inputs, binding)
    assert validation.source_cluster_count == SOURCE_CLUSTER_COUNT
    assert len(validation.cross_input_digest) == 64
    formal_results = evaluate_hf_only_reference_confirmation_metrics(
        inputs,
        binding=binding,
    )
    assert formal_results.cross_input_digest == validation.cross_input_digest
    assert formal_results.primary_null_fixed_fpr.metric_id == (
        "hf_only_reference_primary_null_fixed_fpr"
    )
    assert formal_results.paired_rgb8_quality.pair_count == (
        SOURCE_CLUSTER_COUNT
    )
    validate_hf_only_reference_confirmation_metric_results(formal_results, inputs, binding)
    with pytest.raises(HfOnlyReferenceMetricError, match="results_identity_mismatch"):
        validate_hf_only_reference_confirmation_metric_results(
            replace(formal_results, result_identity=_digest("forged-result")),
            inputs,
            binding,
        )

    fake_quality_payload = {
        "analysis_unit_identity": asdict(
            quality_results[0].identity.analysis_unit_identity
        ),
        "relative_l2": (0.0).hex(),
        "normalized_rgb8_mse": (
            quality_results[0].normalized_rgb8_mse.hex()
        ),
        "clean_image_digest": quality_results[0].clean_image_digest,
        "registered_watermarked_image_digest": (
            quality_results[0].registered_watermarked_image_digest
        ),
        "formula_identity_digest": quality_results[0].formula_identity_digest,
    }
    valid_self_rehashed_fake_quality = replace(
        quality_results[0],
        relative_l2=0.0,
        result_identity=_canonical_digest(fake_quality_payload),
    )
    validate_hf_only_reference_quality_case_result(valid_self_rehashed_fake_quality, binding)
    with pytest.raises(HfOnlyReferenceMetricError, match="artifact_count_or_type"):
        evaluate_hf_only_reference_confirmation_metrics(
            replace(
                inputs,
                raw_rgb8_quality_artifacts=(
                    valid_self_rehashed_fake_quality,
                    *quality_artifacts[1:],
                ),
            ),
            binding=binding,
        )

    wrong_tau = math.nextafter(threshold.tau, math.inf)
    wrong_threshold_without_identity = replace(
        threshold,
        tau=wrong_tau,
        tau_float64_hex=wrong_tau.hex(),
    )
    wrong_threshold_payload = {
        key: getattr(wrong_threshold_without_identity, key)
        for key in wrong_threshold_without_identity.__dataclass_fields__
        if key != "threshold_identity"
    }
    valid_self_rehashed_wrong_threshold = replace(
        wrong_threshold_without_identity,
        threshold_identity=_canonical_digest(wrong_threshold_payload),
    )
    validate_hf_only_reference_threshold_result(
        valid_self_rehashed_wrong_threshold,
        binding,
    )
    with pytest.raises(HfOnlyReferenceMetricError, match="threshold_fit_replay_mismatch"):
        evaluate_hf_only_reference_confirmation_metrics(
            replace(inputs, threshold=valid_self_rehashed_wrong_threshold),
            binding=binding,
        )

    different_marked_path = tmp_path / "different-marked.rgb8"
    different_marked = HfOnlyReferenceRgb8Image(
        height=1,
        width=1,
        channels=3,
        dtype="uint8",
        values_hwc=bytes((12, 20, 30)),
    )
    different_marked_path.write_bytes(different_marked.values_hwc)
    valid_but_cross_drifted_artifacts = (
        replace(
            quality_artifacts[0],
            registered_watermarked_artifact_path=str(
                different_marked_path.resolve()
            ),
            registered_watermarked_artifact_sha256=hashlib.sha256(
                different_marked.values_hwc
            ).hexdigest(),
        ),
        *quality_artifacts[1:],
    )
    with pytest.raises(HfOnlyReferenceMetricError, match="cross_binding_mismatch"):
        validate_hf_only_reference_confirmation_input_bundle(
            replace(
                inputs,
                raw_rgb8_quality_artifacts=valid_but_cross_drifted_artifacts,
            ),
            binding,
        )
