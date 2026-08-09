"""CPU checks for the internal_execution_components method adapter, attacks, and metrics."""

from __future__ import annotations

import ast
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import main
import main.content_chain as main_content_chain
import main.geometry_chain as main_geometry_chain
import main.joint_decision as main_joint_decision
import main.shared as main_shared
from experiments.attacks import (
    AttackArtifact,
    GeometricAttackError,
    GeometricAttackSpec,
    apply_geometric_attack,
    load_attack_registry,
)
from experiments.methods import (
    CegWmExperimentAdapter,
    CegWmExperimentAdapterError,
    load_ceg_wm_experiment_adapter_configuration,
)
from experiments.metrics import (
    BranchOutcomeCase,
    DetectionMetricCase,
    FixedFprThresholdResult,
    InternalMetricError,
    QualityMetricCase,
    RectificationMetricCase,
    ReliabilityMetricCase,
    RescueSafetyCase,
    RoutingPairCase,
    TransformMetricCase,
    aggregate_branch_complementarity,
    aggregate_matched_budget_quality,
    aggregate_rectification_delta,
    aggregate_reliability,
    aggregate_rescue_fpr_safety,
    aggregate_routing_gain,
    aggregate_transform_error,
    evaluate_detection_at_threshold,
    fit_fixed_fpr_threshold,
    load_metric_registry,
)
from experiments.protocol.internal_splits import (
    AnalysisUnitIdentity,
    derive_source_cluster_id,
)
from main.content_chain import (
    ContentDetectorError,
    HfDetectionObservation,
    LfDetectionObservation,
    content_detector,
    hf_detector,
    validate_content_detection_result,
)
from main.geometry_chain import (
    GeometricTransformEstimation,
    GeometryReliabilityResult,
    GeometryReliabilityThresholds,
    ImageRectificationResult,
    QkLayerObservation,
    SimilarityTransform,
)
from main.joint_decision import (
    ConditionalRecoveryResult,
    ContentDetectorBinding,
    JointDecisionThresholds,
)
from main.shared import identify_root_key, rgb8_image_digest
from runtime import RuntimeQkObservationResult


ROOT = Path(__file__).resolve().parents[2]
COMPONENT_CONFIG_PATH = (
    ROOT / "configs/experiments/internal_execution_components.json"
)
ADAPTER_MAIN_PUBLIC_OWNERS = {
    main_content_chain: (
        "BranchNullCalibration",
        "ContentDetectionResult",
        "ContentEmbeddingResult",
        "ContentRoutingResult",
        "HfCarrierResult",
        "HfDetectionObservation",
        "HfDetectionResult",
        "LfCarrierResult",
        "LfDetectionObservation",
        "LfDetectionResult",
        "LfNullWhitenedDetectionResult",
        "LfNullWhiteningAsset",
        "RoutingObservations",
        "content_detector",
        "content_embedder",
        "content_router",
        "hf_carrier",
        "hf_detector",
        "lf_carrier",
        "lf_detector",
        "lf_null_whitened_matched_detector",
    ),
    main_geometry_chain: (
        "GeometricTransformEstimation",
        "GeometryReliabilityResult",
        "GeometryReliabilityThresholds",
        "ImageRectificationResult",
        "QkGeometrySyncResult",
        "geometric_transform_estimator",
        "geometry_reliability",
        "image_rectifier",
        "qk_geometry_sync",
    ),
    main_joint_decision: (
        "ConditionalRecoveryResult",
        "ContentDetectorBinding",
        "GeometryEstimationOperation",
        "JointDecisionThresholds",
        "conditional_recovery_decision",
    ),
    main_shared: (
        "DerivedWrongKeyMaterial",
        "KeyScheduleConfig",
        "KeyStreamResult",
        "RootKeyIdentity",
        "derive_public_noise_stream",
        "derive_wrong_key_material",
        "derive_wrong_key_stream",
        "identify_root_key",
        "key_schedule_sha256_counter",
    ),
}


def _unit(index: int, *, case_id: str | None = None) -> AnalysisUnitIdentity:
    prompt_digest = f"{index + 1:064x}"
    image_lineage_digest = f"{index + 101:064x}"
    key_family_digest = f"{index + 201:064x}"
    return AnalysisUnitIdentity(
        unit_id=f"unit_{index}",
        case_id=case_id or f"case_{index}",
        source_cluster_id=derive_source_cluster_id(
            prompt_digest=prompt_digest,
            generation_seed=index,
            image_lineage_digest=image_lineage_digest,
            registered_key_family_digest=key_family_digest,
        ),
        prompt_digest=prompt_digest,
        generation_seed=index,
        image_lineage_digest=image_lineage_digest,
        registered_key_family_digest=key_family_digest,
    )


class _ActualContentOperation:
    """Real image-to-HF-to-content operation with replay validation."""

    def _detect(self, image: torch.Tensor, detection_key: str):
        observation = HfDetectionObservation.from_public_image_encoding(
            tuple((image.to(dtype=torch.float32) / 255.0).reshape(-1).tolist()),
            tuple(image.shape),
        )
        return replace(
            content_detector(hf_detector(observation, detection_key)),
            content_input_image_digest=rgb8_image_digest(image),
            content_replay_operation=self,
        )

    def __call__(self, image: torch.Tensor, detection_key: str):
        return self._detect(image, detection_key)

    def replay_validate_content_result(
        self,
        result: object,
        input_image: object,
        detection_key: str,
    ):
        if not isinstance(input_image, torch.Tensor):
            raise ContentDetectorError(
                "internal_execution_components content replay requires an RGB8 tensor"
            )
        expected = self._detect(input_image, detection_key)
        validate_content_detection_result(expected)
        if result != expected:
            raise ContentDetectorError("internal_execution_components content replay mismatch")
        return result


def _identity_estimation_record(
    registered_key: str,
) -> GeometricTransformEstimation:
    transform = SimilarityTransform(
        dihedral="identity",
        residual_rotation_degrees=0.0,
        log_scale=0.0,
        translation_x=0.0,
        translation_y=0.0,
        matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        is_exact_identity=True,
        continuous_parameter_on_search_boundary=False,
    )
    return GeometricTransformEstimation(
        candidate_ids=(
            "key_schedule_sha256_counter",
            "qk_relation_similarity",
            "rectification_similarity",
        ),
        transform=transform,
        registered_objective=1.0,
        second_registered_objective=0.8,
        exact_identity_objective=1.0,
        wrong_key_objectives=tuple(0.2 for _ in range(8)),
        canonical_score=1.0,
        observation_score=1.0,
        coverage_forward=1.0,
        coverage_backward=1.0,
        uniqueness_forward=1.0,
        uniqueness_backward=1.0,
        coverage=1.0,
        uniqueness=1.0,
        gap=0.2,
        identity_margin=0.0,
        key_margin=0.8,
        inlier_ratio=1.0,
        mean_residual=0.0,
        epsilon_inlier=0.8,
        anchor_residuals=tuple(0.0 for _ in range(12)),
        registered_root_key_public_digest=identify_root_key(
            registered_key
        ).root_key_public_digest,
        observation_descriptor_digest="1" * 64,
        observation_projection_digest="2" * 64,
        observation_geometry_config_digest="3" * 64,
        search_config_digest="4" * 64,
    )


def _reliability_thresholds() -> GeometryReliabilityThresholds:
    return GeometryReliabilityThresholds(
        gamma_coverage=0.45,
        gamma_uniqueness=0.5,
        gamma_gap=0.05,
        gamma_key=0.1,
        gamma_inlier=0.5,
        gamma_residual=0.5,
        gamma_identity=0.05,
        epsilon_inlier=0.8,
        fit_identity="adapter_geometry_reliability_fit_cpu_synthetic",
    )


def _content_binding(
    operation: _ActualContentOperation,
    image: torch.Tensor,
    detection_key: str,
) -> ContentDetectorBinding:
    prototype = operation(image, detection_key)
    return ContentDetectorBinding(
        content_detection_operation=operation,
        detector_identity=prototype.detector_identity,
        content_config_digest=prototype.content_config_digest,
        hf_detector_identity=prototype.hf_result.detector_identity,
        hf_detector_config_digest=prototype.hf_result.detector_config_digest,
        hf_template_digest=prototype.hf_result.template_digest,
        preprocessing_identity="public_image_rgb8_preprocessing_cpu_synthetic",
        formal_mode=prototype.formal_mode,
        root_key_public_digest=prototype.hf_result.root_key_public_digest,
        key_role=prototype.hf_result.key_role,
        wrong_key_index=prototype.hf_result.wrong_key_index,
    )


@pytest.mark.unit
def test_adapter_main_symbols_are_identity_preserving_top_level_exports() -> None:
    adapter_path = ROOT / "experiments/methods/ceg_wm.py"
    tree = ast.parse(
        adapter_path.read_text(encoding="utf-8"),
        filename=str(adapter_path),
    )
    imported_from_main = tuple(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "main"
        for alias in node.names
    )
    expected = {
        symbol
        for symbols in ADAPTER_MAIN_PUBLIC_OWNERS.values()
        for symbol in symbols
    }
    assert set(imported_from_main) == expected
    for owner, symbols in ADAPTER_MAIN_PUBLIC_OWNERS.items():
        for symbol in symbols:
            assert symbol in main.__all__
            assert getattr(main, symbol) is getattr(owner, symbol)


@pytest.mark.unit
def test_adapter_runs_actual_main_small_tensor_content_chain() -> None:
    configuration = load_ceg_wm_experiment_adapter_configuration(
        COMPONENT_CONFIG_PATH
    )
    adapter = CegWmExperimentAdapter(configuration)
    root_key = "internal-components-actual-main-small-tensor-key"
    shape = (1, 1, 5, 5)

    key_call = adapter.identify_key(root_key)
    secret_domain = {
        "candidate_id": "hf_sparse_tail",
        "operator": "carrier_template",
        "responsibility_domain": "hf_carrier",
        "model_revision": "b940f670f0eda2d07fbb75229e779da1ad11eb80",
        "tensor_role": "base_gaussian",
    }
    registered_stream_call = adapter.derive_registered_key_stream(
        root_key,
        secret_domain,
        (2, 2),
    )
    wrong_stream_call = adapter.derive_wrong_key_stream(
        key_call.result.root_key_public_digest,
        0,
        secret_domain,
        (2, 2),
    )
    public_stream_call = adapter.derive_public_noise(
        {
            "candidate_id": "routing_stqr",
            "operator": "local_sensitivity_public_probe",
            "responsibility_domain": "public_noise",
            "model_revision": "b940f670f0eda2d07fbb75229e779da1ad11eb80",
            "sample_index": 0,
            "tensor_role": "latent_probe",
        },
        (2, 2),
    )
    route_call = adapter.route_content(
        shape,
        mode="routing_uniform_control",
    )
    lf_carrier_call = adapter.build_lf_carrier(
        root_key,
        shape,
        routing_result=route_call.result,
    )
    hf_carrier_call = adapter.build_hf_carrier(
        root_key,
        shape,
        routing_result=route_call.result,
    )
    latent_values = tuple(float(index + 1) / 25.0 for index in range(25))
    embedding_call = adapter.embed_content(
        latent_values,
        hf_carrier_call.result,
        lf_carrier_result=lf_carrier_call.result,
        mixing_coefficient=0.5,
        routing_result=route_call.result,
    )
    public_observation_values = tuple(
        float(index % 7) / 7.0 for index in range(25)
    )
    lf_detection_call = adapter.detect_lf(
        LfDetectionObservation(
            values=public_observation_values,
            shape=shape,
        ),
        root_key,
    )
    hf_detection_call = adapter.detect_hf(
        HfDetectionObservation.from_public_image_encoding(
            public_observation_values,
            shape,
        ),
        root_key,
    )
    content_detection_call = adapter.detect_content(
        hf_detection_call.result,
        lf_detection_call.result,
    )
    generator = torch.Generator().manual_seed(2303)
    qk_runtime_result = RuntimeQkObservationResult(
        candidate_id="runtime_sd35_flowmatch",
        runtime_config_digest="1" * 64,
        model_id="stabilityai/stable-diffusion-3.5-medium",
        model_revision="b940f670f0eda2d07fbb75229e779da1ad11eb80",
        scheduler_class="FlowMatchEulerDiscreteScheduler",
        detection_schedule_index=7,
        detection_timestep=1.0,
        detection_conditioning_protocol="sd3_empty_text_triplet_without_cfg",
        public_noise_domain_digest="2" * 64,
        public_noise_values_float32_be_sha256="3" * 64,
        qk_actual_dtype="float16",
        qk_layer_observations=tuple(
            QkLayerObservation(
                layer_name=layer_name,
                query=torch.randn(2, 9, 5, generator=generator),
                attention_key=torch.randn(2, 9, 5, generator=generator),
                operator_identity="actual_cpu_projected_qk",
            )
            for layer_name in (
                "transformer_blocks.0.attn",
                "transformer_blocks.23.attn",
            )
        ),
    )
    qk_call = adapter.synchronize_qk_observation(
        qk_runtime_result,
        root_key,
    )
    estimation_call = adapter.estimate_geometric_transform(
        qk_call.result,
        root_key,
        epsilon_inlier=0.8,
    )

    calls = (
        key_call,
        route_call,
        lf_carrier_call,
        hf_carrier_call,
        embedding_call,
        lf_detection_call,
        hf_detection_call,
        content_detection_call,
        qk_call,
        estimation_call,
    )
    assert tuple(call.responsibility for call in calls) == (
        "key_schedule",
        "content_router",
        "lf_carrier",
        "hf_carrier",
        "content_embedder",
        "lf_detector",
        "hf_detector",
        "content_detector",
        "qk_geometry_sync",
        "geometric_transform_estimator",
    )
    assert all(call.adapter_config_digest == configuration.config_digest for call in calls)
    assert all(call.result_identity for call in calls)
    assert len(
        {
            registered_stream_call.result_identity,
            wrong_stream_call.result_identity,
            public_stream_call.result_identity,
        }
    ) == 3
    assert (
        registered_stream_call.result.config_digest
        == wrong_stream_call.result.config_digest
        == public_stream_call.result.config_digest
    )
    assert embedding_call.result.mode == "combined"
    assert content_detection_call.result.formal_mode == "hf_only"
    assert qk_call.result.layers[0].relation_shape == (9, 9, 4)
    assert type(estimation_call.result) is GeometricTransformEstimation
    assert (
        estimation_call.result_identity
        == estimation_call.result.estimation_identity_digest
    )
    assert estimation_call.public_callable == (
        "main.geometric_transform_estimator"
    )
    assert (
        key_call.result.root_key_public_digest
        == content_detection_call.result.hf_result.root_key_public_digest
    )
    assert key_call.public_callable == "main.identify_root_key"
    assert (
        registered_stream_call.public_callable
        == "main.key_schedule_sha256_counter"
    )
    assert (
        wrong_stream_call.public_callable
        == (
            "main.derive_wrong_key_material"
            " -> main.derive_wrong_key_stream"
        )
    )
    assert (
        public_stream_call.public_callable
        == "main.derive_public_noise_stream"
    )


@pytest.mark.unit
def test_adapter_runs_actual_reliability_rectifier_and_conditional_recovery() -> None:
    configuration = load_ceg_wm_experiment_adapter_configuration(
        COMPONENT_CONFIG_PATH
    )
    adapter = CegWmExperimentAdapter(configuration)
    root_key = "internal-components-actual-joint-adapter-key"
    estimation = _identity_estimation_record(root_key)
    reliability_call = adapter.assess_geometry_reliability(
        estimation,
        _reliability_thresholds(),
    )
    image = torch.arange(
        3 * 9 * 9,
        dtype=torch.uint8,
    ).reshape(1, 3, 9, 9)
    rectification_call = adapter.rectify_image(
        image,
        estimation,
        reliability_call.result,
    )

    operation = _ActualContentOperation()
    binding = _content_binding(operation, image, root_key)
    raw_score = operation(image, root_key).content_score
    thresholds = JointDecisionThresholds(
        tau=raw_score + 0.1,
        tau_rescue=raw_score,
        detector_binding_digest=binding.detector_binding_digest,
        calibration_identity="same_detector_threshold_calibration_cpu_synthetic",
    )
    decision_call = adapter.decide_conditional_recovery(
        image,
        root_key,
        content_detector_binding=binding,
        thresholds=thresholds,
        geometry_estimation_operation=lambda actual_image, actual_key: estimation,
        geometry_reliability_thresholds=_reliability_thresholds(),
    )

    assert type(reliability_call.result) is GeometryReliabilityResult
    assert reliability_call.result.reliable
    assert (
        reliability_call.result_identity
        == reliability_call.result.reliability_identity_digest
    )
    assert (
        reliability_call.public_callable
        == "main.geometry_reliability"
    )
    assert type(rectification_call.result) is ImageRectificationResult
    assert torch.equal(rectification_call.result.rectified_image, image)
    assert (
        rectification_call.result_identity
        == rectification_call.result.rectified_image_digest
    )
    assert (
        rectification_call.public_callable
        == "main.image_rectifier"
    )
    assert type(decision_call.result) is ConditionalRecoveryResult
    assert decision_call.result.status == "rectified_content_negative"
    assert decision_call.result.geometry_triggered
    assert (
        decision_call.result.detector_identity
        == binding.detector_identity
        == decision_call.result.raw_content_result.detector_identity
        == decision_call.result.rectified_content_result.detector_identity
    )
    assert (
        decision_call.result.threshold_identity
        == thresholds.threshold_identity
    )
    assert decision_call.result.tau == thresholds.tau
    assert decision_call.result.tau_rescue == thresholds.tau_rescue
    assert (
        decision_call.result_identity
        == decision_call.result.decision_identity_digest
    )
    assert decision_call.public_callable == (
        "main.conditional_recovery_decision"
    )


@pytest.mark.unit
def test_adapter_methods_delegate_all_thirteen_responsibilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import experiments.methods.ceg_wm as adapter_module

    configuration = load_ceg_wm_experiment_adapter_configuration(
        COMPONENT_CONFIG_PATH
    )
    adapter = CegWmExperimentAdapter(configuration)
    calls: list[str] = []

    def delegate(name: str, identity_field: str):
        def operation(*args: object, **kwargs: object):
            calls.append(name)
            values = {identity_field: f"{name}_identity"}
            if name == "qk_geometry_sync":
                values["model_revision"] = "revision"
            return SimpleNamespace(**values)

        return operation

    delegated = {
        "identify_root_key": ("key_schedule", "root_key_public_digest"),
        "content_router": ("content_router", "route_identity"),
        "lf_carrier": ("lf_carrier", "carrier_config_digest"),
        "hf_carrier": ("hf_carrier", "carrier_config_digest"),
        "content_embedder": ("content_embedder", "embedding_result_identity"),
        "lf_detector": ("lf_detector", "detector_identity"),
        "hf_detector": ("hf_detector", "detector_identity"),
        "content_detector": ("content_detector", "detector_identity"),
        "qk_geometry_sync": ("qk_geometry_sync", "geometry_config_digest"),
        "geometric_transform_estimator": (
            "geometric_transform_estimator",
            "estimation_identity_digest",
        ),
        "geometry_reliability": (
            "geometry_reliability",
            "reliability_identity_digest",
        ),
        "image_rectifier": ("image_rectifier", "rectified_image_digest"),
        "conditional_recovery_decision": (
            "conditional_recovery_decision",
            "decision_identity_digest",
        ),
    }
    for symbol, (responsibility, identity_field) in delegated.items():
        monkeypatch.setattr(
            adapter_module,
            symbol,
            delegate(responsibility, identity_field),
        )

    qk_runtime_result = RuntimeQkObservationResult(
        candidate_id="runtime_sd35_flowmatch",
        runtime_config_digest="1" * 64,
        model_id="model",
        model_revision="revision",
        scheduler_class="scheduler",
        detection_schedule_index=7,
        detection_timestep=1.0,
        detection_conditioning_protocol="condition",
        public_noise_domain_digest="2" * 64,
        public_noise_values_float32_be_sha256="3" * 64,
        qk_actual_dtype="float16",
        qk_layer_observations=(
            QkLayerObservation(
                layer_name="layer",
                query=torch.ones(1, 4, 1),
                attention_key=torch.ones(1, 4, 1),
                operator_identity="operator",
            ),
        ),
    )
    runtime_spy = SimpleNamespace(
        observe_detection_qk=lambda image: qk_runtime_result
    )
    adapter._runtime_adapter = runtime_spy

    placeholder = SimpleNamespace()
    observations = [
        adapter.identify_key("key"),
        adapter.route_content((1, 1, 2, 2), mode="routing_uniform_control"),
        adapter.build_lf_carrier("key", (1, 1, 2, 2)),
        adapter.build_hf_carrier("key", (1, 1, 2, 2)),
        adapter.embed_content((1.0,), placeholder),
        adapter.detect_lf(placeholder, "key"),
        adapter.detect_hf(placeholder, "key"),
        adapter.detect_content(placeholder),
        adapter.observe_qk_geometry(torch.zeros(1, 3, 2, 2), "key"),
        adapter.estimate_geometric_transform(
            placeholder,
            "key",
            epsilon_inlier=0.1,
        ),
        adapter.assess_geometry_reliability(placeholder),
        adapter.rectify_image(torch.zeros(1, 3, 2, 2), placeholder, placeholder),
        adapter.decide_conditional_recovery(
            torch.zeros(1, 3, 2, 2),
            "key",
            content_detector_binding=placeholder,
            thresholds=placeholder,
            geometry_estimation_operation=lambda image, key: placeholder,
            geometry_reliability_thresholds=None,
        ),
    ]
    assert calls == list(configuration_binding.responsibility for configuration_binding in configuration.component_bindings)
    assert tuple(observation.responsibility for observation in observations) == tuple(calls)


@pytest.mark.unit
def test_adapter_result_identity_is_fail_closed_and_config_has_no_fixed_fusion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import experiments.methods.ceg_wm as adapter_module

    configuration = load_ceg_wm_experiment_adapter_configuration(
        COMPONENT_CONFIG_PATH
    )
    adapter = CegWmExperimentAdapter(configuration)
    monkeypatch.setattr(
        adapter_module,
        "identify_root_key",
        lambda value: SimpleNamespace(root_key_public_digest=""),
    )
    with pytest.raises(
        CegWmExperimentAdapterError,
        match="no observable result identity",
    ):
        adapter.identify_key("key")
    config_document = json.loads(
        COMPONENT_CONFIG_PATH.read_text(encoding="utf-8")
    )
    serialized = json.dumps(config_document, sort_keys=True)
    assert '"weight"' not in serialized
    assert '"lf_weight"' not in serialized
    assert '"hf_weight"' not in serialized
    assert ": 0.7," not in serialized
    assert ": 0.3," not in serialized


@pytest.mark.unit
def test_component_registries_reject_unregistered_config_fields(
    tmp_path: Path,
) -> None:
    document = json.loads(COMPONENT_CONFIG_PATH.read_text(encoding="utf-8"))
    document["unregistered"] = True
    drifted_path = tmp_path / "drifted_components.json"
    drifted_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(
        CegWmExperimentAdapterError,
        match="fields drifted",
    ):
        load_ceg_wm_experiment_adapter_configuration(drifted_path)
    with pytest.raises(GeometricAttackError, match="fields drifted"):
        load_attack_registry(drifted_path)
    with pytest.raises(InternalMetricError, match="fields drifted"):
        load_metric_registry(drifted_path)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "forged_value"),
    (
        ("public_callable", "main.forged_callable"),
        ("result_identity_field", "forged_identity"),
    ),
)
def test_method_adapter_rejects_forged_component_binding(
    tmp_path: Path,
    field: str,
    forged_value: str,
) -> None:
    document = json.loads(COMPONENT_CONFIG_PATH.read_text(encoding="utf-8"))
    document["method_adapter"]["component_bindings"][7][field] = forged_value
    drifted_path = tmp_path / f"forged_{field}.json"
    drifted_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(
        CegWmExperimentAdapterError,
        match="canonical registry",
    ):
        load_ceg_wm_experiment_adapter_configuration(drifted_path)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("mutation_kind", "message"),
    (
        ("component_responsibility", "canonical registry"),
        ("component_callable", "canonical registry"),
        ("component_result_identity", "canonical registry"),
        ("key_operation_id", "key schedule operation"),
        ("key_operation_callable", "key schedule operation"),
        ("config_digest", "configuration digest mismatch"),
    ),
)
def test_method_adapter_revalidates_post_init_configuration_mutation(
    mutation_kind: str,
    message: str,
) -> None:
    configuration = load_ceg_wm_experiment_adapter_configuration(
        COMPONENT_CONFIG_PATH
    )
    if mutation_kind == "component_responsibility":
        object.__setattr__(
            configuration.component_bindings[0],
            "responsibility",
            "forged_responsibility",
        )
    elif mutation_kind == "component_callable":
        object.__setattr__(
            configuration.component_bindings[0],
            "public_callable",
            "forged.module.call",
        )
    elif mutation_kind == "component_result_identity":
        object.__setattr__(
            configuration.component_bindings[0],
            "result_identity_field",
            "forged_identity",
        )
    elif mutation_kind == "key_operation_id":
        object.__setattr__(
            configuration.key_schedule_operations[0],
            "operation_id",
            "forged_operation",
        )
    elif mutation_kind == "key_operation_callable":
        object.__setattr__(
            configuration.key_schedule_operations[0],
            "public_callable",
            "forged.key.call",
        )
    elif mutation_kind == "config_digest":
        object.__setattr__(
            configuration,
            "config_digest",
            "0" * 64,
        )
    else:
        raise AssertionError(f"unknown mutation kind: {mutation_kind}")

    with pytest.raises(CegWmExperimentAdapterError, match=message):
        CegWmExperimentAdapter(configuration)


@pytest.mark.unit
@pytest.mark.parametrize(
    "mutation_access",
    ("constructor_argument", "configuration_property"),
)
@pytest.mark.parametrize(
    ("mutation_kind", "message"),
    (
        ("component_binding", "canonical registry"),
        ("config_digest", "configuration digest mismatch"),
        ("key_operation", "key schedule operation"),
    ),
)
def test_method_adapter_revalidates_post_constructor_mutation_before_call(
    monkeypatch: pytest.MonkeyPatch,
    mutation_access: str,
    mutation_kind: str,
    message: str,
) -> None:
    import experiments.methods.ceg_wm as adapter_module

    configuration = load_ceg_wm_experiment_adapter_configuration(
        COMPONENT_CONFIG_PATH
    )
    adapter = CegWmExperimentAdapter(configuration)
    mutable_configuration = (
        configuration
        if mutation_access == "constructor_argument"
        else adapter.configuration
    )
    method_calls: list[str] = []

    def reject_method_call(*args: object, **kwargs: object) -> object:
        method_calls.append("called")
        raise AssertionError("method call occurred before configuration rejection")

    if mutation_kind == "component_binding":
        object.__setattr__(
            mutable_configuration.component_bindings[1],
            "public_callable",
            "forged.after.constructor",
        )
        monkeypatch.setattr(
            adapter_module,
            "content_router",
            reject_method_call,
        )
        operation = lambda: adapter.route_content(
            (1, 1, 2, 2),
            mode="routing_uniform_control",
        )
    elif mutation_kind == "config_digest":
        object.__setattr__(
            mutable_configuration,
            "config_digest",
            "f" * 64,
        )
        monkeypatch.setattr(
            adapter_module,
            "content_router",
            reject_method_call,
        )
        operation = lambda: adapter.route_content(
            (1, 1, 2, 2),
            mode="routing_uniform_control",
        )
    elif mutation_kind == "key_operation":
        object.__setattr__(
            mutable_configuration.key_schedule_operations[0],
            "public_callable",
            "forged.key.after.constructor",
        )
        monkeypatch.setattr(
            adapter_module,
            "identify_root_key",
            reject_method_call,
        )
        operation = lambda: adapter.identify_key("post-constructor-key")
    else:
        raise AssertionError(f"unknown mutation kind: {mutation_kind}")

    with pytest.raises(CegWmExperimentAdapterError, match=message):
        operation()
    assert method_calls == []


@pytest.mark.unit
def test_method_adapter_all_public_execution_entries_revalidate_configuration() -> None:
    public_execution_entries = (
        "identify_key",
        "derive_registered_key_stream",
        "derive_wrong_key_stream",
        "derive_public_noise",
        "route_content",
        "build_lf_carrier",
        "build_hf_carrier",
        "embed_content",
        "detect_lf",
        "detect_lf_null_whitened",
        "detect_hf",
        "detect_content",
        "observe_qk_geometry",
        "synchronize_qk_observation",
        "estimate_geometric_transform",
        "assess_geometry_reliability",
        "rectify_image",
        "decide_conditional_recovery",
    )
    assert all(
        hasattr(
            getattr(CegWmExperimentAdapter, entry),
            "__wrapped__",
        )
        for entry in public_execution_entries
    )


@pytest.mark.unit
def test_registries_reject_forged_wrong_key_pipeline_and_metric_split(
    tmp_path: Path,
) -> None:
    wrong_key_document = json.loads(
        COMPONENT_CONFIG_PATH.read_text(encoding="utf-8")
    )
    wrong_key_document["method_adapter"]["key_schedule_operations"][2][
        "public_callable"
    ] = "main.derive_wrong_key_stream"
    wrong_key_path = tmp_path / "forged_wrong_key_pipeline.json"
    wrong_key_path.write_text(
        json.dumps(wrong_key_document),
        encoding="utf-8",
    )
    with pytest.raises(
        CegWmExperimentAdapterError,
        match="key schedule operation",
    ):
        load_ceg_wm_experiment_adapter_configuration(wrong_key_path)

    split_document = json.loads(
        COMPONENT_CONFIG_PATH.read_text(encoding="utf-8")
    )
    split_document["metric_registry"]["metric_split_bindings"][0][
        "allowed_splits"
    ] = ["development"]
    split_path = tmp_path / "forged_metric_split.json"
    split_path.write_text(json.dumps(split_document), encoding="utf-8")
    with pytest.raises(
        InternalMetricError,
        match="canonical registry",
    ):
        load_metric_registry(split_path)


@pytest.mark.unit
def test_geometric_attacks_are_deterministic_and_identity_preserving() -> None:
    registry = load_attack_registry(COMPONENT_CONFIG_PATH)
    image = torch.arange(3 * 9 * 9, dtype=torch.uint8).reshape(1, 3, 9, 9)
    artifact = AttackArtifact(_unit(0), image)
    assert registry.crop_fraction_bounds == (0.45, 1.0)
    assert registry.rotation_degrees_bounds == (-32.0, 32.0)

    identity = apply_geometric_attack(
        artifact,
        GeometricAttackSpec("identity"),
        registry=registry,
    )
    assert torch.equal(identity.attacked_artifact.image, image)
    assert identity.source_artifact_digest == identity.attacked_artifact.image_digest

    specifications = (
        GeometricAttackSpec("crop", crop_fraction=0.75),
        GeometricAttackSpec("scale", scale_factor=1.25),
        GeometricAttackSpec("rotation", rotation_degrees=16.0),
        GeometricAttackSpec(
            "crop_scale_rotation",
            crop_fraction=0.8,
            scale_factor=1.1,
            rotation_degrees=-8.0,
        ),
    )
    for specification in specifications:
        first = apply_geometric_attack(
            artifact,
            specification,
            registry=registry,
        )
        second = apply_geometric_attack(
            artifact,
            specification,
            registry=registry,
        )
        assert first.attacked_artifact.image_digest == second.attacked_artifact.image_digest
        assert first.attack_config_digest == specification.attack_config_digest
        assert (
            first.attacked_artifact.analysis_unit_identity
            == artifact.analysis_unit_identity
        )
        assert first.attacked_artifact.image_digest != artifact.image_digest
        assert first.interpolation == "bilinear"
        assert first.padding == "zeros"
        assert first.align_corners is True


@pytest.mark.unit
def test_geometric_attacks_reject_parameter_reinterpretation() -> None:
    with pytest.raises(GeometricAttackError, match="remove border content"):
        GeometricAttackSpec("crop", scale_factor=1.1)
    with pytest.raises(GeometricAttackError, match="at least two"):
        GeometricAttackSpec(
            "crop_scale_rotation",
            rotation_degrees=8.0,
        )
    with pytest.raises(GeometricAttackError, match="\\[-32,32\\]"):
        GeometricAttackSpec("rotation", rotation_degrees=40.0)


def _detection_case(
    index: int,
    *,
    split: str,
    key_role: str,
    score: float,
) -> DetectionMetricCase:
    return DetectionMetricCase(
        analysis_unit_identity=_unit(index),
        split=split,
        detector_identity="synthetic_content_detector",
        key_role=key_role,
        score=score,
    )


@pytest.mark.unit
def test_fixed_fpr_detection_keeps_wrong_key_separate() -> None:
    registry = load_metric_registry(COMPONENT_CONFIG_PATH)
    primary_nulls = tuple(
        _detection_case(
            index,
            split="content_threshold_fit",
            key_role="unwatermarked_primary_null",
            score=score,
        )
        for index, score in enumerate((0.1, 0.2, 0.3, 0.4))
    )
    threshold = fit_fixed_fpr_threshold(
        primary_nulls,
        target_fpr=0.25,
        registry=registry,
    )
    evaluation = (
        _detection_case(
            10,
            split="end_to_end_check",
            key_role="registered_positive",
            score=0.8,
        ),
        _detection_case(
            11,
            split="end_to_end_check",
            key_role="registered_positive",
            score=0.2,
        ),
        _detection_case(
            12,
            split="end_to_end_check",
            key_role="unwatermarked_primary_null",
            score=0.1,
        ),
        _detection_case(
            13,
            split="end_to_end_check",
            key_role="wrong_key",
            score=0.6,
        ),
    )
    aggregate = evaluate_detection_at_threshold(
        evaluation,
        threshold,
        registry=registry,
    )
    assert threshold.empirical_fpr <= 0.25
    assert threshold.fpr_upper_confidence_bound > threshold.empirical_fpr
    assert aggregate.registered_tpr == 0.5
    assert aggregate.primary_null_fpr == 0.0
    assert (
        aggregate.primary_null_fpr_upper_confidence_bound
        > aggregate.primary_null_fpr
    )
    assert aggregate.wrong_key_positive_rate == 1.0
    assert aggregate.primary_null_count == 1
    assert aggregate.wrong_key_count == 1
    assert threshold.split == "content_threshold_fit"
    assert aggregate.split == "end_to_end_check"
    assert all(
        decision.split == aggregate.split for decision in aggregate.decisions
    )


@pytest.mark.unit
def test_fixed_fpr_threshold_result_rejects_direct_and_post_init_forgery() -> None:
    registry = load_metric_registry(COMPONENT_CONFIG_PATH)
    calibration_cases = tuple(
        _detection_case(
            100 + index,
            split="content_threshold_fit",
            key_role="unwatermarked_primary_null",
            score=score,
        )
        for index, score in enumerate((0.1, 0.2, 0.3, 0.4))
    )
    threshold = fit_fixed_fpr_threshold(
        calibration_cases,
        target_fpr=0.25,
        registry=registry,
    )
    reordered_threshold = fit_fixed_fpr_threshold(
        tuple(reversed(calibration_cases)),
        target_fpr=0.25,
        registry=registry,
    )
    changed_score_threshold = fit_fixed_fpr_threshold(
        (
            replace(calibration_cases[0], score=0.15),
            *calibration_cases[1:],
        ),
        target_fpr=0.25,
        registry=registry,
    )
    assert (
        reordered_threshold.calibration_case_digest
        == threshold.calibration_case_digest
    )
    assert reordered_threshold.threshold_identity == threshold.threshold_identity
    assert changed_score_threshold.threshold == threshold.threshold
    assert (
        changed_score_threshold.calibration_case_digest
        != threshold.calibration_case_digest
    )
    assert changed_score_threshold.threshold_identity != threshold.threshold_identity

    nan_payload = asdict(threshold)
    nan_payload["threshold"] = float("nan")
    with pytest.raises(InternalMetricError, match="finite"):
        FixedFprThresholdResult(**nan_payload)

    count_payload = asdict(threshold)
    count_payload["primary_null_count"] = True
    with pytest.raises(InternalMetricError, match="consistent integers"):
        FixedFprThresholdResult(**count_payload)

    digest_payload = asdict(threshold)
    digest_payload["source_cluster_digest"] = "not-a-digest"
    with pytest.raises(InternalMetricError, match="SHA-256"):
        FixedFprThresholdResult(**digest_payload)

    unbounded_finite_threshold = fit_fixed_fpr_threshold(
        tuple(
            _detection_case(
                105 + index,
                split="content_threshold_fit",
                key_role="unwatermarked_primary_null",
                score=score,
            )
            for index, score in enumerate((1.1, 1.2, 1.3, 1.4))
        ),
        target_fpr=0.25,
        registry=registry,
    )
    assert unbounded_finite_threshold.threshold > 1.0

    identity_payload = asdict(threshold)
    identity_payload["threshold_identity"] = "0" * 64
    with pytest.raises(InternalMetricError, match="identity mismatch"):
        FixedFprThresholdResult(**identity_payload)

    with pytest.raises(InternalMetricError, match="does not match.*counts"):
        replace(threshold, empirical_fpr=0.0)

    forged_after_construction = replace(threshold)
    object.__setattr__(
        forged_after_construction,
        "threshold_identity",
        "0" * 64,
    )
    evaluation = (
        _detection_case(
            110,
            split="end_to_end_check",
            key_role="registered_positive",
            score=0.8,
        ),
        _detection_case(
            111,
            split="end_to_end_check",
            key_role="unwatermarked_primary_null",
            score=0.1,
        ),
        _detection_case(
            112,
            split="end_to_end_check",
            key_role="wrong_key",
            score=0.2,
        ),
    )
    with pytest.raises(InternalMetricError, match="identity mismatch"):
        evaluate_detection_at_threshold(
            evaluation,
            forged_after_construction,
            registry=registry,
        )


@pytest.mark.unit
def test_quality_routing_and_branch_metrics_compute_real_paired_values() -> None:
    registry = load_metric_registry(COMPONENT_CONFIG_PATH)
    quality_cases = (
        QualityMetricCase(
            _unit(20),
            "untouched_confirmation",
            "routed",
            "budget_1",
            (1.0, 2.0),
            (1.0, 3.0),
        ),
        QualityMetricCase(
            _unit(21),
            "untouched_confirmation",
            "uniform",
            "budget_1",
            (2.0, 2.0),
            (2.0, 4.0),
        ),
    )
    quality = aggregate_matched_budget_quality(
        quality_cases,
        registry=registry,
    )
    assert quality.mean_relative_l2 > 0.0
    assert quality.mean_squared_error == 1.25
    assert quality.split == "untouched_confirmation"
    assert all(case.split == quality.split for case in quality.cases)

    routing = aggregate_routing_gain(
        (
            RoutingPairCase(
                _unit(22),
                "untouched_confirmation",
                True,
                False,
                0.1,
                0.2,
                "budget_1",
                "budget_1",
            ),
            RoutingPairCase(
                _unit(23),
                "untouched_confirmation",
                True,
                True,
                0.2,
                0.2,
                "budget_1",
                "budget_1",
            ),
        ),
        registry=registry,
    )
    assert routing.mean_detection_gain == 0.5
    assert routing.mean_quality_non_degradation == pytest.approx(0.05)
    assert routing.cases[0].source_cluster_id == _unit(22).source_cluster_id
    assert routing.split == "untouched_confirmation"
    assert all(case.split == routing.split for case in routing.cases)

    branches = aggregate_branch_complementarity(
        (
            BranchOutcomeCase(
                _unit(24),
                "untouched_confirmation",
                "registered_positive",
                False,
                True,
                True,
            ),
            BranchOutcomeCase(
                _unit(25),
                "untouched_confirmation",
                "wrong_key",
                False,
                False,
                False,
            ),
        ),
        registry=registry,
    )
    assert branches.lf_complements_hf_count == 1
    assert branches.combined_gain_over_hf_count == 1
    assert branches.wrong_key_combined_positive_rate == 0.0
    assert branches.cases[0].lf_complements_hf
    assert branches.split == "untouched_confirmation"
    assert all(case.split == branches.split for case in branches.cases)


@pytest.mark.unit
def test_geometry_reliability_rectification_and_rescue_metrics() -> None:
    registry = load_metric_registry(COMPONENT_CONFIG_PATH)
    transform = aggregate_transform_error(
        (
            TransformMetricCase(
                _unit(30),
                "end_to_end_check",
                10.0,
                12.0,
                1.0,
                1.1,
                0.0,
                0.1,
                0.0,
                -0.1,
                0.8,
                0.2,
            ),
        ),
        registry=registry,
    )
    assert transform.mean_rotation_absolute_error == 2.0
    assert transform.mean_translation_euclidean_error == pytest.approx(2**0.5 / 10)
    assert transform.split == "end_to_end_check"
    assert all(case.split == transform.split for case in transform.cases)

    reliability = aggregate_reliability(
        (
            ReliabilityMetricCase(
                _unit(31),
                "end_to_end_check",
                True,
                True,
            ),
            ReliabilityMetricCase(
                _unit(32),
                "end_to_end_check",
                False,
                False,
            ),
        ),
        registry=registry,
    )
    assert reliability.recoverable_accept_rate == 1.0
    assert reliability.unrecoverable_reject_rate == 1.0
    assert reliability.cases[0].expected_recoverable
    assert reliability.split == "end_to_end_check"
    assert all(case.split == reliability.split for case in reliability.cases)

    rectification = aggregate_rectification_delta(
        (
            RectificationMetricCase(
                _unit(33),
                "end_to_end_check",
                "detector",
                "detector",
                "threshold",
                "threshold",
                0.4,
                0.7,
            ),
        ),
        registry=registry,
    )
    assert rectification.mean_score_delta == pytest.approx(0.3)
    assert rectification.improved_fraction == 1.0
    assert rectification.cases[0].score_delta == pytest.approx(0.3)
    assert rectification.split == "end_to_end_check"
    assert all(case.split == rectification.split for case in rectification.cases)

    rescue = aggregate_rescue_fpr_safety(
        tuple(
            RescueSafetyCase(
                _unit(40 + index),
                "end_to_end_check",
                "detector",
                "detector",
                "threshold",
                "threshold",
                raw_positive=index == 0,
                rescue_triggered=index == 1,
                rectified_positive=index == 1,
                watermark_decision_positive=index in {0, 1},
            )
            for index in range(4)
        ),
        target_fpr=0.5,
        registry=registry,
    )
    assert rescue.raw_fpr == 0.25
    assert rescue.rescue_additional_fpr == 0.25
    assert rescue.global_fpr == 0.5
    assert rescue.global_fpr_upper_confidence_bound > rescue.global_fpr
    assert not rescue.global_fpr_within_target
    assert rescue.cases[1].rescue_additional_false_positive
    assert rescue.split == "end_to_end_check"
    assert all(case.split == rescue.split for case in rescue.cases)


def _aggregate_metric_probe(
    metric_name: str,
    splits: tuple[str, str],
) -> object:
    registry = load_metric_registry(COMPONENT_CONFIG_PATH)
    first_split, second_split = splits
    if metric_name == "detection":
        threshold = fit_fixed_fpr_threshold(
            tuple(
                _detection_case(
                    2000 + index,
                    split="content_threshold_fit",
                    key_role="unwatermarked_primary_null",
                    score=score,
                )
                for index, score in enumerate((0.1, 0.2, 0.3, 0.4))
            ),
            target_fpr=0.25,
            registry=registry,
        )
        return evaluate_detection_at_threshold(
            (
                _detection_case(
                    2010,
                    split=first_split,
                    key_role="registered_positive",
                    score=0.8,
                ),
                _detection_case(
                    2011,
                    split=second_split,
                    key_role="unwatermarked_primary_null",
                    score=0.1,
                ),
                _detection_case(
                    2012,
                    split=second_split,
                    key_role="wrong_key",
                    score=0.2,
                ),
            ),
            threshold,
            registry=registry,
        )
    if metric_name == "quality":
        return aggregate_matched_budget_quality(
            (
                QualityMetricCase(
                    _unit(2020),
                    first_split,
                    "condition",
                    "budget",
                    (1.0,),
                    (1.1,),
                ),
                QualityMetricCase(
                    _unit(2021),
                    second_split,
                    "condition",
                    "budget",
                    (1.0,),
                    (1.1,),
                ),
            ),
            registry=registry,
        )
    if metric_name == "routing":
        return aggregate_routing_gain(
            (
                RoutingPairCase(
                    _unit(2030),
                    first_split,
                    True,
                    False,
                    0.1,
                    0.2,
                    "budget",
                    "budget",
                ),
                RoutingPairCase(
                    _unit(2031),
                    second_split,
                    True,
                    False,
                    0.1,
                    0.2,
                    "budget",
                    "budget",
                ),
            ),
            registry=registry,
        )
    if metric_name == "branch":
        return aggregate_branch_complementarity(
            (
                BranchOutcomeCase(
                    _unit(2040),
                    first_split,
                    "registered_positive",
                    False,
                    True,
                    True,
                ),
                BranchOutcomeCase(
                    _unit(2041),
                    second_split,
                    "wrong_key",
                    False,
                    False,
                    False,
                ),
            ),
            registry=registry,
        )
    if metric_name == "transform":
        return aggregate_transform_error(
            (
                TransformMetricCase(
                    _unit(2050),
                    first_split,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                ),
                TransformMetricCase(
                    _unit(2051),
                    second_split,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                ),
            ),
            registry=registry,
        )
    if metric_name == "reliability":
        return aggregate_reliability(
            (
                ReliabilityMetricCase(
                    _unit(2060),
                    first_split,
                    True,
                    True,
                ),
                ReliabilityMetricCase(
                    _unit(2061),
                    second_split,
                    False,
                    False,
                ),
            ),
            registry=registry,
        )
    if metric_name == "rectification":
        return aggregate_rectification_delta(
            (
                RectificationMetricCase(
                    _unit(2070),
                    first_split,
                    "detector",
                    "detector",
                    "threshold",
                    "threshold",
                    0.1,
                    0.2,
                ),
                RectificationMetricCase(
                    _unit(2071),
                    second_split,
                    "detector",
                    "detector",
                    "threshold",
                    "threshold",
                    0.1,
                    0.2,
                ),
            ),
            registry=registry,
        )
    if metric_name == "rescue":
        return aggregate_rescue_fpr_safety(
            (
                RescueSafetyCase(
                    _unit(2080),
                    first_split,
                    "detector",
                    "detector",
                    "threshold",
                    "threshold",
                    False,
                    False,
                    False,
                    False,
                ),
                RescueSafetyCase(
                    _unit(2081),
                    second_split,
                    "detector",
                    "detector",
                    "threshold",
                    "threshold",
                    False,
                    False,
                    False,
                    False,
                ),
            ),
            target_fpr=0.25,
            registry=registry,
        )
    raise AssertionError(f"unknown metric probe: {metric_name}")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("metric_name", "legal_splits"),
    (
        ("detection", ("end_to_end_check", "end_to_end_check")),
        ("quality", ("candidate_selection", "untouched_confirmation")),
        ("routing", ("candidate_selection", "untouched_confirmation")),
        ("branch", ("candidate_selection", "untouched_confirmation")),
        ("transform", ("reliability_fit", "end_to_end_check")),
        ("reliability", ("reliability_fit", "end_to_end_check")),
        ("rectification", ("rescue_threshold_fit", "end_to_end_check")),
        ("rescue", ("rescue_threshold_fit", "end_to_end_check")),
    ),
)
def test_metric_aggregates_reject_mixed_split_identities(
    metric_name: str,
    legal_splits: tuple[str, str],
) -> None:
    first_split, second_split = legal_splits
    if first_split == second_split:
        second_split = "content_threshold_fit"
    with pytest.raises(InternalMetricError, match="cannot mix split"):
        _aggregate_metric_probe(metric_name, (first_split, second_split))


@pytest.mark.unit
@pytest.mark.parametrize(
    "metric_name",
    (
        "detection",
        "quality",
        "routing",
        "branch",
        "transform",
        "reliability",
        "rectification",
        "rescue",
    ),
)
def test_metric_aggregates_reject_uniform_but_illegal_split(
    metric_name: str,
) -> None:
    with pytest.raises(InternalMetricError, match="does not allow split"):
        _aggregate_metric_probe(metric_name, ("development", "development"))


@pytest.mark.unit
def test_rescue_safety_case_enforces_real_rescue_trajectory() -> None:
    raw_positive = RescueSafetyCase(
        _unit(2100),
        "end_to_end_check",
        "detector",
        "detector",
        "threshold",
        "threshold",
        True,
        False,
        False,
        True,
    )
    rescued_positive = RescueSafetyCase(
        _unit(2101),
        "end_to_end_check",
        "detector",
        "detector",
        "threshold",
        "threshold",
        False,
        True,
        True,
        True,
    )
    assert raw_positive.watermark_decision_positive
    assert rescued_positive.watermark_decision_positive

    with pytest.raises(InternalMetricError, match="must not trigger rescue"):
        RescueSafetyCase(
            _unit(2102),
            "end_to_end_check",
            "detector",
            "detector",
            "threshold",
            "threshold",
            True,
            True,
            True,
            True,
        )
    with pytest.raises(InternalMetricError, match="actual rescue trigger"):
        RescueSafetyCase(
            _unit(2103),
            "end_to_end_check",
            "detector",
            "detector",
            "threshold",
            "threshold",
            False,
            False,
            True,
            False,
        )
    with pytest.raises(InternalMetricError, match="trajectory"):
        RescueSafetyCase(
            _unit(2104),
            "end_to_end_check",
            "detector",
            "detector",
            "threshold",
            "threshold",
            False,
            True,
            True,
            False,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("raw_positive", True, "must not trigger rescue"),
        ("rescue_triggered", False, "actual rescue trigger"),
        ("rectified_positive", False, "trajectory"),
        ("watermark_decision_positive", False, "trajectory"),
        ("raw_positive", "true", "must be boolean"),
        (
            "rectified_detector_identity",
            "forged_detector",
            "detector identity mismatch",
        ),
        (
            "rectified_threshold_identity",
            "forged_threshold",
            "threshold identity mismatch",
        ),
    ),
)
def test_rescue_aggregate_revalidates_post_init_case_mutation(
    field: str,
    value: object,
    message: str,
) -> None:
    case = RescueSafetyCase(
        _unit(2110),
        "end_to_end_check",
        "detector",
        "detector",
        "threshold",
        "threshold",
        False,
        True,
        True,
        True,
    )
    object.__setattr__(case, field, value)

    with pytest.raises(InternalMetricError, match=message):
        aggregate_rescue_fpr_safety(
            (case,),
            target_fpr=0.5,
            registry=load_metric_registry(COMPONENT_CONFIG_PATH),
        )


@pytest.mark.unit
def test_metrics_fail_closed_on_held_out_empty_nonfinite_and_identity_drift() -> None:
    registry = load_metric_registry(COMPONENT_CONFIG_PATH)
    with pytest.raises(PermissionError, match="held_out"):
        _detection_case(
            50,
            split="held_out_evaluation",
            key_role="registered_positive",
            score=0.1,
        )
    with pytest.raises(InternalMetricError, match="non-empty"):
        aggregate_reliability((), registry=registry)
    with pytest.raises(InternalMetricError, match="finite"):
        TransformMetricCase(
            _unit(51),
            "end_to_end_check",
            float("nan"),
            0.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        )
    with pytest.raises(InternalMetricError, match="detector identity mismatch"):
        RectificationMetricCase(
            _unit(52),
            "end_to_end_check",
            "raw_detector",
            "rectified_detector",
            "threshold",
            "threshold",
            0.1,
            0.2,
        )


@pytest.mark.unit
def test_experiment_method_adapter_imports_only_top_level_project_surfaces() -> None:
    adapter_path = ROOT / "experiments/methods/ceg_wm.py"
    source = adapter_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(adapter_path))
    project_modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            project_modules.extend(
                alias.name
                for alias in node.names
                if alias.name.split(".", 1)[0]
                in {"main", "runtime", "experiments"}
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".", 1)[0] in {
                "main",
                "runtime",
                "experiments",
            }:
                project_modules.append(node.module)
    assert set(project_modules) == {"main", "runtime"}
    assert all(
        module in {"main", "runtime", "experiments.protocol"}
        or module.startswith("experiments.protocol.")
        for module in project_modules
    )
    assert "importlib" not in source
    assert "__import__" not in source


@pytest.mark.unit
def test_experiment_method_attack_and_metric_layers_keep_project_dependencies_orthogonal() -> None:
    allowed_project_prefixes = {
        ROOT / "experiments/methods/ceg_wm.py": (
            "main",
            "runtime",
            "experiments.protocol",
        ),
        ROOT / "experiments/attacks/geometric.py": ("experiments.protocol",),
        ROOT / "experiments/metrics/internal.py": ("experiments.protocol",),
    }
    forbidden_tokens = {
        ROOT / "experiments/methods/ceg_wm.py": (
            "avg_pool2d",
            "grid_sample",
            "normal_quantile_table_lookup",
            "write_record",
        ),
    }
    for path, allowed_prefixes in allowed_project_prefixes.items():
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        project_imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            else:
                continue
            project_imports.extend(
                name
                for name in names
                if name.split(".", 1)[0]
                in {
                    "main",
                    "runtime",
                    "experiments",
                    "governance",
                    "paper_artifacts",
                }
            )
        assert all(
            any(
                name == prefix or name.startswith(f"{prefix}.")
                for prefix in allowed_prefixes
            )
            for name in project_imports
        )
        for token in forbidden_tokens.get(path, ()):
            assert token not in source
