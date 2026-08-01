from dataclasses import replace
from hashlib import sha256
from math import sqrt
from struct import pack, unpack

import pytest

from main.content_chain.detector import (
    BranchNullCalibration,
    ContentDetectorError,
    NullScoreRecord,
    content_detector,
    validate_content_detection_result,
)
from main.content_chain.embedder import (
    ContentEmbeddingResult,
    ContentEmbedderError,
    ContentMaterializationObservation,
    content_actual_budget_accepts,
    content_materialization_replay_identity,
    content_embedder,
    reconcile_content_materialization_budget,
    scale_content_delta_binary32,
)
from main.content_chain.hf_carrier import hf_carrier
from main.content_chain.hf_detector import (
    HfDetectionObservation,
    hf_detector,
)
from main.content_chain.lf_carrier import lf_carrier
from main.content_chain.lf_detector import (
    LfDetectionObservation,
    lf_detector,
)
from main.content_chain.routing import (
    ContentRouterError,
    ContentRoutingResult,
    RoutingObservations,
    SpatialRoutingObservation,
    content_router,
    validate_content_routing_result,
)
from main.shared.key_schedule import (
    KeyScheduleError,
    derive_wrong_key_material,
    identify_root_key,
    normal_quantile_table_lookup,
    stable_json_utf8,
)

BATCH3_ROOT = "ceg-wm-batch-three-root-π"
BATCH3_SHAPE = (1, 2, 6, 6)


def _source_digest(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


def _spatial(
    values: tuple[float, ...],
    shape: tuple[int, int],
    label: str,
) -> SpatialRoutingObservation:
    return SpatialRoutingObservation(
        values=values,
        spatial_shape=shape,
        source_identity_digest=_source_digest(label),
    )


def _observations(texture_reverse: bool = False) -> RoutingObservations:
    texture = (
        (1.0, 0.75, 0.25, 0.0, 0.5, 0.25)
        if texture_reverse
        else (0.0, 0.25, 0.75, 1.0, 0.5, 0.75)
    )
    return RoutingObservations(
        semantic=_spatial((0.0, 0.8, 0.3, 0.6), (2, 2), "semantic"),
        texture=_spatial(texture, (3, 2), "texture"),
        response=_spatial((0.0, 0.25, 0.5), (1, 3), "response"),
        sensitivity=_spatial((0.1, 0.4), (2, 1), "sensitivity"),
    )


def _route(
    shape: tuple[int, int, int, int] = BATCH3_SHAPE,
    *,
    texture_reverse: bool = False,
) -> ContentRoutingResult:
    return content_router(
        shape,
        mode="routing_stqr",
        observations=_observations(texture_reverse),
    )


def _latent(size: int) -> tuple[float, ...]:
    return tuple(((index % 19) - 9) / 10.0 for index in range(size))


def _float32_digest(values: tuple[float, ...]) -> str:
    return sha256(b"".join(pack(">f", value) for value in values)).hexdigest()


def _test_float32(value: float) -> float:
    return unpack(">f", pack(">f", value))[0]


def _next_positive_float32(value: float) -> float:
    bits = int.from_bytes(pack(">f", value), byteorder="big", signed=False)
    return unpack(">f", (bits + 1).to_bytes(4, byteorder="big"))[0]


def _test_l2_float32(values: tuple[float, ...]) -> float:
    accumulator = 0.0
    for value in values:
        squared = _test_float32(value * value)
        accumulator = _test_float32(accumulator + squared)
    return _test_float32(sqrt(accumulator))


def _related_actual_delta(
    scaled_nominal_delta: tuple[float, ...],
    target_norm: float,
) -> tuple[float, ...]:
    maximum = max(abs(value) for value in scaled_nominal_delta)
    if maximum == 0.0:
        return (0.0,) * len(scaled_nominal_delta)
    direction = tuple(
        _test_float32(value / maximum)
        for value in scaled_nominal_delta
    )
    direction_norm = _test_l2_float32(direction)
    multiplier = _test_float32(target_norm / direction_norm)
    return tuple(
        _test_float32(value * multiplier)
        for value in direction
    )


class _MonotoneActualMaterializer:
    """用于隔离验证 embedder binary32 搜索的单调 actual-dtype 假实现。"""

    def __init__(
        self,
        *,
        greatest_feasible_scale: float,
        zero_through_scale: float = 0.0,
        feasible_utilization: float = 1.0,
        replay_passed: bool = True,
    ) -> None:
        self.greatest_feasible_scale = _test_float32(
            greatest_feasible_scale
        )
        self.zero_through_scale = _test_float32(zero_through_scale)
        self.feasible_utilization = _test_float32(feasible_utilization)
        self.replay_passed = replay_passed
        self.calls: list[float] = []

    def __call__(
        self,
        embedding_result: ContentEmbeddingResult,
        materialization_scale: float,
        /,
    ) -> ContentMaterializationObservation:
        scale = _test_float32(materialization_scale)
        self.calls.append(scale)
        scaled_nominal_delta = scale_content_delta_binary32(
            embedding_result,
            scale,
        )
        scaled_nominal_delta_digest = _float32_digest(
            scaled_nominal_delta
        )
        limit = _test_float32(
            embedding_result.latent_norm * _test_float32(3.0 / 250.0)
        )
        if scale <= self.zero_through_scale:
            delta = (0.0,) * len(embedding_result.delta_content)
            status = "write_disappeared"
        elif scale <= self.greatest_feasible_scale:
            delta = _related_actual_delta(
                scaled_nominal_delta,
                _test_float32(limit * self.feasible_utilization),
            )
            status = "passed"
        else:
            delta = _related_actual_delta(
                scaled_nominal_delta,
                _test_float32(limit * 2.0),
            )
            status = "passed"
        realized_total_l2 = _test_l2_float32(delta)
        replay_identity = content_materialization_replay_identity(
            embedding_result,
            materialization_scale=scale,
            scaled_nominal_delta_digest=scaled_nominal_delta_digest,
            baseline_norm=embedding_result.latent_norm,
            delta_content_actual=delta,
            realized_total_l2=realized_total_l2,
            integrity_status=status,
        )
        return ContentMaterializationObservation(
            materialization_scale=scale,
            baseline_norm=embedding_result.latent_norm,
            scaled_nominal_delta_digest=scaled_nominal_delta_digest,
            delta_content_actual=delta,
            realized_total_l2=realized_total_l2,
            integrity_status=status,
            deterministic_binary16_replay_passed=self.replay_passed,
            materialization_replay_identity=replay_identity,
        )


def _branch_results():
    hf_template = hf_carrier(BATCH3_ROOT, BATCH3_SHAPE)
    lf_template = lf_carrier(BATCH3_ROOT, BATCH3_SHAPE)
    image_encoding = tuple(
        hf_value + lf_value + 0.25
        for hf_value, lf_value in zip(
            hf_template.template,
            lf_template.template,
            strict=True,
        )
    )
    hf_observation = HfDetectionObservation.from_public_image_encoding(
        image_encoding,
        BATCH3_SHAPE,
    )
    lf_observation = LfDetectionObservation.from_public_image_encoding(
        image_encoding,
        BATCH3_SHAPE,
    )
    return (
        hf_detector(hf_observation, BATCH3_ROOT),
        lf_detector(lf_observation, BATCH3_ROOT),
    )


def _null_calibration(
    branch: str,
    detector_identity: str,
    scores: tuple[float, ...],
    *,
    partition: str = "candidate_selection_selection_partition",
    reverse: bool = False,
) -> BranchNullCalibration:
    records = tuple(
        NullScoreRecord(
            score=score,
            source_cluster_id=f"cluster-{index:02d}",
            sample_id=f"sample-{index:02d}",
        )
        for index, score in enumerate(scores)
    )
    if reverse:
        records = tuple(reversed(records))
    return BranchNullCalibration(
        branch=branch,  # type: ignore[arg-type]
        detector_identity=detector_identity,
        partition_identity=partition,
        records=records,
    )


@pytest.mark.unit
def test_lf_domain_and_independent_key() -> None:
    lf_result = lf_carrier(BATCH3_ROOT, BATCH3_SHAPE)
    repeated = lf_carrier(BATCH3_ROOT, BATCH3_SHAPE)
    hf_result = hf_carrier(BATCH3_ROOT, BATCH3_SHAPE)

    assert lf_result.candidate_ids == (
        "key_schedule_sha256_counter",
        "lf_low_pass",
    )
    assert lf_result.template == repeated.template
    assert lf_result.key_domain_digest == repeated.key_domain_digest
    assert lf_result.key_domain_digest != hf_result.key_domain_digest
    assert lf_result.template_digest != hf_result.template_digest
    assert sum(lf_result.template) == pytest.approx(0.0, abs=2e-6)
    assert sqrt(sum(value * value for value in lf_result.template)) == pytest.approx(
        1.0,
        abs=2e-6,
    )
    assert sum(value != 0.0 for value in lf_result.template) > len(
        lf_result.template
    ) * 0.8


@pytest.mark.unit
def test_lf_blind_score_time_centering() -> None:
    carrier = lf_carrier(BATCH3_ROOT, BATCH3_SHAPE)
    positive_shift = LfDetectionObservation.from_public_image_encoding(
        tuple(value + 5.0 for value in carrier.template),
        carrier.shape,
    )
    negative_shift = LfDetectionObservation.from_public_image_encoding(
        tuple(value - 7.5 for value in carrier.template),
        carrier.shape,
    )

    first = lf_detector(positive_shift, BATCH3_ROOT)
    second = lf_detector(negative_shift, BATCH3_ROOT)
    assert first.lf_score > 0.999
    assert second.lf_score == pytest.approx(first.lf_score, abs=2e-6)
    assert first.observation_digest != second.observation_digest
    assert not hasattr(positive_shift, "routing_mask")
    assert not hasattr(positive_shift, "embed_record")


@pytest.mark.unit
def test_lf_wrong_key_rejection() -> None:
    carrier = lf_carrier(BATCH3_ROOT, BATCH3_SHAPE)
    observation = LfDetectionObservation.from_public_image_encoding(
        carrier.template,
        carrier.shape,
    )
    root_identity = identify_root_key(BATCH3_ROOT)
    wrong_key = derive_wrong_key_material(
        root_identity.root_key_public_digest,
        3,
    )

    registered = lf_detector(observation, BATCH3_ROOT)
    wrong = lf_detector(observation, wrong_key)
    assert registered.lf_score > 0.999
    assert wrong.lf_score < 0.5
    assert registered.template_digest != wrong.template_digest
    assert registered.key_role == "registered"
    assert wrong.key_role == "wrong"
    assert wrong.wrong_key_index == 3
    assert registered.root_key_public_digest == wrong.root_key_public_digest


@pytest.mark.unit
def test_routing_mask_partition_and_range() -> None:
    observations = RoutingObservations(
        semantic=_spatial((0.0, 0.8), (1, 2), "s-align"),
        texture=_spatial((0.0, 1.0), (1, 2), "t-align"),
        response=_spatial((0.0,), (1, 1), "r-align"),
        sensitivity=_spatial((0.0,), (1, 1), "q-align"),
    )
    result = content_router(
        (1, 2, 1, 4),
        mode="routing_stqr",
        observations=observations,
    )

    expected_semantic = (0.0, 0.2, 0.6, 0.8)
    expected_texture = (0.0, 0.25, 0.75, 1.0)
    expected_routing_map = tuple(
        (1.0 - value) ** (1.0 / 3.0) for value in expected_semantic
    )
    assert result.routing_map[:4] == pytest.approx(
        expected_routing_map,
        abs=2e-7,
    )
    assert result.routing_map[4:] == result.routing_map[:4]
    assert result.mask_hf[:4] == pytest.approx(
        tuple(
            routing_value * texture
            for routing_value, texture in zip(
                expected_routing_map,
                expected_texture,
                strict=True,
            )
        ),
        abs=2e-7,
    )
    assert all(
        0.0 <= value <= 1.0
        for value in result.routing_map + result.mask_lf + result.mask_hf
    )
    assert all(
        lf_value + hf_value == pytest.approx(routing_value, abs=4e-7)
        for routing_value, lf_value, hf_value in zip(
            result.routing_map,
            result.mask_lf,
            result.mask_hf,
            strict=True,
        )
    )
    assert result.observation_digests[0][0] == "S"
    assert result.routing_observations is observations
    assert not hasattr(result, "mixing_coefficient")
    assert not hasattr(result, "budget_lf")


@pytest.mark.unit
def test_routing_disabled_uniform_control() -> None:
    class ExplodingObservations:
        def __getattribute__(self, name: str):
            raise AssertionError(f"uniform control read observations: {name}")

    result = content_router(
        BATCH3_SHAPE,
        mode="routing_uniform_control",
        observations=ExplodingObservations(),
    )
    routed = _route()
    assert result.candidate_id == "routing_uniform_control"
    assert result.routing_observations is None
    assert result.observation_digests == ()
    assert set(result.routing_map) == {1.0}
    assert set(result.mask_lf) == {1.0}
    assert set(result.mask_hf) == {1.0}
    assert result.route_identity != routed.route_identity


@pytest.mark.unit
def test_routing_formula_recomputation_rejects_coordinated_forgery() -> None:
    result = content_router(
        BATCH3_SHAPE,
        mode="routing_stqr",
        observations=_observations(),
    )
    element_count = len(result.routing_map)
    forged_routing_map = (1.0,) * element_count
    forged_mask_lf = (0.5,) * element_count
    forged_mask_hf = (0.5,) * element_count
    forged_routing_map_digest = _float32_digest(forged_routing_map)
    forged_mask_lf_digest = _float32_digest(forged_mask_lf)
    forged_mask_hf_digest = _float32_digest(forged_mask_hf)
    forged_route_identity = sha256(
        stable_json_utf8(
            {
                "routing_map_digest": forged_routing_map_digest,
                "mask_hf_digest": forged_mask_hf_digest,
                "mask_lf_digest": forged_mask_lf_digest,
                "observation_digests": [
                    list(item) for item in result.observation_digests
                ],
                "route_config_digest": result.route_config_digest,
            }
        )
    ).hexdigest()
    forged = replace(
        result,
        routing_map=forged_routing_map,
        mask_lf=forged_mask_lf,
        mask_hf=forged_mask_hf,
        routing_map_digest=forged_routing_map_digest,
        mask_lf_digest=forged_mask_lf_digest,
        mask_hf_digest=forged_mask_hf_digest,
        mean_routing_map=1.0,
        mean_mask_lf=0.5,
        mean_mask_hf=0.5,
        route_identity=forged_route_identity,
    )

    assert forged.routing_observations is result.routing_observations
    assert forged.observation_digests == result.observation_digests
    with pytest.raises(ContentRouterError, match="authoritative routing formula"):
        validate_content_routing_result(forged)


@pytest.mark.unit
def test_content_embedding_branch_consumption() -> None:
    first_route = content_router(
        BATCH3_SHAPE,
        mode="routing_stqr",
        observations=_observations(),
    )
    second_route = content_router(
        BATCH3_SHAPE,
        mode="routing_stqr",
        observations=_observations(texture_reverse=True),
    )
    lf_first = lf_carrier(
        BATCH3_ROOT,
        BATCH3_SHAPE,
        routing_result=first_route,
    )
    hf_first = hf_carrier(
        BATCH3_ROOT,
        BATCH3_SHAPE,
        routing_result=first_route,
    )
    lf_second = lf_carrier(
        BATCH3_ROOT,
        BATCH3_SHAPE,
        routing_result=second_route,
    )
    hf_second = hf_carrier(
        BATCH3_ROOT,
        BATCH3_SHAPE,
        routing_result=second_route,
    )
    latent = _latent(len(hf_first.direction))

    hf_only = content_embedder(latent, hf_first)
    lf_only = content_embedder(latent, lf_carrier_result=lf_first)
    combined_first = content_embedder(
        latent,
        hf_first,
        lf_carrier_result=lf_first,
        mixing_coefficient=0.25,
        routing_result=first_route,
    )
    combined_second = content_embedder(
        latent,
        hf_second,
        lf_carrier_result=lf_second,
        mixing_coefficient=0.25,
        routing_result=second_route,
    )
    assert {hf_only.mode, lf_only.mode, combined_first.mode} == {
        "hf_only",
        "lf_only",
        "combined",
    }
    assert len(
        {
            hf_only.delta_content_digest,
            lf_only.delta_content_digest,
            combined_first.delta_content_digest,
            combined_second.delta_content_digest,
        }
    ) == 4
    assert combined_first.hf_carrier_config_digest == (
        hf_first.carrier_config_digest
    )
    assert combined_first.lf_carrier_config_digest == (
        lf_first.carrier_config_digest
    )
    assert combined_first.route_identity == first_route.route_identity
    assert combined_second.route_identity == second_route.route_identity


@pytest.mark.unit
def test_content_embedding_route_identity_fail_closed() -> None:
    first_route = content_router(
        BATCH3_SHAPE,
        mode="routing_stqr",
        observations=_observations(),
    )
    second_route = content_router(
        BATCH3_SHAPE,
        mode="routing_stqr",
        observations=_observations(texture_reverse=True),
    )
    lf_first = lf_carrier(
        BATCH3_ROOT,
        BATCH3_SHAPE,
        routing_result=first_route,
    )
    hf_first = hf_carrier(
        BATCH3_ROOT,
        BATCH3_SHAPE,
        routing_result=first_route,
    )
    lf_second = lf_carrier(
        BATCH3_ROOT,
        BATCH3_SHAPE,
        routing_result=second_route,
    )
    latent = _latent(len(hf_first.direction))

    with pytest.raises(ContentEmbedderError, match="routing binding"):
        content_embedder(
            latent,
            hf_first,
            lf_carrier_result=lf_second,
            mixing_coefficient=0.50,
            routing_result=first_route,
        )

    raw_half_mask = (0.5,) * len(hf_first.direction)
    raw_lf = lf_carrier(
        BATCH3_ROOT,
        BATCH3_SHAPE,
        mask_lf=raw_half_mask,
    )
    raw_hf = hf_carrier(
        BATCH3_ROOT,
        BATCH3_SHAPE,
        mask_hf=raw_half_mask,
    )
    with pytest.raises(ContentEmbedderError, match="routing binding"):
        content_embedder(
            latent,
            raw_hf,
            lf_carrier_result=raw_lf,
            mixing_coefficient=0.50,
            routing_result=first_route,
        )

    forged_route = replace(
        first_route,
        mask_lf=raw_half_mask,
    )
    with pytest.raises(ContentEmbedderError, match="routing binding"):
        content_embedder(
            latent,
            hf_first,
            lf_carrier_result=lf_first,
            mixing_coefficient=0.50,
            routing_result=forged_route,
        )


@pytest.mark.unit
def test_content_embedding_total_budget_and_frozen_allocation() -> None:
    route = _route()
    lf_result = lf_carrier(
        BATCH3_ROOT,
        BATCH3_SHAPE,
        routing_result=route,
    )
    hf_result = hf_carrier(
        BATCH3_ROOT,
        BATCH3_SHAPE,
        routing_result=route,
    )
    latent = _latent(len(hf_result.direction))
    controls = [
        content_embedder(latent, hf_result),
        content_embedder(latent, lf_carrier_result=lf_result),
    ]
    combined = [
        content_embedder(
            latent,
            hf_result,
            lf_carrier_result=lf_result,
            mixing_coefficient=coefficient,
            routing_result=route,
        )
        for coefficient in (0.25, 0.50, 0.75)
    ]

    target_norm = controls[0].target_total_norm
    for result in controls + combined:
        assert result.target_relative_l2 == pytest.approx(0.012)
        assert result.target_total_norm == target_norm
        assert sqrt(
            sum(value * value for value in result.delta_content)
        ) == pytest.approx(target_norm, rel=3e-6)
        assert not hasattr(result, "delta_content_actual")
        assert not hasattr(result, "realized_relative_l2")
    for result in combined:
        coefficient = result.mixing_coefficient
        expected_norm = sqrt(
            coefficient**2
            + (1.0 - coefficient) ** 2
            + 2.0
            * coefficient
            * (1.0 - coefficient)
            * result.gamma_lh
        )
        assert result.combined_pre_normalization_norm == pytest.approx(
            expected_norm,
            abs=1e-12,
        )
        reconstructed = tuple(
            lf_value + hf_value
            for lf_value, hf_value in zip(
                result.target_component_lf,
                result.target_component_hf,
                strict=True,
            )
        )
        assert reconstructed == pytest.approx(
            result.delta_content,
            abs=2e-7,
        )
        assert (
            result.target_component_lf_norm
            + result.target_component_hf_norm
        ) != pytest.approx(result.target_total_norm, abs=1e-6)

    materializer = _MonotoneActualMaterializer(
        greatest_feasible_scale=1.0,
        feasible_utilization=0.75,
    )
    actual = reconcile_content_materialization_budget(
        controls[0],
        materializer,
    )
    assert actual.content_relative_l2_nominal == pytest.approx(0.012)
    assert actual.content_relative_l2_limit == pytest.approx(0.012)
    assert actual.materialization_scale == 1.0
    assert actual.attempt_count == 1
    assert actual.integrity_status == "passed"
    assert actual.budget_status == "accepted"
    assert actual.budget_utilization == pytest.approx(0.75, rel=1e-6)
    assert any(
        value != 0.0
        for value in actual.observation.delta_content_actual
    )

    half_delta = tuple(
        _test_float32(value * 0.5)
        for value in controls[0].delta_content
    )
    forged_half = replace(
        controls[0],
        delta_content=half_delta,
        delta_content_digest=_float32_digest(half_delta),
    )
    with pytest.raises(ContentEmbedderError, match="nominal formula replay"):
        reconcile_content_materialization_budget(
            forged_half,
            materializer,
        )
    opposite_delta = tuple(-value for value in controls[0].delta_content)
    forged_opposite = replace(
        controls[0],
        delta_content=opposite_delta,
        delta_content_digest=_float32_digest(opposite_delta),
    )
    with pytest.raises(ContentEmbedderError, match="nominal formula replay"):
        reconcile_content_materialization_budget(
            forged_opposite,
            materializer,
        )


@pytest.mark.unit
def test_actual_content_budget_predicate_is_monotone_and_strict() -> None:
    embedding = content_embedder(
        _latent(BATCH3_SHAPE[1] * BATCH3_SHAPE[2] * BATCH3_SHAPE[3]),
        hf_carrier(BATCH3_ROOT, BATCH3_SHAPE),
    )
    baseline_norm = embedding.latent_norm
    limit = _test_float32(baseline_norm * _test_float32(3.0 / 250.0))
    just_over = _next_positive_float32(limit)
    decisions = [
        content_actual_budget_accepts(baseline_norm, realized)
        for realized in (0.0, _test_float32(limit * 0.5), limit, just_over)
    ]
    assert decisions == [True, True, True, False]
    scale = _test_float32(0.30)
    scaled = scale_content_delta_binary32(embedding, scale)
    assert tuple(pack(">f", value) for value in scaled) == tuple(
        pack(">f", _test_float32(value * scale))
        for value in embedding.delta_content
    )
    with pytest.raises(ContentEmbedderError, match="nonnegative"):
        content_actual_budget_accepts(baseline_norm, -1.0)
    for invalid_baseline in (-1.0, 0.0):
        with pytest.raises(ContentEmbedderError, match="positive"):
            content_actual_budget_accepts(invalid_baseline, 0.0)


@pytest.mark.unit
def test_actual_budget_bisection_selects_maximal_scale_after_plateau() -> None:
    embedding = content_embedder(
        _latent(BATCH3_SHAPE[1] * BATCH3_SHAPE[2] * BATCH3_SHAPE[3]),
        hf_carrier(BATCH3_ROOT, BATCH3_SHAPE),
    )
    materializer = _MonotoneActualMaterializer(
        greatest_feasible_scale=0.375,
        zero_through_scale=0.30,
        feasible_utilization=0.5,
    )
    result = reconcile_content_materialization_budget(
        embedding,
        materializer,
    )
    assert result.materialization_scale == _test_float32(0.375)
    assert result.materialization_scale == max(
        scale
        for scale in materializer.calls
        if 0.30 < scale <= 0.375
    )
    assert any(scale <= 0.30 for scale in materializer.calls)
    assert result.attempt_count == len(materializer.calls)
    assert result.attempt_count < 200
    next_scale = _next_positive_float32(result.materialization_scale)
    probe = _MonotoneActualMaterializer(
        greatest_feasible_scale=0.375,
        zero_through_scale=0.30,
        feasible_utilization=0.5,
    )
    next_observation = probe(embedding, next_scale)
    assert not content_actual_budget_accepts(
        next_observation.baseline_norm,
        next_observation.realized_total_l2,
    )


@pytest.mark.unit
def test_actual_budget_bisection_handles_subnormal_scale_and_terminates() -> None:
    embedding = content_embedder(
        _latent(BATCH3_SHAPE[1] * BATCH3_SHAPE[2] * BATCH3_SHAPE[3]),
        hf_carrier(BATCH3_ROOT, BATCH3_SHAPE),
    )
    largest_subnormal = unpack(">f", b"\x00\x7f\xff\xff")[0]
    materializer = _MonotoneActualMaterializer(
        greatest_feasible_scale=largest_subnormal,
        feasible_utilization=0.75,
    )
    result = reconcile_content_materialization_budget(
        embedding,
        materializer,
    )
    assert result.materialization_scale == largest_subnormal
    assert result.attempt_count == len(materializer.calls)
    assert result.attempt_count <= 151


@pytest.mark.unit
def test_actual_budget_fails_when_no_nonzero_scale_is_feasible() -> None:
    embedding = content_embedder(
        _latent(BATCH3_SHAPE[1] * BATCH3_SHAPE[2] * BATCH3_SHAPE[3]),
        hf_carrier(BATCH3_ROOT, BATCH3_SHAPE),
    )
    largest_below_one = unpack(">f", b"\x3f\x7f\xff\xff")[0]
    materializer = _MonotoneActualMaterializer(
        greatest_feasible_scale=0.0,
        zero_through_scale=largest_below_one,
    )
    with pytest.raises(ContentEmbedderError, match="no nonzero"):
        reconcile_content_materialization_budget(
            embedding,
            materializer,
        )
    assert len(materializer.calls) <= 151


@pytest.mark.unit
def test_actual_budget_integrity_failures_are_fail_closed() -> None:
    embedding = content_embedder(
        _latent(BATCH3_SHAPE[1] * BATCH3_SHAPE[2] * BATCH3_SHAPE[3]),
        hf_carrier(BATCH3_ROOT, BATCH3_SHAPE),
    )
    replay_failure = _MonotoneActualMaterializer(
        greatest_feasible_scale=1.0,
        replay_passed=False,
    )
    with pytest.raises(ContentEmbedderError, match="binary16 replay"):
        reconcile_content_materialization_budget(
            embedding,
            replay_failure,
        )

    vanished_full_scale = _MonotoneActualMaterializer(
        greatest_feasible_scale=1.0,
        zero_through_scale=1.0,
    )
    with pytest.raises(ContentEmbedderError, match="full-scale"):
        reconcile_content_materialization_budget(
            embedding,
            vanished_full_scale,
        )


@pytest.mark.unit
def test_actual_materialization_observation_identity_failures() -> None:
    embedding = content_embedder(
        _latent(BATCH3_SHAPE[1] * BATCH3_SHAPE[2] * BATCH3_SHAPE[3]),
        hf_carrier(BATCH3_ROOT, BATCH3_SHAPE),
    )
    valid = _MonotoneActualMaterializer(
        greatest_feasible_scale=1.0,
        feasible_utilization=0.75,
    )(embedding, 1.0)
    zero_delta = (0.0,) * len(valid.delta_content_actual)
    unrelated_delta = tuple(
        -value for value in valid.delta_content_actual
    )
    cases = (
        (
            replace(valid, materialization_scale=0.5),
            "scale identity",
        ),
        (
            replace(
                valid,
                baseline_norm=_next_positive_float32(valid.baseline_norm),
            ),
            "baseline norm does not match",
        ),
        (
            replace(valid, baseline_norm=-1.0),
            "baseline L2 must be positive",
        ),
        (
            replace(valid, baseline_norm=0.0),
            "baseline L2 must be positive",
        ),
        (
            replace(
                valid,
                realized_total_l2=_next_positive_float32(
                    valid.realized_total_l2
                ),
            ),
            "row-major binary32 replay",
        ),
        (
            replace(valid, materialization_replay_identity="not-a-digest"),
            "replay identity",
        ),
        (
            replace(valid, integrity_status="invalid"),
            "integrity status",
        ),
        (
            replace(
                valid,
                delta_content_actual=zero_delta,
                realized_total_l2=0.0,
                integrity_status="passed",
            ),
            "passed observation has zero",
        ),
        (
            replace(valid, integrity_status="write_disappeared"),
            "write-disappeared observation has nonzero",
        ),
        (
            replace(valid, scaled_nominal_delta_digest="0" * 64),
            "scaled nominal delta digest drifted",
        ),
        (
            replace(valid, delta_content_actual=unrelated_delta),
            "replay identity mismatch",
        ),
    )
    for forged, error_match in cases:
        with pytest.raises(ContentEmbedderError, match=error_match):
            reconcile_content_materialization_budget(
                embedding,
                lambda _embedding, _scale, forged=forged: forged,
            )


@pytest.mark.unit
def test_content_embedding_active_zero_direction_fail_closed() -> None:
    hf_result = hf_carrier(BATCH3_ROOT, BATCH3_SHAPE)
    latent = _latent(len(hf_result.direction))
    baseline = content_embedder(latent, hf_result)
    assert baseline.mode == "hf_only"
    assert baseline.hf_carrier_config_digest == hf_result.carrier_config_digest

    zero = (0.0,) * len(hf_result.direction)
    forged_zero_hf = replace(
        hf_result,
        direction=zero,
        direction_digest=_float32_digest(zero),
    )
    with pytest.raises(ContentEmbedderError, match="active HF direction"):
        content_embedder(latent, forged_zero_hf)

    zero_shape = (1, 1, 6, 6)
    zero_support_index = 14
    zero_route = content_router(
        zero_shape,
        mode="routing_stqr",
        observations=RoutingObservations(
            semantic=_spatial(
                tuple(
                    0.0 if index == zero_support_index else 1.0
                    for index in range(36)
                ),
                (6, 6),
                "combined-zero-semantic",
            ),
            texture=_spatial(
                (0.5,) * 36,
                (6, 6),
                "combined-zero-texture",
            ),
            response=_spatial(
                (0.0,) * 36,
                (6, 6),
                "combined-zero-response",
            ),
            sensitivity=_spatial(
                (0.0,) * 36,
                (6, 6),
                "combined-zero-sensitivity",
            ),
        ),
    )
    routed_hf = hf_carrier(
        "ceg-wm-combined-zero-root-001",
        zero_shape,
        routing_result=zero_route,
    )
    routed_lf = lf_carrier(
        "ceg-wm-combined-zero-root-001",
        zero_shape,
        routing_result=zero_route,
    )
    zero_latent = _latent(len(routed_hf.direction))
    assert routed_lf.direction == tuple(
        -value for value in routed_hf.direction
    )
    with pytest.raises(
        ContentEmbedderError,
        match="combined content direction",
    ):
        content_embedder(
            zero_latent,
            routed_hf,
            lf_carrier_result=routed_lf,
            mixing_coefficient=0.50,
            routing_result=zero_route,
        )
    with pytest.raises(ContentEmbedderError, match="0.25"):
        content_embedder(
            zero_latent,
            routed_hf,
            lf_carrier_result=routed_lf,
            mixing_coefficient=0.70,
            routing_result=zero_route,
        )


@pytest.mark.unit
def test_content_wrong_key_rejection() -> None:
    hf_carrier_result = hf_carrier(BATCH3_ROOT, BATCH3_SHAPE)
    lf_carrier_result = lf_carrier(BATCH3_ROOT, BATCH3_SHAPE)
    hf_observation = HfDetectionObservation.from_public_image_encoding(
        hf_carrier_result.template,
        BATCH3_SHAPE,
    )
    lf_observation = LfDetectionObservation.from_public_image_encoding(
        lf_carrier_result.template,
        BATCH3_SHAPE,
    )
    wrong_key = derive_wrong_key_material(
        identify_root_key(BATCH3_ROOT).root_key_public_digest,
        5,
    )

    hf_registered = hf_detector(hf_observation, BATCH3_ROOT)
    hf_wrong = hf_detector(hf_observation, wrong_key)
    lf_registered = lf_detector(lf_observation, BATCH3_ROOT)
    lf_wrong = lf_detector(lf_observation, wrong_key)
    assert hf_registered.hf_score > 0.999
    assert lf_registered.lf_score > 0.999
    assert hf_wrong.hf_score < 0.5
    assert lf_wrong.lf_score < 0.5
    assert hf_registered.hf_score > hf_wrong.hf_score
    assert lf_registered.lf_score > lf_wrong.lf_score


@pytest.mark.unit
def test_content_scores_independently_observable() -> None:
    hf_template = hf_carrier(BATCH3_ROOT, BATCH3_SHAPE)
    lf_template = lf_carrier(BATCH3_ROOT, BATCH3_SHAPE)
    image_encoding = tuple(
        hf_value + lf_value
        for hf_value, lf_value in zip(
            hf_template.template,
            lf_template.template,
            strict=True,
        )
    )
    hf_result = hf_detector(
        HfDetectionObservation.from_public_image_encoding(
            image_encoding,
            BATCH3_SHAPE,
        ),
        BATCH3_ROOT,
    )
    lf_result = lf_detector(
        LfDetectionObservation.from_public_image_encoding(
            image_encoding,
            BATCH3_SHAPE,
        ),
        BATCH3_ROOT,
    )
    hf_only = content_detector(hf_result)
    result = content_detector(hf_result, lf_result)

    assert result.formal_mode == "hf_only"
    assert result.content_score == result.hf_score == hf_result.hf_score
    assert result.lf_score == lf_result.lf_score
    assert result.combined_score is None
    assert result.hf_result is hf_result
    assert result.lf_result is lf_result
    assert result.detector_identity == hf_only.detector_identity
    assert result.content_config_digest == hf_only.content_config_digest
    assert not hasattr(result, "carrier_direction")
    assert not hasattr(result, "routing_mask")


@pytest.mark.unit
def test_content_detector_cross_image_rejected() -> None:
    hf_template = hf_carrier(BATCH3_ROOT, BATCH3_SHAPE)
    lf_template = lf_carrier(BATCH3_ROOT, BATCH3_SHAPE)
    hf_result = hf_detector(
        HfDetectionObservation.from_public_image_encoding(
            hf_template.template,
            BATCH3_SHAPE,
        ),
        BATCH3_ROOT,
    )
    lf_result = lf_detector(
        LfDetectionObservation.from_public_image_encoding(
            lf_template.template,
            BATCH3_SHAPE,
        ),
        BATCH3_ROOT,
    )

    assert hf_result.observation_digest != lf_result.observation_digest
    with pytest.raises(ContentDetectorError, match="cross-image"):
        content_detector(hf_result, lf_result)


@pytest.mark.unit
def test_content_combination_branch_consumption() -> None:
    hf_template = hf_carrier(BATCH3_ROOT, BATCH3_SHAPE)
    lf_template = lf_carrier(BATCH3_ROOT, BATCH3_SHAPE)
    image_encoding = tuple(
        hf_value + lf_value
        for hf_value, lf_value in zip(
            hf_template.template,
            lf_template.template,
            strict=True,
        )
    )
    hf_result = hf_detector(
        HfDetectionObservation.from_public_image_encoding(
            image_encoding,
            BATCH3_SHAPE,
        ),
        BATCH3_ROOT,
    )
    lf_result = lf_detector(
        LfDetectionObservation.from_public_image_encoding(
            image_encoding,
            BATCH3_SHAPE,
        ),
        BATCH3_ROOT,
    )
    hf_query = replace(hf_result, hf_score=0.0)
    lf_low = replace(lf_result, lf_score=-0.75)
    lf_high = replace(lf_result, lf_score=0.75)
    scores = (-1.0, -0.5, 0.0, 0.5, 1.0)
    hf_null = _null_calibration("hf", hf_result.detector_identity, scores)
    lf_null = _null_calibration("lf", lf_result.detector_identity, scores)

    low = content_detector(
        hf_query,
        lf_low,
        hf_null=hf_null,
        lf_null=lf_null,
        combination="weighted_hf_lf_standardized_score",
        weight=0.50,
    )
    high = content_detector(
        hf_query,
        lf_high,
        hf_null=hf_null,
        lf_null=lf_null,
        combination="weighted_hf_lf_standardized_score",
        weight=0.50,
    )
    hf_only_score_low = content_detector(
        hf_query,
        lf_low,
        hf_null=hf_null,
        combination="hf_only_standardized_score",
    )
    hf_only_score_high = content_detector(
        hf_query,
        lf_high,
        hf_null=hf_null,
        combination="hf_only_standardized_score",
    )
    assert low.combined_score < high.combined_score
    assert hf_only_score_low.combined_score == hf_only_score_high.combined_score
    assert low.hf_score == high.hf_score == 0.0
    assert low.lf_score == -0.75
    assert high.lf_score == 0.75
    assert low.content_score == high.content_score == 0.0


@pytest.mark.unit
def test_content_combination_frozen_formula_identity() -> None:
    hf_template = hf_carrier(BATCH3_ROOT, BATCH3_SHAPE)
    lf_template = lf_carrier(BATCH3_ROOT, BATCH3_SHAPE)
    image_encoding = tuple(
        hf_value + lf_value
        for hf_value, lf_value in zip(
            hf_template.template,
            lf_template.template,
            strict=True,
        )
    )
    hf_result = hf_detector(
        HfDetectionObservation.from_public_image_encoding(
            image_encoding,
            BATCH3_SHAPE,
        ),
        BATCH3_ROOT,
    )
    lf_result = lf_detector(
        LfDetectionObservation.from_public_image_encoding(
            image_encoding,
            BATCH3_SHAPE,
        ),
        BATCH3_ROOT,
    )
    hf_query = replace(hf_result, hf_score=0.0)
    lf_query = replace(lf_result, lf_score=0.5)
    scores = (-1.0, 0.0, 0.0, 1.0)
    hf_null = _null_calibration("hf", hf_result.detector_identity, scores)
    hf_permuted = _null_calibration(
        "hf",
        hf_result.detector_identity,
        scores,
        reverse=True,
    )
    lf_null = _null_calibration("lf", lf_result.detector_identity, scores)

    assert hf_null.calibration_identity == hf_permuted.calibration_identity
    hf_only_score_result = content_detector(
        hf_query,
        hf_null=hf_null,
        combination="hf_only_standardized_score",
    )
    hf_only_score = hf_only_score_result.diagnostic_combination
    assert hf_only_score.hf_standardization.less_count == 1
    assert hf_only_score.hf_standardization.equal_count == 2
    assert hf_only_score.hf_standardization.u_raw == 0.5
    assert hf_only_score.hf_standardization.quantile_index == 524288
    assert pack(">f", hf_only_score.hf_standardization.z_score).hex() == "35a06c99"
    assert pack(">f", normal_quantile_table_lookup(524288)).hex() == "35a06c99"

    combination_results = [
        content_detector(
            hf_query,
            lf_query,
            hf_null=hf_null,
            lf_null=lf_null,
            combination="weighted_hf_lf_standardized_score",
            weight=weight,
        )
        for weight in (0.25, 0.50, 0.75)
    ]
    combinations = [
        result.diagnostic_combination for result in combination_results
    ]
    maximum_score_result = content_detector(
        hf_query,
        lf_query,
        hf_null=hf_null,
        lf_null=lf_null,
        combination="maximum_hf_lf_standardized_score",
    )
    maximum_score = maximum_score_result.diagnostic_combination
    for result in combinations:
        expected = (
            result.weight * result.hf_standardization.z_score
            + sqrt(1.0 - result.weight**2)
            * result.lf_standardization.z_score
        )
        assert result.combined_score == pytest.approx(expected, abs=1e-15)
        assert result.diagnostic_only is True
        assert result.promoted is False
    assert len({result.formula_identity for result in combinations}) == 3
    assert maximum_score.combined_score == max(
        maximum_score.hf_standardization.z_score,
        maximum_score.lf_standardization.z_score,
    )
    weighted_score_result = combination_results[1]
    weighted_score = weighted_score_result.diagnostic_combination
    tampered_score = weighted_score.combined_score + 0.25
    coordinated_score_tamper = replace(
        weighted_score_result,
        combined_score=tampered_score,
        diagnostic_combination=replace(
            weighted_score,
            combined_score=tampered_score,
        ),
    )
    with pytest.raises(ContentDetectorError, match="score replay"):
        validate_content_detection_result(coordinated_score_tamper)
    formula_identity_tamper = replace(
        weighted_score_result,
        diagnostic_combination=replace(
            weighted_score,
            formula_identity="f" * 64,
        ),
    )
    with pytest.raises(ContentDetectorError, match="formula"):
        validate_content_detection_result(formula_identity_tamper)
    standardization_tamper = replace(
        weighted_score_result,
        diagnostic_combination=replace(
            weighted_score,
            hf_standardization=replace(
                weighted_score.hf_standardization,
                less_count=weighted_score.hf_standardization.less_count + 1,
            ),
        ),
    )
    with pytest.raises(ContentDetectorError, match="standardization"):
        validate_content_detection_result(standardization_tamper)
    with pytest.raises(ContentDetectorError, match="hf_only_standardized_score"):
        validate_content_detection_result(
            replace(
                hf_only_score_result,
                diagnostic_combination=replace(hf_only_score, weight=0.25),
            )
        )
    with pytest.raises(
        ContentDetectorError,
        match="weighted_hf_lf_standardized_score",
    ):
        validate_content_detection_result(
            replace(
                weighted_score_result,
                diagnostic_combination=replace(
                    weighted_score,
                    lf_standardization=None,
                ),
            )
        )
    with pytest.raises(
        ContentDetectorError,
        match="maximum_hf_lf_standardized_score",
    ):
        validate_content_detection_result(
            replace(
                maximum_score_result,
                diagnostic_combination=replace(maximum_score, weight=0.25),
            )
        )

    with pytest.raises(
        ContentDetectorError,
        match="weighted_hf_lf_standardized_score weight",
    ):
        content_detector(
            hf_query,
            lf_query,
            hf_null=hf_null,
            lf_null=lf_null,
            combination="weighted_hf_lf_standardized_score",
            weight=0.70,
        )
    with pytest.raises(ContentDetectorError, match="at least two"):
        _null_calibration("hf", hf_result.detector_identity, (0.0,))
    with pytest.raises(KeyScheduleError, match="index"):
        normal_quantile_table_lookup(1 << 20)


@pytest.mark.unit
def test_content_combination_wrong_key_not_masked() -> None:
    hf_registered, lf_registered = _branch_results()
    wrong_key = derive_wrong_key_material(
        identify_root_key(BATCH3_ROOT).root_key_public_digest,
        7,
    )
    hf_template = hf_carrier(BATCH3_ROOT, BATCH3_SHAPE)
    lf_template = lf_carrier(BATCH3_ROOT, BATCH3_SHAPE)
    image_encoding = tuple(
        hf_value + lf_value
        for hf_value, lf_value in zip(
            hf_template.template,
            lf_template.template,
            strict=True,
        )
    )
    hf_observation = HfDetectionObservation.from_public_image_encoding(
        image_encoding,
        BATCH3_SHAPE,
    )
    lf_observation = LfDetectionObservation.from_public_image_encoding(
        image_encoding,
        BATCH3_SHAPE,
    )
    hf_wrong = hf_detector(hf_observation, wrong_key)
    lf_wrong = lf_detector(lf_observation, wrong_key)

    with pytest.raises(ContentDetectorError, match="key semantics"):
        content_detector(hf_registered, lf_wrong)
    with pytest.raises(ContentDetectorError, match="key semantics"):
        content_detector(hf_wrong, lf_registered)

    hf_null = _null_calibration(
        "hf",
        hf_wrong.detector_identity,
        (-1.0, -0.5, 0.0, 0.5, 1.0),
    )
    lf_null = _null_calibration(
        "lf",
        lf_wrong.detector_identity,
        (-1.0, -0.5, 0.0, 0.5, 1.0),
    )
    exaggerated_lf_wrong = replace(lf_wrong, lf_score=10.0)
    result = content_detector(
        hf_wrong,
        exaggerated_lf_wrong,
        hf_null=hf_null,
        lf_null=lf_null,
        combination="maximum_hf_lf_standardized_score",
    )
    assert result.hf_result.key_role == "wrong"
    assert result.lf_result.key_role == "wrong"
    assert result.hf_result.wrong_key_index == 7
    assert result.lf_result.wrong_key_index == 7
    assert result.combined_score > result.hf_score
    assert result.content_score == result.hf_score
    assert result.formal_mode == "hf_only"
    assert result.diagnostic_combination.promoted is False
