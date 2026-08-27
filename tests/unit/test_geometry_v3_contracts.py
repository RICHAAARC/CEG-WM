from __future__ import annotations

import inspect

import pytest

from cegwm.geometry_v3.contracts import (
    ANCHOR_DOMAIN_ID,
    GEOMETRY_DECISION_CEILING,
    METHOD_ID,
    ContentDetectorBinding,
    FeatureRole,
    ObservationProvenance,
    PlacementBasis,
    RecoverabilityPhase,
    WriterBudget,
    WriterPlacement,
    authorize_rectification,
    declare_writer_contract,
    derive_canonical_relation_anchor,
    make_geometry_estimate,
    record_attacked_rgb,
    record_final_rgb,
    record_fresh_qk_observation,
    record_geometry_estimate,
    start_recoverability,
)


KEY_A = b"geometry-key-a-0123456789abcdef"
KEY_B = b"geometry-key-b-0123456789abcdef"


def _declaration():
    return declare_writer_contract(
        placements=(
            WriterPlacement("transformer_blocks.4.attn", FeatureRole.QUERY),
            WriterPlacement("transformer_blocks.17.attn", FeatureRole.KEY),
        ),
        budget=WriterBudget(0.002, 0.001),
        placement_basis=PlacementBasis.INDEPENDENT_PREDECLARED,
        placement_protocol_id="geometry-v3-placement-study-v1",
        interference_test_protocol_id="geometry-v3-content-interference-v1",
    )


def _binding() -> ContentDetectorBinding:
    return ContentDetectorBinding(
        detector_id="content-calibrated-weighted-joint-v1",
        key_semantics_id="content-detection-key-v1",
        preprocessing_id="content-final-rgb-preprocess-v1",
        threshold_id="content-frozen-tau-v1",
    )


def _observed_state():
    state = start_recoverability(_declaration(), detector_binding=_binding())
    state = record_final_rgb(state, final_rgb_id="final-rgb-0001")
    state = record_attacked_rgb(state, attacked_rgb_id="attacked-rgb-0001")
    return record_fresh_qk_observation(
        state,
        observation_id="fresh-qk-observation-0001",
        attacked_rgb_id="attacked-rgb-0001",
        provenance=ObservationProvenance.FRESH_ATTACKED_RGB_QK,
    )


def _estimate(state, *, support: float = 0.9, reliability: float = 0.95):
    return make_geometry_estimate(
        state,
        corners=((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)),
        homography=((1.0, 0.0, 0.02), (0.0, 1.0, -0.01), (0.0, 0.0, 1.0)),
        reliability_protocol_id="geometry-v3-reliability-v1",
        support_fraction=support,
        reliability_score=reliability,
        minimum_support_fraction=0.75,
        minimum_reliability=0.8,
    )


@pytest.mark.unit
def test_canonical_anchor_is_deterministic_domain_separated_and_bounded() -> None:
    first = derive_canonical_relation_anchor(KEY_A, point_count=12)
    repeat = derive_canonical_relation_anchor(KEY_A, point_count=12)
    other_key = derive_canonical_relation_anchor(KEY_B, point_count=12)

    assert first == repeat
    assert first != other_key
    assert first.method_id == METHOD_ID
    assert first.domain_id == ANCHOR_DOMAIN_ID
    assert len(first.points) == 12
    assert all(len(point) == 2 for point in first.points)
    assert all(0.1 < coordinate < 0.9 for point in first.points for coordinate in point)
    assert len(first.public_digest) == 64
    assert not hasattr(first, "geometry_key")
    assert KEY_A.hex() not in repr(first)


@pytest.mark.unit
def test_anchor_shape_validation_fails_closed() -> None:
    for count in (0, 3, 65, True):
        with pytest.raises(ValueError, match="point_count"):
            derive_canonical_relation_anchor(KEY_A, point_count=count)


@pytest.mark.unit
def test_writer_declaration_has_no_default_or_inherited_placements() -> None:
    signature = inspect.signature(declare_writer_contract)
    assert signature.parameters["placements"].default is inspect.Parameter.empty
    assert signature.parameters["placement_basis"].default is inspect.Parameter.empty
    assert _declaration().placements == (
        WriterPlacement("transformer_blocks.4.attn", FeatureRole.QUERY),
        WriterPlacement("transformer_blocks.17.attn", FeatureRole.KEY),
    )

    with pytest.raises(ValueError, match="at least one"):
        declare_writer_contract(
            placements=(),
            budget=WriterBudget(0.002, 0.001),
            placement_basis=PlacementBasis.INDEPENDENT_PREDECLARED,
            placement_protocol_id="geometry-v3-placement-study-v1",
            interference_test_protocol_id="geometry-v3-content-interference-v1",
        )
    with pytest.raises(TypeError):
        # Even a historically observed pair cannot enter implicitly: an
        # independent placement authority is a required argument.
        declare_writer_contract(
            placements=(
                WriterPlacement("transformer_blocks.23.attn", FeatureRole.QUERY),
                WriterPlacement("transformer_blocks.14.attn", FeatureRole.KEY),
            ),
            budget=WriterBudget(0.002, 0.001),
            placement_protocol_id="geometry-v3-placement-study-v1",
            interference_test_protocol_id="geometry-v3-content-interference-v1",
        )


@pytest.mark.unit
def test_duplicate_placement_and_invalid_budget_fail_closed() -> None:
    repeated = WriterPlacement("transformer_blocks.4.attn", FeatureRole.QUERY)
    with pytest.raises(ValueError, match="unique"):
        declare_writer_contract(
            placements=(repeated, repeated),
            budget=WriterBudget(0.002, 0.001),
            placement_basis=PlacementBasis.INDEPENDENT_PREDECLARED,
            placement_protocol_id="geometry-v3-placement-study-v1",
            interference_test_protocol_id="geometry-v3-content-interference-v1",
        )
    for arguments in ((0.0, 0.0), (0.001, 0.002), (float("inf"), 0.001)):
        with pytest.raises(ValueError):
            WriterBudget(*arguments)


@pytest.mark.unit
def test_writer_phase_requires_complete_predeclaration() -> None:
    with pytest.raises(ValueError, match="writer declaration"):
        start_recoverability(None, detector_binding=_binding())
    state = start_recoverability(_declaration(), detector_binding=_binding())
    assert state.phase is RecoverabilityPhase.WRITER_ADMITTED


@pytest.mark.unit
def test_detection_admits_only_fresh_qk_from_current_attacked_rgb() -> None:
    state = start_recoverability(_declaration(), detector_binding=_binding())
    state = record_final_rgb(state, final_rgb_id="final-rgb-0001")
    state = record_attacked_rgb(state, attacked_rgb_id="attacked-rgb-0001")

    for forbidden in ("embed_cached_qk", "embed_side_route", "original_rgb_qk"):
        with pytest.raises(ValueError, match="freshly observed"):
            record_fresh_qk_observation(
                state,
                observation_id="forbidden-observation",
                attacked_rgb_id="attacked-rgb-0001",
                provenance=forbidden,  # type: ignore[arg-type]
            )
    with pytest.raises(ValueError, match="current attacked RGB"):
        record_fresh_qk_observation(
            state,
            observation_id="stale-observation",
            attacked_rgb_id="attacked-rgb-stale",
            provenance=ObservationProvenance.FRESH_ATTACKED_RGB_QK,
        )
    admitted = record_fresh_qk_observation(
        state,
        observation_id="fresh-qk-observation-0001",
        attacked_rgb_id="attacked-rgb-0001",
        provenance=ObservationProvenance.FRESH_ATTACKED_RGB_QK,
    )
    assert admitted.phase is RecoverabilityPhase.FRESH_QK_OBSERVED
    assert not hasattr(admitted, "raw_qk")
    assert not hasattr(admitted, "embed_route")


@pytest.mark.unit
def test_unreliable_estimate_stops_and_cannot_rectify() -> None:
    observed = _observed_state()
    estimate = _estimate(observed, support=0.5, reliability=0.95)
    stopped = record_geometry_estimate(observed, estimate)

    assert estimate.reliable is False
    assert stopped.phase is RecoverabilityPhase.STOPPED_UNRELIABLE
    with pytest.raises(ValueError, match="expected phase geometry_estimated"):
        authorize_rectification(
            stopped,
            detector_binding_after_rectification=_binding(),
        )


@pytest.mark.unit
def test_reliable_rectification_preserves_the_exact_content_detector_binding() -> None:
    observed = _observed_state()
    estimated = record_geometry_estimate(observed, _estimate(observed))
    original = _binding()
    replacements = (
        ContentDetectorBinding(
            "changed-detector",
            original.key_semantics_id,
            original.preprocessing_id,
            original.threshold_id,
        ),
        ContentDetectorBinding(
            original.detector_id,
            "changed-key-semantics",
            original.preprocessing_id,
            original.threshold_id,
        ),
        ContentDetectorBinding(
            original.detector_id,
            original.key_semantics_id,
            "changed-preprocessing",
            original.threshold_id,
        ),
        ContentDetectorBinding(
            original.detector_id,
            original.key_semantics_id,
            original.preprocessing_id,
            "changed-threshold",
        ),
    )
    for changed in replacements:
        with pytest.raises(ValueError, match="preserve detector"):
            authorize_rectification(
                estimated,
                detector_binding_after_rectification=changed,
            )
    final_state, authorization = authorize_rectification(
        estimated,
        detector_binding_after_rectification=_binding(),
    )
    assert final_state.phase is RecoverabilityPhase.RECTIFICATION_AUTHORIZED
    assert authorization.detector_binding == _binding()
    assert authorization.geometry_authority == GEOMETRY_DECISION_CEILING
    assert "positive" in authorization.geometry_authority


@pytest.mark.unit
def test_corners_and_homography_are_constrained() -> None:
    observed = _observed_state()
    with pytest.raises(ValueError, match="four normalized"):
        make_geometry_estimate(
            observed,
            corners=((0.0, 0.0),) * 3,
            homography=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            reliability_protocol_id="geometry-v3-reliability-v1",
            support_fraction=1.0,
            reliability_score=1.0,
            minimum_support_fraction=0.5,
            minimum_reliability=0.5,
        )
    with pytest.raises(ValueError, match="invertible"):
        make_geometry_estimate(
            observed,
            corners=((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)),
            homography=((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            reliability_protocol_id="geometry-v3-reliability-v1",
            support_fraction=1.0,
            reliability_score=1.0,
            minimum_support_fraction=0.5,
            minimum_reliability=0.5,
        )
