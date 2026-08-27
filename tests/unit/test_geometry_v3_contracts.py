from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError

import pytest

from cegwm.geometry_v3.contracts import (
    ANCHOR_DOMAIN_ID,
    GEOMETRY_DECISION_CEILING,
    METHOD_ID,
    CanonicalRelationAnchor,
    ContentDetectorBinding,
    FeatureRole,
    ObservationProvenance,
    PlacementBasis,
    RecoverabilityPhase,
    ReliabilityPolicy,
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


def _anchor(key=KEY_A):
    return derive_canonical_relation_anchor(key, point_count=12)


def _policy(
    minimum_support_fraction: float = 0.75,
    minimum_reliability: float = 0.8,
) -> ReliabilityPolicy:
    return ReliabilityPolicy(
        protocol_id="geometry-v3-reliability-v1",
        minimum_support_fraction=minimum_support_fraction,
        minimum_reliability=minimum_reliability,
    )


def _declaration(canonical_anchor=None):
    anchor = canonical_anchor if canonical_anchor is not None else _anchor()
    return declare_writer_contract(
        canonical_anchor=anchor,
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
    anchor = _anchor()
    state = start_recoverability(
        _declaration(anchor),
        canonical_anchor=anchor,
        reliability_policy=_policy(),
        detector_binding=_binding(),
    )
    state = record_final_rgb(state, final_rgb_id="final-rgb-0001")
    state = record_attacked_rgb(state, attacked_rgb_id="attacked-rgb-0001")
    return record_fresh_qk_observation(
        state,
        observation_id="fresh-qk-observation-0001",
        attacked_rgb_id="attacked-rgb-0001",
        canonical_anchor=anchor,
        provenance=ObservationProvenance.FRESH_ATTACKED_RGB_QK,
    )


def _estimate(state, *, support: float = 0.9, reliability: float = 0.95):
    return make_geometry_estimate(
        state,
        corners=((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)),
        homography=((1.0, 0.0, 0.02), (0.0, 1.0, -0.01), (0.0, 0.0, 1.0)),
        support_fraction=support,
        reliability_score=reliability,
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
def test_public_contract_cannot_substitute_an_underived_anchor() -> None:
    derived = _anchor()
    with pytest.raises(ValueError, match="keyed derivation"):
        CanonicalRelationAnchor(
            method_id=METHOD_ID,
            domain_id=ANCHOR_DOMAIN_ID,
            points=derived.points,
            public_digest=derived.public_digest,
            _derivation_token=object(),
        )


@pytest.mark.unit
def test_writer_declaration_has_no_default_or_inherited_placements() -> None:
    signature = inspect.signature(declare_writer_contract)
    assert signature.parameters["canonical_anchor"].default is inspect.Parameter.empty
    assert signature.parameters["placements"].default is inspect.Parameter.empty
    assert signature.parameters["placement_basis"].default is inspect.Parameter.empty
    assert _declaration().placements == (
        WriterPlacement("transformer_blocks.4.attn", FeatureRole.QUERY),
        WriterPlacement("transformer_blocks.17.attn", FeatureRole.KEY),
    )
    assert _declaration().anchor_identity.domain_id == ANCHOR_DOMAIN_ID
    assert _declaration().anchor_identity.point_count == 12
    assert _declaration().anchor_identity.coordinate_dimension == 2

    with pytest.raises(ValueError, match="at least one"):
        declare_writer_contract(
            canonical_anchor=_anchor(),
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
            canonical_anchor=_anchor(),
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
        start_recoverability(
            None,
            canonical_anchor=_anchor(),
            reliability_policy=_policy(),
            detector_binding=_binding(),
        )
    anchor = _anchor()
    state = start_recoverability(
        _declaration(anchor),
        canonical_anchor=anchor,
        reliability_policy=_policy(),
        detector_binding=_binding(),
    )
    assert state.phase is RecoverabilityPhase.WRITER_ADMITTED
    assert state.anchor_identity == state.writer_declaration.anchor_identity


@pytest.mark.unit
def test_anchor_and_reliability_policy_are_required_before_writer_admission() -> None:
    anchor = _anchor()
    declaration = _declaration(anchor)
    with pytest.raises(ValueError, match="canonical anchor"):
        start_recoverability(
            declaration,
            canonical_anchor=None,
            reliability_policy=_policy(),
            detector_binding=_binding(),
        )
    with pytest.raises(ValueError, match="reliability policy"):
        start_recoverability(
            declaration,
            canonical_anchor=anchor,
            reliability_policy=None,
            detector_binding=_binding(),
        )
    with pytest.raises(TypeError, match="canonical_anchor"):
        declare_writer_contract(
            canonical_anchor=None,  # type: ignore[arg-type]
            placements=(WriterPlacement("transformer_blocks.4.attn", FeatureRole.QUERY),),
            budget=WriterBudget(0.002, 0.001),
            placement_basis=PlacementBasis.INDEPENDENT_PREDECLARED,
            placement_protocol_id="geometry-v3-placement-study-v1",
            interference_test_protocol_id="geometry-v3-content-interference-v1",
        )


@pytest.mark.unit
def test_anchor_mismatch_fails_at_admission_and_fresh_observation() -> None:
    anchor = _anchor(KEY_A)
    other = _anchor(KEY_B)
    declaration = _declaration(anchor)
    with pytest.raises(ValueError, match="anchor identities differ"):
        start_recoverability(
            declaration,
            canonical_anchor=other,
            reliability_policy=_policy(),
            detector_binding=_binding(),
        )
    state = start_recoverability(
        declaration,
        canonical_anchor=anchor,
        reliability_policy=_policy(),
        detector_binding=_binding(),
    )
    state = record_final_rgb(state, final_rgb_id="final-rgb-0001")
    state = record_attacked_rgb(state, attacked_rgb_id="attacked-rgb-0001")
    with pytest.raises(ValueError, match="anchor identity differs"):
        record_fresh_qk_observation(
            state,
            observation_id="fresh-qk-observation-0001",
            attacked_rgb_id="attacked-rgb-0001",
            canonical_anchor=other,
            provenance=ObservationProvenance.FRESH_ATTACKED_RGB_QK,
        )


@pytest.mark.unit
def test_detection_admits_only_fresh_qk_from_current_attacked_rgb() -> None:
    anchor = _anchor()
    state = start_recoverability(
        _declaration(anchor),
        canonical_anchor=anchor,
        reliability_policy=_policy(),
        detector_binding=_binding(),
    )
    state = record_final_rgb(state, final_rgb_id="final-rgb-0001")
    state = record_attacked_rgb(state, attacked_rgb_id="attacked-rgb-0001")

    for forbidden in ("embed_cached_qk", "embed_side_route", "original_rgb_qk"):
        with pytest.raises(ValueError, match="freshly observed"):
            record_fresh_qk_observation(
                state,
                observation_id="forbidden-observation",
                attacked_rgb_id="attacked-rgb-0001",
                canonical_anchor=anchor,
                provenance=forbidden,  # type: ignore[arg-type]
            )
    with pytest.raises(ValueError, match="current attacked RGB"):
        record_fresh_qk_observation(
            state,
            observation_id="stale-observation",
            attacked_rgb_id="attacked-rgb-stale",
            canonical_anchor=anchor,
            provenance=ObservationProvenance.FRESH_ATTACKED_RGB_QK,
        )
    admitted = record_fresh_qk_observation(
        state,
        observation_id="fresh-qk-observation-0001",
        attacked_rgb_id="attacked-rgb-0001",
        canonical_anchor=anchor,
        provenance=ObservationProvenance.FRESH_ATTACKED_RGB_QK,
    )
    assert admitted.phase is RecoverabilityPhase.FRESH_QK_OBSERVED
    assert not hasattr(admitted, "raw_qk")
    assert not hasattr(admitted, "embed_route")


@pytest.mark.unit
@pytest.mark.parametrize(
    "support,reliability",
    ((0.0, 1.0), (1.0, 0.0), (0.0, 0.0)),
)
def test_zero_support_or_score_never_authorizes_rectification(
    support: float,
    reliability: float,
) -> None:
    observed = _observed_state()
    estimate = _estimate(observed, support=support, reliability=reliability)
    stopped = record_geometry_estimate(observed, estimate)
    assert estimate.reliable is False
    assert stopped.phase is RecoverabilityPhase.STOPPED_UNRELIABLE
    with pytest.raises(ValueError, match="expected phase geometry_estimated"):
        authorize_rectification(
            stopped,
            detector_binding_after_rectification=_binding(),
        )


@pytest.mark.unit
def test_reliability_policy_is_frozen_before_observation_and_cannot_be_lowered() -> None:
    observed = _observed_state()
    policy = observed.reliability_policy
    with pytest.raises(FrozenInstanceError):
        policy.minimum_reliability = 0.1  # type: ignore[misc]
    signature = inspect.signature(make_geometry_estimate)
    assert "minimum_support_fraction" not in signature.parameters
    assert "minimum_reliability" not in signature.parameters
    assert "reliability_protocol_id" not in signature.parameters
    estimate = _estimate(observed, support=0.74, reliability=0.79)
    assert estimate.reliability_policy == policy
    assert estimate.anchor_identity == observed.anchor_identity
    assert estimate.reliable is False


@pytest.mark.unit
@pytest.mark.parametrize(
    "minimum_support,minimum_reliability",
    (
        (0.0, 0.5),
        (0.5, 0.0),
        (float("nan"), 0.5),
        (0.5, float("inf")),
    ),
)
def test_reliability_policy_rejects_zero_and_nonfinite_minima(
    minimum_support: float,
    minimum_reliability: float,
) -> None:
    with pytest.raises(ValueError):
        _policy(minimum_support, minimum_reliability)


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
            support_fraction=1.0,
            reliability_score=1.0,
        )
    with pytest.raises(ValueError, match="invertible"):
        make_geometry_estimate(
            observed,
            corners=((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)),
            homography=((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            support_fraction=1.0,
            reliability_score=1.0,
        )
