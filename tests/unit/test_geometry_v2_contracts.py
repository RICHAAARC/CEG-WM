from __future__ import annotations

from dataclasses import replace
import math

import pytest

from cegwm.geometry_v2.contracts import (
    CANONICAL_CORNERS,
    ContentDetectorIdentity,
    GEOMETRY_AUTHORITY,
    GeometryEstimate,
    METHOD_IDENTITY,
    ReliabilityAssessment,
    ReliabilityPolicy,
    assert_content_detector_identity_preserved,
    assess_reliability,
    build_rectification_request,
    derive_keyed_sync_target,
)


IDENTITY_H = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)


def _detector() -> ContentDetectorIdentity:
    return ContentDetectorIdentity(
        detector_identity="content-weighted-joint-v1",
        detection_key_identity="sha256:public-key-binding",
        preprocessing_identity="content-preprocess-v1",
        tau=0.375,
    )


@pytest.mark.unit
def test_keyed_sync_target_is_deterministic_and_domain_separated() -> None:
    key_a = b"geometry-key-a-0123456789abcdef"
    key_b = b"geometry-key-b-0123456789abcdef"

    first = derive_keyed_sync_target(key_a, b"sample-0001", code_length=96)
    repeated = derive_keyed_sync_target(key_a, b"sample-0001", code_length=96)
    other_key = derive_keyed_sync_target(key_b, b"sample-0001", code_length=96)
    other_context = derive_keyed_sync_target(key_a, b"sample-0002", code_length=96)

    assert first == repeated
    assert first.method_identity == METHOD_IDENTITY
    assert first.bipolar_code != other_key.bipolar_code
    assert first.bipolar_code != other_context.bipolar_code
    assert set(first.bipolar_code) == {-1, 1}
    assert not hasattr(first, "geometry_key")


@pytest.mark.unit
@pytest.mark.parametrize("bad_key", [b"short", "not-bytes", bytearray(b"0" * 32)])
def test_keyed_sync_target_rejects_invalid_or_ambiguous_keys(bad_key: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        derive_keyed_sync_target(bad_key, b"sample-0001")  # type: ignore[arg-type]


@pytest.mark.unit
def test_geometry_estimate_accepts_finite_convex_bounded_corners_and_h() -> None:
    estimate = GeometryEstimate(CANONICAL_CORNERS, IDENTITY_H)

    assert estimate.corners == CANONICAL_CORNERS
    assert estimate.homography == IDENTITY_H
    assert len(estimate.binding) == 64


@pytest.mark.unit
@pytest.mark.parametrize(
    ("corners", "homography"),
    [
        (
            ((0.0, 0.0), (1.0, 0.0), (1.0, math.inf), (0.0, 1.0)),
            IDENTITY_H,
        ),
        (
            ((0.0, 0.0), (1.0, 0.0), (0.5, 0.0), (0.0, 1.0)),
            IDENTITY_H,
        ),
        (
            ((0.0, 0.0), (1.5, 0.0), (1.5, 1.0), (0.0, 1.0)),
            ((1.5, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        ),
        (
            CANONICAL_CORNERS,
            ((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ),
        (
            CANONICAL_CORNERS,
            ((1.0, 0.0, 0.1), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        ),
    ],
)
def test_geometry_estimate_fails_closed_on_nonfinite_degenerate_or_mismatched_input(
    corners: object,
    homography: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        GeometryEstimate(corners, homography)  # type: ignore[arg-type]


@pytest.mark.unit
def test_unreliable_or_nonfinite_measurement_cannot_request_rectification() -> None:
    estimate = GeometryEstimate(CANONICAL_CORNERS, IDENTITY_H)
    policy = ReliabilityPolicy(min_confidence=0.8, min_support=0.6)

    below = assess_reliability(estimate, confidence=0.79, support=0.9, policy=policy)
    invalid = assess_reliability(estimate, confidence=math.nan, support=0.9, policy=policy)
    missing = assess_reliability(None, confidence=1.0, support=1.0, policy=policy)

    assert not below.reliable and below.reason == "below_threshold"
    assert not invalid.reliable and invalid.reason == "invalid_measurement"
    assert not missing.reliable and missing.reason == "invalid_geometry"
    assert build_rectification_request(estimate, below, _detector()) is None
    assert build_rectification_request(estimate, invalid, _detector()) is None

    with pytest.raises(ValueError, match="does not satisfy"):
        ReliabilityAssessment(
            reliable=True,
            confidence=0.5,
            support=1.0,
            reason="reliable",
            geometry_binding=estimate.binding,
            policy=policy,
        )


@pytest.mark.unit
def test_reliable_geometry_only_requests_coordinates_and_preserves_detector_identity() -> None:
    corners = ((0.1, 0.2), (0.9, 0.2), (0.9, 0.8), (0.1, 0.8))
    homography = ((0.8, 0.0, 0.1), (0.0, 0.6, 0.2), (0.0, 0.0, 1.0))
    estimate = GeometryEstimate(corners, homography)
    detector = _detector()
    assessment = assess_reliability(
        estimate,
        confidence=0.9,
        support=0.75,
        policy=ReliabilityPolicy(min_confidence=0.8, min_support=0.6),
    )

    request = build_rectification_request(estimate, assessment, detector)

    assert request is not None
    assert request.geometry_authority == GEOMETRY_AUTHORITY
    assert request.positive_watermark_authority is False
    assert request.content_detector_identity is detector
    expected_inverse = (
        (1.25, 0.0, -0.125),
        (0.0, 5.0 / 3.0, -1.0 / 3.0),
        (0.0, 0.0, 1.0),
    )
    for actual_row, expected_row in zip(
        request.attacked_to_canonical_homography,
        expected_inverse,
        strict=True,
    ):
        assert actual_row == pytest.approx(expected_row, abs=1.0e-12)
    assert assert_content_detector_identity_preserved(request, detector) is detector

    for changed in (
        replace(detector, detection_key_identity="sha256:different-key"),
        replace(detector, preprocessing_identity="different-preprocess"),
        replace(detector, tau=0.5),
    ):
        with pytest.raises(ValueError, match="changed across rectification"):
            assert_content_detector_identity_preserved(request, changed)


@pytest.mark.unit
def test_reliability_is_bound_to_the_exact_geometry_estimate() -> None:
    first = GeometryEstimate(CANONICAL_CORNERS, IDENTITY_H)
    second = GeometryEstimate(
        ((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)),
        ((0.8, 0.0, 0.1), (0.0, 0.8, 0.1), (0.0, 0.0, 1.0)),
    )
    assessment = assess_reliability(
        first,
        confidence=1.0,
        support=1.0,
        policy=ReliabilityPolicy(0.8, 0.6),
    )

    with pytest.raises(ValueError, match="different geometry estimate"):
        build_rectification_request(second, assessment, _detector())
