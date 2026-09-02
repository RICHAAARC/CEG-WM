from __future__ import annotations

import pytest

from cegwm.geometry_v7.contracts import (
    CANONICAL_CORNERS_NORMALIZED,
    D4Transform,
    GeometryStatus,
    compose_d4_current_to_canonical,
    compose_d4_observed_to_canonical,
    estimate_geometry,
    homography_current_to_canonical,
    homography_observed_to_canonical,
    normalized_to_pixel_center,
    pixel_center_to_normalized,
    syncseal_raw_to_public_normalized,
)


@pytest.mark.unit
def test_512_pixel_center_and_homography_direction_are_exact() -> None:
    assert pixel_center_to_normalized(0, 512) == -1.0
    assert pixel_center_to_normalized(511, 512) == 1.0
    assert normalized_to_pixel_center(-1.0, 512) == 0.0
    assert normalized_to_pixel_center(1.0, 512) == 511.0
    official_identity_raw = (
        (-1.0, -1.0), (127.0 / 128.0, -1.0),
        (127.0 / 128.0, 127.0 / 128.0), (-1.0, 127.0 / 128.0),
    )
    assert syncseal_raw_to_public_normalized(official_identity_raw) == (
        (-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)
    )
    identity = homography_observed_to_canonical(CANONICAL_CORNERS_NORMALIZED)
    assert identity == ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    predicted_correspondences = (
        (-0.5, -0.5),
        (0.5, -0.5),
        (0.5, 0.5),
        (-0.5, 0.5),
    )
    assert homography_observed_to_canonical(predicted_correspondences) == (
        (0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 1.0)
    )
    assert homography_current_to_canonical(predicted_correspondences) == (
        (0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 1.0)
    )


@pytest.mark.unit
def test_d4_is_left_composed_in_canonical_coordinates() -> None:
    raw = ((2.0, 0.0, 0.25), (0.0, 3.0, -0.5), (0.0, 0.0, 1.0))
    expected = (
        (0.0, 3.0, -0.5), (-2.0, 0.0, -0.25), (0.0, 0.0, 1.0)
    )
    assert compose_d4_observed_to_canonical(raw, D4Transform.ROTATE_90_CCW) == expected
    assert compose_d4_current_to_canonical(raw, D4Transform.ROTATE_90_CCW) == expected


@pytest.mark.unit
def test_raw_output_is_observable_but_never_reliable_without_frozen_gate() -> None:
    estimate = estimate_geometry(0.75, CANONICAL_CORNERS_NORMALIZED)
    assert estimate.status is GeometryStatus.UNRELIABLE
    assert estimate.legal is True and estimate.basic_observable is True
    assert estimate.uncalibrated_sync_logit == 0.75
    assert (
        estimate.corners_current_normalized
        == estimate.observed_corners_in_canonical_normalized
    )
    assert (
        estimate.homography_current_to_canonical
        == estimate.homography_observed_to_canonical
    )
    finite_but_nonconvex = (
        (-1.0, -1.0),
        (1.0, 1.0),
        (1.0, -1.0),
        (-1.0, 1.0),
    )
    raw = (
        (-1.0, -1.0),
        (127.0 / 128.0, 127.0 / 128.0),
        (127.0 / 128.0, -1.0),
        (-1.0, 127.0 / 128.0),
    )
    unsupported = estimate_geometry(
        0.75,
        finite_but_nonconvex,
        raw_syncseal_corners=raw,
    )
    assert unsupported.status is GeometryStatus.UNSUPPORTED
    assert unsupported.raw_syncseal_corners == raw
    assert (
        unsupported.observed_corners_in_canonical_normalized
        == finite_but_nonconvex
    )
    assert unsupported.homography_observed_to_canonical is None
    malformed = estimate_geometry(0.75, ((-1.0, -1.0),))
    assert malformed.status is GeometryStatus.ERROR
    assert malformed.observed_corners_in_canonical_normalized is None
    assert {status.value for status in GeometryStatus} == {
        "RELIABLE", "UNRELIABLE", "UNSUPPORTED", "ERROR"
    }
