from __future__ import annotations

import inspect
import math

from PIL import Image

import cegwm.geometry_v7.r4 as r4
from cegwm.geometry_v7.contracts import GeometryEstimate, GeometryStatus
from cegwm.geometry_v7.r1a import CANONICAL_CORNERS_NORMALIZED, apply_homography


def _matrix(*, angle_degrees: float = 0.0, scale: float = 1.0):
    angle = math.radians(angle_degrees)
    cosine, sine = math.cos(angle) * scale, math.sin(angle) * scale
    return ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0))


def _geometry(matrix=None, *, legal=True, error=None):
    matrix = _matrix(angle_degrees=15.0) if matrix is None else matrix
    points = apply_homography(matrix, CANONICAL_CORNERS_NORMALIZED)
    return GeometryEstimate(
        GeometryStatus.UNRELIABLE,
        0.0,
        points,
        points,
        matrix,
        legal,
        legal,
        error,
    )


def _image():
    return Image.new("RGB", (512, 512), (127, 127, 127))


def test_runtime_signature_is_single_rgb_and_blind():
    signature = inspect.signature(r4.detect_refine_and_recover)
    assert tuple(signature.parameters) == (
        "image_rgb", "score_rgb", "detect_geometry", "cycle_score_rgb"
    )
    source = inspect.getsource(r4.detect_refine_and_recover)
    for forbidden in ("condition_id", "truth", "attack_label", "membership", "outcome"):
        assert forbidden not in source


def test_direct_routes_do_not_call_geometry_or_cycle_and_zero_is_boundary(monkeypatch):
    calls = {"score": 0, "geometry": 0, "cycle": 0, "rectify": 0}

    def detector(_image):
        calls["geometry"] += 1
        return _geometry()

    def cycle(_image, _matrix):
        calls["cycle"] += 1
        return 0.0

    def rectifier(image, matrix):
        calls["rectify"] += 1
        assert matrix == _matrix(angle_degrees=15.0)
        return image.copy()

    monkeypatch.setattr(r4, "rectify_attacked_rgb", rectifier)

    def positive(_image):
        calls["score"] += 1
        return 0.25

    record = r4.detect_refine_and_recover(
        _image(), score_rgb=positive, detect_geometry=detector, cycle_score_rgb=cycle
    )
    assert record.route == "DIRECT_POSITIVE" and record.positive
    assert calls == {"score": 1, "geometry": 0, "cycle": 0, "rectify": 0}

    calls = dict.fromkeys(calls, 0)
    record = r4.detect_refine_and_recover(
        _image(), score_rgb=lambda _image: r4.R3_B_LOW,
        detect_geometry=detector, cycle_score_rgb=cycle,
    )
    assert record.route == "DIRECT_NEGATIVE" and not record.positive
    assert calls == {"score": 0, "geometry": 0, "cycle": 0, "rectify": 0}

    values = iter((0.0, 0.5))

    def boundary_score(_image):
        calls["score"] += 1
        return next(values)

    record = r4.detect_refine_and_recover(
        _image(), score_rgb=boundary_score, detect_geometry=detector, cycle_score_rgb=cycle
    )
    assert record.route == "BOUNDARY" and record.recovered and record.positive
    assert record.pre_score == 0.0 and record.post_score == 0.5
    assert record.refined_gate is not None and record.refined_gate.pure_rotation
    assert calls == {"score": 2, "geometry": 1, "cycle": 0, "rectify": 1}


def test_zoom_cycle_is_called_only_when_needed_and_raw_h_is_used_once(monkeypatch):
    matrix = _matrix(scale=1.25)
    geometry = _geometry(matrix)
    seen = []

    def rectifier(image, used_matrix):
        seen.append(used_matrix)
        return image.copy()

    monkeypatch.setattr(r4, "rectify_attacked_rgb", rectifier)
    scores = iter((0.0, 0.1))
    cycles = []
    record = r4.detect_refine_and_recover(
        _image(), score_rgb=lambda _image: next(scores),
        detect_geometry=lambda _image: geometry,
        cycle_score_rgb=lambda _image, used_matrix: cycles.append(used_matrix) or 8.0,
    )
    assert record.recovered and record.positive
    assert cycles == [matrix] and seen == [matrix]
    assert record.geometry is geometry

    score_calls = 0

    def score(_image):
        nonlocal score_calls
        score_calls += 1
        return 0.0

    record = r4.detect_refine_and_recover(
        _image(), score_rgb=score, detect_geometry=lambda _image: geometry,
        cycle_score_rgb=lambda _image, _matrix: 8.0000001,
    )
    assert not record.recovered and not record.positive and score_calls == 1
    assert record.refined_gate is not None and not record.refined_gate.reliable


def test_invalid_geometry_and_callback_failure_reject_without_retry():
    calls = 0

    def detector(_image):
        nonlocal calls
        calls += 1
        raise RuntimeError("detector interrupted")

    record = r4.detect_refine_and_recover(
        _image(), score_rgb=lambda _image: 0.0,
        detect_geometry=detector, cycle_score_rgb=lambda _image, _matrix: 0.0,
    )
    assert record.route == "ERROR" and not record.positive and calls == 1
    assert "detector interrupted" in record.error

    invalid_matrix = (
        (float("nan"), 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    invalid = GeometryEstimate(
        GeometryStatus.UNRELIABLE, 0.0,
        CANONICAL_CORNERS_NORMALIZED, CANONICAL_CORNERS_NORMALIZED,
        invalid_matrix, True, True, None,
    )
    record = r4.detect_refine_and_recover(
        _image(), score_rgb=lambda _image: 0.0,
        detect_geometry=lambda _image: invalid,
        cycle_score_rgb=lambda _image, _matrix: 0.0,
    )
    assert not record.recovered and not record.positive


def test_refined_gate_inclusive_boundaries():
    zoom = r4.refined_gate_from_features(
        boundary=True, r2_selector_accepted=True, regime_valid=True,
        angle_degrees=2.0, scale=1.15, translation=0.02, perspective=0.01,
        pure_rotation=False, cycle_score_px=8.0,
    )
    assert zoom.zoom_out_like and zoom.reliable
    assert not r4.refined_gate_from_features(
        boundary=True, r2_selector_accepted=True, regime_valid=True,
        angle_degrees=2.000001, scale=1.15, translation=0.02, perspective=0.01,
        pure_rotation=False, cycle_score_px=8.0,
    ).reliable
