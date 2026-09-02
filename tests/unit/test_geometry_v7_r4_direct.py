from __future__ import annotations

import inspect

from PIL import Image

import cegwm.geometry_v7.r4 as r4
from cegwm.geometry_v7.contracts import (
    CANONICAL_CORNERS_NORMALIZED,
    GeometryEstimate,
    GeometryStatus,
)


IDENTITY_H = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _image():
    return Image.new("RGB", (512, 512), (128, 128, 128))


def _geometry(matrix=IDENTITY_H, *, legal=True, error=None):
    return GeometryEstimate(
        GeometryStatus.UNRELIABLE,
        0.0,
        CANONICAL_CORNERS_NORMALIZED,
        CANONICAL_CORNERS_NORMALIZED,
        matrix,
        legal,
        legal,
        error,
    )


def test_direct_runtime_signature_has_no_evaluator_inputs():
    signature = inspect.signature(r4.detect_direct_and_recover)
    assert tuple(signature.parameters) == ("image_rgb", "score_rgb", "detect_geometry")
    source = inspect.getsource(r4.detect_direct_and_recover)
    for forbidden in (
        "condition_id", "attack_label", "truth", "membership", "safe",
        "outcome", "postscore",
    ):
        assert forbidden not in source
    assert callable(r4.detect_refine_and_recover)


def test_direct_content_routes_short_circuit_before_geometry(monkeypatch):
    calls = {"score": 0, "detect": 0, "rectify": 0}

    def detector(_image):
        calls["detect"] += 1
        return _geometry()

    def rectifier(_image, _matrix):
        calls["rectify"] += 1
        return _image.copy()

    monkeypatch.setattr(r4, "rectify_attacked_rgb", rectifier)

    def positive(_image):
        calls["score"] += 1
        return 0.1

    record = r4.detect_direct_and_recover(
        _image(), score_rgb=positive, detect_geometry=detector
    )
    assert record.route == "DIRECT_POSITIVE" and record.final_positive
    assert calls == {"score": 1, "detect": 0, "rectify": 0}

    calls = dict.fromkeys(calls, 0)

    def negative(_image):
        calls["score"] += 1
        return r4.R3_B_LOW

    record = r4.detect_direct_and_recover(
        _image(), score_rgb=negative, detect_geometry=detector
    )
    assert record.route == "DIRECT_NEGATIVE" and not record.final_positive
    assert calls == {"score": 1, "detect": 0, "rectify": 0}


def test_zero_is_boundary_same_scorer_and_raw_h0_rectifies_once(monkeypatch):
    calls = {"detect": 0, "rectify": 0}
    matrices = []

    class BoundScorer:
        def __init__(self):
            self.values = iter((0.0, 0.25))
            self.calls = 0

        def __call__(self, _image):
            self.calls += 1
            return next(self.values)

    scorer = BoundScorer()

    def detector(_image):
        calls["detect"] += 1
        return _geometry()

    def rectifier(image, matrix):
        calls["rectify"] += 1
        matrices.append(matrix)
        return image.copy()

    monkeypatch.setattr(r4, "rectify_attacked_rgb", rectifier)
    record = r4.detect_direct_and_recover(
        _image(), score_rgb=scorer, detect_geometry=detector
    )
    assert record.route == "BOUNDARY"
    assert record.pre_score == 0.0 and record.post_score == 0.25
    assert record.recovered and record.final_positive and record.error is None
    assert scorer.calls == 2 and calls == {"detect": 1, "rectify": 1}
    assert matrices == [IDENTITY_H]


def test_geometry_never_votes_positive_and_invalid_h_fails_closed(monkeypatch):
    rectifications = 0

    def rectifier(image, _matrix):
        nonlocal rectifications
        rectifications += 1
        return image.copy()

    monkeypatch.setattr(r4, "rectify_attacked_rgb", rectifier)
    values = iter((0.0, -0.1))
    record = r4.detect_direct_and_recover(
        _image(), score_rgb=lambda _image: next(values),
        detect_geometry=lambda _image: _geometry(),
    )
    assert record.recovered and not record.final_positive and rectifications == 1

    for geometry in (
        _geometry(legal=False),
        _geometry(error="reported"),
        _geometry(((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))),
        _geometry(((float("nan"), 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
    ):
        record = r4.detect_direct_and_recover(
            _image(), score_rgb=lambda _image: 0.0,
            detect_geometry=lambda _image, geometry=geometry: geometry,
        )
        assert record.route == "BOUNDARY"
        assert not record.recovered and not record.final_positive
        assert record.post_score is None and record.error
    assert rectifications == 1


def test_post_score_failure_retains_completed_recovery(monkeypatch):
    calls = {"score": 0, "rectify": 0}

    def score(_image):
        calls["score"] += 1
        if calls["score"] == 1:
            return 0.0
        raise RuntimeError("post scorer interrupted")

    def rectifier(image, matrix):
        assert matrix == IDENTITY_H
        calls["rectify"] += 1
        return image.copy()

    monkeypatch.setattr(r4, "rectify_attacked_rgb", rectifier)
    record = r4.detect_direct_and_recover(
        _image(), score_rgb=score, detect_geometry=lambda _image: _geometry()
    )
    assert record.route == "BOUNDARY" and record.pre_score == 0.0
    assert record.recovered is True and record.post_score is None
    assert not record.final_positive and "post scorer interrupted" in record.error
    assert calls == {"score": 2, "rectify": 1}


def test_callback_exceptions_are_not_retried():
    calls = {"score": 0, "detect": 0}

    def score(_image):
        calls["score"] += 1
        return 0.0

    def detector(_image):
        calls["detect"] += 1
        raise RuntimeError("detector interrupted")

    record = r4.detect_direct_and_recover(
        _image(), score_rgb=score, detect_geometry=detector
    )
    assert not record.recovered and not record.final_positive
    assert calls == {"score": 1, "detect": 1}
    assert "detector interrupted" in record.error
