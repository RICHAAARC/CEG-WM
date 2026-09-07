import math

import numpy as np
from PIL import Image

from cegwm.geometry_v7.contracts import CANONICAL_CORNERS_NORMALIZED, GeometryEstimate
from cegwm.geometry_v7.r1a import condition_by_id, render_r1a_attack, truth_correspondences
from diagnostics.rotation_renderer.run import CONDITIONS, is_identity, measure, render, summarize


def test_black_bilinear_is_exact_historical_renderer():
    image = Image.fromarray(np.random.default_rng(12).integers(0, 256, (512,512,3), dtype=np.uint8))
    for condition in CONDITIONS:
        actual = render(image, condition, "black", "bilinear")
        old = render_r1a_attack(image, condition_by_id(condition))
        np.testing.assert_array_equal(actual, old)


def test_reflection_keeps_same_interior_coordinates_and_samples_once(monkeypatch):
    yy, xx = np.mgrid[:512, :512]
    image = Image.fromarray(np.stack([xx % 256, yy % 256, (xx+yy) % 256], -1).astype(np.uint8))
    original = Image.Image.transform
    calls = []
    def counting(self, size, method, data, *args, **kwargs):
        calls.append((self.size, size))
        return original(self, size, method, data, *args, **kwargs)
    monkeypatch.setattr(Image.Image, "transform", counting)
    for condition in CONDITIONS:
        for kernel in ("bilinear", "bicubic"):
            black = np.asarray(render(image, condition, "black", kernel))
            reflect = np.asarray(render(image, condition, "reflect", kernel))
            # Well inside the source support: padding must not shift content.
            # Adding an integer to floating transform offsets may shift an
            # exact integer quantization tie by one uint8 level in Pillow.
            delta = np.abs(black[100:412,100:412].astype(int)-reflect[100:412,100:412].astype(int))
            assert delta.max() <= 1
            assert np.mean(delta != 0) < .001
    assert len(calls) == 8
    assert all(out == (512,512) for _, out in calls)
    assert [src for src, _ in calls] == [(512,512),(768,768)] * 4


def test_metric_conversion_and_failures_retained():
    image = Image.new("RGB", (512,512))
    calls = []
    def detector(attacked):
        calls.append(attacked)
        return GeometryEstimate.error_record(RuntimeError("fixture failure"))
    row = measure(detector, image, {"unit_id":"u", "path":"u.png"}, CONDITIONS[0],
        "black", "bilinear", {"prediction_rmse":.003, "errors":[]})
    assert len(calls) == 1 and row["errors"] and row["corner_rmse_px"] is None
    truth = np.asarray(truth_correspondences(condition_by_id(CONDITIONS[0])))
    independent = np.sqrt(np.mean(np.sum(((truth-np.asarray(CANONICAL_CORNERS_NORMALIZED))*255.5)**2, axis=1)))
    assert math.isclose(row["identity_baseline_rmse_px"], independent)
    assert math.isclose(row["historical_corner_rmse_px"], .003*255.5*math.sqrt(2))
    summary = summarize([row])["groups"][0]
    assert summary["denominator"] == 8 and summary["errors"] == 1
    assert summary["median_corner_rmse_px"] is None


def test_identity_is_matrix_property_not_geometry_status():
    assert is_identity(np.eye(3)*2) is True
    assert is_identity([[1,0,.1],[0,1,0],[0,0,1]]) is False
    assert is_identity(None) is None
