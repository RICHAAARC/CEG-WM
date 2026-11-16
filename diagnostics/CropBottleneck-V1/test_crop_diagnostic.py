"""CPU engineering checks only; synthetic images, no models or watermark runs."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import nbformat
import numpy as np
from PIL import Image
import pytest

HERE = Path(__file__).resolve().parent
REPO = next(p for p in (HERE.parents[1], HERE.parents[1] / "alive" / "CEG-WM")
            if (p / "src" / "cegwm").is_dir())
sys.path[:0] = [str(HERE), str(REPO), str(REPO / "src")]
import run_crop_diagnostic as d
from cegwm.geometry_v7.r1b import _pixel_output_to_source, rectify_attacked_rgb
from cegwm.formal_experiment_v2 import apply_attack


def test_oracle_sampler_matches_crop_inverse_and_pixel_center_mapping():
    g = d.crop_geometry()
    assert g["crop_box"] == [27, 27, 485, 485]
    a, b, c, e, f, h, p, q = _pixel_output_to_source(g["sampler_h_observed_to_canonical"])
    assert (a, f, c, h) == pytest.approx((512 / 458, 512 / 458, -27 * 512 / 458, -27 * 512 / 458))
    assert (b, e, p, q) == pytest.approx((0, 0, 0, 0))
    # Independent pixel-center equation for points throughout the observed image.
    physical = np.asarray(g["physical_h_observed_to_canonical"])
    for observed in (0, 30, 255.5, 511):
        r = physical @ np.array([2 * observed / 511 - 1, 0, 1])
        actual = (r[0] / r[2] + 1) * 511 / 2
        assert actual == pytest.approx(27 + (observed + .5) * 458 / 512 - .5)


def test_oracle_restores_support_without_reintroducing_deleted_pixels():
    arr = np.full((512, 512, 3), 100, dtype=np.uint8)
    arr[:, :27] = (255, 0, 0)
    arr[:, 485:] = (0, 255, 0)
    attacked = apply_attack(Image.fromarray(arr), "center_crop_80_restore")
    recovered = np.asarray(rectify_attacked_rgb(attacked, d.crop_geometry()["sampler_h_observed_to_canonical"]))
    expected = np.zeros_like(arr)
    expected[27:485, 27:485] = 100
    np.testing.assert_array_equal(recovered, expected)


def test_oracle_preserves_interior_ramp_position():
    # Pixel values are a coordinate ramp; test actual PIL resize+transform.
    ramp = np.tile((np.arange(512) / 2).astype(np.uint8), (512, 1))
    rgb = np.stack((ramp, ramp.T, np.full_like(ramp, 80)), axis=-1)
    attacked = apply_attack(Image.fromarray(rgb), "center_crop_80_restore")
    recovered = np.asarray(rectify_attacked_rgb(attacked, d.crop_geometry()["sampler_h_observed_to_canonical"]))
    assert np.max(np.abs(recovered[32:480, 32:480].astype(int) - rgb[32:480, 32:480])) <= 1


def test_identity_change_and_corrupted_saved_file_are_rejected(tmp_path):
    d.bind(tmp_path / "identity.json", {"value": 1})
    with pytest.raises(RuntimeError, match="identity differs"):
        d.bind(tmp_path / "identity.json", {"value": 2})
    record = d.save_image(tmp_path / "image.png", Image.new("RGB", (4, 4)))
    (tmp_path / "image.png").write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="hash differs"):
        d.verify_file(record)


def test_failure_details_redact_secrets(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "synthetic-token")
    monkeypatch.setenv("CEG_WM_ROOT_KEY", "synthetic-key")
    public = d.public_error(RuntimeError("request synthetic-token key synthetic-key failed"))
    assert "synthetic-token" not in json.dumps(public)
    assert "synthetic-key" not in json.dumps(public)
    assert "failed" in public["traceback_redacted"]


def test_notebook_is_self_contained_and_does_not_invoke_formal_worker():
    nb = nbformat.read(HERE / "CEG_WM_crop_bottleneck_v1.ipynb", as_version=4)
    nbformat.validate(nb)
    texts = [c.source for c in nb.cells if c.cell_type == "code"]
    for n, source in enumerate(texts):
        compile(source, f"cell-{n}", "exec")
    class Capture:
        def write_text(self, text, encoding):
            self.text = text
    capture = Capture()
    embedded = next(s for s in texts if s.startswith("# Self-contained"))
    exec(embedded, {"script": capture})
    assert capture.text == (HERE / "run_crop_diagnostic.py").read_text()
    assert "experiments.run_paper_v2" not in "\n".join(texts)
    assert "--mode', 'formal" not in "\n".join(texts)


def simulated_runtime(monkeypatch, tmp_path, invalid_geometry=False):
    import torch
    import experiments.run_paper_main_worker_v2 as worker
    from cegwm.geometry_v7.contracts import GeometryEstimate, GeometryStatus
    import cegwm.formal_experiment_v2 as formal
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "synthetic-no-GPU")
    monkeypatch.setattr(d, "version", lambda _: "synthetic")
    monkeypatch.setattr(d.subprocess, "check_output", lambda args, **kw: d.EXACT if "rev-parse" in args else "")
    monkeypatch.setenv("CEG_WM_ROOT_KEY", "00" * 32)
    # The real roster/config parser remains active. No model is loaded.
    if invalid_geometry:
        geometry = GeometryEstimate(GeometryStatus.UNSUPPORTED, None, None, None, None, False, False, "synthetic missing H")
    else:
        geometry = GeometryEstimate(GeometryStatus.RELIABLE, 1., None, None,
                                    ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)), True, True)
    backend = SimpleNamespace(detect_geometry_continuous=lambda _: geometry)
    def runtime(root):
        p = root / "syncseal-test" / "syncmodel.jit.pt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"synthetic checkpoint, never loaded")
        return {"assets": SimpleNamespace(geometry_backend=backend), "key": b"test"}
    monkeypatch.setattr(worker, "_build_runtime", runtime)
    generation_calls = []
    def pair(runtime, prompt, seed):
        generation_calls.append(seed)
        return Image.new("RGB", (512, 512), (80, 80, 80)), Image.new("RGB", (512, 512), (100, 100, 100))
    monkeypatch.setattr(worker, "_main_pair", pair)
    score_calls = []
    def score(runtime, image):
        score_calls.append(1)
        return {"margin_to_reference_tau": -1., "above_reference_tau": False}
    monkeypatch.setattr(d, "score_image", score)
    root = tmp_path / "output"
    d.write(root / "model_revisions.json", {m: "synthetic" for m in d.MODELS})
    return root, generation_calls, score_calls, score


def test_interrupted_scoring_resumes_without_regenerating_completed_pairs(monkeypatch, tmp_path):
    root, generations, scores, score = simulated_runtime(monkeypatch, tmp_path)
    def fail_third(runtime, image):
        if len(scores) == 2:
            scores.append("failed")
            raise RuntimeError("synthetic interruption")
        return score(runtime, image)
    monkeypatch.setattr(d, "score_image", fail_third)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        d.execute(root, REPO, tmp_path / "runtime")
    partial = d.read(root / "summary.json")
    assert len(partial["records"]) == 64
    assert sum(r["status"] == "SCORED" for r in partial["records"]) == 2
    monkeypatch.setattr(d, "score_image", score)
    d.execute(root, REPO, tmp_path / "runtime")
    assert len(generations) == 8
    assert len(scores) == 65  # 64 successful, one failed; no successful score rerun.
    final = d.read(root / "summary.json")
    assert final["status"] == "COMPLETE_DIAGNOSTIC"
    assert all(r["status"] == "SCORED" for r in final["records"])
    d.execute(root, REPO, tmp_path / "runtime")
    assert len(generations) == 8 and len(scores) == 65
    assert any(d.read(p).get("status") == "OPERATIONAL_ERROR" for p in root.glob("units/*/*/*/attempts/*/state.json"))
    first = final["records"][0]["image"]["path"]
    Path(first).write_bytes(b"corrupted after completion")
    with pytest.raises(RuntimeError, match="hash differs"):
        d.execute(root, REPO, tmp_path / "runtime")


def test_invalid_predictions_keep_all_denominators_and_other_routes(monkeypatch, tmp_path):
    root, generations, scores, _ = simulated_runtime(monkeypatch, tmp_path, invalid_geometry=True)
    d.execute(root, REPO, tmp_path / "runtime")
    final = d.read(root / "summary.json")
    assert len(final["records"]) == 64
    assert sum(r["status"] == "NO_VALID_H" for r in final["records"]) == 16
    assert len(scores) == 48 and len(generations) == 8
    assert all(c["planned"] == 8 for c in final["counts"].values())
