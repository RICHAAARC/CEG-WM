"""CPU synthetic checks; none are watermark evidence."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import diagnostic as d

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
apply_attack = d.apply_attack
from cegwm.geometry_v7.r1b import rectify_attacked_rgb
from cegwm.geometry_v7.contracts import GeometryEstimate, GeometryStatus


class Backend:
    attack = staticmethod(apply_attack)
    rectify = staticmethod(rectify_attacked_rgb)
    def score(self, image):
        return 10.
    def geometry(self, image):
        h = d.oracle_geometry(d.CONDITIONS[1])["H_truth_pixel_centers_normalized"]
        return GeometryEstimate(GeometryStatus.RELIABLE, 0., None, None, h, True, True)
    def matrix(self, geometry):
        return geometry.homography_observed_to_canonical
    def detect(self, image):
        return SimpleNamespace(route="GEOMETRY_RECOVERED", positive=False,
            pre=SimpleNamespace(value=-5.), post=SimpleNamespace(value=-4.),
            geometry=self.geometry(image), method_complete=True, operational_error=None)


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (512, 512), (100, 110, 120))

    def test_truth_matches_actual_pillow_attack_coefficients(self):
        observed = []
        original = Image.Image.transform
        def capture(im, size, method, data=None, *args, **kwargs):
            observed.append((im.size, tuple(data)))
            return original(im, size, method, data, *args, **kwargs)
        with patch.object(Image.Image, "transform", capture):
            apply_attack(self.image, d.CONDITIONS[1])
        size, coeff = observed[0]
        a,b,c,e,f,g = coeff
        px,py = (size[0]-512)//2, (size[1]-512)//2
        actual = d.translation(-px,-py) @ np.array([[a,b,c],[e,f,g],[0,0,1]]) @ d.translation(px,py)
        expected = d.oracle_geometry(d.CONDITIONS[1])["pillow_observed_to_original_edge_coordinates"]
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_landmarks_inverse_direction_and_pixel_centers(self):
        yy,xx = np.mgrid[:512,:512]
        for x,y in [(100,120), (390,130), (135,370), (380,390)]:
            blob = np.rint(255*np.exp(-((xx-x)**2+(yy-y)**2)/(2*4**2))).astype("uint8")
            rgb = Image.fromarray(np.repeat(blob[:,:,None],3,axis=2))
            attacked = apply_attack(rgb, d.CONDITIONS[1])
            truth = d.oracle_geometry(d.CONDITIONS[1])
            recovered = rectify_attacked_rgb(attacked, truth["H_oracle_sampler_observed_to_canonical"])
            arr = np.asarray(recovered)[:,:,0].astype(float)
            error = np.hypot((arr*xx).sum()/arr.sum()-x, (arr*yy).sum()/arr.sum()-y)
            self.assertLess(error, .15)
            bad = rectify_attacked_rgb(attacked, np.linalg.inv(truth["H_oracle_sampler_observed_to_canonical"]))
            arr = np.asarray(bad)[:,:,0].astype(float)
            bad_error = np.hypot((arr*xx).sum()/arr.sum()-x, (arr*yy).sum()/arr.sum()-y)
            self.assertGreater(bad_error, 30.)

    def test_identity_oracle_preserves_rgb(self):
        rgb = Image.fromarray(np.random.default_rng(3).integers(0,256,(512,512,3),dtype="uint8"))
        truth = d.oracle_geometry(d.CONDITIONS[0])
        np.testing.assert_array_equal(rgb, rectify_attacked_rgb(rgb, truth["H_oracle_sampler_observed_to_canonical"]))

    def test_half_pixel_conventions_are_explicit(self):
        truth = d.oracle_geometry(d.CONDITIONS[1])
        self.assertFalse(np.allclose(truth["H_truth_pixel_centers_normalized"], truth["H_oracle_sampler_observed_to_canonical"]))
        self.assertGreater(truth["support_mask"].mean(), .7)
        self.assertLess(truth["support_mask"].mean(), 1.)
        self.assertFalse(truth["support_mask"][0,0])

    def test_oracle_positive_cannot_change_production_negative(self):
        row = d.diagnose_image(self.image, d.CONDITIONS[1], Backend())
        self.assertFalse(row["production"]["positive"])
        self.assertEqual(row["production"]["normalized_score"], -4.)
        self.assertEqual(row["oracle_post_score"], 10.)
        self.assertEqual(row["corner_rmse_px"], 0.)
        self.assertEqual(row["diagnostic_adjudication"], "UNADJUDICATED")
        json.dumps(row, allow_nan=False)

    def test_direct_positive_forced_failure_is_only_diagnostic(self):
        class Direct(Backend):
            def detect(self, image):
                return SimpleNamespace(route="DIRECT_POSITIVE", positive=True,
                    pre=SimpleNamespace(value=3.), post=None, geometry=None,
                    method_complete=True, operational_error=None)
            def geometry(self, image):
                raise RuntimeError("synthetic geometry failure")
        row = d.diagnose_image(self.image, d.CONDITIONS[1], Direct())
        self.assertTrue(row["production"]["positive"])
        self.assertTrue(row["forced_post_after_direct_positive"])
        self.assertIsNone(row["syncseal_post_score"])
        self.assertIn("forced_geometry", row["errors"])

    def test_oracle_failure_retains_production_and_missing_score(self):
        class Failure(Backend):
            def rectify(self, image, matrix):
                raise RuntimeError("synthetic sampler failure")
        row = d.diagnose_image(self.image, d.CONDITIONS[1], Failure())
        self.assertEqual(row["syncseal_post_score"], -4.)
        self.assertIsNone(row["oracle_post_score"])
        self.assertIn("oracle_rectification", row["errors"])

    def test_output_is_create_only(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/"row.json"
            d.write_new(path, {"status":"original"})
            with self.assertRaises(FileExistsError):
                d.write_new(path, {"status":"replacement"})
            self.assertEqual(json.loads(path.read_text())["status"], "original")

    def test_all_missing_inputs_remain_400_failed_rows_after_interruption(self):
        with tempfile.TemporaryDirectory() as td:
            root, output = Path(td)/"input", Path(td)/"output"
            root.mkdir()
            entries = json.loads((Path(d.__file__).parent/"input_reference.json").read_text())["entries"]
            (root/"manifest.json").write_text(json.dumps({"entries": entries}))
            sources = {"records": [{"physical_unit_id":e["sample_id"], "condition":c,
                "truth_role":role, "decision":False, "route":"GEOMETRY_RECOVERED", "normalized_score":-1.}
                for e in entries for c in d.CONDITIONS for role in ("negative", "positive")]}
            original_write = d.write_new
            writes = [0]
            def interrupted(path, value):
                writes[0] += 1
                if writes[0] == 16:
                    raise KeyboardInterrupt()
                original_write(path, value)
            # Simulate images disappearing after a successful input audit.
            with patch.object(d, "audit_inputs", return_value={"input_usable":True}), patch.object(d, "load_source_rows", return_value=sources):
                with patch.object(d, "write_new", side_effect=interrupted):
                    with self.assertRaises(KeyboardInterrupt):
                        d.run(root, output, Backend())
                first = next(output.glob("*.json"))
                original_text = first.read_text()
                d.run(root, output, Backend())
            summary = json.loads((output/"summary.json").read_text())
            self.assertEqual(summary["planned_image_condition_rows"], 400)
            self.assertEqual(summary["unit_status_counts"], {"FAILED":400})
            self.assertEqual(first.read_text(), original_text)
            paired = json.loads((output/"paired_differences.json").read_text())
            self.assertEqual(len(paired["rows"]), 200)


if __name__ == "__main__":
    unittest.main()
