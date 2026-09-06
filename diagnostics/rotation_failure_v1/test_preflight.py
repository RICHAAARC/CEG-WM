import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import preflight as p


class PreflightTests(unittest.TestCase):
    def test_input_fixed_without_external_images(self):
        a,b = p.synthetic_input(),p.synthetic_input()
        self.assertEqual(a.mode,"RGB")
        self.assertEqual(a.size,(512,512))
        np.testing.assert_array_equal(a,b)

    def test_negative_and_unsupported_geometry_are_not_performance_failures(self):
        row={"reference_score":-10.,"pre_score":-20.,"oracle_post_score":-30.,
             "production":{"method_complete":True,"positive":False},
             "geometry_record":{"status":"UNSUPPORTED"},"predicted_H":None}
        self.assertEqual(p.assess(row),[])
        row["geometry_record"]["status"]="ERROR"
        self.assertTrue(p.assess(row))

    def test_missing_legal_post_is_failure(self):
        row={"reference_score":-10.,"pre_score":-20.,"oracle_post_score":-30.,
             "production":{"method_complete":True},"geometry_record":{"status":"RELIABLE"},
             "predicted_H":[[1,0,0],[0,1,0],[0,0,1]],"syncseal_post_score":None}
        self.assertIn("legal-H post score unavailable",p.assess(row))

    def test_builder_failure_retains_start_and_terminal_record_no_retry(self):
        with tempfile.TemporaryDirectory() as td:
            output=Path(td)/"output"
            with patch.object(p,"FrozenRuntime",side_effect=RuntimeError("synthetic failure")) as factory:
                result=p.execute(output,lambda:factory())
                self.assertEqual(factory.call_count,1)
            self.assertEqual(result["status"],"PREFLIGHT_FAILED")
            self.assertEqual(result["science_denominator"],0)
            self.assertTrue((output/"started.json").exists())
            self.assertTrue((output/"result.json").exists())
            with self.assertRaises(FileExistsError):
                p.execute(output,lambda:None)


if __name__=="__main__":
    unittest.main()
