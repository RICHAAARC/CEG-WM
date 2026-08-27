import ast, unittest
from pathlib import Path
class ContentV10N96RunnerTests(unittest.TestCase):
 def test_combined_runner_binds_c1_then_paired_n96_without_v9_fitter(self):
  source=(Path(__file__).parents[2]/'experiments/run_content_v10_n96.py').read_text(); ast.parse(source)
  self.assertIn('produce_calibration_payload',source); self.assertIn('run_content_v10_paired_evaluation',source)
  self.assertNotIn('fit_weighted_joint_calibration',source); self.assertIn('len(rows)!=96',source)
  self.assertIn('63c17e8200a92383b061541fc234dfef36e4b7356954c160ce5f048f820cde96',source)
  self.assertIn('CEGWM_CONTENT_V10_N96_PAIRED_SUMMARY',source)
if __name__=='__main__': unittest.main()
