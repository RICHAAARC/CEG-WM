import ast, json, unittest
from pathlib import Path

NOTEBOOK=Path(__file__).parents[2]/'notebooks/content_v10_calibration_colab.ipynb'
EXACT='15f8d743b7a88e18eaf9e0826f33bbd1bc23231f'

class ContentV10CalibrationNotebookTests(unittest.TestCase):
 def test_thin_handoff_contract(self):
  value=json.loads(NOTEBOOK.read_text(encoding='utf-8')); cells=value['cells']; code=[cell for cell in cells if cell['cell_type']=='code']
  self.assertEqual(code[0]['source'],["from google.colab import drive\n","drive.mount('/content/drive')\n"])
  self.assertTrue(all(cell['execution_count'] is None and cell['outputs']==[] for cell in code))
  source='\n'.join(''.join(cell['source']) for cell in code); ast.parse(source)
  self.assertIn(EXACT,source); self.assertIn("Content-V10-Calibration-{EXPECTED_EXACT[:7]}-{RUN_UTC}",source)
  self.assertIn("drive.mount('/content/drive')",source); self.assertNotIn('force_remount',source)
  self.assertEqual(source.count('subprocess.Popen('),1); self.assertIn("experiments.run_content_v10_calibration",source)
  for flag in ('--repo-root','--expected-exact','--local-work-root','--artifact-sink'):
   self.assertIn(flag,source)
  self.assertIn("CEGWM_CONTENT_V10_CALIBRATION_SUMMARY",source); self.assertIn("CEGWM_CONTENT_V10_CALIBRATION_ARTIFACT",source); self.assertIn("CEGWM_CONTENT_V10_CALIBRATION_INCOMPLETE",source)
  self.assertIn("summary_count!=1",source); self.assertIn("'status']=='complete'",source); self.assertIn("'status']=='incomplete'",source)
  self.assertIn("child_env.pop('CEG_WM_ROOT_KEY'",source); self.assertIn("child_env.pop('HF_TOKEN'",source)
  self.assertNotIn('zipfile',source); self.assertNotIn('run_content_v9',source)
if __name__=='__main__': unittest.main()
