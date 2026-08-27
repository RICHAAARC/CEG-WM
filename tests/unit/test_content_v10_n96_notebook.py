import ast,json,unittest
from pathlib import Path
P=Path(__file__).parents[2]/'notebooks/content_v10_n96_colab.ipynb'
class Test(unittest.TestCase):
 def test_contract(self):
  n=json.loads(P.read_text()); c=[x for x in n['cells'] if x['cell_type']=='code']; s=''.join(q for x in c for q in x['source']); ast.parse(s)
  self.assertEqual(c[0]['source'],["from google.colab import drive\n","drive.mount('/content/drive')\n"]); self.assertTrue(all(x['execution_count'] is None and x['outputs']==[] for x in c))
  self.assertIn('60fedf82c14f14a196dfb350a7e743a59f870d09',s); self.assertIn('Content-V10-60fedf8-',s); self.assertIn('experiments.run_content_v10_n96',s); self.assertEqual(s.count('subprocess.Popen('),1)
  self.assertIn("CEGWM_CONTENT_V10_ARTIFACT",s); self.assertIn("CEGWM_CONTENT_V10_INCOMPLETE",s); self.assertIn("git('checkout','--detach',EXPECTED_EXACT)",s); self.assertNotIn('force_remount',s)
if __name__=='__main__': unittest.main()
