import ast
import json
import subprocess
import tempfile
import unittest
import uuid
from pathlib import Path
P = Path(__file__).parents[2] / "notebooks/content_v10_n96_colab.ipynb"
EXECUTION_EXACT = "60fedf82c14f14a196dfb350a7e743a59f870d09"
HANDOFF_EXACT = "5fe7b2fce42e057b85d7502b1d16eb32a08b6e3f"
class Test(unittest.TestCase):
 def test_contract(self):
  n=json.loads(P.read_text()); c=[x for x in n['cells'] if x['cell_type']=='code']; s=''.join(q for x in c for q in x['source']); ast.parse(s)
  self.assertEqual(c[0]['source'],["from google.colab import drive\n","drive.mount('/content/drive')\n"]); self.assertTrue(all(x['execution_count'] is None and x['outputs']==[] for x in c))
  self.assertIn(EXECUTION_EXACT,s); self.assertIn('ATTEMPT_NONCE=uuid.uuid4().hex[:12]',s); self.assertIn('cegwm-content-v10-source-{ATTEMPT_NONCE}',s); self.assertIn('Content-V10-60fedf8-local-{ATTEMPT_NONCE}',s); self.assertIn('Content-V10-60fedf8-{RUN_UTC}',s); self.assertIn('%Y%m%dT%H%M%S%fZ',s)
  self.assertIn("HANDOFF_HEAD=git('rev-parse','HEAD')",s); self.assertIn("merge-base','--is-ancestor',EXPECTED_EXACT,HANDOFF_HEAD",s); self.assertIn('experiments.run_content_v10_n96',s); self.assertEqual(s.count('subprocess.Popen('),1)
  self.assertIn("CEGWM_CONTENT_V10_ARTIFACT",s); self.assertIn("CEGWM_CONTENT_V10_INCOMPLETE",s); self.assertIn("git('checkout','--detach',EXPECTED_EXACT)",s); self.assertIn("git('branch','--show-current') != BRANCH",s); self.assertIn("git('branch','--show-current') != ''",s); self.assertIn('while True:',s); self.assertIn('if not DRIVE_TARGET.exists(): break',s); self.assertNotIn('force_remount',s)

 def test_execution_ancestor_is_accepted_and_unrelated_exact_is_rejected(self):
  root=Path(__file__).parents[2]
  def is_ancestor(ancestor,descendant):
   return subprocess.run(['git','-c',f'safe.directory={root}','merge-base','--is-ancestor',ancestor,descendant],cwd=root,check=False).returncode==0
  self.assertTrue(is_ancestor(EXECUTION_EXACT,HANDOFF_EXACT))
  self.assertFalse(is_ancestor('7917a7da15fbeee79083b4938362d2bdf202a740',HANDOFF_EXACT))

 def test_nonce_paths_and_drive_retry_are_create_only(self):
  nonce_one,nonce_two=uuid.uuid4().hex[:12],uuid.uuid4().hex[:12]
  self.assertNotEqual(nonce_one,nonce_two)
  self.assertNotEqual(f'/content/cegwm-content-v10-source-{nonce_one}',f'/content/cegwm-content-v10-source-{nonce_two}')
  self.assertNotEqual(f'/content/Content-V10-60fedf8-local-{nonce_one}',f'/content/Content-V10-60fedf8-local-{nonce_two}')
  with tempfile.TemporaryDirectory() as temporary:
   root=Path(temporary); stamps=iter(('20260828T010203123456Z','20260828T010203123457Z')); first=root/f'Content-V10-60fedf8-{next(stamps)}'; first.mkdir()
   while True:
    target=root/f'Content-V10-60fedf8-{next(stamps)}'
    if not target.exists(): break
   self.assertNotEqual(first,target); self.assertTrue(first.exists()); self.assertFalse(target.exists())
if __name__=='__main__': unittest.main()
