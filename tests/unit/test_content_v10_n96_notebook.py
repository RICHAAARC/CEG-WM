import ast
import json
import pathlib
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
  self.assertIn(EXECUTION_EXACT,s); self.assertIn('ATTEMPT_NONCE=uuid.uuid4().hex[:12]',s); self.assertIn('cegwm-content-v10-source-{ATTEMPT_NONCE}',s); self.assertIn('Content-V10-60fedf8-local-{ATTEMPT_NONCE}',s); self.assertIn('detach_v10_execution_checkout(SOURCE, BRANCH, EXPECTED_EXACT)',s); self.assertIn('allocate_v10_attempt_paths(',s); self.assertIn('experiments.run_content_v10_n96',s); self.assertEqual(s.count('subprocess.Popen('),1)
  self.assertIn("CEGWM_CONTENT_V10_ARTIFACT",s); self.assertIn("CEGWM_CONTENT_V10_INCOMPLETE",s); self.assertNotIn('force_remount',s)

 def test_execution_ancestor_is_accepted_and_unrelated_exact_is_rejected(self):
  detach=self._notebook_helpers()['detach_v10_execution_checkout']
  with tempfile.TemporaryDirectory() as temporary:
   repo=Path(temporary)/'repo'
   def git(*args): return subprocess.run(['git',*args],cwd=repo,check=True,capture_output=True,text=True).stdout.strip()
   subprocess.run(['git','init','-q',str(repo)],check=True); git('config','user.email','v10-test@example.invalid'); git('config','user.name','V10 Test'); git('checkout','-q','-b','Content-V10')
   (repo/'fixture.txt').write_text('ancestor\n'); git('add','fixture.txt'); git('commit','-qm','ancestor'); execution=git('rev-parse','HEAD')
   (repo/'fixture.txt').write_text('handoff\n'); git('commit','-am','handoff','-q'); git('checkout','-q','--orphan','unrelated'); git('rm','-q','-rf','.'); (repo/'unrelated.txt').write_text('unrelated\n'); git('add','unrelated.txt'); git('commit','-qm','unrelated'); unrelated=git('rev-parse','HEAD'); git('checkout','-q','Content-V10')
   detach(repo,'Content-V10',execution); self.assertEqual(git('rev-parse','HEAD'),execution); self.assertEqual(git('branch','--show-current'),''); git('checkout','-q','Content-V10')
   with self.assertRaises(RuntimeError): detach(repo,'Content-V10',unrelated)

 def test_nonce_paths_and_drive_retry_are_create_only(self):
  allocate=self._notebook_helpers()['allocate_v10_attempt_paths']
  with tempfile.TemporaryDirectory() as temporary:
   root=Path(temporary); content,drive=root/'content',root/'drive'; content.mkdir(); drive.mkdir()
   class Clock:
    def __init__(self,values): self.values=iter(values)
    def __call__(self):
     value=next(self.values); return type('Moment',(),{'strftime':lambda self,_format:value})()
   source_one,local_one,first,_=allocate(content,drive,'nonce-one001',Clock(['20260828T010203123456Z'])); first.mkdir()
   source_two,local_two,target,_=allocate(content,drive,'nonce-two002',Clock(['20260828T010203123456Z','20260828T010203123457Z']))
   self.assertNotEqual((source_one,local_one,first),(source_two,local_two,target)); self.assertTrue(first.exists()); self.assertFalse(target.exists())

 def _notebook_helpers(self):
  n=json.loads(P.read_text()); tree=ast.parse(''.join(q for x in n['cells'] if x['cell_type']=='code' for q in x['source']))
  names={'_v10_git','detach_v10_execution_checkout','allocate_v10_attempt_paths'}; helpers=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
  self.assertEqual({node.name for node in helpers},names); namespace={'pathlib':pathlib,'subprocess':subprocess}; exec(compile(ast.Module(body=helpers,type_ignores=[]),'v10-notebook-helpers','exec'),namespace); return namespace
if __name__=='__main__': unittest.main()
