import hashlib, importlib.util, io, os, sys, tempfile, unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT=Path(__file__).parents[2]
SPEC=importlib.util.spec_from_file_location('v10_calibration_runner',ROOT/'experiments/run_content_v10_calibration.py')
RUNNER=importlib.util.module_from_spec(SPEC); sys.modules['v10_calibration_runner']=RUNNER; SPEC.loader.exec_module(RUNNER)

class ContentV10CalibrationRunnerTests(unittest.TestCase):
 def test_payload_sidecar_is_create_only_and_v10_loader_accepts(self):
  fit=SimpleNamespace(mu_lf=.1,sigma_lf=.2,mu_hf=.3,sigma_hf=.4,rho=.05)
  payload=RUNNER._asset_payload('a'*40,'b'*64,'c'*64,fit)
  with tempfile.TemporaryDirectory() as directory:
   asset=Path(directory)/'run'/RUNNER.ASSET_FILENAME; sidecar=asset.with_name(asset.name+'.sha256')
   digest=RUNNER._publish(asset,sidecar,payload)
   self.assertEqual(digest,hashlib.sha256(payload).hexdigest())
   method_spec=importlib.util.spec_from_file_location('v10_loader',ROOT/'src/cegwm/method/content_v10_texture_neutral.py'); method=importlib.util.module_from_spec(method_spec); sys.modules['v10_loader']=method; method_spec.loader.exec_module(method)
   self.assertEqual(method.load_independent_calibration_asset(asset,sidecar,producer_execution_exact='a'*40,protocol_digest='b'*64,calibration_public_key_digest='c'*64).rho,.05)
   with self.assertRaises(FileExistsError): RUNNER._publish(asset,sidecar,payload)

 def test_stage_validation_failure_cleans_without_final_pair(self):
  with tempfile.TemporaryDirectory() as directory:
   local=Path(directory)/'local'; local.mkdir(); run=local/'run'; run.mkdir(); payload=b'{}'
   with self.assertRaises(RuntimeError): RUNNER._stage_and_validate(run,payload,'a'*40,'b'*64,'c'*64,mock.Mock(side_effect=RuntimeError('readback')))
   self.assertFalse((run/'staging').exists())

 def test_final_publication_failures_clean_only_owned_run_directory(self):
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory); stage=root/'stage'; stage.mkdir(); asset=stage/RUNNER.ASSET_FILENAME; sidecar=stage/(RUNNER.ASSET_FILENAME+'.sha256'); asset.write_bytes(b'asset'); sidecar.write_bytes(b'sidecar')
   final=root/'sink'/'run'/RUNNER.ASSET_FILENAME; final_sidecar=final.with_name(final.name+'.sha256')
   with mock.patch.object(Path,'open',side_effect=OSError('first write')):
    with self.assertRaises(OSError): RUNNER._publish_staged(asset,sidecar,final,final_sidecar)
   self.assertFalse(final.parent.exists())
   original=Path.open; calls=[]
   def second(path,*args,**kwargs):
    calls.append(path)
    if len(calls)==2: raise OSError('second write')
    return original(path,*args,**kwargs)
   with mock.patch.object(Path,'open',new=second):
    with self.assertRaises(OSError): RUNNER._publish_staged(asset,sidecar,final,final_sidecar)
   self.assertFalse(final.parent.exists())
   kept=root/'sink'/'kept'; kept.mkdir(parents=True); protected=kept/RUNNER.ASSET_FILENAME; protected.write_bytes(b'old')
   with self.assertRaises(FileExistsError): RUNNER._publish_staged(asset,sidecar,protected,protected.with_name(protected.name+'.sha256'))
   self.assertEqual(protected.read_bytes(),b'old')

 def test_incomplete_maps_to_one_sanitized_summary_without_asset(self):
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory)/'repo'; root.mkdir(); sink=Path(directory)/'sink'; local=Path(directory)/'local'
   args=SimpleNamespace(repo_root=str(root),expected_exact='a'*40,artifact_sink=str(sink),local_work_root=str(local))
   stream=io.StringIO()
   os.environ[RUNNER.KEY_ENV]='root'; os.environ[RUNNER.TOKEN_ENV]='token'
   def identity(*unused):
    self.assertNotIn(RUNNER.KEY_ENV,os.environ); self.assertNotIn(RUNNER.TOKEN_ENV,os.environ); raise RuntimeError('identity failure')
   with mock.patch.object(RUNNER,'_git_exact',side_effect=identity), redirect_stdout(stream):
    self.assertEqual(RUNNER.execute(args),2)
   lines=[line for line in stream.getvalue().splitlines() if line]
   self.assertEqual(len(lines),1); self.assertTrue(lines[0].startswith(RUNNER.PREFIX+' '))
   self.assertIn('"completeness":"incomplete"',lines[0]); self.assertIn('"asset_path":null',lines[0])
   self.assertNotIn('identity failure',lines[0]); self.assertNotIn(RUNNER.KEY_ENV,os.environ)

class ContentV10CalibrationRunnerStaticTests(unittest.TestCase):
 def test_cli_contract_and_real_runtime_wiring_are_explicit(self):
  source=(ROOT/'experiments/run_content_v10_calibration.py').read_text(encoding='utf-8')
  self.assertIn('run_content_v10_calibration_unit',source); self.assertNotIn('run_content_v9_calibration_unit',source)
  for flag in ('--repo-root','--expected-exact','--local-work-root','--artifact-sink'):
   self.assertIn(flag,source)
if __name__=='__main__': unittest.main()
