import hashlib, importlib.util, io, os, tempfile, unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT=Path(__file__).parents[2]
RUNNER=None
if importlib.util.find_spec('torch') is not None:
 SPEC=importlib.util.spec_from_file_location('v10_calibration_runner',ROOT/'experiments/run_content_v10_calibration.py')
 RUNNER=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(RUNNER)

@unittest.skipUnless(RUNNER is not None,'bundled runtime has no torch; calibration production boundary requires its real tensor dependency')
class ContentV10CalibrationRunnerTests(unittest.TestCase):
 def test_payload_sidecar_is_create_only_and_v10_loader_accepts(self):
  fit=SimpleNamespace(mu_lf=.1,sigma_lf=.2,mu_hf=.3,sigma_hf=.4,rho=.05)
  payload=RUNNER._asset_payload('a'*40,fit)
  with tempfile.TemporaryDirectory() as directory:
   asset=Path(directory)/RUNNER.ASSET_FILENAME; sidecar=asset.with_name(asset.name+'.sha256')
   digest=RUNNER._publish(asset,sidecar,payload)
   self.assertEqual(digest,hashlib.sha256(payload).hexdigest())
   from cegwm.method.content_v10_texture_neutral import load_independent_calibration_asset
   self.assertEqual(load_independent_calibration_asset(asset,sidecar).rho,.05)
   with self.assertRaises(FileExistsError): RUNNER._publish(asset,sidecar,payload)

 def test_incomplete_maps_to_one_sanitized_summary_without_asset(self):
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory)/'repo'; root.mkdir(); sink=Path(directory)/'sink'; local=Path(directory)/'local'
   args=SimpleNamespace(repo_root=str(root),expected_exact='a'*40,artifact_sink=str(sink),local_work_root=str(local))
   stream=io.StringIO()
   with mock.patch.object(RUNNER,'_git_exact',return_value='a'*40), mock.patch.object(RUNNER,'load_content_v10_contract',side_effect=RuntimeError('identity failure')), redirect_stdout(stream):
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
