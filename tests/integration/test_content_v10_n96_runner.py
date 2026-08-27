import ast, unittest
from pathlib import Path
class ContentV10N96RunnerTests(unittest.TestCase):
 def test_combined_runner_binds_c1_then_paired_n96_without_v9_fitter(self):
  source=(Path(__file__).parents[2]/'experiments/run_content_v10_n96.py').read_text(); ast.parse(source)
  self.assertIn('produce_calibration_payload',source); self.assertIn('run_content_v10_paired_evaluation',source)
  self.assertNotIn('fit_weighted_joint_calibration',source); self.assertIn('len(rows)!=96',source)
  self.assertIn('63c17e8200a92383b061541fc234dfef36e4b7356954c160ce5f048f820cde96',source)
  self.assertIn('CEGWM_CONTENT_V10_N96_PAIRED_SUMMARY',source)
  self.assertIn('def _spearman',source); self.assertIn('len(texture)!=96',source); self.assertIn('failure_ledger',source)

 def test_statistics_fixed96_ties_and_constant_fail_closed(self):
  import importlib.util
  spec=importlib.util.spec_from_file_location('n96_runner',Path(__file__).parents[2]/'experiments/run_content_v10_n96.py'); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
  self.assertAlmostEqual(module._spearman(list(range(96)),list(range(96))),1.0)
  self.assertAlmostEqual(module._spearman(list(range(96)),list(reversed(range(96)))),-1.0)
  with self.assertRaises(ValueError): module._spearman([1.0]*96,list(range(96)))
  with self.assertRaises(ValueError): module._spearman(list(range(95)),list(range(95)))

 def test_c1_acceptance_state_never_exposes_unaccepted_provenance(self):
  import importlib.util
  spec=importlib.util.spec_from_file_location('n96_runner_state',Path(__file__).parents[2]/'experiments/run_content_v10_n96.py'); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
  pending=module._c1_state(False,Path('asset'),Path('sidecar'),'a'*64); self.assertEqual(pending['status'],'incomplete_not_accepted'); self.assertIsNone(pending['asset_path'])
  accepted=module._c1_state(True,Path('asset'),Path('sidecar'),'a'*64); self.assertEqual(accepted['status'],'calibration_complete'); self.assertEqual(accepted['asset_sha256'],'a'*64)
  source=(Path(__file__).parents[2]/'experiments/run_content_v10_n96.py').read_text(); self.assertLess(source.index('c1_accepted=True; phase="n96"'),source.index('v9_path='))

 def test_final_readback_rejection_rolls_back_only_published_pair(self):
  import importlib.util, tempfile
  spec=importlib.util.spec_from_file_location('n96_runner_accept',Path(__file__).parents[2]/'experiments/run_content_v10_n96.py'); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
  with tempfile.TemporaryDirectory() as directory:
   asset=Path(directory)/'run'/'asset.json'; asset.parent.mkdir(); asset.write_bytes(b'x'); sidecar=asset.with_name('asset.json.sha256'); sidecar.write_bytes(b'x')
   with self.assertRaises(RuntimeError): module._accept_published_c1(asset,sidecar,exact='a'*40,protocol_digest='b'*64,public_digest='c'*64,loader=lambda *args,**kwargs: (_ for _ in ()).throw(RuntimeError('reject')))
   self.assertFalse(asset.parent.exists())
  source=(Path(__file__).parents[2]/'experiments/run_content_v10_n96.py').read_text(); self.assertLess(source.index('n96_contract=load_content_v10_n96_paired_contract(root)'),source.index('v9_path='))
if __name__=='__main__': unittest.main()
