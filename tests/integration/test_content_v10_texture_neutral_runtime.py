import ast, importlib.util, types, unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

class ContentV10RuntimeTests(unittest.TestCase):
 def test_v10_routes_private_v6_seam_and_public_v6_stays_unselectable(self):
  root=Path(__file__).parents[2]; v10=(root/'src/cegwm/runtime/content_v10_texture_neutral_sd35.py').read_text(); v6=(root/'src/cegwm/runtime/content_iss_sd35_v6.py').read_text()
  self.assertIn('_run_content_v6_pass2',v10); self.assertIn('allocation_factory=allocator',v10); self.assertIn('allocate_texture_neutral',v10)
  tree=ast.parse(v6); public=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_content_v6_evaluation_pair'); self.assertNotIn('allocation_factory',[a.arg for a in public.args.args+public.args.kwonlyargs])
  callback=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ContentV6InjectionCallback'); init=next(n for n in callback.body if isinstance(n,ast.FunctionDef) and n.name=='__init__'); self.assertNotIn('allocation_factory',[a.arg for a in init.args.args+init.args.kwonlyargs])
  self.assertIn('_ContentV6PrivateInjectionCallback',v6)
  self.assertIn('step-18 allocation did not occur exactly once',v10)
  v9=(root/'src/cegwm/runtime/content_weighted_joint_sd35_v9.py').read_text(); self.assertIn('run_content_v6_evaluation_pair',v9); self.assertNotIn('_run_content_v6_pass2',v9)
  self.assertIn('def run_content_v10_paired_evaluation',v10); self.assertIn('allocation_factory=allocate_content',v10)
  self.assertIn('ContentV10PairedArmFailure("v9")',v10); self.assertIn('ContentV10PairedArmFailure("v10")',v10)

 def test_v10_pair_invokes_private_seam_with_neutral_allocator(self):
  if importlib.util.find_spec('torch') is None: self.skipTest('bundled runtime has no torch; production behavior requires its real tensor dependency')
  from cegwm.runtime import content_v10_texture_neutral_sd35 as runtime
  class FakeAssets:
   lf_public_assets=object(); iss_asset=object()
  fake_assets=FakeAssets(); plain=object(); image=object(); captured={}; original={}
  def pass2(pipeline,prompt,key,assets,beta,**kwargs):
   captured.update(kwargs)
   from cegwm.method.content_adaptive_v3 import ContentSignals
   signals=ContentSignals(("semantic",), (1.0,)*16, ("lf-consistency",), ("hf-consistency",), ("lf-sensitivity",), ("hf-sensitivity",))
   original['signals']=signals
   value=kwargs['allocation_factory'](signals)
   self.assertEqual(value,"neutral-v3"); return image,"measurement"
  with mock.patch.object(runtime,'ContentV6EvaluationAssets',FakeAssets), mock.patch.object(runtime,'run_sd35_plain',return_value=plain), mock.patch.object(runtime,'require_ordinary_rgb_image',side_effect=lambda x:x), mock.patch.object(runtime,'_generator',side_effect=lambda seed:('generator',seed)), mock.patch.object(runtime,'content_v6_h',return_value=.1), mock.patch.object(runtime,'iss_beta',return_value=1.0), mock.patch.object(runtime,'_run_content_v6_pass2',side_effect=pass2), mock.patch('cegwm.method.content_adaptive_v3.allocate_content',return_value="neutral-v3") as allocate:
   result=runtime.run_content_v10_evaluation_pair(object(),'prompt',b'key',fake_assets,height=512,width=512,seed=7)
  neutral_signals=allocate.call_args.args[0]; self.assertEqual(neutral_signals,replace(original['signals'],texture_complexity=(0.0,)*16))
  self.assertIs(result.image,image); self.assertIs(result.primary_null,plain); self.assertEqual(captured['generator'],('generator',7))

 def test_calibration_unit_uses_real_v10_pair_and_frozen_33_order(self):
  if importlib.util.find_spec('torch') is None: self.skipTest('bundled runtime has no torch; production behavior requires its real tensor dependency')
  from cegwm.runtime import content_v10_texture_neutral_sd35 as runtime
  class FakeAssets: pass
  unit=types.SimpleNamespace(prompt='prompt',height=512,width=512,seed=9)
  output=runtime.ContentV10RunOutput('candidate','plain','measurement','summary')
  with mock.patch.object(runtime,'ContentV6EvaluationAssets',FakeAssets), mock.patch.object(runtime,'derive_v10_calibration_wrong_keys',return_value=tuple(bytes([index]) for index in range(16))), mock.patch.object(runtime,'run_content_v10_evaluation_pair',return_value=output) as pair, mock.patch.object(runtime,'_blind_calibration_pair',side_effect=lambda image,key,assets:(image,key)) as blind:
   values=runtime.run_content_v10_calibration_unit(object(),unit,b'registered',FakeAssets())
  self.assertEqual(len(values),33); pair.assert_called_once()
  self.assertEqual([value[0] for value in values],['candidate']*16+['plain']*17)
  self.assertEqual([value[1] for value in values],[bytes([index]) for index in range(16)]+[b'registered']+[bytes([index]) for index in range(16)])
  self.assertEqual(blind.call_count,33)

 def test_paired_evaluation_calls_one_plain_beta_and_two_same_seed_passes(self):
  if importlib.util.find_spec('torch') is None: self.skipTest('bundled runtime has no torch; production behavior requires its real tensor dependency')
  from cegwm.runtime import content_v10_texture_neutral_sd35 as runtime
  class FakeAssets: lf_public_assets=object(); iss_asset=object()
  class Image:
   def tobytes(self): return b'plain'
  events=[]; neutral=types.SimpleNamespace(allocation='neutral',texture_summary=types.SimpleNamespace(mean=.5,digest='a'*64))
  def pass2(*args,**kwargs):
   events.append(kwargs)
   if len(events)==2: kwargs['allocation_factory'](types.SimpleNamespace(texture_complexity=(1.0,)*16))
   return object(),f'm{len(events)}'
  with mock.patch.object(runtime,'ContentV6EvaluationAssets',FakeAssets), mock.patch.object(runtime,'run_sd35_plain',return_value=Image()) as plain, mock.patch.object(runtime,'require_ordinary_rgb_image',side_effect=lambda value:value), mock.patch.object(runtime,'_generator',side_effect=lambda seed:('g',seed)), mock.patch.object(runtime,'content_v6_h',return_value=.1) as host, mock.patch.object(runtime,'iss_beta',return_value=.2) as beta, mock.patch.object(runtime,'_run_content_v6_pass2',side_effect=pass2), mock.patch('cegwm.method.content_v10_texture_neutral.allocate_texture_neutral',return_value=neutral):
   result=runtime.run_content_v10_paired_evaluation(object(),'prompt',b'key',FakeAssets(),height=512,width=512,seed=13)
  plain.assert_called_once(); host.assert_called_once(); beta.assert_called_once(); self.assertEqual(len(events),2)
  self.assertEqual([item['generator'] for item in events],[('g',13),('g',13)])
  self.assertEqual(result.texture_raw_digest,'a'*64); self.assertEqual(result.texture_scalar,.5)
if __name__=="__main__": unittest.main()
