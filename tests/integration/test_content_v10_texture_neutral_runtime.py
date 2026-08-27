import ast, importlib.util, types, unittest
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

 def test_v10_pair_invokes_private_seam_with_neutral_allocator(self):
  if importlib.util.find_spec('torch') is None: self.skipTest('bundled runtime has no torch; production behavior requires its real tensor dependency')
  from cegwm.runtime import content_v10_texture_neutral_sd35 as runtime
  fake_assets=object(); plain=object(); image=object(); summary=object(); captured={}
  neutral=types.SimpleNamespace(allocation="neutral-v3",texture_summary=summary)
  def pass2(pipeline,prompt,key,assets,beta,**kwargs):
   captured.update(kwargs); value=kwargs['allocation_factory'](types.SimpleNamespace(texture_complexity=(1.0,)*16)); self.assertEqual(value,"neutral-v3"); return image,"measurement"
  with mock.patch.object(runtime,'ContentV6EvaluationAssets',object), mock.patch.object(runtime,'run_sd35_plain',return_value=plain), mock.patch.object(runtime,'require_ordinary_rgb_image',side_effect=lambda x:x), mock.patch.object(runtime,'_generator',side_effect=lambda seed:('generator',seed)), mock.patch.object(runtime,'content_v6_h',return_value=.1), mock.patch.object(runtime,'iss_beta',return_value=1.0), mock.patch.object(runtime,'_run_content_v6_pass2',side_effect=pass2), mock.patch('cegwm.method.content_v10_texture_neutral.allocate_texture_neutral',return_value=neutral):
   result=runtime.run_content_v10_evaluation_pair(object(),'prompt',b'key',fake_assets,height=512,width=512,seed=7)
  self.assertIs(result.image,image); self.assertIs(result.primary_null,plain); self.assertIs(result.texture_summary,summary); self.assertEqual(captured['generator'],('generator',7))
if __name__=="__main__": unittest.main()
